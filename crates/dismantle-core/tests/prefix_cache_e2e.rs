//! End-to-end prefix-cache parity + bench on the real Qwen-3B model.
//!
//! Requires `models/qwen2.5-3b-instruct-q4_k_m.gguf`. Skips silently
//! when missing so the test suite stays green on CI without weights.
//!
//! ## Gates
//!
//! 1. **Cache MISS parity (cache disabled)** — Generate N tokens with
//!    the env var unset, then with a freshly-empty cache. The decode
//!    loop must produce identical tokens.
//!
//! 2. **Cache HIT parity** — Turn 1 populates the cache for the system
//!    prefix. Turn 2 (system + new user message) generates with the
//!    cache hit active and must produce the same tokens as a no-cache
//!    run of the same turn-2 prompt.
//!
//! ## Bench
//!
//! Final test reports prefill_ms turn 1 (cold), turn 2 (warm), turn 3
//! (warm) to stderr. Not a pass/fail gate — informational only.

#![cfg(target_os = "macos")]

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

const WEIGHTS: &str = "../../models/qwen2.5-3b-instruct-q4_k_m.gguf";
const SYSTEM_PROMPT: &str = "You are a friendly, concise assistant. You answer questions in plain English with no preamble. Stay factual; if you don't know, say so. Do not invent sources. Use Markdown only when asked. Prefer short paragraphs over bullet lists unless the user asks for a list. Avoid emoji. Avoid editorializing about your own answer. Avoid restating the question. Avoid signing off with a closing pleasantry. Avoid hedging language like 'I think' or 'It seems'. Answer directly.";
const USER_TURN_1: &str = "\n\nUser: What is the capital of France?\n\nAssistant:";
const USER_TURN_2: &str = "\n\nUser: What is the capital of France?\n\nAssistant: Paris.\n\nUser: And the largest city in Germany?\n\nAssistant:";
const USER_TURN_3: &str = "\n\nUser: What is the capital of France?\n\nAssistant: Paris.\n\nUser: And the largest city in Germany?\n\nAssistant: Berlin.\n\nUser: Italy?\n\nAssistant:";

const MAX_NEW_TOKENS: usize = 16;

// Engines are expensive to load. Run all tests in this file serially
// against a single shared model. (We re-load between tests anyway
// because each test wants a fresh KV.)
static SERIAL_GATE: OnceLock<Mutex<()>> = OnceLock::new();

fn weights_path() -> Option<PathBuf> {
    let p = PathBuf::from(WEIGHTS);
    if p.exists() {
        Some(p)
    } else {
        eprintln!("prefix_cache_e2e: skipping — no weights at {}", WEIGHTS);
        None
    }
}

fn run_generate(weights: &PathBuf, prompt: &str) -> (Vec<u32>, f64) {
    let cfg = dismantle_core::EngineConfig::default();
    let mut engine = dismantle_core::model::load_engine(weights, cfg).expect("load engine");
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
            dismantle_core::StreamEvent::Done { stats, .. } => {
                prefill_ms = stats.prefill_ms;
            }
        })
        .expect("generate");
    (ids, prefill_ms)
}

#[test]
fn cache_disabled_baseline_matches_itself() {
    let Some(weights) = weights_path() else { return };
    let _g = SERIAL_GATE.get_or_init(|| Mutex::new(())).lock().unwrap();
    // Isolate the DISK tier: the in-RAM prefix cache is default-ON, so
    // disable it explicitly here (this file tests the disk tier only).
    std::env::set_var("DISMANTLE_QWEN_PREFIX_CACHE", "0");
    std::env::remove_var("DISMANTLE_PREFIX_CACHE_DIR");
    let prompt = format!("{SYSTEM_PROMPT}{USER_TURN_1}");
    let (a, _) = run_generate(&weights, &prompt);
    let (b, _) = run_generate(&weights, &prompt);
    assert_eq!(a, b, "deterministic greedy must be self-consistent");
    assert_eq!(a.len(), MAX_NEW_TOKENS);
}

#[test]
fn cache_miss_matches_no_cache() {
    let Some(weights) = weights_path() else { return };
    let _g = SERIAL_GATE.get_or_init(|| Mutex::new(())).lock().unwrap();

    let prompt = format!("{SYSTEM_PROMPT}{USER_TURN_1}");

    // Disk tier only — keep the RAM tier out of this comparison.
    std::env::set_var("DISMANTLE_QWEN_PREFIX_CACHE", "0");
    std::env::remove_var("DISMANTLE_PREFIX_CACHE_DIR");
    let (ids_nocache, _) = run_generate(&weights, &prompt);

    let tmp = tempfile::TempDir::new().unwrap();
    std::env::set_var("DISMANTLE_PREFIX_CACHE_DIR", tmp.path().to_string_lossy().as_ref());
    let (ids_miss, _) = run_generate(&weights, &prompt);
    std::env::remove_var("DISMANTLE_PREFIX_CACHE_DIR");

    assert_eq!(
        ids_nocache, ids_miss,
        "cache-miss path must produce same tokens as no-cache path"
    );
}

#[test]
fn cache_hit_matches_no_cache_and_speeds_prefill() {
    let Some(weights) = weights_path() else { return };
    let _g = SERIAL_GATE.get_or_init(|| Mutex::new(())).lock().unwrap();

    let prompt_t1 = format!("{SYSTEM_PROMPT}{USER_TURN_1}");
    let prompt_t2 = format!("{SYSTEM_PROMPT}{USER_TURN_2}");
    let prompt_t3 = format!("{SYSTEM_PROMPT}{USER_TURN_3}");

    // No-cache reference for turn 2 + turn 3. Disk tier only — the RAM
    // tier is default-ON, so disable it to isolate the disk-cache effect.
    std::env::set_var("DISMANTLE_QWEN_PREFIX_CACHE", "0");
    std::env::remove_var("DISMANTLE_PREFIX_CACHE_DIR");
    let (ids_t2_nocache, t2_nocache_ms) = run_generate(&weights, &prompt_t2);
    let (ids_t3_nocache, t3_nocache_ms) = run_generate(&weights, &prompt_t3);

    // Cache-enabled run: turn 1 populates; turn 2 + turn 3 should hit.
    let tmp = tempfile::TempDir::new().unwrap();
    std::env::set_var("DISMANTLE_PREFIX_CACHE_DIR", tmp.path().to_string_lossy().as_ref());
    let (_ids_t1, t1_warm_ms) = run_generate(&weights, &prompt_t1);
    let (ids_t2_cache, t2_warm_ms) = run_generate(&weights, &prompt_t2);
    let (ids_t3_cache, t3_warm_ms) = run_generate(&weights, &prompt_t3);
    std::env::remove_var("DISMANTLE_PREFIX_CACHE_DIR");

    assert_eq!(
        ids_t2_nocache, ids_t2_cache,
        "turn-2 cache-hit output must match no-cache"
    );
    assert_eq!(
        ids_t3_nocache, ids_t3_cache,
        "turn-3 cache-hit output must match no-cache"
    );

    eprintln!("\n=== prefix-cache prefill timings ===");
    eprintln!(
        "turn 1 (cold, cache populating):  prefill_ms = {:>7.1}",
        t1_warm_ms
    );
    eprintln!(
        "turn 2 no-cache:                  prefill_ms = {:>7.1}",
        t2_nocache_ms
    );
    eprintln!(
        "turn 2 with cache (HIT):          prefill_ms = {:>7.1}  ({:>5.1}% of no-cache)",
        t2_warm_ms,
        100.0 * t2_warm_ms / t2_nocache_ms
    );
    eprintln!(
        "turn 3 no-cache:                  prefill_ms = {:>7.1}",
        t3_nocache_ms
    );
    eprintln!(
        "turn 3 with cache (HIT):          prefill_ms = {:>7.1}  ({:>5.1}% of no-cache)",
        t3_warm_ms,
        100.0 * t3_warm_ms / t3_nocache_ms
    );
    eprintln!("=====================================\n");

    // Sanity check on the speedup. The warm prefill should be at least
    // some not-laughably-bad fraction faster. The theoretical floor is
    // (delta_tokens / total_tokens) × cold_ms. We assert a loose 20%
    // improvement so this test catches a wire-up bug (cache hit but no
    // prefill skipped) without being flaky.
    assert!(
        t2_warm_ms < 0.85 * t2_nocache_ms,
        "turn-2 cache hit should reduce prefill by >15%; got warm={:.1} vs no-cache={:.1}",
        t2_warm_ms,
        t2_nocache_ms
    );
}
