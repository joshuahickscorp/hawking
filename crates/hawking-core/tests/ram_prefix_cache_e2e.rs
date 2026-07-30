#![cfg(target_os = "macos")]
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
const WEIGHTS: &str = "../../models/qwen2.5-3b-instruct-q4_k_m.gguf";
const PREAMBLE: &str = "Here is a Rust file I am working on:\n\nfn fib(n: u64) -> u64 {\n    if n < 2 { n } else { fib(n - 1) + fib(n - 2) }\n}\n\nfn factorial(n: u64) -> u64 {\n    (1..=n).product()\n}\n\nfn main() {\n    for i in 0..10 {\n        println!(\"fib({}) = {}\", i, fib(i));\n    }\n    for i in 0..6 {\n        println!(\"fact({}) = {}\", i, factorial(i));\n    }\n}\n\n";
const TURN1_TAIL: &str = "User: What does this program print?\nAssistant:";
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
fn make_engine(weights: &PathBuf) -> Box<dyn hawking_core::Engine> {
    std::env::set_var("HAWKING_QWEN_TCB", "1");
    let cfg = hawking_core::EngineConfig::default();
    hawking_core::model::load_engine(weights, cfg).expect("load engine")
}
fn gen_on(engine: &mut dyn hawking_core::Engine, prompt: &str) -> (Vec<u32>, f64) {
    let req = hawking_core::GenerateRequest {
        prompt: prompt.into(),
        max_new_tokens: MAX_NEW_TOKENS,
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
    let mut prefill_ms = 0.0f64;
    engine
        .generate(req, &mut |ev| match ev {
            hawking_core::StreamEvent::Token { id, .. } => ids.push(id),
            hawking_core::StreamEvent::Done { stats, .. } => prefill_ms = stats.prefill_ms,
        })
        .expect("generate");
    (ids, prefill_ms)
}
#[test]
fn ram_cache_hit_is_bit_identical() {
    let Some(weights) = weights_path() else {
        return;
    };
    let _g = SERIAL_GATE.get_or_init(|| Mutex::new(())).lock().unwrap();
    let prompt1 = format!("{PREAMBLE}{TURN1_TAIL}");
    let prompt2 = format!("{PREAMBLE}{TURN2_TAIL}");
    std::env::set_var("HAWKING_QWEN_PREFIX_CACHE", "0");
    std::env::remove_var("HAWKING_PREFIX_CACHE_DIR");
    let (ref_ids, off_ms) = {
        let mut e = make_engine(&weights);
        let _ = gen_on(e.as_mut(), &prompt1);
        gen_on(e.as_mut(), &prompt2)
    };
    std::env::set_var("HAWKING_QWEN_PREFIX_CACHE", "1");
    let (hit_ids, on_ms) = {
        let mut e = make_engine(&weights);
        let _ = gen_on(e.as_mut(), &prompt1); // populates the RAM prefix cache
        gen_on(e.as_mut(), &prompt2) // turn 2 extends turn 1 → HIT
    };
    std::env::set_var("HAWKING_QWEN_PREFIX_CACHE", "0");
    assert_eq!(hit_ids.len(), MAX_NEW_TOKENS);
    // THE GATE: bit-identical reuse.
    assert_eq!(ref_ids, hit_ids, "GATE FAILED: RAM prefix-cache hit changed greedy output.\n off={ref_ids:?}\n  on={hit_ids:?}");
    assert!(
        on_ms < 0.85 * off_ms,
        "expected prefix-cache HIT to cut turn-2 prefill >15%; got on={on_ms:.1} vs off={off_ms:.1} \
         (no hit ⇒ check lookup/extends-prefix scenario)"
    );
}
#[test]
fn ram_cache_miss_is_bit_identical() {
    let Some(weights) = weights_path() else {
        return;
    };
    let _g = SERIAL_GATE.get_or_init(|| Mutex::new(())).lock().unwrap();
    let prompt_a = format!("Apples are red. {TURN1_TAIL}");
    let prompt_b = format!("Bananas are yellow. {TURN1_TAIL}");
    std::env::set_var("HAWKING_QWEN_PREFIX_CACHE", "0");
    std::env::remove_var("HAWKING_PREFIX_CACHE_DIR");
    let ref_ids = {
        let mut e = make_engine(&weights);
        let _ = gen_on(e.as_mut(), &prompt_a);
        gen_on(e.as_mut(), &prompt_b).0
    };
    std::env::set_var("HAWKING_QWEN_PREFIX_CACHE", "1");
    let miss_ids = {
        let mut e = make_engine(&weights);
        let _ = gen_on(e.as_mut(), &prompt_a); // unrelated prefix in cache
        gen_on(e.as_mut(), &prompt_b).0 // no shared prefix → miss
    };
    std::env::set_var("HAWKING_QWEN_PREFIX_CACHE", "0");
    assert_eq!(
        ref_ids, miss_ids,
        "cache-miss path must match the no-cache path on the same engine"
    );
}
