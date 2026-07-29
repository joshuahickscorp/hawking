#![cfg(target_os = "macos")]
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
const WEIGHTS: &str = "../../models/qwen2.5-3b-instruct-q4_k_m.gguf";
const MAX_NEW_TOKENS_CAP: usize = 16;
const N_PROMPTS: usize = 20;
static SERIAL_GATE: OnceLock<Mutex<()>> = OnceLock::new();
const LCG_MUL: u64 = 6364136223846793005;
const LCG_ADD: u64 = 1442695040888963407;
const SEED: u64 = 0xdeadbeef_cafebabe_u64;
#[inline]
fn lcg_step(state: u64) -> u64 {
    state.wrapping_mul(LCG_MUL).wrapping_add(LCG_ADD)
}
mod charsets {
    /// Code-y: identifiers, operators, punctuation (favours [a-zA-Z0-9_:;<=>{}()])
    pub fn code_char(state: u64) -> u8 {
        const CODE: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\
                               0123456789_:;=<>{}()[].,!+-*/&|^~# \n";
        CODE[((state >> 33) as usize) % CODE.len()]
    }
    /// Mixed ASCII: all printable characters (broader distribution).
    pub fn mixed_char(state: u64) -> u8 {
        0x20 + ((state >> 33) as u8 % 95)
    }
    /// Prose-y: biased toward lowercase letters, spaces, and punctuation.
    pub fn prose_char(state: u64) -> u8 {
        const PROSE: &[u8] = b"abcdefghijklmnopqrstuvwxyz      ,. !?abcdefghijklmnopqrstuvwxyz";
        PROSE[((state >> 33) as usize) % PROSE.len()]
    }
}
fn pseudo_rand_prompts(n: usize, seed: u64) -> Vec<String> {
    let mut state = seed;
    let mut prompts = Vec::with_capacity(n);
    for i in 0..n {
        let base_len: usize = 40 + (i * 7 % 120); // 40..160
        let char_fn: fn(u64) -> u8 = if i < 7 {
            charsets::code_char
        } else if i < 14 {
            charsets::mixed_char
        } else {
            charsets::prose_char
        };
        let mut buf = Vec::with_capacity(base_len);
        for _ in 0..base_len {
            state = lcg_step(state);
            let ch = char_fn(state);
            buf.push(ch);
        }
        let s = String::from_utf8(buf).expect("LCG generated non-UTF8 byte — invariant broken");
        prompts.push(s);
    }
    prompts
}
fn weights_path() -> Option<PathBuf> {
    let p = PathBuf::from(WEIGHTS);
    if p.exists() {
        Some(p)
    } else {
        eprintln!("event_horizon_parity_prop: skipping — no weights at {WEIGHTS}");
        None
    }
}
fn make_engine(weights: &PathBuf) -> Box<dyn hawking_core::Engine> {
    std::env::set_var("HAWKING_QWEN_TCB", "1");
    std::env::set_var("HAWKING_QWEN_PREFIX_CACHE", "0");
    std::env::set_var("HAWKING_QWEN_USER_DRAFT", "1");
    std::env::set_var("HAWKING_QWEN_PAIR_2R_INLINE", "0");
    let cfg = hawking_core::EngineConfig::default();
    hawking_core::model::load_engine(weights, cfg).expect("load engine")
}
fn make_engine_nospec(weights: &PathBuf) -> Box<dyn hawking_core::Engine> {
    std::env::set_var("HAWKING_QWEN_TCB", "1");
    std::env::set_var("HAWKING_QWEN_PREFIX_CACHE", "0");
    std::env::set_var("HAWKING_QWEN_USER_DRAFT", "0");
    std::env::set_var("HAWKING_QWEN_PAIR_2R_INLINE", "0");
    std::env::remove_var("HAWKING_QWEN_EVENT_HORIZON");
    let cfg = hawking_core::EngineConfig::default();
    hawking_core::model::load_engine(weights, cfg).expect("load engine")
}
fn gen_on_n(engine: &mut dyn hawking_core::Engine, prompt: &str, max_new_tokens: usize) -> Vec<u32> {
    let req = hawking_core::GenerateRequest {
        prompt: prompt.into(),
        max_new_tokens,
        sampling: hawking_core::SamplingParams { temperature: 0.0, seed: Some(42), ..Default::default() },
        stop: vec![],
        abort: None,
        max_stall_ms: 0,
        json_mode: false,
    };
    let mut ids: Vec<u32> = Vec::new();
    engine
        .generate(req, &mut |ev| {
            if let hawking_core::StreamEvent::Token { id, .. } = ev {
                ids.push(id);
            }
        })
        .expect("generate");
    ids
}
// ── THE GATE ──────────────────────────────────────────────────────────────
#[test]
fn event_horizon_parity_property() {
    let Some(weights) = weights_path() else {
        return;
    };
    let _g = SERIAL_GATE.get_or_init(|| Mutex::new(())).lock().unwrap();
    let prompts = pseudo_rand_prompts(N_PROMPTS, SEED);
    assert_eq!(prompts.len(), N_PROMPTS);
    let mut failures: Vec<(usize, String)> = Vec::new();
    let mut pass_count = 0usize;
    for (i, prompt) in prompts.iter().enumerate() {
        let _logical_max = if i < 7 {
            16
        } else if i < 14 {
            64
        } else {
            256
        };
        let max_new_tokens = MAX_NEW_TOKENS_CAP;
        std::env::remove_var("HAWKING_QWEN_EVENT_HORIZON");
        let ref_ids = {
            let mut e = make_engine_nospec(&weights);
            gen_on_n(e.as_mut(), prompt, max_new_tokens)
        };
        std::env::set_var("HAWKING_QWEN_EVENT_HORIZON", "1");
        let on_ids = {
            let mut e = make_engine(&weights);
            gen_on_n(e.as_mut(), prompt, max_new_tokens)
        };
        std::env::remove_var("HAWKING_QWEN_EVENT_HORIZON");
        let off_ids = {
            let mut e = make_engine(&weights);
            gen_on_n(e.as_mut(), prompt, max_new_tokens)
        };
        let on_ok = on_ids == ref_ids;
        let off_ok = off_ids == ref_ids;
        if on_ok && off_ok {
            pass_count += 1;
        } else {
            let snippet: String = prompt.chars().take(60).collect();
            let which = match (on_ok, off_ok) {
                (false, false) => "EH-ON and OFF(spec) BOTH",
                (false, true) => "EH-ON",
                (true, false) => "OFF(legacy spec)",
                _ => unreachable!(),
            };
            let msg = format!(
                "prompt[{i}] {which} != no-spec greedy (snippet: {snippet:?})\n  \
                 REF(greedy): {ref_ids:?}\n  \
                 EH-ON      : {on_ids:?}\n  \
                 OFF(spec)  : {off_ids:?}"
            );
            failures.push((i, msg));
        }
    }
    std::env::set_var("HAWKING_QWEN_USER_DRAFT", "0");
    if !failures.is_empty() {
        let summary = failures.iter().map(|(_, m)| m.as_str()).collect::<Vec<_>>().join("\n\n");
        panic!("EH PARITY PROPERTY FAILED: {}/{} prompts had mismatches:\n\n{}", failures.len(), N_PROMPTS, summary);
    }
}
