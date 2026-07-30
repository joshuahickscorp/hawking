use hawking_core::model::rwkv7::RwkvSeven;
use hawking_core::{Engine, EngineConfig};
use std::path::{Path, PathBuf};
mod common;
use common::argmax;
fn read_ids(path: &Path) -> Vec<u32> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read fixture {path:?}: {e}"))
        .split_whitespace()
        .map(|t| t.parse::<u32>().expect("fixture id parse"))
        .collect()
}
fn greedy_from_prompt(
    engine: &mut RwkvSeven,
    prompt_ids: &[u32],
    n: usize,
) -> (Vec<u32>, Vec<f32>) {
    engine.reset_kv_for_test();
    let positions: Vec<usize> = (0..prompt_ids.len()).collect();
    let prompt_logits = engine
        .forward_tokens_for_test(prompt_ids, &positions)
        .expect("prefill forward");
    let logits0 = prompt_logits.last().expect("prompt logits").clone();
    let mut out = Vec::with_capacity(n);
    let mut next = argmax(&logits0);
    out.push(next);
    for _ in 1..n {
        let lg = engine
            .forward_tokens_for_test(&[next], &[0])
            .expect("decode forward")
            .pop()
            .unwrap();
        next = argmax(&lg);
        out.push(next);
    }
    (out, logits0)
}
fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/rwkv7")
        .join(name)
}
#[test]
fn rwkv7_argmax_parity_f32_exact() {
    let f32_path = std::env::var("HAWKING_RWKV7_F32_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/rwkv_ref/rwkv7-04-f32.gguf"));
    if !f32_path.exists() {
        eprintln!(
            "skipping rwkv7_argmax_parity_f32_exact: no F32 RWKV-7 GGUF at {f32_path:?}\n  \
             (produce with: llama-quantize models/rwkv7-04/rwkv7-0.4B-world.Q4_K_M.gguf \
             /tmp/rwkv_ref/rwkv7-04-f32.gguf F32, or set HAWKING_RWKV7_F32_GGUF)"
        );
        return;
    }
    let mut engine = RwkvSeven::load(&f32_path, EngineConfig::default()).expect("load f32 rwkv7");
    for stem in ["capital_france", "village"] {
        let prompt_ids = read_ids(&fixture(&format!("{stem}.prompt_ids")));
        let ref_gen = read_ids(&fixture(&format!("{stem}.gen_ids")));
        let n = ref_gen.len();
        let (mine, _logits0) = greedy_from_prompt(&mut engine, &prompt_ids, n);
        let matched = mine
            .iter()
            .zip(ref_gen.iter())
            .take_while(|(a, b)| a == b)
            .count();
        assert_eq!(
            mine, ref_gen,
            "rwkv7 F32 greedy decode must match llama.cpp exactly for {n} tokens (prompt={stem}); \
             matched {matched}/{n}\n  mine={mine:?}\n  ref ={ref_gen:?}"
        );
    }
}
#[test]
fn rwkv7_loads_and_runs_q4k() {
    let weights = PathBuf::from("../../models/rwkv7-04/rwkv7-0.4B-world.Q4_K_M.gguf");
    if !weights.exists() {
        eprintln!("skipping rwkv7_loads_and_runs_q4k: no rwkv7-0.4B Q4_K weights at {weights:?}");
        return;
    }
    let boxed = hawking_core::model::load_engine(&weights, EngineConfig::default())
        .expect("load_engine routes rwkv7");
    assert_eq!(boxed.model_arch(), "rwkv7", "arch must dispatch to rwkv7");
    drop(boxed);
    let mut engine = RwkvSeven::load(&weights, EngineConfig::default()).expect("load q4k rwkv7");
    let bytes = engine.state.size_bytes();
    let expected = 24 * (16 * 64 * 64 + 2 * 1024) * 4;
    assert_eq!(bytes, expected, "rwkv7 0.4B state size (constant, KV-free)");
    let prompt_ids = read_ids(&fixture("capital_france_q4k.prompt_ids"));
    let ref_gen = read_ids(&fixture("capital_france_q4k.gen_ids"));
    let n = ref_gen.len();
    let (mine, logits0) = greedy_from_prompt(&mut engine, &prompt_ids, n);
    let matched = mine
        .iter()
        .zip(ref_gen.iter())
        .take_while(|(a, b)| a == b)
        .count();
    assert!(logits0.len() == 65536, "vocab logits width");
    assert_eq!(
        mine[0], ref_gen[0],
        "rwkv7 Q4_K first greedy token must match llama.cpp ({} vs {})",
        mine[0], ref_gen[0]
    );
    assert!(
        matched >= 3,
        "expected >=3 leading tokens to match under Q4_K (got {matched})"
    );
}
