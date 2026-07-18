//! Integration parity test: GGUF parse + tokenizer encode + full 16-token greedy decode must be
//! bit-identical to the sealed golden (token ids and completion-text sha256). Gated on the model
//! fixture being present (skips cleanly in fixture-less environments), so `cargo test` stays green
//! everywhere while proving real decode parity where the 105 MB fixture exists.

use hawking_seed_b::adapter::LlamaConfig;
use hawking_seed_b::gguf::GgufFile;
use hawking_seed_b::model::Model;
use hawking_seed_b::record::sha256_hex;
use hawking_seed_b::tokenizer::Tokenizer;
use std::path::Path;

const WEIGHTS: &str = "../../models/SmolLM2-135M-Instruct-Q4_K_M.gguf";
const GOLDEN: &str = "../../reports/condense/gravity_forge/condensation/seed_b_golden.json";

fn load_golden() -> Option<serde_json::Value> {
    serde_json::from_str(&std::fs::read_to_string(GOLDEN).ok()?).ok()
}
fn ids(v: &serde_json::Value, k: &str) -> Vec<u32> {
    v[k].as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}

#[test]
fn full_decode_is_bit_identical_to_golden() {
    let w = Path::new(WEIGHTS);
    if !w.exists() {
        eprintln!("skip: model fixture absent at {WEIGHTS}");
        return;
    }
    let gold = load_golden().expect("golden fixture");

    // (a) GGUF parse + config
    let g = GgufFile::open(w).expect("gguf parse");
    let cfg = LlamaConfig::from_gguf(&g).expect("llama config");
    assert_eq!(cfg.n_layers, 30);
    assert_eq!(cfg.hidden, 576);
    assert_eq!(cfg.n_heads, 9);
    assert_eq!(cfg.n_kv_heads, 3);
    assert_eq!(cfg.head_dim, 64);

    // (b) tokenizer parity
    let tok = Tokenizer::from_gguf(&g).expect("tokenizer");
    let prompt_ids = tok.encode("The capital of France is").expect("encode");
    assert_eq!(prompt_ids, ids(&gold, "prompt_token_ids"), "tokenizer parity");

    // (c) full greedy decode parity — token ids + completion-text sha256
    let mut model = Model::load(w).expect("model load");
    let out = model.generate(&prompt_ids, 16, tok.eos_id()).expect("generate");
    assert_eq!(out, ids(&gold, "completion_token_ids"), "token-id parity");
    let text = tok.decode(&out);
    let sha = sha256_hex(text.trim().as_bytes());
    assert_eq!(
        sha,
        gold["completion_text_sha256"].as_str().unwrap(),
        "completion-text sha256 parity (golden 2d1559cf)"
    );
}
