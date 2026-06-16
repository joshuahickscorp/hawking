//! Read-only gold dumper for the pure-torch RWKV-7 parity gate.
//!
//! Loads an F32 RWKV-7 GGUF via the validated Rust oracle (`RwkvSeven::load`,
//! bit-exact vs llama.cpp) and emits, for each committed fixture prompt:
//!   <out>/<stem>.logits0  — 65536 little-endian f32 last-position logits
//!   <out>/<stem>.gen_ids  — space-separated greedy continuation ids
//!   <out>/<stem>.prompt_ids — the prompt ids (copied through for convenience)
//!
//! This produces *apples-to-apples* gold for the torch model: point both the
//! torch loader and this dumper at the SAME F32 GGUF (e.g. one converted from
//! the g1 BF16 safetensors with `convert_hf_to_gguf.py --outtype f32`, which is
//! a plain bf16->f32 cast, matching the torch loader's BF16->fp32). Then the
//! only residual is f32 op-order rounding (<=0.03 max-abs logit diff).
//!
//! Gated on `DISMANTLE_RWKV7_F32_GGUF` (skips if unset / file absent), so this
//! never runs in CI without the (uncommitted, multi-GB) F32 GGUF. Output dir
//! defaults to the GGUF's parent; override with `DISMANTLE_RWKV7_GOLD_OUT`.
//! Number of continuation tokens via `DISMANTLE_RWKV7_GOLD_N` (default 48).

use dismantle_core::model::rwkv7::RwkvSeven;
use dismantle_core::{Engine, EngineConfig};
use std::path::{Path, PathBuf};

fn read_ids(path: &Path) -> Vec<u32> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read fixture {path:?}: {e}"))
        .split_whitespace()
        .map(|t| t.parse::<u32>().expect("fixture id parse"))
        .collect()
}

fn argmax(v: &[f32]) -> u32 {
    let mut bi = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i as u32;
        }
    }
    bi
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/rwkv7")
        .join(name)
}

#[test]
fn rwkv7_dump_gold_f32() {
    let f32_path = match std::env::var("DISMANTLE_RWKV7_F32_GGUF") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!("skipping rwkv7_dump_gold_f32: set DISMANTLE_RWKV7_F32_GGUF to an F32 RWKV-7 GGUF");
            return;
        }
    };
    if !f32_path.exists() {
        eprintln!("skipping rwkv7_dump_gold_f32: no GGUF at {f32_path:?}");
        return;
    }
    let out_dir = std::env::var("DISMANTLE_RWKV7_GOLD_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| f32_path.parent().unwrap().to_path_buf());
    let n: usize = std::env::var("DISMANTLE_RWKV7_GOLD_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(48);
    std::fs::create_dir_all(&out_dir).expect("create out dir");

    let mut engine = RwkvSeven::load(&f32_path, EngineConfig::default()).expect("load f32 rwkv7");

    for stem in ["capital_france", "village"] {
        let prompt_ids = read_ids(&fixture(&format!("{stem}.prompt_ids")));

        // Fresh state, full-sequence prefill.
        engine.reset_kv_for_test();
        let positions: Vec<usize> = (0..prompt_ids.len()).collect();
        let prompt_logits = engine
            .forward_tokens_for_test(&prompt_ids, &positions)
            .expect("prefill forward");
        let logits0 = prompt_logits.last().expect("prompt logits").clone();
        assert_eq!(logits0.len(), 65536, "vocab logits width");

        // Greedy continuation (first token = argmax of last prompt position).
        let mut gen = Vec::with_capacity(n);
        let mut next = argmax(&logits0);
        gen.push(next);
        for _ in 1..n {
            let lg = engine
                .forward_tokens_for_test(&[next], &[0])
                .expect("decode forward")
                .pop()
                .unwrap();
            next = argmax(&lg);
            gen.push(next);
        }

        // Write logits0 as little-endian f32.
        let mut bytes = Vec::with_capacity(logits0.len() * 4);
        for &x in &logits0 {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        std::fs::write(out_dir.join(format!("{stem}.logits0")), &bytes).expect("write logits0");
        // gen_ids + prompt_ids as space-separated text.
        let gen_txt = gen
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        std::fs::write(out_dir.join(format!("{stem}.gen_ids")), gen_txt).expect("write gen_ids");
        let prompt_txt = prompt_ids
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        std::fs::write(out_dir.join(format!("{stem}.prompt_ids")), prompt_txt)
            .expect("write prompt_ids");

        eprintln!(
            "dumped gold [{stem}] -> {out_dir:?}: logits0(65536 f32), gen_ids({n}), argmax={}",
            argmax(&logits0)
        );
    }
}
