#![cfg(target_os = "macos")]
use hawking_core::model::rwkv7::RwkvSeven;
use hawking_core::{Engine, EngineConfig};
use std::path::PathBuf;
mod common;
use common::argmax;
fn locate(rel: &str, env_key: &str) -> Option<PathBuf> {
    if let Ok(p) = std::env::var(env_key) {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        let cand = dir.join(rel);
        if cand.exists() {
            return Some(cand);
        }
        if !dir.pop() {
            return None;
        }
    }
}
fn load() -> Option<RwkvSeven> {
    let path = locate("models/rwkv7-g1-04-sft-Q4_K_M.gguf", "HAWKING_RWKV7_GGUF")?;
    let engine = RwkvSeven::load(&path, EngineConfig::default()).expect("load rwkv7");
    if !engine.has_gpu() {
        eprintln!("skip: no Metal GPU");
        return None;
    }
    Some(engine)
}
const N_DECODE: usize = 3;
fn solo_gpu_decode(engine: &mut RwkvSeven, prompt_ids: &[u32]) -> Vec<u32> {
    engine.reset_kv_for_test();
    let mut last = vec![0.0f32; 1];
    for &tok in prompt_ids {
        last = engine.forward_token_gpu(tok).expect("solo prefill step");
    }
    let mut cur = argmax(&last);
    let mut seq = vec![cur];
    for _ in 1..N_DECODE {
        let logits = engine.forward_token_gpu(cur).expect("solo decode step");
        cur = argmax(&logits);
        seq.push(cur);
    }
    seq
}
fn multiseq_decode(engine: &mut RwkvSeven, slot: usize, prompt_ids: &[u32]) -> Vec<u32> {
    engine.reset_kv_for_test();
    let prompt_len = prompt_ids.len();
    let first = engine.prefill_slot(slot, prompt_ids).expect("prefill_slot");
    let mut cur = first;
    let mut seq = vec![cur];
    for step in 1..N_DECODE {
        let logits = engine
            .forward_multiseq_batched(&[cur], &[prompt_len + step - 1], &[slot])
            .expect("multiseq decode step");
        cur = argmax(&logits[0]);
        seq.push(cur);
    }
    seq
}
fn run_parity(slot: usize) {
    let Some(mut engine) = load() else {
        return;
    };
    let prompt_ids = engine
        .encode_prompt_for_batch("The capital of France is")
        .expect("encode prompt");
    assert!(!prompt_ids.is_empty(), "empty prompt_ids");
    let eos = engine.eos_id_for_batch();
    let solo = solo_gpu_decode(&mut engine, &prompt_ids);
    let multi = multiseq_decode(&mut engine, slot, &prompt_ids);
    if let Some(e) = eos {
        assert_ne!(
            multi[0], e,
            "served first token is an immediate EOS (slot={slot}) — the serve prefill/handoff bug"
        );
    }
    assert_eq!(
        solo, multi,
        "RWKV prefill->multiseq handoff diverges from single-stream GPU decode \
         (slot={slot}) — copy_cpu_state_to_gpu_slot / immediate-EOS bug"
    );
}
#[test]
#[ignore = "loads the rwkv7 model + Metal; run with --ignored --test-threads=1"]
fn rwkv7_prefill_slot0_multiseq_matches_solo() {
    run_parity(0);
}
#[test]
#[ignore = "loads the rwkv7 model + Metal; run with --ignored --test-threads=1"]
fn rwkv7_prefill_slot3_multiseq_matches_solo() {
    run_parity(3);
}
