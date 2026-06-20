//! RWKV-7 numerical parity gate vs llama.cpp.
//!
//! The deliverable for slices 1-2: hawking's CPU-reference RWKV-7 forward must
//! agree with llama.cpp's RWKV-7 on the SAME GGUF. Exact float bit-match cannot
//! hold (the recurrence is float and the two implementations order ops
//! differently), so the gate is the codebase's standard token-parity standard:
//! the greedy argmax-token sequence must MATCH for >=N tokens, and the
//! first-step logit argmax must match.
//!
//! ## Two gates
//!
//! 1. `rwkv7_argmax_parity_f32_exact` — the rigorous gate. Requires an **F32**
//!    RWKV-7 GGUF (all weights dequantized to f32, produced by
//!    `llama-quantize <q4k.gguf> <out> F32`). With identical-precision weights on
//!    both sides, hawking's forward reproduces llama.cpp's greedy decode
//!    **exactly** for all N tokens (the only residual is f32 op-order rounding,
//!    measured at <=0.03 max-abs logit diff). Gated on env
//!    `HAWKING_RWKV7_F32_GGUF` (or `/tmp/rwkv_ref/rwkv7-04-f32.gguf`); skips if
//!    absent. The reference token ids are the committed fixtures under
//!    `tests/fixtures/rwkv7/*` (dumped by the `rwkv_ref` llama.cpp harness).
//!
//! 2. `rwkv7_loads_and_runs_q4k` — always-on smoke against the shipped Q4_K
//!    model in `models/`. Asserts the GGUF routes to the `rwkv7` engine, the
//!    forward runs, the constant recurrent state is the expected size, and the
//!    first greedy token matches the Q4_K reference. Q4_K-vs-f32-dequant
//!    precision drift means later tokens may diverge on near-ties, so this gate
//!    only asserts the first token + reports the full match count.
//!
//! ## WKV-7 recurrence implemented (per head, state S is head_size x head_size)
//! With `a_op = -kk`, `b_op = kk * iclr` (kk = l2norm_per_head(k * k_k)):
//! ```text
//!   sa[i]   = sum_j a_op[j] * S_prev[i][j]
//!   S[i][j] = S_prev[i][j]*w[j] + v[i]*k[j] + sa[i]*b_op[j]
//!   out[i]  = sum_j S[i][j] * r[j]
//! ```
//! Mirrors `ggml_compute_forward_rwkv_wkv7_f32` and `build_rwkv7_time_mix`.

use hawking_core::model::rwkv7::RwkvSeven;
use hawking_core::{Engine, EngineConfig};
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

/// Feed `prompt_ids` through a fresh RWKV-7 state, then greedy-decode `n`
/// tokens (argmax, temp=0), returning the decoded id sequence. The first
/// decoded token is the argmax of the last prompt position (matching how the
/// reference harness captures step 0).
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

/// Rigorous gate: exact greedy-argmax parity against llama.cpp on F32 weights.
#[test]
fn rwkv7_argmax_parity_f32_exact() {
    // Locate the F32 GGUF: env override, else the conventional /tmp path the
    // de-risk harness writes.
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

    // Both committed prompts must reproduce llama.cpp's F32 greedy decode exactly.
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
        eprintln!("rwkv7 F32 parity [{stem}]: {matched}/{n} leading argmax tokens match llama.cpp");
        assert_eq!(
            mine, ref_gen,
            "rwkv7 F32 greedy decode must match llama.cpp exactly for {n} tokens (prompt={stem}); \
             matched {matched}/{n}\n  mine={mine:?}\n  ref ={ref_gen:?}"
        );
    }
}

/// Always-on smoke: the shipped Q4_K model loads, routes to the rwkv7 engine,
/// runs the forward, carries the constant recurrent state, and produces the
/// expected first greedy token. Later tokens may diverge under Q4_K-vs-f32
/// precision drift, so only the first token is asserted (full count reported).
#[test]
fn rwkv7_loads_and_runs_q4k() {
    let weights = PathBuf::from("../../models/rwkv7-04/rwkv7-0.4B-world.Q4_K_M.gguf");
    if !weights.exists() {
        eprintln!("skipping rwkv7_loads_and_runs_q4k: no rwkv7-0.4B Q4_K weights at {weights:?}");
        return;
    }

    // Route through the public arch dispatcher to prove the `rwkv7` arm wires up.
    let boxed = hawking_core::model::load_engine(&weights, EngineConfig::default())
        .expect("load_engine routes rwkv7");
    assert_eq!(boxed.model_arch(), "rwkv7", "arch must dispatch to rwkv7");
    drop(boxed);

    let mut engine = RwkvSeven::load(&weights, EngineConfig::default()).expect("load q4k rwkv7");

    // Constant recurrent state: 0.4B = 24 layers * (16*64*64 wkv + 2*1024 shift) * 4B.
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
    eprintln!(
        "rwkv7 Q4_K vs llama.cpp Q4_K: first-token argmax mine={} ref={}; {}/{} leading tokens match",
        mine[0], ref_gen[0], matched, n
    );
    assert!(logits0.len() == 65536, "vocab logits width");
    assert_eq!(
        mine[0], ref_gen[0],
        "rwkv7 Q4_K first greedy token must match llama.cpp ({} vs {})",
        mine[0], ref_gen[0]
    );
    // Sanity: at least the first few tokens should survive quant drift.
    assert!(
        matched >= 3,
        "expected >=3 leading tokens to match under Q4_K (got {matched})"
    );
}
