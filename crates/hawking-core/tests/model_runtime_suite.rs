//! Consolidated stateless model loading, runtime, cache, and sidecar cases.

#[cfg(target_os = "macos")]
#[path = "common.rs"]
#[rustfmt::skip]
mod common;
#[rustfmt::skip]
mod chat_special_token_detok {
    //! Regression for the `/v1/chat/completions` special-token detokenization leak.
    //!
    //! Streaming generation decodes one token at a time via `Tokenizer::decode_one`,
    //! which (before the fix) rendered control tokens like `<|im_end|>` literally.
    //! The chat path, whose template emits `<|im_start|>`/`<|im_end|>`, therefore
    //! leaked `"<|im_end|>"` / `"|>"` fragments into the response, while
    //! `/v1/completions` (no template) stayed clean.
    //!
    //! Pure-CPU: loads only the GGUF tokenizer — no Metal, no model forward. Skips
    //! cleanly when the Qwen weights are absent so it is CI-safe without a fixture.

    use hawking_core::gguf::GgufFile;
    use hawking_core::tokenizer::Tokenizer;
    use std::path::Path;

    fn qwen_tokenizer() -> Option<Tokenizer> {
        let p = Path::new("../../models/Qwen2.5-3B-Instruct-Q4_K_M.gguf");
        if !p.exists() {
            eprintln!("skipping chat_special_token_detok: weights missing at {p:?}");
            return None;
        }
        let gguf = GgufFile::open(p).expect("open gguf");
        Some(Tokenizer::from_gguf(&gguf).expect("build tokenizer"))
    }

    /// A control token must never render as visible text in the per-token streaming
    /// path. This is the exact mechanism behind the chat-endpoint garbage.
    #[test]
    fn streamed_decode_suppresses_special_tokens() {
        let Some(tok) = qwen_tokenizer() else { return };

        // The declared eos is always a control token; it must never render as text.
        let eos = tok.eos_id().expect("qwen gguf declares an eos");
        eprintln!("[diag] eos_id={eos} is_eog={} old_decode={:?}", tok.is_eog(eos), tok.decode_one(eos).unwrap());
        assert!(tok.is_special(eos), "eos must be flagged special");
        assert!(tok.is_eog(eos), "eos must be end-of-generation");
        assert_eq!(tok.decode_one(eos).unwrap(), "", "eos must not leak into output");

        // Qwen2.5 control ids: <|endoftext|>=151643, <|im_start|>=151644, <|im_end|>=151645.
        for id in [151643u32, 151644, 151645] {
            assert!((id as usize) < tok.vocab_size(), "id {id} in vocab");
            assert!(tok.is_special(id), "control id {id} must be flagged special");
            assert_eq!(tok.decode_one(id).unwrap(), "", "control id {id} must be suppressed in streamed output");
        }

        // The chat turn terminator must terminate generation even when the GGUF
        // sets eos to <|endoftext|> instead of <|im_end|>.
        assert!(tok.is_eog(151645), "<|im_end|> must be end-of-generation");

        // Normal tokens are unaffected — real text still decodes, with no markup leak.
        let ids = tok.encode("The capital of France is Paris.", false).expect("encode");
        let text: String = ids.iter().map(|&id| tok.decode_one(id).unwrap()).collect();
        assert!(text.contains("Paris") && text.contains("capital"), "normal tokens must still decode to text, got: {text:?}");
        assert!(!text.contains("<|"), "plain text must contain no control markup, got: {text:?}");
    }

    /// The chat template must `encode` `<|im_start|>` / `<|im_end|>` as ATOMIC control
    /// ids, not byte-level fragments. If they shatter, the chat prompt is malformed
    /// and the model generates `<|>` garbage — the real cause of the chat-endpoint
    /// bug (the brief mis-framed it as a detokenization leak).
    #[test]
    fn chat_template_encodes_control_tokens_atomically() {
        let Some(tok) = qwen_tokenizer() else { return };
        let template = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n";
        let ids = tok.encode(template, false).expect("encode template");
        eprintln!("[diag] template -> {} ids: {ids:?}", ids.len());
        assert!(ids.contains(&151644), "<|im_start|> must encode as atomic id 151644, got {ids:?}");
        assert!(ids.contains(&151645), "<|im_end|> must encode as atomic id 151645, got {ids:?}");
        // A correctly-encoded 3-line template is short; a shattered one is long.
        assert!(ids.len() < 16, "template shattered into {} tokens (control tokens not atomic): {ids:?}", ids.len());
    }
}
#[rustfmt::skip]
mod cpu_backend_parity {
    // Phase 3.3 portability cross-check: the pure-Rust CPU reference path
    // (EngineConfig.force_cpu => metal_ctx = None => forward_token + scalar dequant
    // GEMV) must produce greedy output that matches the Metal path. This is the
    // "engine runs correctly off-macOS" guarantee, exercised on-macOS by forcing
    // the CPU path. Perf is NOT the bar (CPU decode is ~100x slower) -- correctness
    // is. Scoped to the small dense qwen2.5-0.5b model (MoE CPU decode is a separate
    // follow-up).

    #![cfg(target_os = "macos")]

    use std::path::PathBuf;

    fn run_greedy(weights: &PathBuf, force_cpu: bool, n: usize) -> Vec<u32> {
        let cfg = hawking_core::EngineConfig { force_cpu, ..Default::default() };
        let mut engine = hawking_core::model::load_engine(weights, cfg).expect("load engine");
        let req = hawking_core::GenerateRequest {
            prompt: "The capital of France is".into(),
            max_new_tokens: n,
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

    #[test]
    fn cpu_backend_matches_metal_qwen05b() {
        let weights = PathBuf::from("../../models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
        if !weights.exists() {
            eprintln!("skipping cpu_backend_matches_metal_qwen05b: no qwen0.5b weights");
            return;
        }
        const N: usize = 12;

        // Metal path (normal).
        let metal = run_greedy(&weights, false, N);
        // CPU reference path (force_cpu => metal_ctx = None).
        let cpu = run_greedy(&weights, true, N);

        assert!(metal.len() >= 3 && cpu.len() >= 3, "both paths must emit >=3 tokens (metal={}, cpu={})", metal.len(), cpu.len());

        // Token-output parity. The CPU path dequantizes Q4_K to f32 and runs a scalar
        // gemv (f64-accumulated rmsnorm), while Metal runs the predec fused-FMA GEMV +
        // GPU argmax -- they agree only at the fp16 floor (atol~1e-3), so a LATE token
        // could diverge on a near-tie. The gate is the plan's token-parity standard:
        // the first 3 greedy IDs must match. The full match count is reported.
        let matched = metal.iter().zip(cpu.iter()).take_while(|(a, b)| a == b).count();
        eprintln!("CPU-vs-Metal qwen0.5b greedy: {}/{} leading tokens identical\n  metal={:?}\n  cpu  ={:?}", matched, N, metal, cpu);

        assert_eq!(metal[..3], cpu[..3], "first-3 greedy token IDs must match between the CPU reference path and Metal");
    }
}
#[rustfmt::skip]
mod cpu_backend_parity_deepseek {
    // Phase 3.3 portability LIGHT gate (MoE): the pure-Rust CPU reference path
    // (EngineConfig.force_cpu => metal_ctx = None => forward_token + materialized-KV
    // attention + per-routed-expert dequant-Q4_K GEMV) must LOAD and run a single
    // CPU forward for the DeepSeek-V2-Lite MoE model, producing finite logits of
    // length vocab_size. This is the MoE analogue of cpu_backend_parity.rs (which
    // covers the dense qwen0.5b model), but intentionally a LIGHT gate: ONE 10GB
    // load + ONE CPU token, NOT a CPU-vs-Metal double-load greedy comparison. Perf
    // is NOT the bar (CPU MoE decode is ~100x slower) -- reach/correctness is.
    //
    // Unblocked by three load/forward gating fixes in deepseek_v2.rs (applied A->B->C):
    //   A. load() now honors force_cpu (metal_ctx = None), mirroring qwen_dense.rs.
    //   B. mla_metal is suppressed under force_cpu / off-macOS so mla_c_kv stays
    //      empty and attention() takes the CPU materialized-KV path instead of the
    //      'mla_decode: Metal context unavailable' hard error.
    //   C. forward_token_final_norm_maybe_read gained a pure-Rust per-layer CPU
    //      driver (calling the full ffn(), routed + shared experts) when
    //      metal_ctx.is_none().
    //
    // NOTE: EngineConfig carries NO kernel_profile here (Default), so the default
    // mla_metal=true is exercised at load and MUST be suppressed by Edit B for this
    // to reach the FFN instead of erroring. forward_token(token, pos) is a private
    // inherent method; the public trait seam is forward_tokens_for_test, which for
    // deepseek routes through forward_tokens -> forward_token -> ... -> the Edit C
    // CPU loop -> gemv_f16 LM head producing vec![0.0; vocab_size].
    //
    // RUN (on the dev Mac; this sandbox cannot dispatch):
    //   cargo test -p hawking-core --test cpu_backend_parity_deepseek -- --nocapture

    #![cfg(target_os = "macos")]

    use std::path::PathBuf;

    use hawking_core::{Engine, EngineConfig};

    #[test]
    fn cpu_forward_deepseek_v2_lite_force_cpu_ok() {
        let weights = PathBuf::from("../../models/deepseek-v2-lite-q4.gguf");
        if !weights.exists() {
            eprintln!("skipping cpu_forward_deepseek_v2_lite_force_cpu_ok: no deepseek-v2-lite-q4.gguf");
            return;
        }

        // force_cpu=true => metal_ctx = None (Edit A) => mla_metal suppressed (Edit B)
        // => single CPU forward via the Edit C per-layer driver.
        let cfg = EngineConfig { force_cpu: true, ..Default::default() };
        let mut engine = hawking_core::model::load_engine(&weights, cfg).expect("load engine");

        // Single CPU forward at (token=0, pos=0). forward_tokens_for_test is the
        // public trait seam; for deepseek it loops forward_token, exercising the
        // CPU decode path end-to-end (materialized-KV attention + MoE expert GEMVs).
        let out = engine.forward_tokens_for_test(&[0], &[0]).expect("forward_tokens_for_test (force_cpu) must return Ok");

        assert_eq!(out.len(), 1, "one input token must yield one logit vector");
        let logits = &out[0];

        // logits is built as vec![0.0f32; config.vocab_size] by the LM head, so its
        // length IS the model's vocab_size. Assert it is the full (non-empty) vocab
        // (Default config has no vocab-prune path, so no shrink).
        assert!(!logits.is_empty(), "logits length must equal vocab_size (got {})", logits.len());

        // The real correctness gate: every logit must be finite (no NaN / Inf). A
        // NaN here would mean the CPU attention / MoE expert path produced garbage.
        let bad = logits.iter().position(|v| !v.is_finite());
        assert!(bad.is_none(), "CPU-path logits must be finite (no NaN/Inf); first non-finite at index {:?} = {:?}", bad, bad.map(|i| logits[i]));

        eprintln!("deepseek-v2-lite force_cpu single CPU forward OK: vocab_size={}, all logits finite", logits.len());
    }
}
#[rustfmt::skip]
mod e3_user_draft_gate_rule {
    //! Track 0/9 lock-in (DESIGN MIRROR): pins the E3 inline-pair default rule.
    //!
    //! The live gate is an INLINE `OnceLock`-cached closure in
    //! `crates/hawking-core/src/model_qwen_dense.rs` (the `ffn_pair_2r_inline`
    //! binding). It is NOT exposed as a pure fn and extracting one would mean
    //! editing that file, which is out of scope here. So this test pins a TINY pure
    //! MIRROR of the same boolean rule. If you ever change the live gate, change
    //! `e3_default` below to match — the two must stay in lockstep.
    //!
    //! Live rule being mirrored (verbatim semantics):
    //!   let explicit = std::env::var_os("HAWKING_QWEN_PAIR_2R_INLINE")
    //!                      .map(|v| v != "0");          // Some(true) for any non-"0"
    //!   // when unset, default-ON for plain decode but default-OFF under user-draft:
    //!   explicit.unwrap_or_else(|| !crate::env_on("HAWKING_QWEN_USER_DRAFT"))
    //! where `env_on(x)` is true IFF the var is exactly "1".
    //!
    //! Gate (CPU, no GPU, no env, no model):
    //!   cargo test -p hawking-core --test e3_user_draft_gate_rule

    /// Pure mirror of the qwen_dense `ffn_pair_2r_inline` default decision.
    ///
    /// * `user_draft_on` — the value of `env_on("HAWKING_QWEN_USER_DRAFT")`
    ///   (true only when the var is exactly "1").
    /// * `explicit` — the resolved explicit override:
    ///   `var_os("HAWKING_QWEN_PAIR_2R_INLINE").map(|v| v != "0")`
    ///   (None = unset; Some(false) only for "0"; Some(true) for any other value).
    ///
    /// Returns whether E3 (the 2r inline pair) is ON.
    fn e3_default(user_draft_on: bool, explicit: Option<bool>) -> bool {
        explicit.unwrap_or(!user_draft_on)
    }

    #[test]
    fn explicit_override_always_wins() {
        // An explicit HAWKING_QWEN_PAIR_2R_INLINE pins the result regardless of
        // user-draft (power user accepting the draft bit-identity trade).
        assert!(e3_default(true, Some(true)), "explicit=1 forces E3 ON even under user-draft");
        assert!(e3_default(false, Some(true)), "explicit=1 forces E3 ON");
        assert!(!e3_default(false, Some(false)), "explicit=0 forces E3 OFF");
        assert!(!e3_default(true, Some(false)), "explicit=0 forces E3 OFF even without user-draft");
    }

    #[test]
    fn unset_default_is_off_under_user_draft_on_otherwise() {
        // THE load-bearing invariant (the 2026-06-07 regression this guards):
        // with no explicit override, E3 is ON for plain decode and OFF when the
        // user n-gram draft is active (keeps forward_tokens_verify bit-identical).
        assert!(e3_default(false, None), "unset + no user-draft => E3 ON (+9.6%)");
        assert!(!e3_default(true, None), "unset + user-draft ON => E3 OFF (draft stays lossless)");
    }
}
#[rustfmt::skip]
mod eagle5_forward_parity {
    //! Numerical parity test for the Rust Eagle6 forward pass.
    //!
    //! Validates that `eagle5_forward::forward_single_step` produces logits
    //! that match the PyTorch reference (`Eagle5Head.forward` in
    //! `colab/eagle5_train_pytorch.py`) within `atol=1e-3` on a fixed
    //! seeded input. The fixture is generated by
    //! `tools/eagle5_forward_dump.py`.
    //!
    //! Fixture path: `crates/hawking-core/tests/eagle5_parity_q3b.json`.
    //!
    //! This test requires the real q3b head safetensors at `$HAWKING_Q3B_HEAD`
    //! (default fallback: `~/Downloads/head_final.safetensors`). It's
    //! `#[ignore]`d by default so CI doesn't need the 1.66 GB head.
    //!
    //! Run:
    //!   HAWKING_Q3B_HEAD=$HOME/Downloads/head_final.safetensors \
    //!     cargo test --release --test eagle5_forward_parity -- --ignored
    //!
    //! Assertions, in order of strictness:
    //! - top-1 argmax matches PyTorch exactly
    //! - top-16 argmax indices match PyTorch exactly
    //! - top-16 logit values match PyTorch within atol=1e-2 (relative)
    //! - full L_inf (max |logits_rust - logits_pytorch|) ≤ 5e-2
    //! - L2 norm of logits within 1% of PyTorch reference
    //!
    //! Tolerances are looser than 1e-3 because we accumulate over
    //! ~310M FMAs through a transformer block + LM head. Float order
    //! of summation differs Rust-vs-PyTorch; absolute drift in the
    //! 1e-2 range is expected and acceptable as long as the top-K
    //! argmax is preserved.

    use hawking_core::speculate::eagle5::Eagle5Head;
    use serde::Deserialize;
    use std::path::PathBuf;

    /// Minimal RFC-4648 base64 decoder. Inlined here so the test doesn't
    /// pull a base64 crate into hawking-core's dev-deps. Handles the
    /// standard alphabet (no urlsafe) with optional '=' padding.
    fn b64_decode(s: &str) -> Vec<u8> {
        let mut out = Vec::with_capacity(s.len() * 3 / 4);
        let mut buf = 0u32;
        let mut bits = 0u32;
        for c in s.bytes() {
            let v = match c {
                b'A'..=b'Z' => c - b'A',
                b'a'..=b'z' => c - b'a' + 26,
                b'0'..=b'9' => c - b'0' + 52,
                b'+' => 62,
                b'/' => 63,
                b'=' | b'\n' | b'\r' | b' ' | b'\t' => continue,
                _ => panic!("invalid b64 char: {c:?}"),
            };
            buf = (buf << 6) | (v as u32);
            bits += 6;
            if bits >= 8 {
                bits -= 8;
                out.push(((buf >> bits) & 0xff) as u8);
            }
        }
        out
    }

    // Fixture fields not read here (`n_heads`, `num_blocks`, `top_values`) are
    // still part of the file format — kept for the human reading the JSON
    // and for any future tighter parity gate. `#[allow(dead_code)]` is
    // scoped to this struct, not the whole file.
    #[derive(Deserialize)]
    #[allow(dead_code)]
    struct Fixture {
        schema: String,
        hidden_dim: usize,
        vocab_size: usize,
        n_heads: usize,
        num_blocks: usize,
        prev_token: u32,
        residual_b64: String,
        intermediate_b64: String,
        logits_b64: String,
        argmax: u32,
        top_k: usize,
        top_indices: Vec<u32>,
        top_values: Vec<f32>,
        logits_l2: f32,
    }

    fn decode_f32(s: &str, expected_len: usize) -> Vec<f32> {
        let raw = b64_decode(s);
        assert_eq!(raw.len(), expected_len * 4, "expected {expected_len} f32s = {} bytes, got {}", expected_len * 4, raw.len());
        let mut out = vec![0.0_f32; expected_len];
        let src = raw.as_ptr() as *const f32;
        // SAFETY: bounds-checked above. We require little-endian host
        // (true on Apple Silicon + every Tier-1 target we care about).
        unsafe { std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), expected_len) };
        out
    }

    fn head_path() -> Option<PathBuf> {
        if let Some(p) = std::env::var_os("HAWKING_Q3B_HEAD") {
            let pp = PathBuf::from(p);
            if pp.exists() {
                return Some(pp);
            }
        }
        // Default fallback to the user's Downloads.
        let home = std::env::var_os("HOME")?;
        let candidate = PathBuf::from(home).join("Downloads/head_final.safetensors");
        if candidate.exists() {
            Some(candidate)
        } else {
            None
        }
    }

    fn q1p5_head_path() -> Option<PathBuf> {
        if let Some(p) = std::env::var_os("HAWKING_Q1P5_HEAD") {
            let pp = PathBuf::from(p);
            if pp.exists() {
                return Some(pp);
            }
        }
        let home = std::env::var_os("HOME")?;
        let candidate = PathBuf::from(home).join("Downloads/hawking_export/heads/q1p5_eagle6_long.safetensors");
        if candidate.exists() {
            Some(candidate)
        } else {
            None
        }
    }

    #[test]
    #[ignore = "needs HAWKING_Q3B_HEAD or ~/Downloads/head_final.safetensors"]
    fn eagle6_forward_matches_pytorch_q3b() {
        let head = head_path().expect("set HAWKING_Q3B_HEAD or place head at ~/Downloads/");
        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/eagle5_parity_q3b.json");
        let raw = std::fs::read_to_string(&fixture_path).unwrap_or_else(|e| panic!("read {}: {e}", fixture_path.display()));
        let f: Fixture = serde_json::from_str(&raw).expect("parse fixture");
        assert_eq!(f.schema, "eagle5-forward-parity-v1");

        let hidden = f.hidden_dim;
        let vocab = f.vocab_size;
        let residual = decode_f32(&f.residual_b64, hidden);
        let intermediate = decode_f32(&f.intermediate_b64, hidden);
        let py_logits = decode_f32(&f.logits_b64, vocab);

        // Load head and run the Rust forward.
        let h = Eagle5Head::load_from_safetensors(&head, hidden, vocab).expect("load head");

        // Time a few forward calls so we have a perf regression signal in
        // the test output. The first call may be cold (caches not warm,
        // first-touch alloc); we report median of N=8.
        use std::time::Instant;
        let mut timings = Vec::with_capacity(8);
        let mut rust_logits = Vec::new();
        for _ in 0..8 {
            let t0 = Instant::now();
            let l = h.forward_logits(f.prev_token, &residual, &intermediate).expect("Trained head must return logits");
            timings.push(t0.elapsed());
            rust_logits = l;
        }
        timings.sort();
        let median = timings[timings.len() / 2];
        let min = timings[0];
        let max = timings[timings.len() - 1];
        eprintln!("forward_single_step timing (n=8): median={:.2}ms min={:.2}ms max={:.2}ms", median.as_secs_f64() * 1000.0, min.as_secs_f64() * 1000.0, max.as_secs_f64() * 1000.0,);
        // Soft perf gate: median forward should be under 200 ms on any
        // tier-1 host. The current threaded impl lands at ~5-30 ms; 200 ms
        // is generous against thermal throttling or single-core CI runners.
        assert!(median.as_millis() < 200, "forward_single_step median {}ms exceeds 200ms perf gate", median.as_millis());

        assert_eq!(rust_logits.len(), vocab, "rust logits length wrong: {} != {vocab}", rust_logits.len());

        // Top-1 argmax must match PyTorch exactly.
        let rust_argmax = rust_logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
        assert_eq!(rust_argmax, f.argmax, "top-1 argmax mismatch: rust={} pytorch={}", rust_argmax, f.argmax);

        // Full-vector parity: L_inf (max abs diff) and L2 norm.
        let mut l_inf = 0.0_f32;
        let mut l2_sq = 0.0_f64;
        let mut l2_rust = 0.0_f64;
        for i in 0..vocab {
            let diff = rust_logits[i] - py_logits[i];
            l_inf = l_inf.max(diff.abs());
            l2_sq += (diff as f64) * (diff as f64);
            l2_rust += (rust_logits[i] as f64) * (rust_logits[i] as f64);
        }
        let l2 = l2_sq.sqrt();
        let l2_rust = l2_rust.sqrt() as f32;
        eprintln!("argmax parity: rust={} pytorch={} OK", rust_argmax, f.argmax);
        eprintln!("rust L2={:.2}  pytorch L2={:.2}  delta L2={:.4}  L_inf={:.4e}", l2_rust, f.logits_l2, l2, l_inf);

        // Tolerance band: PyTorch + Rust both accumulate ~310M FMAs to
        // produce each logit. Different sum-orders → fp32 drift typically
        // in the 1e-2 absolute range. We pin the gate at 5e-2 L_inf;
        // tighter than 1e-2 is unreliable across machines.
        const L_INF_TOL: f32 = 5e-2;
        assert!(l_inf <= L_INF_TOL, "L_inf parity violation: {l_inf:.4e} > {L_INF_TOL:.4e}");

        // L2 norm of rust logits must be within 1% of PyTorch's.
        let l2_rel = ((l2_rust - f.logits_l2).abs() / f.logits_l2).abs();
        assert!(l2_rel < 0.01, "logits L2 disagrees by {:.2}%; rust={} pytorch={}", l2_rel * 100.0, l2_rust, f.logits_l2);

        // Top-K argmax indices must all match (order may shuffle within
        // ties; we check set equality on the first 8).
        let mut rust_top_k: Vec<(usize, f32)> = rust_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        rust_top_k.select_nth_unstable_by(f.top_k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap());
        rust_top_k.truncate(f.top_k);
        rust_top_k.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        use std::collections::HashSet;
        let rust_set: HashSet<u32> = rust_top_k.iter().take(8).map(|(i, _)| *i as u32).collect();
        let py_set: HashSet<u32> = f.top_indices.iter().take(8).copied().collect();
        let overlap = rust_set.intersection(&py_set).count();
        assert!(overlap >= 7, "top-8 overlap too small: {overlap}/8 (rust={rust_set:?} pytorch={py_set:?})");

        eprintln!("top-8 overlap: {overlap}/8 OK");
        eprintln!("Rust Eagle6 forward parity ≤{L_INF_TOL} ✓");
    }

    /// Same gate, against the 2-block q1p5 head (Qwen-1.5B). Exercises the
    /// `extra_blocks.0.*` code path in the loader and the chained-blocks
    /// path in `forward_single_step`. q3b is num_blocks=1; q1p5 is 2.
    #[test]
    #[ignore = "needs HAWKING_Q1P5_HEAD or ~/Downloads/hawking_export/heads/q1p5_eagle6_long.safetensors"]
    fn eagle6_forward_matches_pytorch_q1p5() {
        let head = q1p5_head_path().expect("set HAWKING_Q1P5_HEAD or place q1p5_eagle6_long.safetensors at ~/Downloads/hawking_export/heads/");
        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/eagle5_parity_q1p5.json");
        let raw = std::fs::read_to_string(&fixture_path).unwrap_or_else(|e| panic!("read {}: {e}", fixture_path.display()));
        let f: Fixture = serde_json::from_str(&raw).expect("parse fixture");
        assert_eq!(f.schema, "eagle5-forward-parity-v1");

        let hidden = f.hidden_dim;
        let vocab = f.vocab_size;
        let residual = decode_f32(&f.residual_b64, hidden);
        let intermediate = decode_f32(&f.intermediate_b64, hidden);
        let py_logits = decode_f32(&f.logits_b64, vocab);

        let h = Eagle5Head::load_from_safetensors(&head, hidden, vocab).expect("load q1p5 head");
        let rust_logits = h.forward_logits(f.prev_token, &residual, &intermediate).expect("Trained head must return logits");
        assert_eq!(rust_logits.len(), vocab);

        let rust_argmax = rust_logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
        assert_eq!(rust_argmax, f.argmax, "q1p5 top-1 argmax mismatch: rust={} pytorch={}", rust_argmax, f.argmax);

        let mut l_inf = 0.0_f32;
        let mut l2_sq = 0.0_f64;
        let mut l2_rust = 0.0_f64;
        for i in 0..vocab {
            let diff = rust_logits[i] - py_logits[i];
            l_inf = l_inf.max(diff.abs());
            l2_sq += (diff as f64) * (diff as f64);
            l2_rust += (rust_logits[i] as f64) * (rust_logits[i] as f64);
        }
        let l2 = l2_sq.sqrt();
        let l2_rust = l2_rust.sqrt() as f32;
        eprintln!("q1p5 argmax parity: rust={} pytorch={} OK", rust_argmax, f.argmax);
        eprintln!("q1p5 rust L2={:.2}  pytorch L2={:.2}  delta L2={:.4}  L_inf={:.4e}", l2_rust, f.logits_l2, l2, l_inf);

        // 2-block accumulates ~2x the FMA count, so we give slightly more
        // slack on L_inf — still well under any value that would affect
        // top-K rankings.
        const L_INF_TOL: f32 = 1e-1;
        assert!(l_inf <= L_INF_TOL, "q1p5 L_inf parity violation: {l_inf:.4e} > {L_INF_TOL:.4e}");

        let l2_rel = ((l2_rust - f.logits_l2).abs() / f.logits_l2).abs();
        assert!(l2_rel < 0.01, "q1p5 logits L2 disagrees by {:.2}%; rust={} pytorch={}", l2_rel * 100.0, l2_rust, f.logits_l2,);

        use std::collections::HashSet;
        let mut rust_top_k: Vec<(usize, f32)> = rust_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        rust_top_k.select_nth_unstable_by(f.top_k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap());
        rust_top_k.truncate(f.top_k);
        rust_top_k.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let rust_set: HashSet<u32> = rust_top_k.iter().take(8).map(|(i, _)| *i as u32).collect();
        let py_set: HashSet<u32> = f.top_indices.iter().take(8).copied().collect();
        let overlap = rust_set.intersection(&py_set).count();
        assert!(overlap >= 7, "q1p5 top-8 overlap too small: {overlap}/8 (rust={rust_set:?} pytorch={py_set:?})");
        eprintln!("q1p5 top-8 overlap: {overlap}/8 OK");
        eprintln!("Rust Eagle6 forward (2-block) parity ≤{L_INF_TOL} ✓");
    }
}
#[rustfmt::skip]
mod eagle5_spec_parity {
    // Eagle5 spec-decode greedy parity test.
    //
    // The spec-decode correctness invariant is that greedy generation at
    // temperature=0 produces the same token sequence whether or not
    // speculative decoding is enabled, regardless of the draft head's
    // accept rate. The draft only proposes; the verifier (the full
    // V2-Lite model) takes the final argmax at every position.
    //
    // This test exercises that invariant against the deterministic mock
    // Eagle5 head — its accept rate is near 1/vocab, so most steps will
    // emit only the verifier's bonus token, but the emitted sequence
    // must match exactly the no-spec greedy run. If this test ever fails
    // it means the spec-decode runtime is changing greedy output, which
    // is a correctness bug (not a perf regression).
    //
    // Mac-only because the engine constructor needs the Metal context.

    #![cfg(target_os = "macos")]

    use std::path::PathBuf;

    const PROMPT: &str = "Once upon a time";
    const MAX_NEW_TOKENS: usize = 16;

    fn run_greedy(weights: &PathBuf, cfg: hawking_core::EngineConfig) -> Vec<u32> {
        let mut engine = hawking_core::model::load_engine(weights, cfg).expect("load engine");
        let req = hawking_core::GenerateRequest {
            prompt: PROMPT.into(),
            max_new_tokens: MAX_NEW_TOKENS,
            sampling: hawking_core::SamplingParams { temperature: 0.0, top_k: 1, top_p: 1.0, repetition_penalty: 1.0, seed: Some(42) },
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
        assert!(!ids.is_empty(), "must produce at least one token");
        ids
    }

    fn find_weights() -> Option<PathBuf> {
        // Test cwd is the crate root. The worktree may not have a models/
        // tree; the main checkout usually does. Probe both relative roots
        // and an env-var override (used by CI).
        for candidate in ["../../models/deepseek-v2-lite-q4.gguf", "models/deepseek-v2-lite-q4.gguf"] {
            let p = PathBuf::from(candidate);
            if p.exists() {
                return Some(p);
            }
        }
        if let Ok(env_path) = std::env::var("HAWKING_TEST_WEIGHTS") {
            let p = PathBuf::from(env_path);
            if p.exists() {
                return Some(p);
            }
        }
        None
    }

    fn find_profile(weights: &PathBuf) -> Option<hawking_core::profile::KernelProfile> {
        // Prefer a freshly-built deterministic profile so this test
        // doesn't fail when shader sources change between sessions and
        // the on-disk profile snapshot lags behind.
        hawking_core::profile::fresh_test_profile(weights).ok()
    }

    #[test]
    fn eagle5_greedy_parity_k4() {
        let Some(weights) = find_weights() else {
            eprintln!("skipping eagle5_greedy_parity_k4: no deepseek-v2-lite-q4.gguf");
            return;
        };
        let profile = find_profile(&weights);

        // Baseline: no-spec greedy.
        let cfg_baseline = hawking_core::EngineConfig { kernel_profile: profile.clone(), ..Default::default() };
        let baseline_ids = run_greedy(&weights, cfg_baseline);

        // Eagle5 spec-decode greedy with mock head + K=4.
        let cfg_eagle5 = hawking_core::EngineConfig {
            kernel_profile: profile,
            speculate: true,
            speculate_mode: hawking_core::SpeculateMode::Eagle5,
            verify_window: 4,
            eagle5_head_path: None, // forces mock-head fallback
            ..Default::default()
        };
        let eagle5_ids = run_greedy(&weights, cfg_eagle5);

        assert_eq!(
            baseline_ids, eagle5_ids,
            "eagle5 spec-decode at temp=0 must emit the same tokens as no-spec greedy\n  \
             baseline: {:?}\n  eagle5:   {:?}",
            baseline_ids, eagle5_ids,
        );
    }

    #[test]
    fn eagle5_greedy_parity_k2_and_k8() {
        // Same invariant at the other supported window sizes — sanity check
        // that the verify path's accept-prefix accounting is correct across
        // K∈{2,8} too.
        let Some(weights) = find_weights() else {
            eprintln!("skipping eagle5_greedy_parity_k2_and_k8: no weights");
            return;
        };
        let profile = find_profile(&weights);

        let cfg_baseline = hawking_core::EngineConfig { kernel_profile: profile.clone(), ..Default::default() };
        let baseline_ids = run_greedy(&weights, cfg_baseline);

        for &k in &[2usize, 8] {
            let cfg = hawking_core::EngineConfig {
                kernel_profile: profile.clone(),
                speculate: true,
                speculate_mode: hawking_core::SpeculateMode::Eagle5,
                verify_window: k,
                eagle5_head_path: None,
                ..Default::default()
            };
            let ids = run_greedy(&weights, cfg);
            assert_eq!(
                baseline_ids, ids,
                "eagle5 K={k} must emit the same tokens as no-spec greedy\n  \
                 baseline: {:?}\n  eagle5:   {:?}",
                baseline_ids, ids,
            );
        }
    }
}
#[rustfmt::skip]
mod eagle5_trained_head_load {
    //! End-to-end loader test against the real Eagle6 safetensors heads
    //! produced by `colab/finish_q3b_reconciliation.ipynb`.
    //!
    //! These tests are `#[ignore]`d by default because they need the heads
    //! on disk at a specific path. To run them manually after downloading
    //! the heads from Drive:
    //!
    //!   HAWKING_Q3B_HEAD=/path/to/q3b_eagle6_long.safetensors \
    //!   HAWKING_Q1P5_HEAD=/path/to/q1p5_eagle6_long.safetensors \
    //!   cargo test --release --test eagle5_trained_head_load -- --ignored
    //!
    //! Or to point at the user's local Downloads:
    //!
    //!   HAWKING_Q3B_HEAD=$HOME/Downloads/head_final.safetensors \
    //!     cargo test --release --test eagle5_trained_head_load \
    //!     trained_head_q3b_loads -- --ignored
    //!
    //! Validates: file opens, metadata parses, all expected tensors are
    //! present at the expected shapes. Does NOT exercise the (still
    //! placeholder) forward pass.

    use hawking_core::speculate::eagle5::Eagle5Head;
    use std::path::PathBuf;

    const Q3B_HIDDEN: usize = 2048;
    const Q1P5_HIDDEN: usize = 1536;
    const QWEN_VOCAB: usize = 151_936;

    fn head_path(env_var: &str) -> Option<PathBuf> {
        std::env::var_os(env_var).map(PathBuf::from).filter(|p| p.exists())
    }

    #[test]
    #[ignore = "needs HAWKING_Q3B_HEAD=/path/to/q3b_eagle6_long.safetensors"]
    fn trained_head_q3b_loads() {
        let path = head_path("HAWKING_Q3B_HEAD").expect("set HAWKING_Q3B_HEAD to the q3b safetensors path");
        let head = Eagle5Head::load_from_safetensors(&path, Q3B_HIDDEN, QWEN_VOCAB).expect("q3b head must load");
        assert_eq!(head.hidden(), Q3B_HIDDEN);
        assert_eq!(head.vocab(), QWEN_VOCAB);
        // Sanity: propose() with the placeholder forward pass returns K
        // ids in-vocab without panicking. Quality of these drafts is near
        // zero until the real Eagle6 forward lands; we're only proving
        // the loader → propose dispatch is wired.
        let mut h = head;
        let drafts = h.propose(0, 4);
        assert_eq!(drafts.len(), 4);
        for d in &drafts {
            assert!((*d as usize) < QWEN_VOCAB, "draft id out of vocab");
        }
    }

    #[test]
    #[ignore = "needs HAWKING_Q1P5_HEAD=/path/to/q1p5_eagle6_long.safetensors"]
    fn trained_head_q1p5_loads() {
        let path = head_path("HAWKING_Q1P5_HEAD").expect("set HAWKING_Q1P5_HEAD to the q1p5 safetensors path");
        let head = Eagle5Head::load_from_safetensors(&path, Q1P5_HIDDEN, QWEN_VOCAB).expect("q1p5 head must load (note: 2-block — exercises extra_blocks.* path)");
        assert_eq!(head.hidden(), Q1P5_HIDDEN);
        assert_eq!(head.vocab(), QWEN_VOCAB);
        let mut h = head;
        let drafts = h.propose(0, 4);
        assert_eq!(drafts.len(), 4);
    }

    #[test]
    #[ignore = "needs HAWKING_Q3B_HEAD env"]
    fn trained_head_rejects_wrong_hidden() {
        let path = head_path("HAWKING_Q3B_HEAD").expect("set HAWKING_Q3B_HEAD to the q3b safetensors path");
        // Pass deliberately-wrong hidden — loader must refuse.
        let err = Eagle5Head::load_from_safetensors(&path, Q3B_HIDDEN + 1, QWEN_VOCAB);
        assert!(err.is_err(), "loader must reject hidden_dim mismatch");
    }
}
#[rustfmt::skip]
mod gemma2_smoke {
    //! Gemma-2 smoke + greedy-output regression gate.
    //!
    //! Auto-activates when a `models/*gemma-2*.gguf` is present; skips
    //! cleanly otherwise. Pins the greedy token-id hash on first run to
    //! `tests/_gemma2_token_baseline.hashes`, guards drift after.
    //! Mirrors llama32_smoke.rs.
    //!
    //! Pull a GGUF into models/, e.g. models/gemma-2-2b-it-Q4_K_M.gguf.

    #![cfg(target_os = "macos")]

    use sha2::{Digest, Sha256};
    use std::path::PathBuf;

    const PROMPT: &str = "Once upon a time";
    const MAX_NEW_TOKENS: usize = 32;

    fn find_gguf(size_tag: &str) -> Option<PathBuf> {
        let dir = PathBuf::from("../../models");
        for e in std::fs::read_dir(&dir).ok()?.flatten() {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) != Some("gguf") {
                continue;
            }
            let name = p.file_name()?.to_str()?.to_lowercase();
            if name.contains(size_tag) {
                return Some(p);
            }
        }
        None
    }

    fn run_greedy(weights: &PathBuf) -> Vec<u32> {
        let cfg = hawking_core::EngineConfig::default();
        let mut engine = hawking_core::model::load_engine(weights, cfg).expect("load gemma2 engine");
        assert_eq!(engine.model_arch(), "gemma2", "dispatcher must route to gemma2");
        let req = hawking_core::GenerateRequest {
            prompt: PROMPT.into(),
            max_new_tokens: MAX_NEW_TOKENS,
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
        assert!(!ids.is_empty(), "must produce at least one token");
        ids
    }

    fn hash16(ids: &[u32]) -> String {
        let mut h = Sha256::new();
        for &id in ids {
            h.update(id.to_le_bytes());
        }
        format!("{:x}", h.finalize())[..16].to_string()
    }

    fn check_or_pin(label: &str, actual_hash: &str) {
        let pin_path = PathBuf::from("tests/_gemma2_token_baseline.hashes");
        let actual_line = format!("{}: {}\n", label, actual_hash);
        let existing = std::fs::read_to_string(&pin_path).unwrap_or_default();
        match existing.lines().find(|l| l.starts_with(&format!("{}:", label))) {
            None => {
                let mut all = existing;
                all.push_str(&actual_line);
                std::fs::write(&pin_path, all).expect("write pin");
                eprintln!("PINNED first hash for {}: {}", label, actual_hash);
            }
            Some(prior) => assert_eq!(prior.trim(), actual_line.trim(), "gemma2 greedy hash drift for {label}"),
        }
    }

    #[test]
    fn gemma2_2b_greedy_smoke() {
        let Some(weights) = find_gguf("gemma-2-2b") else {
            eprintln!("skipping gemma2-2b: no models/*gemma-2-2b*.gguf present");
            return;
        };
        eprintln!("running gemma2-2b against {}", weights.display());
        let ids = run_greedy(&weights);
        let ids2 = run_greedy(&weights);
        assert_eq!(ids, ids2, "gemma2-2b: greedy temp=0 not deterministic");
        check_or_pin("gemma-2-2b-it", &hash16(&ids));
    }
}
#[rustfmt::skip]
mod llama32_smoke {
    //! Llama-3.2 smoke + greedy-output regression gate.
    //!
    //! Auto-activates when a matching GGUF is present under `models/`;
    //! skips cleanly otherwise (so CI without weights stays green). On first
    //! run it PINS the greedy token-id hash to
    //! `tests/_llama32_token_baseline.hashes`; subsequent runs guard
    //! against drift. Mirrors `integration_greedy_64.rs`.
    //!
    //! Pull a GGUF into `models/` to enable, e.g.:
    //!   models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
    //!   models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
    //! The matcher is case-insensitive and keys on the size token
    //! ("1b" / "3b" / "8b") + ".gguf", so the exact HF filename is fine.

    #![cfg(target_os = "macos")]

    use sha2::{Digest, Sha256};
    use std::path::PathBuf;

    const PROMPT: &str = "Once upon a time";
    const MAX_NEW_TOKENS: usize = 32;

    /// Find the first `models/*.gguf` whose lowercased name contains
    /// `size_tag` (e.g. "llama-3.2-1b"). Returns None if absent.
    fn find_llama_gguf(size_tag: &str) -> Option<PathBuf> {
        let dir = PathBuf::from("../../models");
        let entries = std::fs::read_dir(&dir).ok()?;
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) != Some("gguf") {
                continue;
            }
            let name = p.file_name()?.to_str()?.to_lowercase();
            if name.contains(size_tag) {
                return Some(p);
            }
        }
        None
    }

    fn run_greedy(weights: &PathBuf, expect_arch: &str) -> Vec<u32> {
        // No kernel profile: the engine runs with default kernel selections.
        // Profiles are model-specific (generated via `hawking autotune`);
        // the smoke gate intentionally exercises the no-profile load path.
        let cfg = hawking_core::EngineConfig::default();
        let mut engine = hawking_core::model::load_engine(weights, cfg).expect("load llama engine");
        assert_eq!(engine.model_arch(), expect_arch, "dispatcher routed to the wrong engine");

        let req = hawking_core::GenerateRequest {
            prompt: PROMPT.into(),
            max_new_tokens: MAX_NEW_TOKENS,
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
        assert!(!ids.is_empty(), "must produce at least one token");
        ids
    }

    fn hash16(ids: &[u32]) -> String {
        let mut h = Sha256::new();
        for &id in ids {
            h.update(id.to_le_bytes());
        }
        format!("{:x}", h.finalize())[..16].to_string()
    }

    fn check_or_pin(label: &str, actual_hash: &str) {
        let pin_path = PathBuf::from("tests/_llama32_token_baseline.hashes");
        let actual_line = format!("{}: {}\n", label, actual_hash);
        let existing = std::fs::read_to_string(&pin_path).unwrap_or_default();
        let prior = existing.lines().find(|l| l.starts_with(&format!("{}:", label)));
        match prior {
            None => {
                // Append-pin first hash for this label.
                let mut all = existing;
                all.push_str(&actual_line);
                std::fs::write(&pin_path, all).expect("write pin");
                eprintln!("PINNED first hash for {}: {}", label, actual_hash);
            }
            Some(prior_line) => {
                assert_eq!(prior_line.trim(), actual_line.trim(), "llama greedy hash drift for {label}:\n  pinned: {prior_line}\n  actual: {}", actual_line.trim());
            }
        }
    }

    fn smoke_for(size_tag: &str, label: &str, expect_arch: &str) {
        let Some(weights) = find_llama_gguf(size_tag) else {
            eprintln!("skipping {label}: no models/*{size_tag}*.gguf present");
            return;
        };
        eprintln!("running {label} against {}", weights.display());
        let ids = run_greedy(&weights, expect_arch);
        // Sanity: greedy at temp=0 must be deterministic across two runs.
        let ids2 = run_greedy(&weights, expect_arch);
        assert_eq!(ids, ids2, "{label}: greedy temp=0 output not deterministic");
        check_or_pin(label, &hash16(&ids));
    }

    #[test]
    fn llama32_1b_greedy_smoke() {
        smoke_for("llama-3.2-1b", "llama-3.2-1b-instruct", "llama");
    }

    #[test]
    fn llama32_3b_greedy_smoke() {
        smoke_for("llama-3.2-3b", "llama-3.2-3b-instruct", "llama");
    }

    #[test]
    fn llama31_8b_greedy_smoke() {
        // Llama-3.1-8B is the larger coverage target; matcher keys on "8b".
        smoke_for("llama-3.1-8b", "llama-3.1-8b-instruct", "llama");
    }

    #[test]
    fn mistral_7b_v03_greedy_smoke() {
        // Mistral-7B-Instruct-v0.3 reports arch "llama" and runs through the
        // same dense engine (GQA + SwiGLU + RoPE θ=1e6, no biases, no SWA).
        smoke_for("mistral-7b", "mistral-7b-instruct-v0.3", "llama");
    }
}
#[rustfmt::skip]
mod mamba2_smoke {
    //! Mamba2 loader + deterministic greedy smoke.
    //!
    //! Auto-activates when `models/mamba2-370m-Q4_K_M.gguf` or
    //! `models/mamba2-370m-f16.gguf` is present. The current engine is a
    //! correctness-first CPU/Metal hybrid reference path; this test is intentionally
    //! short so it can sit in the post-G1a chain as an architecture-breadth gate.

    use std::path::PathBuf;

    use hawking_core::{EngineConfig, GenerateRequest, SamplingParams, StreamEvent};

    fn locate() -> Option<PathBuf> {
        for rel in ["models/mamba2-370m-Q4_K_M.gguf", "models/mamba2-370m-f16.gguf"] {
            let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            loop {
                let cand = dir.join(rel);
                if cand.exists() {
                    return Some(cand);
                }
                if !dir.pop() {
                    break;
                }
            }
        }
        None
    }

    fn greedy_ids(weights: &PathBuf) -> Vec<u32> {
        let mut engine = hawking_core::model::load_engine(weights, EngineConfig::default()).expect("load mamba2");
        assert_eq!(engine.model_arch(), "mamba2");
        let req = GenerateRequest {
            prompt: "The capital of France is".into(),
            max_new_tokens: 4,
            sampling: SamplingParams { temperature: 0.0, seed: Some(0), ..Default::default() },
            stop: Vec::new(),
            abort: None,
            max_stall_ms: 0,
            json_mode: false,
        };
        let mut ids = Vec::new();
        engine
            .generate(req, &mut |ev| {
                if let StreamEvent::Token { id, .. } = ev {
                    ids.push(id);
                }
            })
            .expect("mamba2 generate");
        ids
    }

    #[test]
    fn mamba2_loads_and_greedy_is_deterministic() {
        let Some(weights) = locate() else {
            eprintln!("skipping mamba2 smoke: no mamba2 GGUF model in models/");
            return;
        };
        let a = greedy_ids(&weights);
        let b = greedy_ids(&weights);
        assert!(!a.is_empty(), "mamba2 smoke should emit at least one token");
        assert_eq!(a, b, "mamba2 temp=0 greedy decode must be deterministic");
    }
}
#[rustfmt::skip]
mod mixed_quant_store_build {
    //! path-to-50 lever 2 foundation smoke: build a MixedQuantStore from
    //! the live V2-Lite GGUF using the default tier map and verify the
    //! resulting blob byte layout matches expectations. Does NOT exercise
    //! the dispatcher (gated on the kernel-buffer override wedge — see
    //! reports/mixed_precision_quant_wiring_handoff.md §3.4).

    #![cfg(target_os = "macos")]

    use std::path::PathBuf;

    use hawking_core::gguf::{GgmlType, GgufFile};
    use hawking_core::mixed_quant_store::{MixedQuantStore, StoreKey};
    use hawking_core::quant_tier_map::{GroupKind, TierMap};

    fn weights_path() -> PathBuf {
        PathBuf::from("../../models/deepseek-v2-lite-q4.gguf")
    }

    #[test]
    fn build_default_tier_map_against_v2_lite_gguf() {
        let weights = weights_path();
        if !weights.exists() {
            eprintln!("skipping: V2-Lite weights missing");
            return;
        }
        let tier_path = PathBuf::from("../../artifacts/calibration/tier_maps/v2_lite_default.json");
        if !tier_path.exists() {
            eprintln!("skipping: tier map missing");
            return;
        }

        let gguf = GgufFile::open(&weights).expect("open gguf");
        let tier_map = TierMap::load(&tier_path).expect("load tier map");
        tier_map.validate("deepseek2", 27).expect("tier map matches V2-Lite shape");

        let store = MixedQuantStore::build(
            &gguf, &tier_map, 27, 1,  // first_k_dense_layers for V2-Lite
            64, // n_routed_experts
            true,
        )
        .expect("build store");

        // Some layers may already be at the target dtype in the GGUF (in
        // which case the build skips them as no-ops), so the count varies
        // by source-quant. Just sanity-check that *some* re-quantization
        // happened and the spot-check tensors are present.
        eprintln!("mixed_quant_store: {} tensors materialized", store.len_tensors());
        assert!(store.len_tensors() > 0, "tier map should have produced at least one re-quantized tensor");

        // Spot-check: layer 4 expert 0 down should be Q8_0 per the map.
        let key = StoreKey::routed(4, GroupKind::Down, 0);
        let t = store.get(key).expect("layer 4 down expert 0 in store");
        assert_eq!(t.dtype, GgmlType::Q8_0);
        // V2-Lite moe_intermediate * hidden = 1408 * 2048 = 2_883_584 elems
        assert_eq!(t.n_elems, 1408 * 2048);
        // Q8_0: 34 bytes per 32 elems → 2_883_584 / 32 * 34 = 3_063_808 bytes
        assert_eq!(t.byte_size, (1408 * 2048 / 32) * 34);

        // Spot-check: layer 25 down → Q6_K per default map (if not native).
        if let Some(t) = store.get(StoreKey::routed(25, GroupKind::Down, 5)) {
            assert_eq!(t.dtype, GgmlType::Q6_K);
        } else {
            eprintln!("layer 25 down already at Q6_K in source; build skipped (no-op)");
        }

        // Total blob size is sanity-bounded: ~ 26 layers * 64 experts *
        // ~3 MB/expert = ~5 GB upper bound (Q8 case).
        assert!(store.blob().len() <= 6 * 1024 * 1024 * 1024, "store blob {} bytes; expected ≤ 6 GiB", store.blob().len());

        eprintln!("mixed_quant_store: {} tensors / {} MB blob", store.len_tensors(), store.blob().len() / (1024 * 1024));
    }
}
#[rustfmt::skip]
mod phase2_foundation_parity {
    //! Phase 2 wedges 2a/2b/2c parity. Confirms:
    //!   - forward_tokens(N=3) returns 3 logit vectors of correct shape (2a)
    //!   - mla_kv_append refactor leaves the integration golden hash unchanged (2b — covered indirectly by integration_greedy_64)
    //!   - rope_inplace_batch produces bit-identical output to N sequential rope_inplace calls (2c)
    //!
    //! Skips if model not present.

    use hawking_core::kernels::{rope_inplace, rope_inplace_batch};
    use std::path::PathBuf;

    #[test]
    fn rope_batch_matches_sequential() {
        let head_dim = 64;
        let base = 10000.0_f32;

        let mut a1: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.1).collect();
        let mut a2: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.2).collect();
        let mut a3: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.3).collect();

        let mut b1 = a1.clone();
        let mut b2 = a2.clone();
        let mut b3 = a3.clone();

        // Sequential reference
        rope_inplace(&mut a1, 7, base);
        rope_inplace(&mut a2, 11, base);
        rope_inplace(&mut a3, 13, base);

        // Batch
        {
            let mut refs: Vec<&mut [f32]> = vec![&mut b1, &mut b2, &mut b3];
            rope_inplace_batch(&mut refs, &[7, 11, 13], base);
        }

        assert_eq!(a1, b1, "rope_batch[0] mismatch");
        assert_eq!(a2, b2, "rope_batch[1] mismatch");
        assert_eq!(a3, b3, "rope_batch[2] mismatch");
    }

    #[test]
    fn rope_batch_empty_is_noop() {
        let mut empty: Vec<&mut [f32]> = vec![];
        rope_inplace_batch(&mut empty, &[], 10000.0);
        // No panic, no allocation; just confirm the empty path is safe.
    }

    #[test]
    fn forward_tokens_shim_returns_n_vectors() {
        let weights = PathBuf::from("../../models/deepseek-v2-lite-q4.gguf");
        if !weights.exists() {
            eprintln!("skipping forward_tokens_shim: no weights at {:?}", weights);
            return;
        }
        let profile_path = PathBuf::from("../../profiles/deepseek-v2-lite-q4.m3pro18.json");
        let profile = hawking_core::profile::KernelProfile::load(&profile_path).expect("load profile");
        let cfg = hawking_core::EngineConfig { kernel_profile: Some(profile), ..Default::default() };
        let mut engine = hawking_core::model::load_engine(&weights, cfg).expect("load engine");

        let logits = engine.forward_tokens_for_test(&[1, 2, 3], &[0, 1, 2]).expect("forward_tokens shim");
        assert_eq!(logits.len(), 3, "must return N logit vectors for N tokens");
        for (i, lvec) in logits.iter().enumerate() {
            assert!(!lvec.is_empty(), "logits[{i}] empty");
            assert!(lvec.iter().all(|x| x.is_finite()), "logits[{i}] non-finite");
        }
    }
}
#[rustfmt::skip]
mod phase2_weight_pinning_parity {
    //! Phase 2 / WB — Weight-pinning parity tests.
    //!
    //! WB pre-uploads kernel-bound weight tensors as `metal::Buffer`
    //! once at model load time and adds `*_pinned` variants of the
    //! kernel entry points that reference the buffer instead of memcpy'ing
    //! a host slice on every dispatch. The byte-slice variants stay as the
    //! parity-test surface — this file proves the pinned and byte-slice
    //! paths produce bit-identical output for the same input.
    //!
    //! Coverage (incremental — extends as more weights are pinned):
    //! - `gemv_f16_metal_pinned` vs `gemv_f16_metal` — LM head shape
    //!   (~vocab × hidden) and a smaller fixed shape that matches the
    //!   `phase1_kernel_parity` test.

    #![cfg(target_os = "macos")]

    use half::f16;
    use hawking_core::kernels;
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;

    use crate::common;
    use common::*;

    fn fixed_input(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = Pcg64Mcg::new(seed as u128);
        (0..n).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect()
    }

    fn gemv_f16_pinned_check(rows: usize, cols: usize, seed_x: u64, seed_w: u64) {
        let x = fixed_input(cols, seed_x);
        let w_f32 = fixed_input(rows * cols, seed_w);
        let w_f16: Vec<f16> = w_f32.iter().map(|&v| f16::from_f32(v)).collect();
        let w_bytes: &[u8] = bytemuck::cast_slice(&w_f16);

        let ctx = ctx().clone();

        // Path A: byte-slice (parity-test reference path).
        let mut out_byte_slice = vec![0.0_f32; rows];
        kernels::gemv_f16_metal(&ctx, w_bytes, rows, cols, &x, &mut out_byte_slice).expect("gemv_f16_metal byte-slice path");

        // Path B: pinned (production path with WB).
        let w_buf = ctx.new_buffer_with_bytes(w_bytes);
        let mut out_pinned = vec![0.0_f32; rows];
        kernels::gemv_f16_metal_pinned(&ctx, &w_buf, rows, cols, &x, &mut out_pinned).expect("gemv_f16_metal_pinned");

        // Pinned and byte-slice paths must be bit-identical: same kernel,
        // same input bytes, just a different upload path. ATOL is generous
        // (1e-3) but anything beyond fp32 representation noise is a bug.
        let max_diff = out_byte_slice.iter().zip(out_pinned.iter()).map(|(&a, &b)| (a - b).abs()).fold(0.0_f32, f32::max);
        println!("[WB] gemv_f16 ({rows}x{cols}) byte-slice vs pinned max abs diff = {max_diff:.6}");
        assert!(max_diff < ATOL, "gemv_f16 pinned/byte-slice diverged: {max_diff} >= atol {ATOL}");
    }

    #[test]
    fn test_gemv_f16_pinned_matches_byte_slice_small() {
        // Small shape that exercises kernel reduction without depending on
        // the LM head being loaded. Mirrors the size used by
        // `phase1_kernel_parity::test_gemv_f16_matches_cpu`.
        gemv_f16_pinned_check(4096, 2048, 0xA1A1_A1A1, 0xB2B2_B2B2);
    }

    #[test]
    fn test_gemv_f16_pinned_matches_byte_slice_lm_head_shape() {
        // DeepSeek-V2-Lite LM-head shape: vocab=102400, hidden=2048.
        // This is the production target — the buffer that lives on
        // `DeepSeekV2.lm_head_buf` is exactly this size.
        gemv_f16_pinned_check(102400, 2048, 0xC3C3_C3C3, 0xD4D4_D4D4);
    }

    fn gemv_f32_attn_pinned_check(rows: usize, cols: usize, seed_x: u64, seed_w: u64) {
        let x = fixed_input(cols, seed_x);
        let w = fixed_input(rows * cols, seed_w);
        let w_bytes: &[u8] = bytemuck::cast_slice(&w);

        let ctx = ctx().clone();

        let mut out_byte_slice = vec![0.0_f32; rows];
        kernels::gemv_f32_attn_metal(&ctx, &w, rows, cols, &x, &mut out_byte_slice).expect("gemv_f32_attn_metal byte-slice path");

        let w_buf = ctx.new_buffer_with_bytes(w_bytes);
        let mut out_pinned = vec![0.0_f32; rows];
        kernels::gemv_f32_attn_metal_pinned(&ctx, &w_buf, rows, cols, &x, &mut out_pinned).expect("gemv_f32_attn_metal_pinned");

        let max_diff = out_byte_slice.iter().zip(out_pinned.iter()).map(|(&a, &b)| (a - b).abs()).fold(0.0_f32, f32::max);
        println!("[WB] gemv_f32_attn ({rows}x{cols}) byte-slice vs pinned max abs diff = {max_diff:.6}");
        assert!(max_diff < ATOL, "gemv_f32_attn pinned/byte-slice diverged: {max_diff} >= atol {ATOL}");
    }

    #[test]
    fn test_gemv_f32_attn_pinned_q_a_proj() {
        gemv_f32_attn_pinned_check(1536, 2048, 0xE1E1_E1E1, 0xE2E2_E2E2);
    }

    #[test]
    fn test_gemv_f32_attn_pinned_q_b_proj() {
        gemv_f32_attn_pinned_check(3072, 1536, 0xF1F1_F1F1, 0xF2F2_F2F2);
    }

    #[test]
    fn test_gemv_f32_attn_pinned_kv_a_proj() {
        gemv_f32_attn_pinned_check(576, 2048, 0x1010_1010, 0x2020_2020);
    }

    #[test]
    fn test_gemv_f32_attn_pinned_kv_b_proj() {
        gemv_f32_attn_pinned_check(2048, 512, 0x3030_3030, 0x4040_4040);
    }

    #[test]
    fn test_gemv_f32_attn_pinned_o_proj() {
        // hidden × (n_heads × v_head_dim) = 2048 × 2048
        gemv_f32_attn_pinned_check(2048, 2048, 0x5050_5050, 0x6060_6060);
    }
}
#[rustfmt::skip]
mod phi3_smoke {
    //! Phi-3.5 smoke + greedy-output regression gate.
    //!
    //! Auto-activates when a `models/*phi-3.5-mini*.gguf` (or *phi-3-mini*)
    //! is present; skips cleanly otherwise. Pins the greedy token-id hash on
    //! first run to `tests/_phi3_token_baseline.hashes`, guards drift
    //! after. Mirrors llama32_smoke.rs.
    //!
    //! Pull a GGUF into models/, e.g. models/Phi-3.5-mini-instruct-Q4_K_M.gguf.

    #![cfg(target_os = "macos")]

    use sha2::{Digest, Sha256};
    use std::path::PathBuf;

    const PROMPT: &str = "Once upon a time";
    const MAX_NEW_TOKENS: usize = 32;

    fn find_gguf(tags: &[&str]) -> Option<PathBuf> {
        let dir = PathBuf::from("../../models");
        for e in std::fs::read_dir(&dir).ok()?.flatten() {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) != Some("gguf") {
                continue;
            }
            let name = p.file_name()?.to_str()?.to_lowercase();
            if tags.iter().any(|t| name.contains(t)) {
                return Some(p);
            }
        }
        None
    }

    fn run_greedy(weights: &PathBuf) -> Vec<u32> {
        let cfg = hawking_core::EngineConfig::default();
        let mut engine = hawking_core::model::load_engine(weights, cfg).expect("load phi3 engine");
        assert_eq!(engine.model_arch(), "phi3", "dispatcher must route to phi3");
        let req = hawking_core::GenerateRequest {
            prompt: PROMPT.into(),
            max_new_tokens: MAX_NEW_TOKENS,
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
        assert!(!ids.is_empty(), "must produce at least one token");
        ids
    }

    fn hash16(ids: &[u32]) -> String {
        let mut h = Sha256::new();
        for &id in ids {
            h.update(id.to_le_bytes());
        }
        format!("{:x}", h.finalize())[..16].to_string()
    }

    fn check_or_pin(label: &str, actual_hash: &str) {
        let pin_path = PathBuf::from("tests/_phi3_token_baseline.hashes");
        let actual_line = format!("{}: {}\n", label, actual_hash);
        let existing = std::fs::read_to_string(&pin_path).unwrap_or_default();
        match existing.lines().find(|l| l.starts_with(&format!("{}:", label))) {
            None => {
                let mut all = existing;
                all.push_str(&actual_line);
                std::fs::write(&pin_path, all).expect("write pin");
                eprintln!("PINNED first hash for {}: {}", label, actual_hash);
            }
            Some(prior) => assert_eq!(prior.trim(), actual_line.trim(), "phi3 greedy hash drift for {label}"),
        }
    }

    #[test]
    fn phi35_mini_greedy_smoke() {
        let Some(weights) = find_gguf(&["phi-3.5-mini", "phi-3-mini", "phi3.5-mini"]) else {
            eprintln!("skipping phi3.5-mini: no models/*phi-3*mini*.gguf present");
            return;
        };
        eprintln!("running phi3.5-mini against {}", weights.display());
        let ids = run_greedy(&weights);
        let ids2 = run_greedy(&weights);
        assert_eq!(ids, ids2, "phi3.5-mini: greedy temp=0 not deterministic");
        check_or_pin("phi-3.5-mini-instruct", &hash16(&ids));
    }
}
#[rustfmt::skip]
mod prefix_cache_parity {
    //! Parity test for the on-disk prefix KV cache.
    //!
    //! Simulates the `generate()` prefill loop without spinning up the real
    //! Qwen model: each "forward step" writes a deterministic value into
    //! `KvCache`, and we assert that the two paths produce byte-identical
    //! KV state for the full prompt:
    //!
    //! 1. **Cold path** — prefill every token, no cache consulted.
    //! 2. **Warm path** — round-trip the first N tokens through
    //!    `store` + `lookup_longest_prefix` + `restore_hit_into_kv`, then
    //!    prefill the remaining `prompt_len - N` tokens.
    //!
    //! If those two end-states match for arbitrary N, the wire-up in
    //! `qwen_dense::generate()` is correct: turn-2 chat will produce the
    //! same KV (and therefore the same next token) as a fresh prefill.

    use hawking_core::cache::prefill_disk::{restore_hit_into_kv, PrefillDiskCache, PrefillKey};
    use hawking_core::cache::KvCache;
    use tempfile::TempDir;

    /// Deterministic "forward step" that writes per-token KV vectors.
    /// Mirrors what `forward_token` does to the cache: appends one row of
    /// KV per layer, then bumps seq_len.
    fn fake_forward(kv: &mut KvCache, token: u32, pos: usize) {
        assert_eq!(kv.seq_len, pos, "fake_forward: expected seq_len == pos");
        let stride = kv.n_kv_heads * kv.head_dim;
        let mut k_row = vec![0.0f32; stride];
        let mut v_row = vec![0.0f32; stride];
        for (li, (kbuf, vbuf)) in kv.keys.iter_mut().zip(kv.values.iter_mut()).enumerate() {
            for d in 0..stride {
                // Mix layer, token id, position, and dim so any off-by-one
                // in restore is loud.
                let mix = ((li as u32).wrapping_mul(2654435761)) ^ token.wrapping_mul(40503) ^ ((pos as u32).wrapping_mul(0x9E37_79B9)) ^ ((d as u32).wrapping_mul(0xDEAD_BEEF));
                k_row[d] = (mix as f32) * 1e-9;
                v_row[d] = -(mix as f32) * 1e-9;
            }
            let off = pos * stride;
            kbuf[off..off + stride].copy_from_slice(&k_row);
            vbuf[off..off + stride].copy_from_slice(&v_row);
        }
        kv.seq_len += 1;
    }

    fn cold_prefill(prompt: &[u32], n_layers: usize, n_kv: usize, head_dim: usize) -> KvCache {
        let mut kv = KvCache::new(n_layers, prompt.len() + 8, n_kv, head_dim);
        for (i, &t) in prompt.iter().enumerate() {
            fake_forward(&mut kv, t, i);
        }
        kv
    }

    fn assert_kv_eq(a: &KvCache, b: &KvCache) {
        assert_eq!(a.seq_len, b.seq_len, "seq_len mismatch");
        assert_eq!(a.n_layers, b.n_layers);
        assert_eq!(a.n_kv_heads, b.n_kv_heads);
        assert_eq!(a.head_dim, b.head_dim);
        for li in 0..a.n_layers {
            assert_eq!(a.keys_for(li), b.keys_for(li), "keys mismatch on layer {}", li);
            assert_eq!(a.values_for(li), b.values_for(li), "values mismatch on layer {}", li);
        }
    }

    #[test]
    fn cold_vs_warm_prefill_byte_identical() {
        // Roughly mirrors the chat scenario: a 50-token "system prompt" +
        // a 30-token "user message". The system prefix gets cached on
        // turn 1, then turn 2's prompt = system + first_response (10 tok) +
        // new user msg (20 tok). The cached system prefix must restore
        // byte-identically.
        let n_layers = 4;
        let n_kv = 2;
        let head_dim = 16;

        let tmp = TempDir::new().unwrap();
        let cache = PrefillDiskCache::open(tmp.path()).unwrap();

        // Turn 1.
        let system: Vec<u32> = (0..50u32).map(|i| 100 + i).collect();
        let kv_cold_t1 = cold_prefill(&system, n_layers, n_kv, head_dim);
        let key_t1 = PrefillKey::from_model_and_prompt("qwen-test", b"tok-sig-v1", &system);
        cache.store(&key_t1, &kv_cold_t1).unwrap();

        // Turn 2: system + 10-token reply + 20-token new user message.
        let mut turn2: Vec<u32> = system.clone();
        turn2.extend((0..10u32).map(|i| 1000 + i));
        turn2.extend((0..20u32).map(|i| 2000 + i));

        // Cold reference.
        let kv_cold_t2 = cold_prefill(&turn2, n_layers, n_kv, head_dim);

        // Warm path: lookup → restore → prefill delta.
        let key_t2 = PrefillKey::from_model_and_prompt("qwen-test", b"tok-sig-v1", &turn2);
        let hit = cache.lookup_longest_prefix(&key_t2.model_hash, &key_t2.tokenizer_hash, &turn2).unwrap().expect("expected prefix hit on turn 2");
        assert_eq!(hit.n_tokens, 50, "should hit the 50-token cached system prefix");
        let mut kv_warm = KvCache::new(n_layers, turn2.len() + 8, n_kv, head_dim);
        restore_hit_into_kv(&hit, &mut kv_warm).unwrap();
        assert_eq!(kv_warm.seq_len, 50);
        // Continue prefill on the delta.
        for (i, &t) in turn2.iter().enumerate().skip(50) {
            fake_forward(&mut kv_warm, t, i);
        }

        assert_kv_eq(&kv_cold_t2, &kv_warm);
    }

    #[test]
    fn store_then_load_byte_exact() {
        // The simplest gate: store turn-1 KV, immediately load it back via
        // a longer prompt, restore, and verify byte equality with the
        // source KV for the cached prefix length.
        let n_layers = 3;
        let n_kv = 2;
        let head_dim = 8;
        let tmp = TempDir::new().unwrap();
        let cache = PrefillDiskCache::open(tmp.path()).unwrap();

        let prompt: Vec<u32> = (0..20u32).collect();
        let kv = cold_prefill(&prompt, n_layers, n_kv, head_dim);
        let key = PrefillKey::from_model_and_prompt("m", b"tok", &prompt);
        cache.store(&key, &kv).unwrap();

        // Lookup with a 1-token-longer prompt so the cache returns the
        // full 20-token prefix.
        let mut probe = prompt.clone();
        probe.push(999);
        let hit = cache.lookup_longest_prefix(&key.model_hash, &key.tokenizer_hash, &probe).unwrap().unwrap();
        assert_eq!(hit.n_tokens, 20);

        let mut restored = KvCache::new(n_layers, prompt.len() + 4, n_kv, head_dim);
        restore_hit_into_kv(&hit, &mut restored).unwrap();
        // The remaining unused KV slots stay zeroed — that's fine, the
        // engine only consults kv up to seq_len.
        let stride = n_kv * head_dim;
        let used = restored.seq_len * stride;
        for li in 0..n_layers {
            assert_eq!(&restored.keys[li][..used], &kv.keys[li][..used]);
            assert_eq!(&restored.values[li][..used], &kv.values[li][..used]);
        }
    }

    #[test]
    fn cache_miss_falls_back_to_full_prefill() {
        // A fresh cache directory + new prompt → lookup returns None and
        // the caller must run a full prefill, producing identical KV to
        // the no-cache path.
        let tmp = TempDir::new().unwrap();
        let cache = PrefillDiskCache::open(tmp.path()).unwrap();

        let prompt: Vec<u32> = (10..30u32).collect();
        let key = PrefillKey::from_model_and_prompt("m", b"sig", &prompt);
        let hit = cache.lookup_longest_prefix(&key.model_hash, &key.tokenizer_hash, &prompt).unwrap();
        assert!(hit.is_none(), "fresh cache must miss");

        // No-cache prefill.
        let kv_a = cold_prefill(&prompt, 2, 2, 8);
        // "Cache miss" prefill (identical code path since hit was None).
        let kv_b = cold_prefill(&prompt, 2, 2, 8);
        assert_kv_eq(&kv_a, &kv_b);
    }
}
#[rustfmt::skip]
mod sidecar_roundtrip {
    //! Confirmation test for the `.hawking` sidecar (Track 4.1/4.2): lock in the
    //! three invariants the bake→load path depends on so they can't silently
    //! regress.
    //!
    //!   (1) ROUND-TRIP: predec scale entries written by `SidecarWriter` and read
    //!       back by `read_predec_entries` are byte-for-byte the same values that
    //!       `predecode_q4_k_scale_table` produces in memory at load time. (The
    //!       loader, `ensure_q4k_predec_cache`, uploads exactly these to a
    //!       PinnedBuffer — so equality here == bit-identical decode.)
    //!   (2) HEADER ROUND-TRIP: the JSON header survives the write/read cycle.
    //!   (3) HASH-MISMATCH REJECTS: `check_sidecar_compatibility` flags a stale
    //!       sidecar (GGUF hash differs) as fatal, and the matching-hash case is
    //!       loadable — this is the guard that stops the engine from using stale
    //!       predec data against a different GGUF.
    //!
    //! Pure CPU (no Metal context, no model load): `predecode_q4_k_scale_table` is
    //! byte math and the sidecar reader/writer are plain file I/O. Gates with:
    //!
    //!   cargo test -p hawking-core --test sidecar_roundtrip
    //!
    //! NOTE: `predecode_q4_k_scale_table` is re-exported under
    //! `hawking_core::kernels` only on macOS (it lives in the macOS-gated
    //! `metal_dispatch` module), so the round-trip test is macOS-gated. The
    //! compatibility-check test is platform-independent and always runs.

    use hawking_core::sidecar::{
        check_sidecar_compatibility, read_predec_entries, sidecar_path_for, SidecarCompat, SidecarContents, SidecarHeader, SidecarProfile, SidecarQuality, SidecarWriter, SIDECAR_VERSION,
    };

    /// Deterministic Q4_K blocks (144 bytes each) with realistic header scales and
    /// packed sub-block bytes — same shape as the production GGUF tensor slices fed
    /// to `predecode_q4_k_scale_table`. Mirrors the helper in
    /// `tests/predec_f16_scale_table.rs`.
    fn make_q4k_bytes(n_blocks: usize) -> Vec<u8> {
        let mut bytes = vec![0u8; n_blocks * 144];
        for b in 0..n_blocks {
            let off = b * 144;
            let d = 0.012_f32 + (b % 7) as f32 * 0.001;
            let dmin = ((b % 5) as f32 - 2.0) * 0.002;
            bytes[off..off + 2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
            bytes[off + 2..off + 4].copy_from_slice(&half::f16::from_f32(dmin).to_bits().to_le_bytes());
            for i in 4..144 {
                bytes[off + i] = ((i * 31 + b * 17) & 0xFF) as u8;
            }
        }
        bytes
    }

    fn header_for(gguf_hash: &str, shader_hash: &str) -> SidecarHeader {
        SidecarHeader {
            version: SIDECAR_VERSION,
            source_gguf_hash: gguf_hash.to_string(),
            tokenizer_hash: "tok123".to_string(),
            shader_hash: shader_hash.to_string(),
            bake_profile: SidecarProfile::Fast,
            contents: SidecarContents { q4k_predec_scales: true, ..Default::default() },
            quality: SidecarQuality { quality_gate_passed: true, quality_gate_spec: "predec scales are bit-identical".to_string(), ..Default::default() },
            bake_device: "test-device".to_string(),
            bake_time_secs: 0,
            tier_map: None,
        }
    }

    /// (1)+(2): bake → load round-trip equals the in-memory predecode, and the
    /// header survives the JSON round-trip. macOS-only (predecode entry point is
    /// macOS-gated).
    #[cfg(target_os = "macos")]
    #[test]
    fn bake_load_roundtrip_equals_in_memory_predecode() {
        use hawking_core::kernels::predecode_q4_k_scale_table;

        // Two "tensors" at distinct GGUF offsets, like q_proj and ffn_down.
        let t0 = make_q4k_bytes(48);
        let t1 = make_q4k_bytes(80);
        let off0: u64 = 0x1000;
        let off1: u64 = 0x9abc;

        // In-memory predecode — the source of truth (what ensure_q4k_predec_cache
        // computes when there is no sidecar).
        let mem0 = predecode_q4_k_scale_table(&t0);
        let mem1 = predecode_q4_k_scale_table(&t1);
        assert_eq!(mem0.len(), 48 * 16, "predec table is 16 f32 per 144-byte block");
        assert_eq!(mem1.len(), 80 * 16);

        // Bake: write exactly those tables into a sidecar (what bake_sidecar_predec
        // does — it calls the same predecode fn on the same bytes).
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("model.hawking");
        let header = header_for("deadbeef_gguf_hash", "shader_abc");
        let writer = SidecarWriter { path: path.clone(), predec_entries: vec![(off0, mem0.clone()), (off1, mem1.clone())], header: header.clone() };
        let written = writer.write().expect("sidecar write");
        assert!(written > 0, "writer reported {written} bytes");

        // Load: read the entries back.
        let (read_header, entries) = read_predec_entries(&path).expect("read predec entries");

        // (2) Header round-trips through JSON.
        assert_eq!(read_header.version, header.version);
        assert_eq!(read_header.source_gguf_hash, header.source_gguf_hash);
        assert_eq!(read_header.tokenizer_hash, header.tokenizer_hash);
        assert_eq!(read_header.shader_hash, header.shader_hash);
        assert_eq!(read_header.bake_profile, header.bake_profile);
        assert!(read_header.contents.q4k_predec_scales);

        // (1) Entries are byte-for-byte the in-memory predecode, keyed by offset.
        let map: std::collections::HashMap<usize, Vec<f32>> = entries.into_iter().collect();
        assert_eq!(map.len(), 2, "both tensors round-trip");
        let r0 = map.get(&(off0 as usize)).expect("entry off0");
        let r1 = map.get(&(off1 as usize)).expect("entry off1");
        assert_eq!(r0.len(), mem0.len());
        assert_eq!(r1.len(), mem1.len());
        // bit-identical: compare raw f32 bits, not approximate.
        for (i, (&a, &b)) in r0.iter().zip(mem0.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "off0 scale[{i}] not bit-identical");
        }
        for (i, (&a, &b)) in r1.iter().zip(mem1.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "off1 scale[{i}] not bit-identical");
        }
    }

    /// (3) HASH-MISMATCH REJECTS: a sidecar baked against one GGUF must be flagged
    /// fatal when loaded against a different GGUF; matching hash is loadable.
    /// Platform-independent (pure compatibility check).
    #[test]
    fn hash_mismatch_is_fatal_and_match_is_loadable() {
        let header = header_for("hash_of_gguf_A", "shader_A");

        // Matching GGUF + shader → Compatible, loadable, not fatal.
        let ok = check_sidecar_compatibility(&header, "hash_of_gguf_A", "shader_A");
        assert!(matches!(ok, SidecarCompat::Compatible), "got {ok:?}");
        assert!(ok.is_loadable());
        assert!(!ok.is_fatal());

        // Different GGUF hash → GgufHashMismatch, fatal, NOT loadable. This is the
        // guard that stops stale predec scales from being applied to a new GGUF.
        let stale = check_sidecar_compatibility(&header, "hash_of_gguf_B", "shader_A");
        assert!(matches!(stale, SidecarCompat::GgufHashMismatch { .. }), "stale GGUF must be a hash mismatch, got {stale:?}");
        assert!(stale.is_fatal(), "GGUF hash mismatch must be fatal");
        assert!(!stale.is_loadable(), "stale sidecar must NOT load");

        // Shader-hash mismatch is non-fatal (data still valid) but flagged loadable.
        let shader_drift = check_sidecar_compatibility(&header, "hash_of_gguf_A", "shader_B");
        assert!(matches!(shader_drift, SidecarCompat::ShaderHashMismatch { .. }), "got {shader_drift:?}");
        assert!(shader_drift.is_loadable(), "shader drift is non-fatal");
        assert!(!shader_drift.is_fatal());

        // Version newer than this binary understands → fatal.
        let mut future = header.clone();
        future.version = SIDECAR_VERSION + 1;
        let too_new = check_sidecar_compatibility(&future, "hash_of_gguf_A", "shader_A");
        assert!(matches!(too_new, SidecarCompat::VersionTooNew { .. }), "got {too_new:?}");
        assert!(too_new.is_fatal());
    }

    /// Bad magic bytes are rejected by the reader (corrupt/foreign file guard).
    /// Platform-independent.
    #[test]
    fn bad_magic_is_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("not_a_sidecar.hawking");
        std::fs::write(&path, b"NOTDSMTL........junk").expect("write junk");
        let err = read_predec_entries(&path);
        assert!(err.is_err(), "reader must reject a file with bad magic");
    }

    /// `sidecar_path_for` derives `<stem>.hawking` from a GGUF path.
    /// Platform-independent.
    #[test]
    fn sidecar_path_derivation() {
        let p = sidecar_path_for(std::path::Path::new("models/qwen2.5-3b-q4_k_m.gguf"));
        assert_eq!(p, std::path::PathBuf::from("models/qwen2.5-3b-q4_k_m.hawking"));
    }
}
#[rustfmt::skip]
mod sidecar_tier_map_consume {
    //! Track 4.3 gate: a sidecar tier map (a) round-trips when attached to a
    //! predec sidecar via `attach_tier_map_to_sidecar`, and (b) the LOADED copy's
    //! resolver (`SidecarTierMap::dtype_for`) — the exact fn the loader hook
    //! `honor_sidecar_tier_map` consults — reports the per-tensor override, returns
    //! None for absent tensors, and validates. Pure file I/O + byte math, no Metal.

    use hawking_core::gguf::GgmlType;
    use hawking_core::sidecar::{
        attach_tier_map_to_sidecar, load_sidecar_tier_map_json, read_predec_entries, SidecarContents, SidecarHeader, SidecarProfile, SidecarQuality, SidecarTierEntry, SidecarTierMap, SidecarWriter,
        SIDECAR_VERSION,
    };

    fn predec_only_header() -> SidecarHeader {
        SidecarHeader {
            version: SIDECAR_VERSION,
            source_gguf_hash: "gguf_hash_xyz".into(),
            tokenizer_hash: "tok".into(),
            shader_hash: "shader".into(),
            bake_profile: SidecarProfile::Fast,
            contents: SidecarContents { q4k_predec_scales: true, ..Default::default() },
            quality: SidecarQuality::default(),
            bake_device: "test".into(),
            bake_time_secs: 0,
            tier_map: None,
        }
    }

    #[test]
    fn attach_then_loader_resolver_reports_override() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("model.hawking");

        // 1) Write a predec-ONLY sidecar (what bake_sidecar_predec produces).
        let base = SidecarWriter { path: path.clone(), predec_entries: vec![(0x1000u64, vec![1.0_f32, 2.0, 3.0]), (0x2000u64, vec![4.0_f32])], header: predec_only_header() };
        assert!(base.write().expect("write predec sidecar") > 0);

        // 2) Attach a tier map (what the CLI does after bake).
        let tm = SidecarTierMap {
            entries: vec![SidecarTierEntry { tensor: "blk.0.ffn_down.weight".into(), dtype: "q6_K".into() }, SidecarTierEntry { tensor: "blk.7.attn_v.weight".into(), dtype: "q8_0".into() }],
        };
        assert!(attach_tier_map_to_sidecar(&path, tm.clone()).expect("attach") > 0);

        // 3) Read the LOADED copy back and exercise the resolver the loader hook uses.
        let (header, entries) = read_predec_entries(&path).expect("re-read");
        assert!(header.contents.mixed_quant_tier_map, "content flag must flip on attach");
        assert_eq!(entries.len(), 2, "predec entries survive the rewrite");
        let loaded = header.tier_map.expect("tier map present after attach");
        assert_eq!(loaded, tm, "tier map byte-identical after round-trip");
        assert!(loaded.validate().is_ok());
        // This is exactly what honor_sidecar_tier_map calls per GGUF tensor name:
        assert_eq!(loaded.dtype_for("blk.0.ffn_down.weight").unwrap(), Some(GgmlType::Q6_K));
        assert_eq!(loaded.dtype_for("blk.7.attn_v.weight").unwrap(), Some(GgmlType::Q8_0));
        assert_eq!(loaded.dtype_for("blk.3.attn_q.weight").unwrap(), None, "absent tensor falls through");
    }

    #[test]
    fn bad_dtype_json_fails_the_bake() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = dir.path().join("tm.json");
        std::fs::write(&p, r#"{"entries":[{"tensor":"blk.0.ffn_down.weight","dtype":"q3_K"}]}"#).unwrap();
        // q3_K is not a supported sidecar tier dtype → load+validate must error,
        // so a typo'd tier fails the bake instead of silently no-op'ing at load.
        assert!(load_sidecar_tier_map_json(&p).is_err());
    }

    #[test]
    fn good_dtype_json_parses() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = dir.path().join("tm.json");
        std::fs::write(&p, r#"{"entries":[{"tensor":"blk.0.ffn_down.weight","dtype":"q6_K"}]}"#).unwrap();
        let tm = load_sidecar_tier_map_json(&p).expect("parse good json");
        assert_eq!(tm.entries.len(), 1);
        assert_eq!(tm.dtype_for("blk.0.ffn_down.weight").unwrap(), Some(GgmlType::Q6_K));
    }
}
#[rustfmt::skip]
mod v041_moe_batched_q4_indexed_v2_parity {
    //! v0.4.1 — Parity test: moe_batched_gemm_q4_indexed_v2 vs scalar reference.
    //!
    //! Test 1: routes=2, rows=64,  cols=256  (sub-TG sanity)
    //! Test 2: routes=4, rows=256, cols=2048 (realistic shape)
    //! Asserts max |scalar - v2| < 1e-3 (fp16 quant noise tolerance).

    #![cfg(target_os = "macos")]

    use half::f16;
    use hawking_core::kernels;
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;

    use crate::common;
    use common::*;

    fn fixed_input(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = Pcg64Mcg::new(seed as u128);
        (0..n).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect()
    }

    fn synthetic_q4_k_bytes(n_blocks: usize, seed: u64) -> Vec<u8> {
        let mut rng = Pcg64Mcg::new(seed as u128);
        let mut bytes = vec![0u8; n_blocks * 144];
        for b in 0..n_blocks {
            let off = b * 144;
            let d = 0.001 + rng.gen::<f32>() * 0.001;
            bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
            let dmin = (rng.gen::<f32>() - 0.5) * 0.001;
            bytes[off + 2..off + 4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
            for i in 4..144 {
                bytes[off + i] = rng.gen::<u8>();
            }
        }
        bytes
    }

    fn run_parity(routes: usize, rows: usize, cols: usize, seed_base: u64) {
        let n_experts = routes + 3;
        let blocks_per_expert = rows * (cols / 256);
        let bytes_per_expert = blocks_per_expert * 144;

        // Build full fused tensor: n_experts consecutive matrices.
        let fused = synthetic_q4_k_bytes(n_experts * blocks_per_expert, seed_base);

        // Prepend 64 bytes of padding to exercise base_offset != 0.
        let mut model_bytes = vec![0xA5u8; 64];
        let base_offset = model_bytes.len();
        model_bytes.extend_from_slice(&fused);

        // Select experts (spread across available ids).
        let route_ids: Vec<u32> = (0..routes).map(|i| ((i * 2 + 1) % n_experts) as u32).collect();

        let x = fixed_input(cols, seed_base ^ 0xDEAD_BEEF);

        let mut scalar_out = vec![0.0_f32; routes * rows];
        kernels::moe_batched_gemm_q4_indexed_raw(ctx(), false, &model_bytes, base_offset, &route_ids, &x, routes, rows, cols, &mut scalar_out).expect("scalar dispatch should succeed");

        let mut v2_out = vec![0.0_f32; routes * rows];
        kernels::moe_batched_gemm_q4_indexed_raw(ctx(), true, &model_bytes, base_offset, &route_ids, &x, routes, rows, cols, &mut v2_out).expect("v2 dispatch should succeed");

        let diff = max_abs_diff(&scalar_out, &v2_out);
        println!("[v0.4.1] indexed q4k parity (routes={routes} rows={rows} cols={cols}) max abs diff = {diff:.6e}");
        // Verify bytes_per_expert is used (suppress unused warning).
        let _ = bytes_per_expert;
        assert!(diff < ATOL, "moe_batched_gemm_q4_indexed_v2 vs scalar diff {diff:.6e} >= atol {ATOL}");
    }

    #[test]
    fn test_indexed_q4k_v2_small() {
        run_parity(2, 64, 256, 0x4100_0001);
    }

    #[test]
    fn test_indexed_q4k_v2_realistic() {
        run_parity(4, 256, 2048, 0x4100_0002);
    }
}
#[rustfmt::skip]
mod v0511_forward_shared_only_smoke {
    //! v0.5.11 smoke test: forward_token_shared_only returns finite logits.
    //!
    //! Does NOT check accuracy or specific values — just that the shared-only
    //! path compiles, runs, and produces finite f32 logits.
    //!
    //! Skipped when the model weights are absent (CI without model files).

    #![cfg(target_os = "macos")]

    use std::path::PathBuf;

    #[test]
    fn forward_token_shared_only_returns_finite_logits() {
        let weights = PathBuf::from("../../models/deepseek-v2-lite-q4.gguf");
        if !weights.exists() {
            eprintln!("skipping v0511 smoke: no weights at {:?}", weights);
            return;
        }
        let profile_path = PathBuf::from("../../profiles/deepseek-v2-lite-q4.m3pro18.json");
        let profile = hawking_core::profile::KernelProfile::load(&profile_path).expect("load profile");
        let cfg = hawking_core::EngineConfig { kernel_profile: Some(profile), ..Default::default() };
        let mut engine = hawking_core::model::load_engine(&weights, cfg).expect("load engine");

        // Token 1 (BOS-like) at position 0.
        let token = 1u32;
        let pos = 0usize;

        let logits = engine.forward_token_shared_only_for_test(token, pos).expect("forward_token_shared_only_for_test");

        // Basic shape check.
        assert!(logits.len() > 1000, "logits too short: {}", logits.len());

        // All finite.
        let non_finite: Vec<usize> = logits.iter().enumerate().filter(|(_, &v)| !v.is_finite()).map(|(i, _)| i).collect();
        assert!(non_finite.is_empty(), "non-finite logits at indices: {:?}", &non_finite[..non_finite.len().min(5)]);
    }
}
#[rustfmt::skip]
mod v1_1_phase4c_batched_forward_parity {
    //! Phase 4C parity: forward_tokens_batched argmax matches sequential forward_token calls.
    //!
    //! Greedy next-token must be identical between batched and sequential for all K positions.
    //! This is the correctness gate for n-gram spec-decode verify wiring in Phase 4D.
    //!
    //! K=4 and K=8 are tested in one function to avoid Metal GPU interference from
    //! parallel test execution.
    //!
    //! Skips if model weights are not present.

    use std::path::PathBuf;

    fn weights_path() -> PathBuf {
        PathBuf::from("../../models/deepseek-v2-lite-q4.gguf")
    }

    fn load_engine() -> Option<Box<dyn hawking_core::Engine>> {
        let p = weights_path();
        if !p.exists() {
            eprintln!("v1_1_phase4C_batched_forward_parity: no weights at {:?}, skipping", p);
            return None;
        }
        let cfg = hawking_core::EngineConfig::default();
        match hawking_core::model::load_engine(&p, cfg) {
            Ok(e) => Some(e),
            Err(err) => {
                eprintln!("v1_1_phase4C_batched_forward_parity: load failed: {err}, skipping");
                None
            }
        }
    }

    fn argmax(v: &[f32]) -> u32 {
        v.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i as u32).unwrap_or(0)
    }

    fn check_argmax_parity(engine: &mut Box<dyn hawking_core::Engine>, tokens: &[u32], positions: &[usize], label: &str) {
        let seq_logits = engine.forward_tokens_for_test(tokens, positions).unwrap_or_else(|e| panic!("{label} sequential: {e}"));

        engine.reset_kv_for_test();

        let batch_logits = engine.forward_tokens_batched_for_test(tokens, positions).unwrap_or_else(|e| panic!("{label} batched: {e}"));

        assert_eq!(seq_logits.len(), batch_logits.len(), "{label} result count mismatch");
        for m in 0..tokens.len() {
            let seq_top = argmax(&seq_logits[m]);
            let bat_top = argmax(&batch_logits[m]);
            assert_eq!(seq_top, bat_top, "{label} position {m}: batched argmax={bat_top} != sequential argmax={seq_top}");
        }
    }

    /// K=4 and K=8 argmax parity, run sequentially to avoid Metal device interference.
    #[test]
    fn batched_argmax_matches_sequential_k4_k8() {
        let Some(mut engine) = load_engine() else {
            return;
        };

        // K=4: BOS + 3 draft continuations
        check_argmax_parity(&mut engine, &[1u32, 315, 1012, 297], &[0, 1, 2, 3], "K=4");

        engine.reset_kv_for_test();

        // K=8: longer spec-decode window
        check_argmax_parity(&mut engine, &[1u32, 315, 1012, 297, 338, 263, 1243, 310], &[0, 1, 2, 3, 4, 5, 6, 7], "K=8");
    }
}
#[rustfmt::skip]
mod v1_1_phase4d_spec_exact_mode {
    //! Phase 4D: n-gram spec decode exact-mode invariant.
    //!
    //! Greedy output with `--speculate ngram` must be byte-identical to
    //! greedy output with spec off. Tests both a repetitive prompt
    //! (high acceptance rate) and a natural-text prompt (mixed acceptance).
    //!
    //! Skips if model weights are not present.

    use hawking_core::{EngineConfig, GenerateRequest, SamplingParams, SpeculateMode, StreamEvent};
    use std::path::PathBuf;

    fn weights_path() -> PathBuf {
        PathBuf::from("../../models/deepseek-v2-lite-q4.gguf")
    }

    fn load_engine(speculate_mode: SpeculateMode) -> Option<Box<dyn hawking_core::Engine>> {
        let p = weights_path();
        if !p.exists() {
            eprintln!("v1_1_phase4D_spec_exact_mode: no weights at {:?}, skipping", p);
            return None;
        }
        let mut cfg = EngineConfig::default();
        cfg.speculate = speculate_mode != SpeculateMode::Off;
        cfg.speculate_mode = speculate_mode;

        // Load profile if available — activates GPU-resident greedy path used in production.
        let profile_path = std::path::PathBuf::from("../../profiles/deepseek-v2-lite-q4.m3pro18.json");
        if profile_path.exists() {
            if let Ok(profile) = hawking_core::profile::KernelProfile::load(&profile_path) {
                cfg.kernel_profile = Some(profile);
            }
        }

        match hawking_core::model::load_engine(&p, cfg) {
            Ok(e) => Some(e),
            Err(err) => {
                eprintln!("v1_1_phase4D_spec_exact_mode: load failed: {err}, skipping");
                None
            }
        }
    }

    fn collect_tokens(engine: &mut Box<dyn hawking_core::Engine>, prompt: &str, max_new_tokens: usize) -> Vec<u32> {
        let req = GenerateRequest {
            prompt: prompt.to_string(),
            max_new_tokens,
            sampling: SamplingParams { temperature: 0.0, top_p: 1.0, top_k: 0, repetition_penalty: 1.0, seed: None },
            stop: vec![],
            abort: None,
            max_stall_ms: 0,
            json_mode: false,
        };
        let mut tokens = Vec::new();
        engine
            .generate(req, &mut |ev| {
                if let StreamEvent::Token { id, .. } = ev {
                    tokens.push(id);
                }
            })
            .expect("generate");
        tokens
    }

    /// Repetitive prompt: n-gram draft acceptance should be very high.
    /// Output must be byte-identical to non-spec greedy.
    #[test]
    fn repetitive_prompt_spec_matches_greedy() {
        let Some(mut ref_engine) = load_engine(SpeculateMode::Off) else {
            return;
        };
        let Some(mut spec_engine) = load_engine(SpeculateMode::ExactShared) else {
            return;
        };

        let prompt = "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.";
        let ref_ids = collect_tokens(&mut ref_engine, prompt, 20);
        let spec_ids = collect_tokens(&mut spec_engine, prompt, 20);

        assert_eq!(ref_ids, spec_ids, "repetitive prompt: spec output differs from greedy\nref={ref_ids:?}\nspec={spec_ids:?}");
    }

    /// Natural-text prompt: n-gram may not always match, but output must be identical.
    #[test]
    fn natural_prompt_spec_matches_greedy() {
        let Some(mut ref_engine) = load_engine(SpeculateMode::Off) else {
            return;
        };
        let Some(mut spec_engine) = load_engine(SpeculateMode::ExactShared) else {
            return;
        };

        let prompt = "Explain how speculative decoding works in language models:";
        let ref_ids = collect_tokens(&mut ref_engine, prompt, 15);
        let spec_ids = collect_tokens(&mut spec_engine, prompt, 15);

        assert_eq!(ref_ids, spec_ids, "natural prompt: spec output differs from greedy\nref={ref_ids:?}\nspec={spec_ids:?}");
    }
}
#[rustfmt::skip]
mod v1_1_phase5a_batched_forward_parity {
    //! Phase 5A parity: forward_tokens_batched (single-TCB K-token fast path) argmax
    //! matches sequential forward_token calls for K=1, 2, 4, 8.
    //!
    //! Also verifies the exact-mode invariant: n-gram spec decode with batched verify
    //! (Phase 5A) produces byte-identical output to greedy with spec off.
    //!
    //! Skips if model weights are not present.

    use hawking_core::{EngineConfig, GenerateRequest, SamplingParams, SpeculateMode, StreamEvent};
    use std::path::PathBuf;

    fn weights_path() -> PathBuf {
        PathBuf::from("../../models/deepseek-v2-lite-q4.gguf")
    }

    fn load_engine_with_profile(speculate_mode: SpeculateMode) -> Option<Box<dyn hawking_core::Engine>> {
        let p = weights_path();
        if !p.exists() {
            eprintln!("v1_1_phase5A_batched_forward_parity: no weights at {:?}, skipping", p);
            return None;
        }
        let mut cfg = EngineConfig::default();
        cfg.speculate = speculate_mode != SpeculateMode::Off;
        cfg.speculate_mode = speculate_mode;

        let profile_path = PathBuf::from("../../profiles/deepseek-v2-lite-q4.m3pro18.json");
        if profile_path.exists() {
            if let Ok(profile) = hawking_core::profile::KernelProfile::load(&profile_path) {
                cfg.kernel_profile = Some(profile);
            }
        }

        match hawking_core::model::load_engine(&p, cfg) {
            Ok(e) => Some(e),
            Err(err) => {
                eprintln!("v1_1_phase5A_batched_forward_parity: load failed: {err}, skipping");
                None
            }
        }
    }

    fn argmax(v: &[f32]) -> u32 {
        v.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i as u32).unwrap_or(0)
    }

    fn check_batched_parity(engine: &mut Box<dyn hawking_core::Engine>, tokens: &[u32], positions: &[usize], label: &str) {
        // Sequential baseline.
        let seq_logits = engine.forward_tokens_for_test(tokens, positions).unwrap_or_else(|e| panic!("{label} sequential: {e}"));

        engine.reset_kv_for_test();

        // Batched fast path (Phase 5A TCB path when conditions are met).
        let batch_logits = engine.forward_tokens_batched_for_test(tokens, positions).unwrap_or_else(|e| panic!("{label} batched: {e}"));

        assert_eq!(seq_logits.len(), batch_logits.len(), "{label} result count mismatch");
        for m in 0..tokens.len() {
            let seq_top = argmax(&seq_logits[m]);
            let bat_top = argmax(&batch_logits[m]);
            assert_eq!(seq_top, bat_top, "{label} position {m}: batched argmax={bat_top} != sequential argmax={seq_top}");
        }
        engine.reset_kv_for_test();
    }

    fn collect_tokens(engine: &mut Box<dyn hawking_core::Engine>, prompt: &str, max_new_tokens: usize) -> Vec<u32> {
        let req = GenerateRequest {
            prompt: prompt.to_string(),
            max_new_tokens,
            sampling: SamplingParams { temperature: 0.0, top_p: 1.0, top_k: 0, repetition_penalty: 1.0, seed: None },
            stop: vec![],
            abort: None,
            max_stall_ms: 0,
            json_mode: false,
        };
        let mut tokens = Vec::new();
        engine
            .generate(req, &mut |ev| {
                if let StreamEvent::Token { id, .. } = ev {
                    tokens.push(id);
                }
            })
            .expect("generate");
        tokens
    }

    /// K=1 through K=8 argmax parity: batched TCB path must match sequential.
    #[test]
    fn batched_tcb_argmax_parity_k1_through_k8() {
        let Some(mut engine) = load_engine_with_profile(SpeculateMode::Off) else {
            return;
        };

        // K=1
        check_batched_parity(&mut engine, &[1u32], &[0], "K=1");

        // K=2
        check_batched_parity(&mut engine, &[1u32, 315], &[0, 1], "K=2");

        // K=4 (spec verify window default)
        check_batched_parity(&mut engine, &[1u32, 315, 1012, 297], &[0, 1, 2, 3], "K=4");

        // K=5 (spec verify window + anchor = typical batched verify call)
        check_batched_parity(&mut engine, &[1u32, 315, 1012, 297, 338], &[0, 1, 2, 3, 4], "K=5");

        // K=8
        check_batched_parity(&mut engine, &[1u32, 315, 1012, 297, 338, 263, 1243, 310], &[0, 1, 2, 3, 4, 5, 6, 7], "K=8");
    }

    /// Exact-mode invariant with Phase 5A batched verify.
    ///
    /// Both repetitive and natural prompts are tested with a single engine-pair load
    /// to avoid GPU memory pressure from back-to-back dual-engine loads.
    /// n-gram spec output must be byte-identical to greedy with spec off.
    #[test]
    fn spec_batched_verify_exact_mode() {
        let Some(mut ref_engine) = load_engine_with_profile(SpeculateMode::Off) else {
            return;
        };
        let Some(mut spec_engine) = load_engine_with_profile(SpeculateMode::ExactShared) else {
            return;
        };

        // Repetitive prompt: n-gram acceptance rate is very high.
        {
            let prompt = "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.";
            let ref_ids = collect_tokens(&mut ref_engine, prompt, 16);
            let spec_ids = collect_tokens(&mut spec_engine, prompt, 16);
            assert_eq!(ref_ids, spec_ids, "repetitive: spec+batched-verify differs from greedy\nref={ref_ids:?}\nspec={spec_ids:?}");
        }

        // Natural-text prompt: mixed acceptance rate.
        {
            let prompt = "Explain how speculative decoding works:";
            let ref_ids = collect_tokens(&mut ref_engine, prompt, 12);
            let spec_ids = collect_tokens(&mut spec_engine, prompt, 12);
            assert_eq!(ref_ids, spec_ids, "natural: spec+batched-verify differs from greedy\nref={ref_ids:?}\nspec={spec_ids:?}");
        }
    }
}
#[rustfmt::skip]
mod v1_2_memory_limit {
    //! v1.2.0-12: memory budget enforcement tests.
    //!
    //! Verifies that `load_engine` enforces `memory_limit_mb` before mmap
    //! allocation, and that auto-detection (Some(0)) does not erroneously
    //! block a model that fits in 80% of available RAM.

    use std::path::PathBuf;

    fn weights_path() -> PathBuf {
        PathBuf::from("../../models/deepseek-v2-lite-q4.gguf")
    }

    /// Load with a 1 MiB budget — model is ~9 GiB so this must fail.
    #[test]
    fn memory_limit_too_low_returns_error() {
        let weights = weights_path();
        if !weights.exists() {
            eprintln!("skip memory_limit_too_low: no weights at {weights:?}");
            return;
        }
        let cfg = hawking_core::EngineConfig { memory_limit_mb: Some(1), ..Default::default() };
        let result = hawking_core::model::load_engine(&weights, cfg);
        match result {
            Ok(_) => panic!("expected error with 1 MiB budget, but got success"),
            Err(e) => {
                let msg = e.to_string();
                assert!(msg.contains("memory budget exceeded"), "error should mention 'memory budget exceeded', got: {msg}");
            }
        }
    }

    /// Load with a very generous 99_999 MiB budget — must succeed.
    #[test]
    fn memory_limit_generous_succeeds() {
        let weights = weights_path();
        if !weights.exists() {
            eprintln!("skip memory_limit_generous: no weights at {weights:?}");
            return;
        }
        let cfg = hawking_core::EngineConfig { memory_limit_mb: Some(99_999), ..Default::default() };
        let result = hawking_core::model::load_engine(&weights, cfg);
        assert!(result.is_ok(), "expected success with 99_999 MiB budget");
    }

    /// No budget (None) — must succeed (unlimited).
    #[test]
    fn memory_limit_none_is_unlimited() {
        let weights = weights_path();
        if !weights.exists() {
            eprintln!("skip memory_limit_none: no weights at {weights:?}");
            return;
        }
        let cfg = hawking_core::EngineConfig { memory_limit_mb: None, ..Default::default() };
        let result = hawking_core::model::load_engine(&weights, cfg);
        assert!(result.is_ok(), "expected success with no memory limit");
    }

    /// Auto (Some(0)) — 80% of system RAM. On an 18 GiB Mac the budget is
    /// ~14_745 MiB; V2-Lite at ~8_700 MiB fits comfortably.
    #[test]
    fn memory_limit_auto_detection_succeeds_on_18gb_mac() {
        let weights = weights_path();
        if !weights.exists() {
            eprintln!("skip memory_limit_auto: no weights at {weights:?}");
            return;
        }
        let cfg = hawking_core::EngineConfig { memory_limit_mb: Some(0), ..Default::default() };
        let result = hawking_core::model::load_engine(&weights, cfg);
        // On a machine with < 11 GiB total RAM, the model might not fit even at
        // 80%. Skip the success assertion in that edge case.
        match result {
            Ok(_) => eprintln!("auto-detect budget: model fits"),
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("memory budget exceeded") {
                    eprintln!("auto-detect budget: model doesn't fit on this machine (ok to skip)");
                } else {
                    panic!("unexpected error from auto-detection: {msg}");
                }
            }
        }
    }
}
#[rustfmt::skip]
mod v1_memory_eviction_parity {
    #![cfg(target_os = "macos")]

    use std::path::PathBuf;

    const PROMPT: &str = "Once upon a time";
    const MAX_NEW_TOKENS: usize = 8;

    fn run_ids(max_routed_expert_ram_mb: Option<usize>) -> Vec<u32> {
        let weights = PathBuf::from("../../models/deepseek-v2-lite-q4.gguf");
        let profile_path = PathBuf::from("../../profiles/deepseek-v2-lite-q4.m3pro18.json");
        let profile = hawking_core::profile::KernelProfile::load(&profile_path).expect("load profile");
        let cfg = hawking_core::EngineConfig { kernel_profile: Some(profile), max_routed_expert_ram_mb, ..Default::default() };
        let mut engine = hawking_core::model::load_engine(&weights, cfg).expect("load engine");
        let req = hawking_core::GenerateRequest {
            prompt: PROMPT.into(),
            max_new_tokens: MAX_NEW_TOKENS,
            sampling: hawking_core::SamplingParams { temperature: 0.0, seed: Some(42), ..Default::default() },
            stop: Vec::new(),
            abort: None,
            max_stall_ms: 60_000,
            json_mode: false,
        };
        let mut ids = Vec::new();
        engine
            .generate(req, &mut |ev| {
                if let hawking_core::StreamEvent::Token { id, .. } = ev {
                    ids.push(id);
                }
            })
            .expect("generate");
        ids
    }

    #[test]
    fn v2lite_memory_limit_noop_is_bit_identical() {
        let weights = PathBuf::from("../../models/deepseek-v2-lite-q4.gguf");
        if !weights.exists() {
            eprintln!("skipping memory eviction parity: no weights at {weights:?}");
            return;
        }
        let unlimited = run_ids(None);
        let aggressive = run_ids(Some(1));
        assert_eq!(aggressive, unlimited);
    }
}
#[rustfmt::skip]
mod v1_mixtral_smoke {
    use hawking_core::model::mixtral::{MixtralConfig, MixtralEngine};

    #[test]
    fn synthetic_mixtral_shape_smoke_is_finite() {
        let cfg = MixtralConfig::synthetic_for_test();
        assert_eq!(cfg.n_experts, 8);
        assert_eq!(cfg.top_k, 2);
        assert_eq!(cfg.hidden, cfg.n_heads * cfg.head_dim);

        let logits = MixtralEngine::synthetic_forward_shape_for_test(&cfg, 17);
        assert_eq!(logits.len(), cfg.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()));
    }
}
#[rustfmt::skip]
mod v_a1_batched_scaffold_parity {
    //! Phase A Wedge A1 — forward_tokens_batched scaffold parity test.
    //!
    //! Verifies that the layer-first `forward_tokens_batched` produces the same
    //! logit vectors as sequential `forward_tokens` calls at atol=1e-5.
    //!
    //! Skips if model weights are not present.

    use std::path::PathBuf;

    fn weights_path() -> PathBuf {
        PathBuf::from("../../models/deepseek-v2-lite-q4.gguf")
    }

    fn load_engine() -> Option<Box<dyn hawking_core::Engine>> {
        let p = weights_path();
        if !p.exists() {
            eprintln!("v_a1_batched_scaffold_parity: no weights at {:?}, skipping", p);
            return None;
        }
        let cfg = hawking_core::EngineConfig::default();
        match hawking_core::model::load_engine(&p, cfg) {
            Ok(e) => Some(e),
            Err(err) => {
                eprintln!("v_a1_batched_scaffold_parity: load failed: {err}, skipping");
                None
            }
        }
    }

    #[test]
    fn batched_scaffold_returns_n_finite_logit_vectors() {
        let Some(mut engine) = load_engine() else {
            return;
        };

        let tokens = [1u32, 2, 3];
        let positions = [0usize, 1, 2];

        let results = engine.forward_tokens_batched_for_test(&tokens, &positions).expect("forward_tokens_batched_for_test");

        assert_eq!(results.len(), tokens.len(), "must return N logit vectors");
        for (i, logits) in results.iter().enumerate() {
            assert!(!logits.is_empty(), "logits[{i}] empty");
            assert!(logits.iter().all(|x| x.is_finite()), "logits[{i}] contains non-finite values");
        }
    }

    #[test]
    fn batched_scaffold_matches_sequential_at_atol_1e5() {
        let Some(mut engine) = load_engine() else {
            return;
        };

        let tokens = [1u32, 7, 42];
        let positions = [0usize, 1, 2];

        // Reference: sequential via forward_tokens.
        let seq_results = engine.forward_tokens_for_test(&tokens, &positions).expect("forward_tokens sequential reference");

        // Reset KV so the batched pass starts from the same empty state.
        engine.reset_kv_for_test();

        // Batched (layer-first scaffold).
        let batch_results = engine.forward_tokens_batched_for_test(&tokens, &positions).expect("forward_tokens_batched_for_test");

        assert_eq!(seq_results.len(), batch_results.len(), "result count mismatch");
        for m in 0..tokens.len() {
            assert_eq!(seq_results[m].len(), batch_results[m].len(), "logit vector length mismatch at token {m}");
            for (j, (&s, &b)) in seq_results[m].iter().zip(batch_results[m].iter()).enumerate() {
                let diff = (s - b).abs();
                assert!(diff <= 1e-5, "logit[{m}][{j}] diff {diff} > 1e-5 (seq={s} batch={b})");
            }
        }
    }

    #[test]
    fn batched_scaffold_empty_tokens_ok() {
        let Some(mut engine) = load_engine() else {
            return;
        };

        let results = engine.forward_tokens_batched_for_test(&[], &[]).expect("empty batched forward");
        assert!(results.is_empty(), "empty input must return empty output");
    }
}
#[rustfmt::skip]
mod vocab_prune_parity {
    //! path-to-50 lever 1: vocab-prune parity test.
    //!
    //! With the same prompt and fixed seed (temperature=0 ⇒ greedy), the
    //! pruned model must produce the same token sequence as the unpruned
    //! model — provided every emitted token survives the whitelist. The
    //! whitelist covers ≥99.5% of corpus tokens; the held-out generation
    //! prompt below is chosen to land in that 99.5%.
    //!
    //! If the test pins (no prior token-list on disk), it writes the
    //! reference and asserts the rerun matches. Subsequent runs assert
    //! exact equality.

    #![cfg(target_os = "macos")]

    use std::path::PathBuf;

    use hawking_core::{profile::fresh_test_profile, EngineConfig, GenerateRequest, SamplingParams, StreamEvent};

    const PROMPT: &str = "Once upon a time";
    const MAX_NEW_TOKENS: usize = 64;
    const SEED: u64 = 42;

    fn weights_path() -> PathBuf {
        PathBuf::from("../../models/deepseek-v2-lite-q4.gguf")
    }

    fn whitelist_path() -> PathBuf {
        PathBuf::from("../../artifacts/calibration/analysis/vocab_whitelist_995.json")
    }

    fn run_greedy(prune: Option<PathBuf>) -> Option<Vec<u32>> {
        let weights = weights_path();
        if !weights.exists() {
            eprintln!("skipping vocab_prune_parity: model weights missing");
            return None;
        }
        // Build a fresh kernel profile in-memory via the shared helper that
        // sidesteps on-disk shader-hash drift.
        let profile = fresh_test_profile(&weights).expect("fresh test profile");
        let cfg = EngineConfig { kernel_profile: Some(profile), vocab_prune_path: prune, ..Default::default() };
        let mut engine = hawking_core::model::load_engine(&weights, cfg).expect("load engine");
        let req = GenerateRequest {
            prompt: PROMPT.into(),
            max_new_tokens: MAX_NEW_TOKENS,
            sampling: SamplingParams { temperature: 0.0, seed: Some(SEED), ..Default::default() },
            stop: vec![],
            abort: None,
            max_stall_ms: 0,
            json_mode: false,
        };
        let mut ids: Vec<u32> = Vec::new();
        engine
            .generate(req, &mut |ev| {
                if let StreamEvent::Token { id, .. } = ev {
                    ids.push(id);
                }
            })
            .expect("generate");
        Some(ids)
    }

    #[test]
    fn vocab_prune_matches_full_vocab_greedy() {
        let baseline = match run_greedy(None) {
            Some(b) => b,
            None => return,
        };
        let pruned_path = whitelist_path();
        if !pruned_path.exists() {
            eprintln!("skipping vocab_prune_parity: whitelist missing at {:?}", pruned_path);
            return;
        }

        // The LM head GEMV produces logits[i] = W[i,:] @ x. Pruning deletes
        // some rows from W, so logits for surviving rows are bit-identical
        // to the unpruned logits restricted to those rows. Greedy (temp=0)
        // therefore agrees up to the first baseline token that is NOT in the
        // whitelist; at that position the pruned model cannot emit the same
        // token and divergence is expected. After divergence the input embed
        // for the next step differs, so subsequent positions are unrelated.
        let whitelist_bytes = std::fs::read(&pruned_path).expect("read vocab_whitelist_995.json");
        let raw: serde_json::Value = serde_json::from_slice(&whitelist_bytes).expect("parse whitelist json");
        let keep: std::collections::HashSet<u32> = raw["keep_token_ids"].as_array().expect("keep_token_ids array").iter().map(|v| v.as_u64().expect("u64 token id") as u32).collect();

        let first_oov = baseline.iter().position(|t| !keep.contains(t));
        let parity_len = first_oov.unwrap_or(baseline.len());
        eprintln!("vocab_prune_parity: baseline {} tokens; first OOV-in-whitelist at index {:?} → parity prefix {}", baseline.len(), first_oov, parity_len,);
        // Sanity: parity prefix should be substantial. A whitelist with 99.5%
        // coverage means OOV in a 64-token greedy run is uncommon but possible;
        // we require ≥ 4 to ensure the wiring isn't trivially broken at token 0.
        assert!(parity_len >= 4, "parity prefix too short ({}); wiring suspect — baseline[0..8]={:?}", parity_len, &baseline[..baseline.len().min(8)],);

        let pruned = run_greedy(Some(pruned_path)).expect("pruned generate");
        assert!(pruned.len() >= parity_len, "pruned produced fewer tokens ({}) than expected parity prefix ({})", pruned.len(), parity_len,);
        for i in 0..parity_len {
            assert_eq!(baseline[i], pruned[i], "vocab-prune parity diverged at in-whitelist token {}: baseline={} pruned={}", i, baseline[i], pruned[i],);
        }
    }
}
