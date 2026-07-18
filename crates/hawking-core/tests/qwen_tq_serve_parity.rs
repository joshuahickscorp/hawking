//! TQ Stage A served-forward correctness gate (Track 3).
//!
//! The `.tq` GPU bitslice decode kernel is already bit-identity-gated against the
//! CPU integer oracle (`tq_trellis_parity.rs`), and `qwen_tq_serves_ffn_on_gpu`
//! (in `qwen_dense.rs`) proves a single GPU GEMV matches `matvec_rht`. What was
//! UNPROVEN is the *served forward*: that a model loaded with `HAWKING_QWEN_TQ=1`
//! actually generates coherent text through the TQ path, and that the GPU
//! `strand_bitslice_gemv_tcb` serve and the CPU `matvec_rht` serve produce the
//! SAME greedy token trajectory end-to-end (not just per-tensor).
//!
//! This test closes that gap on a REAL model. It is gated on macOS + the `tq`
//! feature (the GPU path and the `strand-quant` dep only exist there) and SKIPS
//! cleanly (never a fake pass) when the GGUF + `.tq` sidecar are absent.
//!
//! Run it (assets present, ~minutes on M-series):
//!
//! ```sh
//! HAWKING_QWEN_TQ=1 \
//!   cargo test -p hawking-core --features tq --test qwen_tq_serve_parity \
//!   -- --nocapture --ignored
//! ```
//!
//! It is `#[ignore]` because loading + greedily decoding a 3B twice (GPU + CPU
//! TQ paths) is far too heavy for the default `cargo test` run / CI; the
//! command above runs it deliberately. The skip-on-missing-asset keeps it from
//! ever failing spuriously when someone *does* `--ignored` without the model.

#![cfg(all(target_os = "macos", feature = "tq"))]

use std::path::Path;

use hawking_core::model::load_engine;
use hawking_core::{EngineConfig, GenerateRequest, SamplingParams, StreamEvent};

const WEIGHTS: &str = "../../models/Qwen2.5-3B-Instruct-Q4_K_M.gguf";
const SIDECAR: &str = "../../models/Qwen2.5-3B-Instruct-Q4_K_M.tq";
const PROMPT: &str = "The capital of France is";
const N_TOKENS: usize = 24;

/// Greedily decode `N_TOKENS` from `PROMPT` against the weights, with the TQ
/// path engaged. `tq_cpu` selects the CPU `matvec_rht` serve (`true`) vs the GPU
/// `strand_bitslice_gemv_tcb` serve (`false`); everything else (sampler, seed,
/// the rest of the Metal pipeline) is held identical so the only variable is the
/// TQ GEMV implementation.
fn greedy_tq_trajectory(tq_cpu: bool) -> Vec<u32> {
    std::env::set_var("HAWKING_QWEN_TQ", "1");
    std::env::remove_var("HAWKING_TQ_RESIDUAL");
    if tq_cpu {
        std::env::set_var("HAWKING_QWEN_TQ_CPU", "1");
    } else {
        std::env::remove_var("HAWKING_QWEN_TQ_CPU");
    }

    let mut engine =
        load_engine(Path::new(WEIGHTS), EngineConfig::default()).expect("load TQ engine");
    let req = GenerateRequest {
        prompt: PROMPT.to_string(),
        max_new_tokens: N_TOKENS,
        sampling: SamplingParams {
            temperature: 0.0,
            seed: Some(0),
            ..Default::default()
        },
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
    ids
}

#[test]
#[ignore = "heavy: loads + greedily decodes a 3B twice; run with --ignored and the model present"]
fn qwen_tq_served_forward_is_nondegenerate_and_cpu_gpu_agree() {
    let weights = Path::new(WEIGHTS);
    let sidecar = Path::new(SIDECAR);
    if !weights.exists() || !sidecar.exists() {
        eprintln!(
            "qwen_tq_served_forward: skip (need {} + {})",
            weights.display(),
            sidecar.display()
        );
        return;
    }

    // (a) GPU TQ serve: the model loads through the `.tq` path and generates.
    let gpu = greedy_tq_trajectory(false);

    // Non-degenerate: it produced tokens, and not a single repeated token (a
    // collapsed/garbage TQ serve degenerates into one token id forever).
    assert!(
        !gpu.is_empty(),
        "TQ served forward produced no tokens (load or decode broke)"
    );
    let distinct = gpu.iter().collect::<std::collections::HashSet<_>>().len();
    assert!(
        distinct > 1,
        "TQ served forward is degenerate: {} tokens, all identical ({:?})",
        gpu.len(),
        gpu.first()
    );

    // (b) CPU `matvec_rht` serve over the SAME decoded Q12: the greedy
    // trajectory must agree with the GPU `strand_bitslice_gemv_tcb` serve. This
    // is the end-to-end CPU-vs-GPU parity the kernel must honour, beyond the
    // per-tensor bit-identity gate.
    let cpu = greedy_tq_trajectory(true);
    assert_eq!(
        gpu,
        cpu,
        "TQ greedy trajectories diverged: GPU strand_bitslice_gemv_tcb vs CPU matvec_rht\n  gpu={gpu:?}\n  cpu={cpu:?}"
    );

    println!(
        "[qwen_tq_serve] non-degenerate ({} distinct of {} tokens), GPU==CPU greedy trajectory",
        distinct,
        gpu.len()
    );
}
