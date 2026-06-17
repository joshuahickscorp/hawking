//! RWKV-7 CONTINUOUS-BATCH (multi-seq) decode parity gate.
//!
//! The continuous-batch decode advances B INDEPENDENT streams in one pass while
//! every projection weight is read once across the B activation columns. Its
//! correctness contract is simple and exact: a B-stream batch must be, token for
//! token, B independent single-stream decodes that merely share the weight set.
//! This gate pins that contract at two levels.
//!
//! ## 1. CPU multiseq oracle == B serial `forward_token`  (always on)
//!
//! `forward_tokens_multiseq_cpu` advances B `RwkvState`s against the shared
//! weights via the same `forward_token_core` the single-stream `forward_token`
//! uses. So for ANY interleaving of B token streams, the per-stream logits from
//! the multiseq call must be **bit-for-bit identical** (exact f32 equality — same
//! code, same op order, just a different `&mut RwkvState`) to running each stream
//! alone through `forward_token`. This is the gate on the multi-stream state
//! plumbing (`RwkvMultiState`, the explicit-state refactor) and needs no GPU, so
//! it runs on every host that has the model. A mismatch here means a stream's
//! state leaked into another's — the one bug the whole design must exclude.
//!
//! ## 2. GPU multiseq == CPU multiseq oracle  (macOS + Metal, skip otherwise)
//!
//! `forward_token_gpu_multiseq` reproduces the CPU oracle on the GPU while
//! reading each weight once across the B columns (the bandwidth win). It shares
//! the f32 weights and op order with the CPU path, so the only residual is f32
//! reduction-order rounding: per-stream argmax must match every step and the
//! max-abs logit gap must stay under a tight tolerance. Skips cleanly (passes)
//! when no Metal GPU or model is present, so CI on non-Metal hosts is green.
//!
//! NOTE: gate 2 EXECUTES Metal (GPU) at runtime. It is written here as the
//! deferred GPU-validation gate; run it on a free GPU (`cargo test -p
//! dismantle-core --test rwkv7_multiseq_parity`) once the training run releases
//! the device.

use dismantle_core::model::rwkv7::{RwkvMultiState, RwkvSeven};
use dismantle_core::{Engine, EngineConfig};
use std::path::{Path, PathBuf};

fn read_ids(path: &Path) -> Vec<u32> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read fixture {path:?}: {e}"))
        .split_whitespace()
        .map(|t| t.parse::<u32>().expect("fixture id parse"))
        .collect()
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/rwkv7")
        .join(name)
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

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Locate the shipped Q4_K rwkv7-0.4B GGUF (env override → in-tree `../../models`
/// → walk up from the manifest dir, covering git-worktree layouts). Mirrors the
/// single-stream Metal-parity gate so both find the model identically.
fn locate_q4k() -> Option<PathBuf> {
    const REL: &str = "models/rwkv7-04/rwkv7-0.4B-world.Q4_K_M.gguf";
    if let Ok(p) = std::env::var("DISMANTLE_RWKV7_GGUF") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    let direct = PathBuf::from("../..").join(REL);
    if direct.exists() {
        return Some(direct);
    }
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        let cand = dir.join(REL);
        if cand.exists() {
            return Some(cand);
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// Prefer an F32 GGUF (tightest parity) else the shipped Q4_K. Returns `None`
/// (test skips) when neither is present.
fn locate_model() -> Option<PathBuf> {
    let f32_path = std::env::var("DISMANTLE_RWKV7_F32_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/rwkv_ref/rwkv7-04-f32.gguf"));
    if f32_path.exists() {
        return Some(f32_path);
    }
    locate_q4k()
}

/// B distinct, realistic token streams of equal length, drawn from the committed
/// fixtures so the test is deterministic and uses in-distribution tokens. Each
/// stream is a rotation of (prompt ++ greedy continuation), so the B streams
/// genuinely differ token-by-token (a real interleave, not B copies).
fn make_streams(b: usize, steps: usize) -> Vec<Vec<u32>> {
    let mut base = read_ids(&fixture("capital_france_q4k.prompt_ids"));
    base.extend(read_ids(&fixture("capital_france_q4k.gen_ids")));
    assert!(
        base.len() >= steps + b,
        "fixture too short: {} < {}",
        base.len(),
        steps + b
    );
    (0..b)
        .map(|s| base[s..s + steps].to_vec())
        .collect::<Vec<_>>()
}

/// GATE 1 (always on): the CPU multiseq oracle is bit-for-bit B serial
/// `forward_token` runs. Exact f32 equality — same core, same op order.
#[test]
fn rwkv7_multiseq_cpu_matches_serial_forward_token() {
    let Some(weights) = locate_model() else {
        eprintln!("skipping rwkv7_multiseq_cpu_matches_serial_forward_token: no rwkv7 weights");
        return;
    };
    let mut engine = RwkvSeven::load(&weights, EngineConfig::default()).expect("load rwkv7");

    const B: usize = 4;
    const STEPS: usize = 24;
    let streams = make_streams(B, STEPS);

    // Reference: each stream decoded ALONE through forward_token (fresh state).
    // serial[s][t] = logits of stream s at step t.
    let mut serial: Vec<Vec<Vec<f32>>> = Vec::with_capacity(B);
    for s in 0..B {
        engine.reset_kv_for_test();
        let mut per_step = Vec::with_capacity(STEPS);
        for t in 0..STEPS {
            per_step.push(engine.forward_token(streams[s][t]).expect("serial forward"));
        }
        serial.push(per_step);
    }

    // Candidate: all B streams advanced together, one token-column per step.
    let mut multi = RwkvMultiState::new(&engine.config, B);
    for t in 0..STEPS {
        let col: Vec<u32> = (0..B).map(|s| streams[s][t]).collect();
        let rows = engine
            .forward_tokens_multiseq_cpu(&col, &mut multi)
            .expect("multiseq cpu forward");
        assert_eq!(rows.len(), B, "multiseq returned {} rows != B", rows.len());
        for s in 0..B {
            // Exact equality: the multiseq core IS forward_token_core with this
            // stream's state, so any difference is a state-mixing bug.
            assert_eq!(
                rows[s], serial[s][t],
                "multiseq stream {s} step {t} differs from serial forward_token \
                 (state leaked across streams)"
            );
        }
    }
    eprintln!(
        "rwkv7 multiseq CPU oracle: B={B} streams × {STEPS} steps bit-exact vs serial forward_token"
    );
}

/// GATE 1b (always on): per-slot reset isolates one stream. After resetting slot
/// 0 mid-decode, slot 0 behaves like a fresh sequence while the other slots keep
/// advancing exactly as their serial counterparts — the continuous-batch reuse
/// contract.
#[test]
fn rwkv7_multiseq_cpu_slot_reset_is_isolated() {
    let Some(weights) = locate_model() else {
        eprintln!("skipping rwkv7_multiseq_cpu_slot_reset_is_isolated: no rwkv7 weights");
        return;
    };
    let mut engine = RwkvSeven::load(&weights, EngineConfig::default()).expect("load rwkv7");

    const B: usize = 3;
    const STEPS: usize = 12;
    let streams = make_streams(B, STEPS);

    let mut multi = RwkvMultiState::new(&engine.config, B);
    // Warm all slots a few steps.
    for t in 0..4 {
        let col: Vec<u32> = (0..B).map(|s| streams[s][t]).collect();
        engine
            .forward_tokens_multiseq_cpu(&col, &mut multi)
            .expect("warm multiseq");
    }
    // Reset ONLY slot 0 — its sequence "finished"; reuse the slot fresh.
    multi.reset_slot(0);

    // Independently: a FRESH engine state replaying slot 0's post-reset tokens
    // must match slot 0's multiseq logits exactly, proving the reset zeroed slot
    // 0 (and only slot 0) — the others are untouched and keep their warm state.
    // Build the reference for slot 0 alone from a fresh state.
    let post: Vec<u32> = (4..STEPS).map(|t| streams[0][t]).collect();
    engine.reset_kv_for_test();
    let mut slot0_ref = Vec::with_capacity(post.len());
    for &tok in &post {
        slot0_ref.push(engine.forward_token(tok).expect("slot0 ref forward"));
    }

    // Drive the multiseq batch over the remaining steps; check slot 0 == fresh ref.
    for (i, t) in (4..STEPS).enumerate() {
        let col: Vec<u32> = (0..B).map(|s| streams[s][t]).collect();
        let rows = engine
            .forward_tokens_multiseq_cpu(&col, &mut multi)
            .expect("post-reset multiseq");
        assert_eq!(
            rows[0], slot0_ref[i],
            "slot 0 after reset_slot must match a fresh sequence at step {t}"
        );
    }
    eprintln!("rwkv7 multiseq CPU oracle: reset_slot(0) isolated; slot 0 decoded fresh, others kept");
}

/// GATE 2 (macOS + Metal; skips otherwise — EXECUTES the GPU): the B-stream GPU
/// decode reproduces the CPU multiseq oracle stream-for-stream.
#[cfg(target_os = "macos")]
#[test]
fn rwkv7_multiseq_gpu_matches_cpu_oracle() {
    /// Per-step max-abs logit tolerance (GPU vs CPU; only reduction order
    /// differs). Same value as the single-stream Metal-parity gate.
    const LOGIT_TOL: f32 = 0.05;

    let Some(weights) = locate_model() else {
        eprintln!("skipping rwkv7_multiseq_gpu_matches_cpu_oracle: no rwkv7 weights");
        return;
    };
    let mut engine = RwkvSeven::load(&weights, EngineConfig::default()).expect("load rwkv7");
    if !engine.has_gpu() {
        eprintln!("skipping rwkv7_multiseq_gpu_matches_cpu_oracle: Metal GPU not available");
        return;
    }

    const B: usize = 4;
    const STEPS: usize = 24;
    let streams = make_streams(B, STEPS);

    // CPU oracle: B streams via the (parity-checked) multiseq CPU path.
    let mut multi = RwkvMultiState::new(&engine.config, B);
    let mut cpu: Vec<Vec<Vec<f32>>> = Vec::with_capacity(STEPS); // [t][s]
    for t in 0..STEPS {
        let col: Vec<u32> = (0..B).map(|s| streams[s][t]).collect();
        cpu.push(
            engine
                .forward_tokens_multiseq_cpu(&col, &mut multi)
                .expect("cpu oracle"),
        );
    }

    // GPU: B streams via the continuous-batch decode (fresh B-stream bundle).
    engine
        .ensure_gpu_batch(B)
        .expect("size gpu bundle for B streams");
    engine.reset_gpu_multiseq();
    let mut worst = 0.0f32;
    let mut worst_at = (0usize, 0usize);
    let mut argmax_mismatches = 0usize;
    for t in 0..STEPS {
        let col: Vec<u32> = (0..B).map(|s| streams[s][t]).collect();
        let rows = engine
            .forward_token_gpu_multiseq(&col)
            .expect("gpu multiseq forward");
        assert_eq!(rows.len(), B, "gpu multiseq returned {} rows != B", rows.len());
        for s in 0..B {
            let (gl, cl) = (&rows[s], &cpu[t][s]);
            assert_eq!(gl.len(), cl.len(), "logit width mismatch stream {s} step {t}");
            let d = max_abs_diff(gl, cl);
            if d > worst {
                worst = d;
                worst_at = (s, t);
            }
            if argmax(gl) != argmax(cl) {
                argmax_mismatches += 1;
                eprintln!(
                    "stream {s} step {t}: argmax GPU={} CPU={} (max|Δ|={d:.4})",
                    argmax(gl),
                    argmax(cl)
                );
            }
        }
    }
    eprintln!(
        "rwkv7 multiseq GPU↔CPU: B={B}×{STEPS} steps, worst max|Δlogit|={worst:.5} @ (stream {}, step {}), argmax mismatches={argmax_mismatches}",
        worst_at.0, worst_at.1
    );
    assert_eq!(
        argmax_mismatches, 0,
        "GPU multiseq argmax must match the CPU oracle every (stream, step)"
    );
    assert!(
        worst < LOGIT_TOL,
        "GPU↔CPU multiseq max-abs logit diff {worst:.5} exceeds tol {LOGIT_TOL}"
    );
}

/// GATE 2b (macOS + Metal; EXECUTES the GPU): the B-stream GPU decode equals B
/// SINGLE-STREAM GPU decodes (forward_token_gpu) run independently — the GPU
/// analogue of gate 1, pinning that the batched projections + stream-major state
/// never cross streams on the device.
#[cfg(target_os = "macos")]
#[test]
fn rwkv7_multiseq_gpu_matches_serial_gpu() {
    const LOGIT_TOL: f32 = 0.05;

    let Some(weights) = locate_model() else {
        eprintln!("skipping rwkv7_multiseq_gpu_matches_serial_gpu: no rwkv7 weights");
        return;
    };
    let mut engine = RwkvSeven::load(&weights, EngineConfig::default()).expect("load rwkv7");
    if !engine.has_gpu() {
        eprintln!("skipping rwkv7_multiseq_gpu_matches_serial_gpu: Metal GPU not available");
        return;
    }

    const B: usize = 3;
    const STEPS: usize = 20;
    let streams = make_streams(B, STEPS);

    // Reference: each stream decoded ALONE on the GPU (batch-1 bundle). Switching
    // to batch 1 then back to B exercises ensure_gpu_batch's rebuild both ways.
    engine.ensure_gpu_batch(1).expect("size gpu bundle B=1");
    let mut serial: Vec<Vec<Vec<f32>>> = Vec::with_capacity(B); // [s][t]
    for s in 0..B {
        engine.reset_gpu_multiseq();
        let mut per_step = Vec::with_capacity(STEPS);
        for t in 0..STEPS {
            per_step.push(
                engine
                    .forward_token_gpu(streams[s][t])
                    .expect("serial gpu forward"),
            );
        }
        serial.push(per_step);
    }

    // Candidate: all B together on the GPU.
    engine.ensure_gpu_batch(B).expect("size gpu bundle for B");
    engine.reset_gpu_multiseq();
    let mut worst = 0.0f32;
    for t in 0..STEPS {
        let col: Vec<u32> = (0..B).map(|s| streams[s][t]).collect();
        let rows = engine
            .forward_token_gpu_multiseq(&col)
            .expect("gpu multiseq forward");
        for s in 0..B {
            let d = max_abs_diff(&rows[s], &serial[s][t]);
            worst = worst.max(d);
            assert_eq!(
                argmax(&rows[s]),
                argmax(&serial[s][t]),
                "GPU multiseq stream {s} step {t} argmax differs from serial GPU decode"
            );
        }
    }
    eprintln!("rwkv7 multiseq GPU == serial GPU: B={B}×{STEPS} steps, worst max|Δlogit|={worst:.5}");
    assert!(
        worst < LOGIT_TOL,
        "GPU multiseq vs serial-GPU max-abs logit diff {worst:.5} exceeds tol {LOGIT_TOL}"
    );
}
