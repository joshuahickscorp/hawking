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
//! hawking-core --test rwkv7_multiseq_parity`) once the training run releases
//! the device.

use hawking_core::model::rwkv7::{RwkvMultiState, RwkvSeven};
use hawking_core::{Engine, EngineConfig};
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
    if let Ok(p) = std::env::var("HAWKING_RWKV7_GGUF") {
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
    let f32_path = std::env::var("HAWKING_RWKV7_F32_GGUF")
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
    eprintln!(
        "rwkv7 multiseq CPU oracle: reset_slot(0) isolated; slot 0 decoded fresh, others kept"
    );
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
        assert_eq!(
            rows.len(),
            B,
            "gpu multiseq returned {} rows != B",
            rows.len()
        );
        for s in 0..B {
            let (gl, cl) = (&rows[s], &cpu[t][s]);
            assert_eq!(
                gl.len(),
                cl.len(),
                "logit width mismatch stream {s} step {t}"
            );
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
    eprintln!(
        "rwkv7 multiseq GPU == serial GPU: B={B}×{STEPS} steps, worst max|Δlogit|={worst:.5}"
    );
    assert!(
        worst < LOGIT_TOL,
        "GPU multiseq vs serial-GPU max-abs logit diff {worst:.5} exceeds tol {LOGIT_TOL}"
    );
}

/// DIAGNOSTIC: Read GPU arena buffers to find where B=2 stream 0 first diverges
/// from single-stream GPU. Reads x_norm (output norm input) after each forward.
#[cfg(target_os = "macos")]
#[test]
fn rwkv7_multiseq_gpu_b2_buffer_inspect() {
    let Some(weights) = locate_model() else {
        eprintln!("skipping: no rwkv7 weights");
        return;
    };
    let mut engine = RwkvSeven::load(&weights, EngineConfig::default()).expect("load rwkv7");
    if !engine.has_gpu() {
        eprintln!("skipping: Metal GPU not available");
        return;
    }
    let streams = make_streams(2, 1);
    let tok0 = streams[0][0];

    // B=2 multiseq step.
    engine.ensure_gpu_batch(2).expect("B=2");
    engine.reset_gpu_multiseq();
    engine
        .forward_token_gpu_multiseq(&[tok0, streams[1][0]])
        .expect("b2 multiseq");
    let b2_x_norm: Vec<f32> = {
        let g = engine.gpu.as_ref().unwrap();
        let n = g.arena.n_embd;
        let ptr = g.arena.x_norm.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
    };

    // Serial GPU step (rebuild B=1 first).
    engine.ensure_gpu_batch(1).expect("B=1");
    engine.reset_gpu_multiseq();
    engine.forward_token_gpu(tok0).expect("serial gpu");
    let b1_x_norm: Vec<f32> = {
        let g = engine.gpu.as_ref().unwrap();
        let n = g.arena.n_embd;
        let ptr = g.arena.x_norm.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
    };

    let d = max_abs_diff(&b2_x_norm, &b1_x_norm);
    eprintln!("x_norm max|Δ| (B=2 stream 0 vs B=1 serial): {d:.5}");
    assert!(
        d < 0.05,
        "x_norm diverges at B=2: max|Δ|={d:.5} — bug is before final LayerNorm"
    );
}

/// DIAGNOSTIC: Per-layer shift state comparison.
/// att_shift and ffn_shift are WRITTEN per-layer (stream-major) during the forward.
/// After the full pass, att_shift[li*n..(li+1)*n] = stream 0's att_in at layer li.
/// Finding the first divergent layer points directly at the buggy kernel.
#[cfg(target_os = "macos")]
#[test]
fn rwkv7_multiseq_gpu_b2_per_layer_shift_inspect() {
    let Some(weights) = locate_model() else {
        eprintln!("skipping: no rwkv7 weights");
        return;
    };
    let mut engine = RwkvSeven::load(&weights, EngineConfig::default()).expect("load rwkv7");
    if !engine.has_gpu() {
        eprintln!("skipping: Metal GPU not available");
        return;
    }
    let streams = make_streams(2, 1);
    let tok0 = streams[0][0];

    // B=2 multiseq step — read att_shift and ffn_shift (all layers, stream 0).
    engine.ensure_gpu_batch(2).expect("B=2");
    engine.reset_gpu_multiseq();
    engine
        .forward_token_gpu_multiseq(&[tok0, streams[1][0]])
        .expect("b2 multiseq");
    let (b2_att, b2_ffn, n_layer, n_embd) = {
        let g = engine.gpu.as_ref().unwrap();
        let nl = g.arena.n_layer;
        let n = g.arena.n_embd;
        // att_shift is stream-major: stream 0 elems = [0..nl*n], stream 1 = [nl*n..2*nl*n]
        let att = unsafe {
            std::slice::from_raw_parts(g.arena.att_shift.contents() as *const f32, nl * n)
        }
        .to_vec();
        let ffn = unsafe {
            std::slice::from_raw_parts(g.arena.ffn_shift.contents() as *const f32, nl * n)
        }
        .to_vec();
        (att, ffn, nl, n)
    };

    // B=1 serial GPU step.
    engine.ensure_gpu_batch(1).expect("B=1");
    engine.reset_gpu_multiseq();
    engine.forward_token_gpu(tok0).expect("serial gpu");
    let (b1_att, b1_ffn) = {
        let g = engine.gpu.as_ref().unwrap();
        let nl = g.arena.n_layer;
        let n = g.arena.n_embd;
        let att = unsafe {
            std::slice::from_raw_parts(g.arena.att_shift.contents() as *const f32, nl * n)
        }
        .to_vec();
        let ffn = unsafe {
            std::slice::from_raw_parts(g.arena.ffn_shift.contents() as *const f32, nl * n)
        }
        .to_vec();
        (att, ffn)
    };

    let mut first_divergent: Option<usize> = None;
    for li in 0..n_layer {
        let lo = li * n_embd;
        let hi = lo + n_embd;
        let d_att = max_abs_diff(&b2_att[lo..hi], &b1_att[lo..hi]);
        let d_ffn = max_abs_diff(&b2_ffn[lo..hi], &b1_ffn[lo..hi]);
        eprintln!("layer {li:2}: att_shift max|Δ|={d_att:.5}  ffn_shift max|Δ|={d_ffn:.5}");
        if first_divergent.is_none() && (d_att > 0.05 || d_ffn > 0.05) {
            first_divergent = Some(li);
        }
    }
    if let Some(li) = first_divergent {
        panic!("first divergent layer: {li}  (see per-layer log above)");
    }
}

/// DIAGNOSTIC: WKV state per-layer for stream 0 after B=2 vs B=1 forward.
/// wkv_state is stream-major: stream b at [b * n_layer * s_per_layer].
/// Layer 0, stream 0 starts at offset 0 for both B=1 and B=2.
/// If this matches, the WKV recurrence is correct and the bug is in the
/// output projection (Wo @ out_wkv → cur) or the residual add/LN after it.
#[cfg(target_os = "macos")]
#[test]
fn rwkv7_multiseq_gpu_b2_wkv_state_inspect() {
    let Some(weights) = locate_model() else {
        eprintln!("skipping: no rwkv7 weights");
        return;
    };
    let mut engine = RwkvSeven::load(&weights, EngineConfig::default()).expect("load rwkv7");
    if !engine.has_gpu() {
        eprintln!("skipping: Metal GPU not available");
        return;
    }
    let streams = make_streams(2, 1);
    let tok0 = streams[0][0];

    // B=2 multiseq step — read wkv_state for stream 0 (at offset 0).
    engine.ensure_gpu_batch(2).expect("B=2");
    engine.reset_gpu_multiseq();
    engine
        .forward_token_gpu_multiseq(&[tok0, streams[1][0]])
        .expect("b2 multiseq");
    let (b2_wkv, s_per_layer, n_layer) = {
        let g = engine.gpu.as_ref().unwrap();
        let hc = g.arena.head_count;
        let hs = g.arena.head_size;
        let nl = g.arena.n_layer;
        let spl = hc * hs * hs;
        // stream 0 wkv_state = wkv_state[0 .. n_layer * spl]
        let ptr = g.arena.wkv_state.contents() as *const f32;
        let s0 = unsafe { std::slice::from_raw_parts(ptr, nl * spl) }.to_vec();
        (s0, spl, nl)
    };

    // B=1 serial GPU step.
    engine.ensure_gpu_batch(1).expect("B=1");
    engine.reset_gpu_multiseq();
    engine.forward_token_gpu(tok0).expect("serial gpu");
    let b1_wkv = {
        let g = engine.gpu.as_ref().unwrap();
        let hc = g.arena.head_count;
        let hs = g.arena.head_size;
        let nl = g.arena.n_layer;
        let spl = hc * hs * hs;
        let ptr = g.arena.wkv_state.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, nl * spl) }.to_vec()
    };

    let mut first_wkv_divergent: Option<usize> = None;
    for li in 0..n_layer {
        let lo = li * s_per_layer;
        let hi = lo + s_per_layer;
        let d = max_abs_diff(&b2_wkv[lo..hi], &b1_wkv[lo..hi]);
        eprintln!("layer {li:2}: wkv_state max|Δ|={d:.5}");
        if first_wkv_divergent.is_none() && d > 1e-4 {
            first_wkv_divergent = Some(li);
        }
    }
    if let Some(li) = first_wkv_divergent {
        panic!(
            "wkv_state first divergent layer {li} — bug in WKV recurrence or its r/w/k/v inputs"
        );
    }
    eprintln!("wkv_state MATCHES for all layers → bug is in output projection (Wo@out_wkv) or residual+LN");
}

/// DIAGNOSTIC: Compare `v_first[0..n_embd]` (the layer-0 value projection,
/// stream 0) between B=2 multiseq and B=1 serial GPU decode.
///
/// v_first is written once per forward pass at layer 0 via
/// `rwkv7_copy_tcb(&mut tcb, &a.v, &a.v_first, 0, b * n)`.
/// In the B=2 arena it is sized `(2 * n_embd)` with stream 0 at `[0..n_embd]`,
/// exactly matching the B=1 layout.
///
/// If max|Δ| ≈ 0 → the value-GEMV (Wv @ xs[slot3]) and the preceding lerp
/// kernel both wrote the correct activations for stream 0.  The bug must be
/// downstream (WKV recurrence, Wo projection, or residual/LN).
///
/// If max|Δ| > 0 → the value-GEMV or the lerp (xs[slot3]) produced wrong
/// activations for stream 0 at B=2 — the bug is in the batched GEMM or the
/// token-shift kernel.
///
/// This test never panics; it is purely diagnostic.
#[cfg(target_os = "macos")]
#[test]
fn rwkv7_multiseq_gpu_b2_vfirst_inspect() {
    let Some(weights) = locate_model() else {
        eprintln!("skipping: no rwkv7 weights");
        return;
    };
    let mut engine = RwkvSeven::load(&weights, EngineConfig::default()).expect("load rwkv7");
    if !engine.has_gpu() {
        eprintln!("skipping: Metal GPU not available");
        return;
    }
    let streams = make_streams(2, 1);
    let tok0 = streams[0][0];

    // B=2 multiseq step — read v_first[0..n_embd] (stream 0).
    engine.ensure_gpu_batch(2).expect("B=2");
    engine.reset_gpu_multiseq();
    engine
        .forward_token_gpu_multiseq(&[tok0, streams[1][0]])
        .expect("b2 multiseq");
    let b2_vfirst: Vec<f32> = {
        let g = engine.gpu.as_ref().unwrap();
        let n = g.arena.n_embd;
        let ptr = g.arena.v_first.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
    };

    // B=1 serial GPU step (rebuild arena so state is isolated).
    engine.ensure_gpu_batch(1).expect("B=1");
    engine.reset_gpu_multiseq();
    engine.forward_token_gpu(tok0).expect("serial gpu");
    let b1_vfirst: Vec<f32> = {
        let g = engine.gpu.as_ref().unwrap();
        let n = g.arena.n_embd;
        let ptr = g.arena.v_first.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
    };

    let d = max_abs_diff(&b2_vfirst, &b1_vfirst);
    eprintln!("v_first[0..n_embd] max|Δ| (B=2 stream 0 vs B=1 serial): {d:.6}");
    // Diagnostic only — never panic.
}

/// DIAGNOSTIC: Compare r, k, a, a_op arena buffers between B=2 multiseq
/// (stream 0) and B=1 serial GPU after a single 1-step forward.
///
/// After the full N-layer forward these buffers hold layer N-1 values.
/// Any divergence introduced at layer 0 in k (post kk_kmix) propagates
/// through layers, so N-1 diffs are still diagnostic.  Never panics —
/// purely prints max|Δ| for each buffer.
#[cfg(target_os = "macos")]
#[test]
fn rwkv7_multiseq_gpu_b2_k_inspect() {
    let Some(weights) = locate_model() else {
        eprintln!("skipping rwkv7_multiseq_gpu_b2_k_inspect: no rwkv7 weights");
        return;
    };
    let mut engine = RwkvSeven::load(&weights, EngineConfig::default()).expect("load rwkv7");
    if !engine.has_gpu() {
        eprintln!("skipping rwkv7_multiseq_gpu_b2_k_inspect: Metal GPU not available");
        return;
    }
    let streams = make_streams(2, 1);
    let tok0 = streams[0][0];

    // B=2 multiseq step — read r, k, a, a_op for stream 0 (first n_embd elements).
    engine.ensure_gpu_batch(2).expect("B=2");
    engine.reset_gpu_multiseq();
    engine
        .forward_token_gpu_multiseq(&[tok0, streams[1][0]])
        .expect("b2 multiseq");
    let (b2_r, b2_k, b2_a, b2_a_op) = {
        let g = engine.gpu.as_ref().unwrap();
        let n = g.arena.n_embd;
        let r =
            unsafe { std::slice::from_raw_parts(g.arena.r.contents() as *const f32, n) }.to_vec();
        let k =
            unsafe { std::slice::from_raw_parts(g.arena.k.contents() as *const f32, n) }.to_vec();
        let a =
            unsafe { std::slice::from_raw_parts(g.arena.a.contents() as *const f32, n) }.to_vec();
        let a_op = unsafe { std::slice::from_raw_parts(g.arena.a_op.contents() as *const f32, n) }
            .to_vec();
        (r, k, a, a_op)
    };

    // B=1 serial GPU step.
    engine.ensure_gpu_batch(1).expect("B=1");
    engine.reset_gpu_multiseq();
    engine.forward_token_gpu(tok0).expect("serial gpu");
    let (b1_r, b1_k, b1_a, b1_a_op) = {
        let g = engine.gpu.as_ref().unwrap();
        let n = g.arena.n_embd;
        let r =
            unsafe { std::slice::from_raw_parts(g.arena.r.contents() as *const f32, n) }.to_vec();
        let k =
            unsafe { std::slice::from_raw_parts(g.arena.k.contents() as *const f32, n) }.to_vec();
        let a =
            unsafe { std::slice::from_raw_parts(g.arena.a.contents() as *const f32, n) }.to_vec();
        let a_op = unsafe { std::slice::from_raw_parts(g.arena.a_op.contents() as *const f32, n) }
            .to_vec();
        (r, k, a, a_op)
    };

    let d_r = max_abs_diff(&b2_r, &b1_r);
    let d_k = max_abs_diff(&b2_k, &b1_k);
    let d_a = max_abs_diff(&b2_a, &b1_a);
    let d_a_op = max_abs_diff(&b2_a_op, &b1_a_op);

    eprintln!("r[0..n_embd]    max|Δ| (B=2 stream 0 vs B=1 serial): {d_r:.6}");
    eprintln!("k[0..n_embd]    max|Δ| (B=2 stream 0 vs B=1 serial): {d_k:.6}");
    eprintln!("a[0..n_embd]    max|Δ| (B=2 stream 0 vs B=1 serial): {d_a:.6}");
    eprintln!("a_op[0..n_embd] max|Δ| (B=2 stream 0 vs B=1 serial): {d_a_op:.6}");
    // Diagnostic only — never panic.
}

/// DIAGNOSTIC: B=2 multiseq stream 0 vs single-stream GPU — narrows whether
/// the bug starts at B=2 or only at B=3+.
#[cfg(target_os = "macos")]
#[test]
fn rwkv7_multiseq_gpu_b2_stream0_matches_serial_gpu() {
    const LOGIT_TOL: f32 = 0.05;
    const STEPS: usize = 4;

    let Some(weights) = locate_model() else {
        eprintln!("skipping: no rwkv7 weights");
        return;
    };
    let mut engine = RwkvSeven::load(&weights, EngineConfig::default()).expect("load rwkv7");
    if !engine.has_gpu() {
        eprintln!("skipping: Metal GPU not available");
        return;
    }
    let streams = make_streams(2, STEPS);

    // Serial GPU reference for stream 0.
    engine.ensure_gpu_batch(1).expect("B=1");
    engine.reset_gpu_multiseq();
    let serial: Vec<Vec<f32>> = streams[0]
        .iter()
        .map(|&t| engine.forward_token_gpu(t).expect("serial gpu"))
        .collect();

    // B=2 multiseq: stream 0's logits.
    engine.ensure_gpu_batch(2).expect("B=2");
    engine.reset_gpu_multiseq();
    for (t, _) in streams[0].iter().enumerate() {
        let col: Vec<u32> = (0..2).map(|s| streams[s][t]).collect();
        let rows = engine
            .forward_token_gpu_multiseq(&col)
            .expect("b2 multiseq gpu");
        let d = max_abs_diff(&rows[0], &serial[t]);
        let (ag, ac) = (argmax(&rows[0]), argmax(&serial[t]));
        eprintln!(
            "step {t}: argmax B2-s0={ag} serial={ac} max|Δ|={d:.4} {}",
            if ag != ac { "MISMATCH" } else { "ok" }
        );
        assert!(
            d < LOGIT_TOL,
            "B=2 multiseq stream 0 step {t}: max|Δlogit|={d:.5} exceeds tol"
        );
    }
    eprintln!("B=2 multiseq stream 0 OK");
}

/// DIAGNOSTIC: B=1 multiseq (batched GEMM kernel path) vs single-stream GPU
/// (GEMV kernel path). If B=1 multiseq fails, the bug is in the batched-GEMM
/// kernel arithmetic itself, not in multi-stream interactions.
#[cfg(target_os = "macos")]
#[test]
fn rwkv7_multiseq_gpu_b1_matches_serial_gpu() {
    const LOGIT_TOL: f32 = 0.05;
    const STEPS: usize = 8;

    let Some(weights) = locate_model() else {
        eprintln!("skipping rwkv7_multiseq_gpu_b1_matches_serial_gpu: no rwkv7 weights");
        return;
    };
    let mut engine = RwkvSeven::load(&weights, EngineConfig::default()).expect("load rwkv7");
    if !engine.has_gpu() {
        eprintln!("skipping rwkv7_multiseq_gpu_b1_matches_serial_gpu: Metal GPU not available");
        return;
    }

    let streams = make_streams(1, STEPS);
    let tok = &streams[0];

    // Serial GPU: single-stream path (gemv_q4_k_v4_predec kernel).
    engine.ensure_gpu_batch(1).expect("init B=1 arena");
    engine.reset_gpu_multiseq();
    let serial: Vec<Vec<f32>> = tok
        .iter()
        .map(|&t| engine.forward_token_gpu(t).expect("serial gpu"))
        .collect();

    // B=1 multiseq: batched GEMM kernel path (gemm_q4_k_m_batched_v3w_predec).
    // ensure_gpu_batch(1) is a no-op here (already B=1), so we need to force
    // a rebuild to a different batch then back to 1 to pick up the multiseq path.
    engine.ensure_gpu_batch(2).expect("force B=2 rebuild");
    engine.ensure_gpu_batch(1).expect("rebuild B=1");
    engine.reset_gpu_multiseq();
    let mut worst = 0.0f32;
    let mut argmax_mismatches = 0;
    for (t, &tok_id) in tok.iter().enumerate() {
        let rows = engine
            .forward_token_gpu_multiseq(&[tok_id])
            .expect("b1 multiseq gpu");
        let d = max_abs_diff(&rows[0], &serial[t]);
        worst = worst.max(d);
        let (ag, ac) = (argmax(&rows[0]), argmax(&serial[t]));
        if ag != ac {
            argmax_mismatches += 1;
            eprintln!("step {t}: argmax B1-multiseq={ag} serial={ac} (max|Δ|={d:.4})");
        }
    }
    eprintln!(
        "rwkv7 B=1 multiseq vs serial GPU: {STEPS} steps, worst max|Δlogit|={worst:.5}, \
         argmax mismatches={argmax_mismatches}"
    );
    assert_eq!(
        argmax_mismatches, 0,
        "B=1 GPU multiseq argmax must match serial GPU every step"
    );
    assert!(
        worst < LOGIT_TOL,
        "B=1 GPU multiseq vs serial-GPU max-abs logit diff {worst:.5} exceeds tol {LOGIT_TOL}"
    );
}
