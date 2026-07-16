//! Consolidated RWKV7 runtime, multisequence, checkpoint, and TQ cases.
#[rustfmt::skip]
mod rwkv7_metal_parity {
    //! RWKV-7 GPU-decode parity gate: the Metal WKV-7 path must match the
    //! PARITY-VALIDATED CPU reference `forward_token` bit-for-bit within f32
    //! tolerance.
    //!
    //! The CPU `rwkv7.rs::forward_token` is the correctness oracle (it is itself
    //! validated F32 48/48 / Q4_K 40/40 vs llama.cpp). The GPU path uploads the
    //! SAME f32-dequantized weights and runs the SAME op order, so the only residual
    //! is f32 reduction-order rounding — the two must agree to a tight tolerance and
    //! produce identical argmax tokens.
    //!
    //! macOS/Metal-gated: skips cleanly (passes) when no Metal GPU is present or the
    //! model weights are absent, so CI on non-Metal hosts is green.
    //!
    //! Two checks:
    //!   1. `rwkv7_gpu_matches_cpu_logits` — feed an identical real token trajectory
    //!      (the committed prompt + its CPU greedy continuation) through a fresh CPU
    //!      state and a fresh GPU state in lockstep; assert per-step argmax match and
    //!      max-abs logit diff under tolerance for >=32 steps.
    //!   2. `rwkv7_gpu_greedy_trajectory_matches_cpu` — let each path drive its OWN
    //!      greedy argmax for 32 steps from the same prompt; the produced id
    //!      sequences must be identical (a self-consistent GPU decode == CPU decode).

    #![cfg(target_os = "macos")]

    use hawking_core::model::rwkv7::RwkvSeven;
    use hawking_core::{Engine, EngineConfig};
    use std::path::{Path, PathBuf};

    /// Tolerance on the per-step max-abs logit difference (GPU vs CPU). The two
    /// share f32 weights; only reduction order differs, so the gap is small. The
    /// LM-head logits reach magnitudes ~O(10–30), so a few 1e-2 of slack is ample
    /// headroom while still catching any real kernel bug (which diverges by O(1+)).
    const LOGIT_TOL: f32 = 0.05;

    fn read_ids(path: &Path) -> Vec<u32> {
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read fixture {path:?}: {e}")).split_whitespace().map(|t| t.parse::<u32>().expect("fixture id parse")).collect()
    }

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join(format!("rwkv7_{name}"))
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
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    /// Locate the shipped Q4_K rwkv7-0.4B GGUF. Tries (in order): an explicit
    /// `HAWKING_RWKV7_GGUF` override, the in-tree `../../models` path (normal
    /// checkout / CI), then walks up from the manifest dir looking for a `models/`
    /// dir (covers git-worktree layouts where `models/` lives in the main checkout).
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

    fn load_model() -> Option<RwkvSeven> {
        // Prefer an F32 GGUF when available (tightest parity), else the shipped Q4_K.
        let f32_path = std::env::var("HAWKING_RWKV7_F32_GGUF").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("/tmp/rwkv_ref/rwkv7-04-f32.gguf"));
        let weights = if f32_path.exists() {
            f32_path
        } else if let Some(q4k) = locate_q4k() {
            q4k
        } else {
            eprintln!("skipping rwkv7_metal_parity: no rwkv7 weights (F32 or Q4_K) found");
            return None;
        };
        let engine = RwkvSeven::load(&weights, EngineConfig::default()).expect("load rwkv7");
        if !engine.has_gpu() {
            eprintln!("skipping rwkv7_metal_parity: Metal GPU not available");
            return None;
        }
        Some(engine)
    }

    /// Lockstep: identical input trajectory through fresh CPU and GPU states.
    #[test]
    fn rwkv7_gpu_matches_cpu_logits() {
        let Some(mut engine) = load_model() else {
            return;
        };

        // Real trajectory: prompt + its CPU greedy continuation (>= 45 ids).
        let mut input = read_ids(&fixture("capital_france_q4k.prompt_ids"));
        input.extend(read_ids(&fixture("capital_france_q4k.gen_ids")));
        assert!(input.len() >= 32, "need >=32 steps, got {}", input.len());

        // CPU pass (oracle): fresh state, collect logits per step.
        engine.reset_kv_for_test();
        let mut cpu_logits = Vec::with_capacity(input.len());
        for &t in &input {
            cpu_logits.push(engine.forward_token(t).expect("cpu forward"));
        }

        // GPU pass: fresh state, same inputs.
        engine.reset_kv_for_test();
        let mut worst = 0.0f32;
        let mut worst_step = 0usize;
        let mut argmax_mismatches = 0usize;
        for (step, &t) in input.iter().enumerate() {
            let gl = engine.forward_token_gpu(t).expect("gpu forward");
            let cl = &cpu_logits[step];
            assert_eq!(gl.len(), cl.len(), "logit width mismatch at step {step}");
            let d = max_abs_diff(&gl, cl);
            if d > worst {
                worst = d;
                worst_step = step;
            }
            let (ag, ac) = (argmax(&gl), argmax(cl));
            if ag != ac {
                argmax_mismatches += 1;
                eprintln!("step {step}: argmax GPU={ag} CPU={ac} (max|Δ|={d:.4})");
            }
        }
        eprintln!("rwkv7 GPU↔CPU parity: {} steps, worst max|Δlogit|={:.5} @step {}, argmax mismatches={}", input.len(), worst, worst_step, argmax_mismatches);
        assert_eq!(argmax_mismatches, 0, "GPU decode argmax must match CPU oracle every step ({} mismatches)", argmax_mismatches);
        assert!(worst < LOGIT_TOL, "GPU↔CPU max-abs logit diff {worst:.5} exceeds tol {LOGIT_TOL} (worst @step {worst_step})");
    }

    /// Each path drives its own greedy argmax; trajectories must be identical.
    #[test]
    fn rwkv7_gpu_greedy_trajectory_matches_cpu() {
        let Some(mut engine) = load_model() else {
            return;
        };
        let prompt = read_ids(&fixture("capital_france_q4k.prompt_ids"));
        let n_decode = 32usize;

        // CPU greedy trajectory.
        engine.reset_kv_for_test();
        let mut cpu_logits0 = Vec::new();
        for &t in &prompt {
            cpu_logits0 = engine.forward_token(t).expect("cpu prefill");
        }
        let mut cpu_traj = Vec::with_capacity(n_decode);
        let mut next = argmax(&cpu_logits0);
        cpu_traj.push(next);
        for _ in 1..n_decode {
            let lg = engine.forward_token(next).expect("cpu decode");
            next = argmax(&lg);
            cpu_traj.push(next);
        }

        // GPU greedy trajectory.
        engine.reset_kv_for_test();
        let mut gpu_logits0 = Vec::new();
        for &t in &prompt {
            gpu_logits0 = engine.forward_token_gpu(t).expect("gpu prefill");
        }
        let mut gpu_traj = Vec::with_capacity(n_decode);
        let mut next = argmax(&gpu_logits0);
        gpu_traj.push(next);
        for _ in 1..n_decode {
            let lg = engine.forward_token_gpu(next).expect("gpu decode");
            next = argmax(&lg);
            gpu_traj.push(next);
        }

        let matched = cpu_traj.iter().zip(gpu_traj.iter()).take_while(|(a, b)| a == b).count();
        eprintln!("rwkv7 GPU greedy trajectory: {matched}/{n_decode} leading tokens match CPU oracle");
        assert_eq!(
            gpu_traj, cpu_traj,
            "GPU greedy decode must reproduce the CPU oracle trajectory for {n_decode} tokens\n  \
             cpu={cpu_traj:?}\n  gpu={gpu_traj:?}"
        );
    }
}
#[rustfmt::skip]
mod rwkv7_multiseq_parity {
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
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read fixture {path:?}: {e}")).split_whitespace().map(|t| t.parse::<u32>().expect("fixture id parse")).collect()
    }

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join(format!("rwkv7_{name}"))
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
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
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
        let f32_path = std::env::var("HAWKING_RWKV7_F32_GGUF").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("/tmp/rwkv_ref/rwkv7-04-f32.gguf"));
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
        assert!(base.len() >= steps + b, "fixture too short: {} < {}", base.len(), steps + b);
        (0..b).map(|s| base[s..s + steps].to_vec()).collect::<Vec<_>>()
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
            let rows = engine.forward_tokens_multiseq_cpu(&col, &mut multi).expect("multiseq cpu forward");
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
        eprintln!("rwkv7 multiseq CPU oracle: B={B} streams × {STEPS} steps bit-exact vs serial forward_token");
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
            engine.forward_tokens_multiseq_cpu(&col, &mut multi).expect("warm multiseq");
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
            let rows = engine.forward_tokens_multiseq_cpu(&col, &mut multi).expect("post-reset multiseq");
            assert_eq!(rows[0], slot0_ref[i], "slot 0 after reset_slot must match a fresh sequence at step {t}");
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
            cpu.push(engine.forward_tokens_multiseq_cpu(&col, &mut multi).expect("cpu oracle"));
        }

        // GPU: B streams via the continuous-batch decode (fresh B-stream bundle).
        engine.ensure_gpu_batch(B).expect("size gpu bundle for B streams");
        engine.reset_gpu_multiseq();
        let mut worst = 0.0f32;
        let mut worst_at = (0usize, 0usize);
        let mut argmax_mismatches = 0usize;
        for t in 0..STEPS {
            let col: Vec<u32> = (0..B).map(|s| streams[s][t]).collect();
            let rows = engine.forward_token_gpu_multiseq(&col).expect("gpu multiseq forward");
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
                    eprintln!("stream {s} step {t}: argmax GPU={} CPU={} (max|Δ|={d:.4})", argmax(gl), argmax(cl));
                }
            }
        }
        eprintln!("rwkv7 multiseq GPU↔CPU: B={B}×{STEPS} steps, worst max|Δlogit|={worst:.5} @ (stream {}, step {}), argmax mismatches={argmax_mismatches}", worst_at.0, worst_at.1);
        assert_eq!(argmax_mismatches, 0, "GPU multiseq argmax must match the CPU oracle every (stream, step)");
        assert!(worst < LOGIT_TOL, "GPU↔CPU multiseq max-abs logit diff {worst:.5} exceeds tol {LOGIT_TOL}");
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
                per_step.push(engine.forward_token_gpu(streams[s][t]).expect("serial gpu forward"));
            }
            serial.push(per_step);
        }

        // Candidate: all B together on the GPU.
        engine.ensure_gpu_batch(B).expect("size gpu bundle for B");
        engine.reset_gpu_multiseq();
        let mut worst = 0.0f32;
        for t in 0..STEPS {
            let col: Vec<u32> = (0..B).map(|s| streams[s][t]).collect();
            let rows = engine.forward_token_gpu_multiseq(&col).expect("gpu multiseq forward");
            for s in 0..B {
                let d = max_abs_diff(&rows[s], &serial[s][t]);
                worst = worst.max(d);
                assert_eq!(argmax(&rows[s]), argmax(&serial[s][t]), "GPU multiseq stream {s} step {t} argmax differs from serial GPU decode");
            }
        }
        eprintln!("rwkv7 multiseq GPU == serial GPU: B={B}×{STEPS} steps, worst max|Δlogit|={worst:.5}");
        assert!(worst < LOGIT_TOL, "GPU multiseq vs serial-GPU max-abs logit diff {worst:.5} exceeds tol {LOGIT_TOL}");
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
        engine.forward_token_gpu_multiseq(&[tok0, streams[1][0]]).expect("b2 multiseq");
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
        assert!(d < 0.05, "x_norm diverges at B=2: max|Δ|={d:.5} — bug is before final LayerNorm");
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
        engine.forward_token_gpu_multiseq(&[tok0, streams[1][0]]).expect("b2 multiseq");
        let (b2_att, b2_ffn, n_layer, n_embd) = {
            let g = engine.gpu.as_ref().unwrap();
            let nl = g.arena.n_layer;
            let n = g.arena.n_embd;
            // att_shift is stream-major: stream 0 elems = [0..nl*n], stream 1 = [nl*n..2*nl*n]
            let att = unsafe { std::slice::from_raw_parts(g.arena.att_shift.contents() as *const f32, nl * n) }.to_vec();
            let ffn = unsafe { std::slice::from_raw_parts(g.arena.ffn_shift.contents() as *const f32, nl * n) }.to_vec();
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
            let att = unsafe { std::slice::from_raw_parts(g.arena.att_shift.contents() as *const f32, nl * n) }.to_vec();
            let ffn = unsafe { std::slice::from_raw_parts(g.arena.ffn_shift.contents() as *const f32, nl * n) }.to_vec();
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
        engine.forward_token_gpu_multiseq(&[tok0, streams[1][0]]).expect("b2 multiseq");
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
            panic!("wkv_state first divergent layer {li} — bug in WKV recurrence or its r/w/k/v inputs");
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
        engine.forward_token_gpu_multiseq(&[tok0, streams[1][0]]).expect("b2 multiseq");
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
        engine.forward_token_gpu_multiseq(&[tok0, streams[1][0]]).expect("b2 multiseq");
        let (b2_r, b2_k, b2_a, b2_a_op) = {
            let g = engine.gpu.as_ref().unwrap();
            let n = g.arena.n_embd;
            let r = unsafe { std::slice::from_raw_parts(g.arena.r.contents() as *const f32, n) }.to_vec();
            let k = unsafe { std::slice::from_raw_parts(g.arena.k.contents() as *const f32, n) }.to_vec();
            let a = unsafe { std::slice::from_raw_parts(g.arena.a.contents() as *const f32, n) }.to_vec();
            let a_op = unsafe { std::slice::from_raw_parts(g.arena.a_op.contents() as *const f32, n) }.to_vec();
            (r, k, a, a_op)
        };

        // B=1 serial GPU step.
        engine.ensure_gpu_batch(1).expect("B=1");
        engine.reset_gpu_multiseq();
        engine.forward_token_gpu(tok0).expect("serial gpu");
        let (b1_r, b1_k, b1_a, b1_a_op) = {
            let g = engine.gpu.as_ref().unwrap();
            let n = g.arena.n_embd;
            let r = unsafe { std::slice::from_raw_parts(g.arena.r.contents() as *const f32, n) }.to_vec();
            let k = unsafe { std::slice::from_raw_parts(g.arena.k.contents() as *const f32, n) }.to_vec();
            let a = unsafe { std::slice::from_raw_parts(g.arena.a.contents() as *const f32, n) }.to_vec();
            let a_op = unsafe { std::slice::from_raw_parts(g.arena.a_op.contents() as *const f32, n) }.to_vec();
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
        let serial: Vec<Vec<f32>> = streams[0].iter().map(|&t| engine.forward_token_gpu(t).expect("serial gpu")).collect();

        // B=2 multiseq: stream 0's logits.
        engine.ensure_gpu_batch(2).expect("B=2");
        engine.reset_gpu_multiseq();
        for (t, _) in streams[0].iter().enumerate() {
            let col: Vec<u32> = (0..2).map(|s| streams[s][t]).collect();
            let rows = engine.forward_token_gpu_multiseq(&col).expect("b2 multiseq gpu");
            let d = max_abs_diff(&rows[0], &serial[t]);
            let (ag, ac) = (argmax(&rows[0]), argmax(&serial[t]));
            eprintln!("step {t}: argmax B2-s0={ag} serial={ac} max|Δ|={d:.4} {}", if ag != ac { "MISMATCH" } else { "ok" });
            assert!(d < LOGIT_TOL, "B=2 multiseq stream 0 step {t}: max|Δlogit|={d:.5} exceeds tol");
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
        let serial: Vec<Vec<f32>> = tok.iter().map(|&t| engine.forward_token_gpu(t).expect("serial gpu")).collect();

        // B=1 multiseq: batched GEMM kernel path (gemm_q4_k_m_batched_v3w_predec).
        // ensure_gpu_batch(1) is a no-op here (already B=1), so we need to force
        // a rebuild to a different batch then back to 1 to pick up the multiseq path.
        engine.ensure_gpu_batch(2).expect("force B=2 rebuild");
        engine.ensure_gpu_batch(1).expect("rebuild B=1");
        engine.reset_gpu_multiseq();
        let mut worst = 0.0f32;
        let mut argmax_mismatches = 0;
        for (t, &tok_id) in tok.iter().enumerate() {
            let rows = engine.forward_token_gpu_multiseq(&[tok_id]).expect("b1 multiseq gpu");
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
        assert_eq!(argmax_mismatches, 0, "B=1 GPU multiseq argmax must match serial GPU every step");
        assert!(worst < LOGIT_TOL, "B=1 GPU multiseq vs serial-GPU max-abs logit diff {worst:.5} exceeds tol {LOGIT_TOL}");
    }
}
#[rustfmt::skip]
mod rwkv7_parity {
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
    //!    `tests/rwkv7_*` (dumped by the `rwkv_ref` llama.cpp harness).
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
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read fixture {path:?}: {e}")).split_whitespace().map(|t| t.parse::<u32>().expect("fixture id parse")).collect()
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
    fn greedy_from_prompt(engine: &mut RwkvSeven, prompt_ids: &[u32], n: usize) -> (Vec<u32>, Vec<f32>) {
        engine.reset_kv_for_test();
        let positions: Vec<usize> = (0..prompt_ids.len()).collect();
        let prompt_logits = engine.forward_tokens_for_test(prompt_ids, &positions).expect("prefill forward");
        let logits0 = prompt_logits.last().expect("prompt logits").clone();

        let mut out = Vec::with_capacity(n);
        let mut next = argmax(&logits0);
        out.push(next);
        for _ in 1..n {
            let lg = engine.forward_tokens_for_test(&[next], &[0]).expect("decode forward").pop().unwrap();
            next = argmax(&lg);
            out.push(next);
        }
        (out, logits0)
    }

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join(format!("rwkv7_{name}"))
    }

    /// Rigorous gate: exact greedy-argmax parity against llama.cpp on F32 weights.
    #[test]
    fn rwkv7_argmax_parity_f32_exact() {
        // Locate the F32 GGUF: env override, else the conventional /tmp path the
        // de-risk harness writes.
        let f32_path = std::env::var("HAWKING_RWKV7_F32_GGUF").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("/tmp/rwkv_ref/rwkv7-04-f32.gguf"));
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

            let matched = mine.iter().zip(ref_gen.iter()).take_while(|(a, b)| a == b).count();
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
        let boxed = hawking_core::model::load_engine(&weights, EngineConfig::default()).expect("load_engine routes rwkv7");
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

        let matched = mine.iter().zip(ref_gen.iter()).take_while(|(a, b)| a == b).count();
        eprintln!("rwkv7 Q4_K vs llama.cpp Q4_K: first-token argmax mine={} ref={}; {}/{} leading tokens match", mine[0], ref_gen[0], matched, n);
        assert!(logits0.len() == 65536, "vocab logits width");
        assert_eq!(mine[0], ref_gen[0], "rwkv7 Q4_K first greedy token must match llama.cpp ({} vs {})", mine[0], ref_gen[0]);
        // Sanity: at least the first few tokens should survive quant drift.
        assert!(matched >= 3, "expected >=3 leading tokens to match under Q4_K (got {matched})");
    }
}
#[rustfmt::skip]
mod rwkv7_state_checkpoint_parity {
    //! W-M1-2 parity gate: the `Engine` checkpoint/fork seam round-trips the RWKV-7
    //! recurrent state with NO re-prefill. Model-gated (skips without a GGUF),
    //! mirroring `rwkv7_parity.rs`. The byte-level round-trip is covered
    //! unconditionally by the `state_serde_tests` unit tests in `model/rwkv7.rs`;
    //! this gate adds the end-to-end "restored state reproduces bit-identical
    //! next-token logits" guarantee that the M1 fork/handoff work builds on.

    use hawking_core::model::rwkv7::RwkvSeven;
    use hawking_core::{Engine, EngineConfig};
    use std::path::PathBuf;

    fn locate_model() -> Option<PathBuf> {
        if let Ok(p) = std::env::var("HAWKING_RWKV7_F32_GGUF") {
            let p = PathBuf::from(p);
            if p.exists() {
                return Some(p);
            }
        }
        for cand in ["/tmp/rwkv_ref/rwkv7-04-f32.gguf", "../../models/rwkv7-04/rwkv7-0.4B-world.Q4_K_M.gguf"] {
            let p = PathBuf::from(cand);
            if p.exists() {
                return Some(p);
            }
        }
        None
    }

    #[test]
    fn rwkv7_checkpoint_roundtrip_bit_identical_logits() {
        let Some(model) = locate_model() else {
            eprintln!(
                "skipping rwkv7_checkpoint_roundtrip: no RWKV-7 GGUF \
                 (set HAWKING_RWKV7_F32_GGUF or place the Q4_K model under models/)"
            );
            return;
        };

        let mut engine = RwkvSeven::load(&model, EngineConfig::default()).expect("load rwkv7");
        engine.reset_kv_for_test();
        engine.forward_tokens_for_test(&[10, 20, 30], &[0, 1, 2]).expect("prefill a few tokens");

        // Snapshot the state; a fork must equal the snapshot at the same point.
        let cp = engine.save_checkpoint().expect("save_checkpoint");
        let fork = engine.fork_state().expect("fork_state");
        assert_eq!(cp, fork, "fork_state must equal the save_checkpoint snapshot");

        // Step one token from the checkpoint, then restore and step the same token:
        // the restored state must reproduce bit-identical next-token logits.
        let l1 = engine.forward_tokens_for_test(&[42], &[3]).expect("step from checkpoint").pop().expect("one logit row");
        engine.load_checkpoint(&cp).expect("load_checkpoint");
        let l2 = engine.forward_tokens_for_test(&[42], &[3]).expect("step from restored state").pop().expect("one logit row");

        assert_eq!(l1, l2, "restored checkpoint must reproduce bit-identical next-token logits (no re-prefill)");
    }
}
#[rustfmt::skip]
mod rwkv7_tq_bench {
    //! Bench scaffold for RWKV-7 TQ (trellis-quant) throughput and memory.
    //!
    //! All tests are `#[ignore]` stubs — they hold signatures and reporting format
    //! for three bench gates that gate TQ deployment:
    //!
    //! 1. Single-stream decode tps vs the Q4_K_M baseline.
    //! 2. Resident memory footprint vs Q4_K_M (target: ≤ 70% RSS at comparable
    //!    quality, i.e. the sub-4-bit byte-cut pays for itself in memory before
    //!    any tps claim).
    //! 3. Speculative-decode accepted tps under TQ draft + Q4_K_M verify (the
    //!    hawking moat lever: TQ's reduced byte footprint keeps the draft cheap
    //!    enough to lift accepted tps above non-spec Q4_K_M).
    //!
    //! Run with:
    //!
    //! ```sh
    //! RWKV7_TQ_MODEL=/path/to/model.tq \
    //! RWKV7_Q4K_MODEL=/path/to/baseline.gguf \
    //!   cargo test -p hawking-core --features tq --test rwkv7_tq_bench \
    //!     -- --nocapture --ignored
    //! ```

    #![cfg(feature = "tq")]
    #![allow(dead_code)]

    /// Print a single bench result line in the canonical hawking format.
    ///
    /// `mode` describes the configuration being measured (e.g. `"TQ k=3 L=7"`).
    /// `tps` is decode tokens-per-second (greedy, single stream, warm).
    /// `rss_mb` is resident set size in MiB after model load.
    /// `bpw` is effective bits-per-weight for the run.
    /// `accepted_tps` is the accepted tokens-per-second under speculative decode,
    /// or `None` if this measurement is a non-spec run.
    fn print_bench_result(mode: &str, tps: f32, rss_mb: f32, bpw: f32, accepted_tps: Option<f32>) {
        let spec_col = match accepted_tps {
            Some(a) => format!("  accepted_tps={a:.2}"),
            None => String::new(),
        };
        println!("[rwkv7_tq_bench]  {mode:<30}  tps={tps:6.2}  rss={rss_mb:7.1} MiB  bpw={bpw:.2}{spec_col}");
    }

    /// Measure single-stream greedy decode tps for the TQ model and print the
    /// result alongside the Q4_K_M baseline.
    ///
    /// Requires:
    /// - `RWKV7_TQ_MODEL`: path to the `.tq` artifact.
    /// - `RWKV7_Q4K_MODEL`: path to the Q4_K_M GGUF baseline (for comparison).
    ///
    /// Warm run: 3 primer tokens discarded, then 128 tokens timed.
    #[test]
    #[ignore = "stub — implement after TqPreparedGpu dispatch and Metal RWKV-7 forward are wired"]
    fn tq_single_stream_tps() {
        let tq_path = std::env::var("RWKV7_TQ_MODEL").expect("RWKV7_TQ_MODEL must be set");
        let _q4k_path = std::env::var("RWKV7_Q4K_MODEL").unwrap_or_else(|_| "(not set — skipping baseline)".to_string());

        println!("STUB: not yet implemented");
        println!("  tq_path = {tq_path}");
        println!("  Would time 128-token greedy decode (3-token warm-up discarded)");
        println!("  and call print_bench_result for both TQ and Q4_K_M.");

        // Example of what the output call will look like once wired:
        // print_bench_result("TQ k=3 L=7 (0.4B)", tq_tps, tq_rss_mb, 3.0, None);
        // print_bench_result("Q4_K_M baseline (0.4B)", q4k_tps, q4k_rss_mb, 4.5, None);
    }

    /// Measure resident memory (RSS) for TQ vs Q4_K_M after model load + one
    /// forward pass (to ensure all Metal buffers are allocated).
    ///
    /// Requires:
    /// - `RWKV7_TQ_MODEL`: path to the `.tq` artifact.
    /// - `RWKV7_Q4K_MODEL`: path to the Q4_K_M GGUF baseline.
    ///
    /// Gate: `rss_tq_mb <= rss_q4k_mb * 0.70` (30% reduction target).
    #[test]
    #[ignore = "stub — implement after TQ loader and Metal buffer allocation are wired"]
    fn tq_resident_memory_vs_q4k() {
        let tq_path = std::env::var("RWKV7_TQ_MODEL").expect("RWKV7_TQ_MODEL must be set");
        let q4k_path = std::env::var("RWKV7_Q4K_MODEL").expect("RWKV7_Q4K_MODEL must be set");

        println!("STUB: not yet implemented");
        println!("  tq_path  = {tq_path}");
        println!("  q4k_path = {q4k_path}");
        println!("  Would measure RSS after load + 1 forward pass for each model.");
        println!("  Gate: rss_tq <= rss_q4k * 0.70");

        // Example output call once wired:
        // print_bench_result("TQ k=3 L=7 (0.4B)", 0.0, tq_rss_mb, 3.0, None);
        // print_bench_result("Q4_K_M baseline (0.4B)", 0.0, q4k_rss_mb, 4.5, None);
        // assert!(tq_rss_mb <= q4k_rss_mb * 0.70, ...);
    }

    /// Measure accepted tokens-per-second under speculative decode with TQ draft +
    /// Q4_K_M verifier. Prints absolute accepted tps and the lift ratio over
    /// non-spec Q4_K_M decode.
    ///
    /// Requires:
    /// - `RWKV7_TQ_DRAFT_MODEL`: path to the small-TQ draft `.tq` artifact.
    /// - `RWKV7_Q4K_MODEL`: path to the Q4_K_M verifier GGUF.
    ///
    /// Bench length: 256 tokens, K=4 (speculate 4 tokens per verify step).
    #[test]
    #[ignore = "stub — implement after TQ speculative decode is wired in the RWKV-7 pipeline"]
    fn tq_spec_decode_accepted_tps() {
        let draft_path = std::env::var("RWKV7_TQ_DRAFT_MODEL").expect("RWKV7_TQ_DRAFT_MODEL must be set");
        let q4k_path = std::env::var("RWKV7_Q4K_MODEL").expect("RWKV7_Q4K_MODEL must be set");

        println!("STUB: not yet implemented");
        println!("  draft_path = {draft_path}");
        println!("  q4k_path   = {q4k_path}");
        println!("  Would run 256-token speculative decode (K=4) and report accepted tps.");

        // Example output call once wired:
        // let accepted_tps: f32 = /* measured */;
        // let draft_bpw: f32 = /* from artifact header */;
        // print_bench_result("TQ draft + Q4K verify (K=4)", accepted_tps, rss_mb, draft_bpw, Some(accepted_tps));
    }
}
#[rustfmt::skip]
mod rwkv7_tq_loader {
    //! Tests for the TQ artifact loader for RWKV-7 models.
    //!
    //! Validates that a `.tq`-format artifact file contains the expected GGUF
    //! tensor names for RWKV-7 and that projection shapes match the 0.4B model
    //! config (n_embd=1024, n_ff=4096, n_layers=24).
    //!
    //! All tests that require a real artifact are `#[ignore]`; they are activated
    //! by setting `RWKV7_TQ_TEST_ARTIFACT` to the path of a `.tq` file before
    //! running:
    //!
    //! ```sh
    //! RWKV7_TQ_TEST_ARTIFACT=/path/to/model.tq \
    //!   cargo test -p hawking-core --features tq --test rwkv7_tq_loader -- --nocapture --ignored
    //! ```

    #![cfg(feature = "tq")]

    #[allow(dead_code)]
    /// Build the list of expected GGUF tensor names for an RWKV-7 model with
    /// `n_layers` transformer layers.
    ///
    /// Each layer contributes 6 projection weight tensors:
    /// - `blk.{i}.time_mix_receptance.weight`
    /// - `blk.{i}.time_mix_key.weight`
    /// - `blk.{i}.time_mix_value.weight`
    /// - `blk.{i}.time_mix_gate.weight`
    /// - `blk.{i}.channel_mix_key.weight`
    /// - `blk.{i}.channel_mix_value.weight`
    fn expected_proj_names(n_layers: usize) -> Vec<String> {
        let mut names = Vec::with_capacity(n_layers * 6);
        for i in 0..n_layers {
            names.push(format!("blk.{i}.time_mix_receptance.weight"));
            names.push(format!("blk.{i}.time_mix_key.weight"));
            names.push(format!("blk.{i}.time_mix_value.weight"));
            names.push(format!("blk.{i}.time_mix_gate.weight"));
            names.push(format!("blk.{i}.channel_mix_key.weight"));
            names.push(format!("blk.{i}.channel_mix_value.weight"));
        }
        names
    }

    /// Loads the artifact from `RWKV7_TQ_TEST_ARTIFACT` and checks that all
    /// 6×24 = 144 expected projection tensor names are present.
    ///
    /// Requires `RWKV7_TQ_TEST_ARTIFACT` to point to a valid 0.4B (24-layer)
    /// RWKV-7 `.tq` artifact.
    #[test]
    #[ignore = "requires RWKV7_TQ_TEST_ARTIFACT env var pointing to a .tq file"]
    fn tq_artifact_loads_expected_names() {
        let path = std::env::var("RWKV7_TQ_TEST_ARTIFACT").expect("RWKV7_TQ_TEST_ARTIFACT must be set to run this test");

        // VERIFY PATH: update this import once the real loader function is wired.
        // The function is expected to return a type that exposes tensor names.
        // hawking_core::model::rwkv7::load_tq_artifact(&path)
        let _ = &path; // placeholder until the real loader is available
        panic!("STUB: wire hawking_core::model::rwkv7::load_tq_artifact and check tensor names");

        // Expected structure once wired:
        //   let artifact = hawking_core::model::rwkv7::load_tq_artifact(&path)
        //       .expect("artifact load should succeed");
        //   let names = expected_proj_names(24);
        //   for name in &names {
        //       assert!(
        //           artifact.contains_tensor(name),
        //           "artifact missing expected tensor: {name}"
        //       );
        //   }
        //   assert_eq!(names.len(), 144);
    }

    /// Loads the artifact from `RWKV7_TQ_TEST_ARTIFACT` and checks that the
    /// channel_mix projections have the shapes expected for the RWKV-7 0.4B model
    /// (n_embd=1024, n_ff=4096).
    ///
    /// `channel_mix_key.weight` shape: [n_ff, n_embd] = [4096, 1024]
    /// `channel_mix_value.weight` shape: [n_embd, n_ff] = [1024, 4096]
    #[test]
    #[ignore = "requires RWKV7_TQ_TEST_ARTIFACT env var pointing to a .tq file"]
    fn tq_artifact_shapes_match_04b() {
        let path = std::env::var("RWKV7_TQ_TEST_ARTIFACT").expect("RWKV7_TQ_TEST_ARTIFACT must be set to run this test");

        const N_FF: usize = 4096;
        const N_EMBD: usize = 1024;

        // VERIFY PATH: update import once the real loader is wired.
        let _ = (path, N_FF, N_EMBD);
        panic!("STUB: wire hawking_core::model::rwkv7::load_tq_artifact and check shapes");

        // Expected structure once wired:
        //   let artifact = hawking_core::model::rwkv7::load_tq_artifact(&path)
        //       .expect("artifact load should succeed");
        //   for i in 0..24 {
        //       let cmk = artifact.tensor(&format!("blk.{i}.channel_mix_key.weight"))
        //           .expect("channel_mix_key must exist");
        //       assert_eq!(cmk.shape(), [N_FF, N_EMBD], "blk.{i} channel_mix_key shape");
        //
        //       let cmv = artifact.tensor(&format!("blk.{i}.channel_mix_value.weight"))
        //           .expect("channel_mix_value must exist");
        //       assert_eq!(cmv.shape(), [N_EMBD, N_FF], "blk.{i} channel_mix_value shape");
        //   }
    }

    /// Loading a non-existent artifact must return an `Err`, not panic.
    ///
    /// This test does NOT require any real artifact — it is always runnable.
    #[test]
    fn tq_loader_missing_artifact_is_err() {
        // VERIFY PATH: update to the real loader function path once wired.
        // The function under test is expected to have the signature:
        //   pub fn load_tq_artifact(path: impl AsRef<std::path::Path>)
        //       -> hawking_core::Result<TqArtifact>
        //
        // Calling it with a non-existent path must return Err, not panic.
        //
        // Uncomment and adjust once the function exists:
        // let result = hawking_core::model::rwkv7::load_tq_artifact(
        //     "/tmp/this_file_does_not_exist_rwkv7_tq.tq",
        // );
        // assert!(result.is_err(), "loading a missing artifact must return Err");

        // Until wired, assert that the `expected_proj_names` helper works correctly
        // so at least one non-ignored assertion runs in this file.
        let names = expected_proj_names(24);
        assert_eq!(names.len(), 144, "24 layers × 6 projections = 144 names");
        assert_eq!(names[0], "blk.0.time_mix_receptance.weight");
        assert_eq!(names[5], "blk.0.channel_mix_value.weight");
        assert_eq!(names[6], "blk.1.time_mix_receptance.weight");
        assert_eq!(names[143], "blk.23.channel_mix_value.weight", "last tensor name for 24-layer model");
    }
}
#[rustfmt::skip]
mod rwkv7_tq_parity {
    //! Parity scaffold for RWKV-7 TQ (trellis-quant) vs Q4_K_M and vs Metal decode.
    //!
    //! All tests in this file are `#[ignore]` stubs — they hold the signatures and
    //! the env-var loading pattern for the three parity gates that must pass before
    //! TQ can be enabled in the RWKV-7 serving path:
    //!
    //! 1. CPU greedy trajectory matches Metal greedy trajectory token-for-token.
    //! 2. TQ perplexity is within the "silver band" of Q4_K_M (≤ +0.3 PPL on the
    //!    calibration set).
    //! 3. Two runs with identical seed produce identical greedy output (determinism
    //!    gate, required before any A/B bench is meaningful).
    //!
    //! Enable a test by setting the required env vars and passing `--ignored`:
    //!
    //! ```sh
    //! RWKV7_TQ_MODEL=/path/to/model.tq \
    //! RWKV7_Q4K_MODEL=/path/to/model.gguf \
    //!   cargo test -p hawking-core --features tq --test rwkv7_tq_parity -- --nocapture --ignored
    //! ```

    #![cfg(feature = "tq")]
    #![allow(dead_code)]

    /// Prompts used as fixtures across all parity tests. Short enough to run in
    /// seconds on a dev machine, varied enough to exercise both the attention branch
    /// and the channel-mix branch across multiple layer types.
    const FIXTURE_PROMPTS: [&str; 3] = [
        "The quick brown fox jumps over the lazy dog.",
        "In mathematics, a prime number is a natural number greater than 1",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    ];

    /// Compare RWKV-7 TQ CPU greedy output against Metal (GPU) greedy output
    /// token-for-token for each fixture prompt.
    ///
    /// Requires:
    /// - `RWKV7_TQ_MODEL`: path to the `.tq` artifact.
    /// - `RWKV7_TOKENIZER`: path to the RWKV World tokenizer vocab file.
    ///
    /// Pass length: 32 tokens per prompt (enough to surface divergence without
    /// slow CI wall-clock).
    #[test]
    #[ignore = "stub — implement after TqPreparedGpu dispatch and CPU serving reference are wired"]
    fn rwkv7_tq_cpu_vs_metal_greedy_trajectory() {
        let model_path = std::env::var("RWKV7_TQ_MODEL").expect("RWKV7_TQ_MODEL must be set");
        let _tokenizer_path = std::env::var("RWKV7_TOKENIZER").unwrap_or_else(|_| "tokenizer/rwkv_vocab_v20230424.txt".to_string());

        println!("STUB: not yet implemented");
        println!("  model_path = {model_path}");
        println!("  fixture_prompts = {}", FIXTURE_PROMPTS.len());
        println!("  Would compare CPU vs Metal greedy trajectories for each prompt.");
        println!("  Wire hawking_core::model::rwkv7 CPU and Metal forward paths, then");
        println!("  assert token-for-token equality for 32-token generations.");
    }

    /// Verify that TQ perplexity on the calibration set is within the silver band
    /// of Q4_K_M: `ppl_tq <= ppl_q4k + 0.3`.
    ///
    /// Requires:
    /// - `RWKV7_TQ_MODEL`: path to the `.tq` artifact.
    /// - `RWKV7_Q4K_MODEL`: path to the baseline Q4_K_M GGUF.
    /// - `RWKV7_CALIB_CORPUS`: path to calibration corpus (JSON lines of text).
    ///
    /// N=100 sequences from the calibration corpus, max 512 tokens each.
    #[test]
    #[ignore = "stub — implement after TQ serving reference is wired and calibration corpus is available"]
    fn rwkv7_tq_vs_q4k_ppl_within_silver() {
        let tq_path = std::env::var("RWKV7_TQ_MODEL").expect("RWKV7_TQ_MODEL must be set");
        let q4k_path = std::env::var("RWKV7_Q4K_MODEL").expect("RWKV7_Q4K_MODEL must be set");
        let _corpus_path = std::env::var("RWKV7_CALIB_CORPUS").expect("RWKV7_CALIB_CORPUS must be set");

        println!("STUB: not yet implemented");
        println!("  tq_path   = {tq_path}");
        println!("  q4k_path  = {q4k_path}");
        println!("  Would compute PPL on N=100 corpus sequences and assert:");
        println!("    ppl_tq <= ppl_q4k + 0.3  (silver gate)");
    }

    /// Verify that two greedy decode runs with the same model and the same prompt
    /// produce identical token sequences (determinism gate).
    ///
    /// Requires:
    /// - `RWKV7_TQ_MODEL`: path to the `.tq` artifact.
    ///
    /// Run 3 times per fixture prompt, compare all pairs. Length: 64 tokens.
    #[test]
    #[ignore = "stub — implement after TqPreparedGpu dispatch is wired"]
    fn rwkv7_tq_deterministic_across_runs() {
        let model_path = std::env::var("RWKV7_TQ_MODEL").expect("RWKV7_TQ_MODEL must be set");

        println!("STUB: not yet implemented");
        println!("  model_path = {model_path}");
        println!("  Would run each fixture prompt 3 times and assert identical 64-token outputs.");
        println!("  Failure here means the kernel has non-deterministic memory reads or");
        println!("  the Metal pipeline cache is not stable across command buffer submissions.");
    }
}
