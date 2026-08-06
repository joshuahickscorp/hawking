use hawking_core::model::rwkv7::{RwkvMultiState, RwkvSeven};
use hawking_core::{Engine, EngineConfig};
use std::path::{Path, PathBuf};
mod common;
use common::*;
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
fn locate_model() -> Option<PathBuf> {
    let f32_path = std::env::var("HAWKING_RWKV7_F32_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/rwkv_ref/rwkv7-04-f32.gguf"));
    if f32_path.exists() {
        return Some(f32_path);
    }
    locate_q4k()
}
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
    let mut serial: Vec<Vec<Vec<f32>>> = Vec::with_capacity(B);
    for s in 0..B {
        engine.reset_kv_for_test();
        let mut per_step = Vec::with_capacity(STEPS);
        for t in 0..STEPS {
            per_step.push(engine.forward_token(streams[s][t]).expect("serial forward"));
        }
        serial.push(per_step);
    }
    let mut multi = RwkvMultiState::new(&engine.config, B);
    for t in 0..STEPS {
        let col: Vec<u32> = (0..B).map(|s| streams[s][t]).collect();
        let rows = engine
            .forward_tokens_multiseq_cpu(&col, &mut multi)
            .expect("multiseq cpu forward");
        assert_eq!(rows.len(), B, "multiseq returned {} rows != B", rows.len());
        for s in 0..B {
            assert_eq!(
                rows[s], serial[s][t],
                "multiseq stream {s} step {t} differs from serial forward_token \
                 (state leaked across streams)"
            );
        }
    }
}
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
    for t in 0..4 {
        let col: Vec<u32> = (0..B).map(|s| streams[s][t]).collect();
        engine
            .forward_tokens_multiseq_cpu(&col, &mut multi)
            .expect("warm multiseq");
    }
    multi.reset_slot(0);
    let post: Vec<u32> = (4..STEPS).map(|t| streams[0][t]).collect();
    engine.reset_kv_for_test();
    let mut slot0_ref = Vec::with_capacity(post.len());
    for &tok in &post {
        slot0_ref.push(engine.forward_token(tok).expect("slot0 ref forward"));
    }
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
}
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
            }
        }
    }
    assert_eq!(
        argmax_mismatches, 0,
        "GPU multiseq argmax must match the CPU oracle every (stream, step)"
    );
    assert!(
        worst < LOGIT_TOL,
        "GPU↔CPU multiseq max-abs logit diff {worst:.5} exceeds tol {LOGIT_TOL}"
    );
}
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
    assert!(
        worst < LOGIT_TOL,
        "GPU multiseq vs serial-GPU max-abs logit diff {worst:.5} exceeds tol {LOGIT_TOL}"
    );
}
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
    assert!(
        d < 0.05,
        "x_norm diverges at B=2: max|Δ|={d:.5} — bug is before final LayerNorm"
    );
}
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
    engine.ensure_gpu_batch(2).expect("B=2");
    engine.reset_gpu_multiseq();
    engine
        .forward_token_gpu_multiseq(&[tok0, streams[1][0]])
        .expect("b2 multiseq");
    let (b2_att, b2_ffn, n_layer, n_embd) = {
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
        (att, ffn, nl, n)
    };
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
        if first_divergent.is_none() && (d_att > 0.05 || d_ffn > 0.05) {
            first_divergent = Some(li);
        }
    }
    if let Some(li) = first_divergent {
        panic!("first divergent layer: {li}  (see per-layer log above)");
    }
}
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
        let ptr = g.arena.wkv_state.contents() as *const f32;
        let s0 = unsafe { std::slice::from_raw_parts(ptr, nl * spl) }.to_vec();
        (s0, spl, nl)
    };
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
        if first_wkv_divergent.is_none() && d > 1e-4 {
            first_wkv_divergent = Some(li);
        }
    }
    if let Some(li) = first_wkv_divergent {
        panic!(
            "wkv_state first divergent layer {li} — bug in WKV recurrence or its r/w/k/v inputs"
        );
    }
}
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
}
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
}
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
    engine.ensure_gpu_batch(1).expect("B=1");
    engine.reset_gpu_multiseq();
    let serial: Vec<Vec<f32>> = streams[0]
        .iter()
        .map(|&t| engine.forward_token_gpu(t).expect("serial gpu"))
        .collect();
    engine.ensure_gpu_batch(2).expect("B=2");
    engine.reset_gpu_multiseq();
    for (t, _) in streams[0].iter().enumerate() {
        let col: Vec<u32> = (0..2).map(|s| streams[s][t]).collect();
        let rows = engine
            .forward_token_gpu_multiseq(&col)
            .expect("b2 multiseq gpu");
        let d = max_abs_diff(&rows[0], &serial[t]);
        let (ag, ac) = (argmax(&rows[0]), argmax(&serial[t]));
        assert!(
            d < LOGIT_TOL,
            "B=2 multiseq stream 0 step {t}: max|Δlogit|={d:.5} exceeds tol"
        );
    }
}
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
    engine.ensure_gpu_batch(1).expect("init B=1 arena");
    engine.reset_gpu_multiseq();
    let serial: Vec<Vec<f32>> = tok
        .iter()
        .map(|&t| engine.forward_token_gpu(t).expect("serial gpu"))
        .collect();
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
        }
    }
    assert_eq!(
        argmax_mismatches, 0,
        "B=1 GPU multiseq argmax must match serial GPU every step"
    );
    assert!(
        worst < LOGIT_TOL,
        "B=1 GPU multiseq vs serial-GPU max-abs logit diff {worst:.5} exceeds tol {LOGIT_TOL}"
    );
}
