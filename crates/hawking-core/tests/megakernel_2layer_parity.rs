#![cfg(target_os = "macos")]
use half::f16;
use hawking_core::kernels::megakernel::{
    megakernel_2layer_dispatch, megakernel_nlayer_dispatch, MegakernelRunner, MK_PROBE_ATTN_OUT,
    MK_PROBE_FFN_DOWN, MK_PROBE_O_PROJ, MK_PROBE_Q_ROT, MK_PROBE_RESIDUAL, MK_PROBE_RESIDUAL_L0,
    MK_PROBE_XNORM_A, MK_PROBE_XNORM_FFN,
};
use hawking_core::metal::MetalContext;
use hawking_core::model::qwen_dense::{MegakernelLayerWeightsF16, QwenDense};
use hawking_core::{Engine, EngineConfig};
use std::path::PathBuf;
mod common;
use common::*;
const TOKEN: u32 = 42;
const POS: usize = 0;
const LAST_LAYER: usize = 1;
const MAX_SEQ: u32 = 256;
const HIDDEN: usize = 2048;
const N_HEADS: usize = 16;
const N_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 128;
const Q_DIM: usize = N_HEADS * HEAD_DIM; // 2048
const KV_DIM: usize = N_KV_HEADS * HEAD_DIM; // 256
const INTERMEDIATE: usize = 11008;
const RMS_EPS: f32 = 1e-6;
const ROPE_THETA: f32 = 1_000_000.0;
const RTOL: f32 = 2e-3;
const RTOL_MULTILAYER: f32 = 2e-2;
/// Multi-layer absolute tolerance — looser than single-stage to absorb
/// fp16 cancellation noise on values that pass through additive paths
/// (residual streams routinely contain near-zero entries where small
/// f16 rounding errors dominate the magnitude). Tracks published
/// guidance on comparing networks up to ~1% relative without flagging
/// model-correctness regressions.
const ATOL_MULTILAYER: f32 = 5e-3;
fn weights_path() -> PathBuf {
    if let Ok(p) = std::env::var("HAWKING_QWEN_GGUF") {
        return PathBuf::from(p);
    }
    PathBuf::from("../../models/qwen2.5-3b-instruct-q4_k_m.gguf")
}
#[test]
#[ignore = "megakernel POC: requires Qwen-3B weights via HAWKING_QWEN_GGUF"]
fn megakernel_2layer_parity_qwen3b() {
    let weights = weights_path();
    if !weights.exists() {
        eprintln!("SKIP: model not at {}", weights.display());
        return;
    }
    let cfg = EngineConfig::default();
    let mut model = <QwenDense as Engine>::load(&weights, cfg).expect("load QwenDense");
    let h = model.config.hidden;
    let q_dim = model.config.n_heads * model.config.head_dim;
    let kv_dim = model.config.n_kv_heads * model.config.head_dim;
    let mid = model.config.intermediate;
    assert_eq!(h, HIDDEN);
    assert_eq!(q_dim, Q_DIM);
    assert_eq!(kv_dim, KV_DIM);
    let ref_x = model
        .forward_layers_subset(TOKEN, POS, LAST_LAYER)
        .expect("forward_layers_subset");
    assert_eq!(ref_x.len(), h, "ref residual has wrong length");
    assert!(
        ref_x.iter().all(|v: &f32| v.is_finite()),
        "ref residual contains NaN/Inf"
    );
    let layer0 = model
        .prep_megakernel_layer_f16(0)
        .expect("prep_megakernel_layer_f16(0)");
    let layer1 = model
        .prep_megakernel_layer_f16(1)
        .expect("prep_megakernel_layer_f16(1)");
    assert_layer_shapes(&layer0, h, q_dim, kv_dim, mid, "layer 0");
    assert_layer_shapes(&layer1, h, q_dim, kv_dim, mid, "layer 1");
    let x_in: Vec<f16> = (0..h)
        .map(|i| f16::from_f32((i as f32) * 0.001 - 1.0))
        .collect();
    let x_in_f32: Vec<f32> = x_in.iter().map(|v| v.to_f32()).collect();
    let ctx = MetalContext::new().expect("MetalContext::new");
    {
        let x_out = megakernel_2layer_dispatch(
            &ctx,
            &layer0,
            &layer1,
            &x_in,
            POS as u32,
            (POS + 1) as u32,
            MAX_SEQ,
            MK_PROBE_XNORM_A,
        )
        .expect("megakernel dispatch (stage A)");
        assert_eq!(x_out.len(), h);
        let ref_xnorm = cpu_rmsnorm(&x_in_f32, &layer0.attn_norm, RMS_EPS);
        let (worst, idx, gv, wv) = max_violation_f16_vs_f32(&x_out, &ref_xnorm);
        assert!(
            worst <= 0.0,
            "stage-A rmsnorm parity FAIL: violation={worst:.3e} at i={idx} \
             (got {gv}, want {wv}, allowed atol+rtol·|want|, atol={ATOL:.0e}, rtol={RTOL:.0e})",
        );
    }
    {
        let x_out = megakernel_2layer_dispatch(
            &ctx,
            &layer0,
            &layer1,
            &x_in,
            POS as u32,
            (POS + 1) as u32,
            MAX_SEQ,
            MK_PROBE_Q_ROT,
        )
        .expect("megakernel dispatch (stage D)");
        assert_eq!(x_out.len(), h);
        let x_norm = cpu_rmsnorm(&x_in_f32, &layer0.attn_norm, RMS_EPS);
        let mut q = cpu_gemv_f16(&layer0.q_proj, q_dim, h, &x_norm);
        for i in 0..q_dim {
            q[i] += layer0.q_bias[i];
        }
        for hh in 0..N_HEADS {
            let off = hh * HEAD_DIM;
            cpu_rope_inplace(&mut q[off..off + HEAD_DIM], POS as u32, ROPE_THETA);
        }
        let (worst, idx, gv, wv) = max_violation_f16_vs_f32(&x_out, &q);
        assert!(
            worst <= 0.0,
            "stage-D Q (post-RoPE) parity FAIL: violation={worst:.3e} at i={idx} \
             (got {gv}, want {wv}, allowed atol+rtol·|want|, atol={ATOL:.0e}, rtol={RTOL:.0e})",
        );
    }
    {
        let x_out = megakernel_2layer_dispatch(
            &ctx,
            &layer0,
            &layer1,
            &x_in,
            POS as u32,
            (POS + 1) as u32,
            MAX_SEQ,
            MK_PROBE_ATTN_OUT,
        )
        .expect("megakernel dispatch (stage F)");
        assert_eq!(x_out.len(), h);
        let attn_out_ref = cpu_layer0_attn_out_pos0(&x_in_f32, &layer0);
        let (worst, idx, gv, wv) = max_violation_f16_vs_f32(&x_out, &attn_out_ref);
        assert!(
            worst <= 0.0,
            "stage-F attn_out (pos=0) parity FAIL: violation={worst:.3e} at i={idx} \
             (got {gv}, want {wv}, allowed atol+rtol·|want|, atol={ATOL:.0e}, rtol={RTOL:.0e})",
        );
    }
    {
        let x_out = megakernel_2layer_dispatch(
            &ctx,
            &layer0,
            &layer1,
            &x_in,
            POS as u32,
            (POS + 1) as u32,
            MAX_SEQ,
            MK_PROBE_O_PROJ,
        )
        .expect("megakernel dispatch (stage G)");
        assert_eq!(x_out.len(), h);
        let attn_out = cpu_layer0_attn_out_pos0(&x_in_f32, &layer0);
        let o = cpu_gemv_f16(&layer0.o_proj, HIDDEN, Q_DIM, &attn_out);
        let (worst, idx, gv, wv) = max_violation_f16_vs_f32(&x_out, &o);
        assert!(
            worst <= 0.0,
            "stage-G o_proj parity FAIL: violation={worst:.3e} at i={idx} \
             (got {gv}, want {wv}, allowed atol+rtol·|want|, atol={ATOL:.0e}, rtol={RTOL:.0e})",
        );
    }
    {
        let x_out = megakernel_2layer_dispatch(
            &ctx,
            &layer0,
            &layer1,
            &x_in,
            POS as u32,
            (POS + 1) as u32,
            MAX_SEQ,
            MK_PROBE_XNORM_FFN,
        )
        .expect("megakernel dispatch (stage H)");
        assert_eq!(x_out.len(), h);
        let attn_out = cpu_layer0_attn_out_pos0(&x_in_f32, &layer0);
        let o = cpu_gemv_f16(&layer0.o_proj, HIDDEN, Q_DIM, &attn_out);
        let mut residual = x_in_f32.clone();
        for i in 0..HIDDEN {
            residual[i] += o[i];
        }
        let xnorm_ffn = cpu_rmsnorm(&residual, &layer0.ffn_norm, RMS_EPS);
        let (worst, idx, gv, wv) = max_violation_f16_vs_f32(&x_out, &xnorm_ffn);
        assert!(
            worst <= 0.0,
            "stage-H xnorm_ffn parity FAIL: violation={worst:.3e} at i={idx} \
             (got {gv}, want {wv}, allowed atol+rtol·|want|, atol={ATOL:.0e}, rtol={RTOL:.0e})",
        );
    }
    {
        let x_out = megakernel_2layer_dispatch(
            &ctx,
            &layer0,
            &layer1,
            &x_in,
            POS as u32,
            (POS + 1) as u32,
            MAX_SEQ,
            MK_PROBE_FFN_DOWN,
        )
        .expect("megakernel dispatch (stage K)");
        assert_eq!(x_out.len(), h);
        let attn_out = cpu_layer0_attn_out_pos0(&x_in_f32, &layer0);
        let o = cpu_gemv_f16(&layer0.o_proj, HIDDEN, Q_DIM, &attn_out);
        let mut residual = x_in_f32.clone();
        for i in 0..HIDDEN {
            residual[i] += o[i];
        }
        let xnorm_ffn = cpu_rmsnorm(&residual, &layer0.ffn_norm, RMS_EPS);
        let g = cpu_gemv_f16(&layer0.ffn_gate, INTERMEDIATE, HIDDEN, &xnorm_ffn);
        let u = cpu_gemv_f16(&layer0.ffn_up, INTERMEDIATE, HIDDEN, &xnorm_ffn);
        let mut act = vec![0.0f32; INTERMEDIATE];
        for i in 0..INTERMEDIATE {
            let s = g[i] / (1.0 + (-g[i]).exp());
            act[i] = s * u[i];
        }
        let ffn_down = cpu_gemv_f16(&layer0.ffn_down, HIDDEN, INTERMEDIATE, &act);
        let (worst, idx, gv, wv) = max_violation_f16_vs_f32(&x_out, &ffn_down);
        assert!(
            worst <= 0.0,
            "stage-K ffn_down parity FAIL: violation={worst:.3e} at i={idx} \
             (got {gv}, want {wv}, allowed atol+rtol·|want|, atol={ATOL:.0e}, rtol={RTOL:.0e})",
        );
    }
    {
        let x_out = megakernel_2layer_dispatch(
            &ctx,
            &layer0,
            &layer1,
            &x_in,
            POS as u32,
            (POS + 1) as u32,
            MAX_SEQ,
            MK_PROBE_RESIDUAL_L0,
        )
        .expect("megakernel dispatch (stage L, post-l0)");
        assert_eq!(x_out.len(), h);
        let residual = cpu_layer_forward(&x_in_f32, &layer0, POS as u32);
        let (worst, idx, gv, wv) = max_violation_f16_vs_f32(&x_out, &residual);
        assert!(
            worst <= 0.0,
            "stage-L residual (post-layer-0) parity FAIL: violation={worst:.3e} at i={idx} \
             (got {gv}, want {wv}, allowed atol+rtol·|want|, atol={ATOL:.0e}, rtol={RTOL:.0e})",
        );
    }
    {
        let x_out = megakernel_2layer_dispatch(
            &ctx,
            &layer0,
            &layer1,
            &x_in,
            POS as u32,
            (POS + 1) as u32,
            MAX_SEQ,
            MK_PROBE_RESIDUAL,
        )
        .expect("megakernel dispatch (post-l1 final)");
        assert_eq!(x_out.len(), h);
        let residual_l0 = cpu_layer_forward(&x_in_f32, &layer0, POS as u32);
        let residual_l1 = cpu_layer_forward(&residual_l0, &layer1, POS as u32);
        let (worst, idx, gv, wv) =
            max_violation_f16_vs_f32_tol(&x_out, &residual_l1, ATOL_MULTILAYER, RTOL_MULTILAYER);
        assert!(
            worst <= 0.0,
            "2-layer post-l1 residual parity FAIL: violation={worst:.3e} at i={idx} \
             (got {gv}, want {wv}, allowed atol+rtol·|want|, atol={ATOL_MULTILAYER:.0e}, rtol={RTOL_MULTILAYER:.0e})",
        );
    }
}
#[test]
#[ignore = "megakernel POC: requires Qwen-3B weights via HAWKING_QWEN_GGUF"]
fn megakernel_nlayer_parity_qwen3b() {
    let weights = weights_path();
    if !weights.exists() {
        eprintln!("SKIP: model not at {}", weights.display());
        return;
    }
    let cfg = EngineConfig::default();
    let model = <QwenDense as Engine>::load(&weights, cfg).expect("load QwenDense");
    let h = model.config.hidden;
    assert_eq!(h, HIDDEN);
    let ctx = MetalContext::new().expect("MetalContext::new");
    let x_in: Vec<f16> = (0..h)
        .map(|i| f16::from_f32((i as f32) * 0.001 - 1.0))
        .collect();
    let x_in_f32: Vec<f32> = x_in.iter().map(|v| v.to_f32()).collect();
    {
        let two: Vec<MegakernelLayerWeightsF16> = (0..2)
            .map(|li| model.prep_megakernel_layer_f16(li).expect("prep"))
            .collect();
        let want = megakernel_2layer_dispatch(
            &ctx,
            &two[0],
            &two[1],
            &x_in,
            POS as u32,
            (POS + 1) as u32,
            MAX_SEQ,
            MK_PROBE_RESIDUAL,
        )
        .expect("2-layer dispatch");
        let got =
            megakernel_nlayer_dispatch(&ctx, &two, &x_in, POS as u32, (POS + 1) as u32, MAX_SEQ)
                .expect("n-layer dispatch (N=2)");
        assert_eq!(got.len(), h);
        let want_f32: Vec<f32> = want.iter().map(|v| v.to_f32()).collect();
        let (worst, idx, gv, wv) = max_violation_f16_vs_f32_tol(&got, &want_f32, ATOL, RTOL);
        assert!(
            worst <= 0.0,
            "N=2 megakernel vs 2-layer kernel FAIL: violation={worst:.3e} at i={idx} \
             (got {gv}, want {wv}, atol={ATOL:.0e}, rtol={RTOL:.0e})",
        );
        let exact = got
            .iter()
            .zip(want.iter())
            .filter(|(a, b)| a.to_bits() == b.to_bits())
            .count();
    }
    {
        const N8: usize = 8;
        assert!(
            model.config.n_layers >= N8,
            "model has {} layers (< {N8})",
            model.config.n_layers
        );
        let layers: Vec<MegakernelLayerWeightsF16> = (0..N8)
            .map(|li| model.prep_megakernel_layer_f16(li).expect("prep"))
            .collect();
        let got =
            megakernel_nlayer_dispatch(&ctx, &layers, &x_in, POS as u32, (POS + 1) as u32, MAX_SEQ)
                .expect("n-layer dispatch (N=8)");
        assert_eq!(got.len(), h);
        assert!(
            got.iter().all(|v| v.to_f32().is_finite()),
            "N=8 megakernel output has NaN/Inf"
        );
        let mut ref_res = x_in_f32.clone();
        for layer in &layers {
            ref_res = cpu_layer_forward(&ref_res, layer, POS as u32);
        }
        let scale = (N8 as f32) / 2.0;
        let atol = ATOL_MULTILAYER * scale;
        let rtol = RTOL_MULTILAYER * scale;
        let (worst, idx, gv, wv) = max_violation_f16_vs_f32_tol(&got, &ref_res, atol, rtol);
        assert!(
            worst <= 0.0,
            "N=8 megakernel parity FAIL: violation={worst:.3e} at i={idx} \
             (got {gv}, want {wv}, atol={atol:.1e}, rtol={rtol:.1e})",
        );
    }
}
#[test]
#[ignore = "megakernel bench: requires Qwen-3B weights via HAWKING_QWEN_GGUF"]
fn megakernel_nlayer_bench_qwen3b() {
    let weights = weights_path();
    if !weights.exists() {
        eprintln!("SKIP: model not at {}", weights.display());
        return;
    }
    let cfg = EngineConfig::default();
    let mut model = <QwenDense as Engine>::load(&weights, cfg).expect("load QwenDense");
    let h = model.config.hidden;
    let ctx = MetalContext::new().expect("MetalContext::new");
    const N: usize = 8;
    const ITERS: usize = 40;
    const WARMUP: usize = 5;
    let layers: Vec<MegakernelLayerWeightsF16> = (0..N)
        .map(|li| model.prep_megakernel_layer_f16(li).expect("prep"))
        .collect();
    let runner = MegakernelRunner::new(&ctx, &layers, MAX_SEQ).expect("runner");
    let x_in: Vec<f16> = (0..h)
        .map(|i| f16::from_f32((i as f32) * 0.001 - 1.0))
        .collect();
    for _ in 0..WARMUP {
        let _ = runner.step(&ctx, &x_in, 0, 1).expect("mk step");
    }
    let t0 = std::time::Instant::now();
    for _ in 0..ITERS {
        let _ = runner.step(&ctx, &x_in, 0, 1).expect("mk step");
    }
    let mk_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
    for _ in 0..WARMUP {
        let _ = model.forward_layers_subset(TOKEN, 0, N - 1).expect("fwd");
    }
    let t1 = std::time::Instant::now();
    for _ in 0..ITERS {
        let _ = model.forward_layers_subset(TOKEN, 0, N - 1).expect("fwd");
    }
    let perop_us = t1.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
    let ratio = perop_us / mk_us;
}
fn cpu_layer_forward(
    x_in_f32: &[f32],
    layer: &MegakernelLayerWeightsF16,
    _pos: u32, // POS=0 → MHA softmax degenerate (attn = V replicated)
) -> Vec<f32> {
    let attn_out = cpu_layer0_attn_out_pos0(x_in_f32, layer);
    let o = cpu_gemv_f16(&layer.o_proj, HIDDEN, Q_DIM, &attn_out);
    let mut residual: Vec<f32> = x_in_f32.iter().zip(o.iter()).map(|(a, b)| a + b).collect();
    let x_norm_ffn = cpu_rmsnorm(&residual, &layer.ffn_norm, RMS_EPS);
    let g = cpu_gemv_f16(&layer.ffn_gate, INTERMEDIATE, HIDDEN, &x_norm_ffn);
    let u = cpu_gemv_f16(&layer.ffn_up, INTERMEDIATE, HIDDEN, &x_norm_ffn);
    let act: Vec<f32> = g
        .iter()
        .zip(u.iter())
        .map(|(gi, ui)| (gi / (1.0 + (-gi).exp())) * ui)
        .collect();
    let ffn_down = cpu_gemv_f16(&layer.ffn_down, HIDDEN, INTERMEDIATE, &act);
    for i in 0..HIDDEN {
        residual[i] += ffn_down[i];
    }
    residual
}
fn cpu_layer0_attn_out_pos0(x_in_f32: &[f32], layer0: &MegakernelLayerWeightsF16) -> Vec<f32> {
    let x_norm = cpu_rmsnorm(x_in_f32, &layer0.attn_norm, RMS_EPS);
    let mut v = cpu_gemv_f16(&layer0.v_proj, KV_DIM, HIDDEN, &x_norm);
    for i in 0..KV_DIM {
        v[i] += layer0.v_bias[i];
    }
    let group_size = N_HEADS / N_KV_HEADS;
    let mut attn_out = vec![0.0f32; N_HEADS * HEAD_DIM];
    for h in 0..N_HEADS {
        let kv_h = h / group_size;
        for d in 0..HEAD_DIM {
            attn_out[h * HEAD_DIM + d] = v[kv_h * HEAD_DIM + d];
        }
    }
    attn_out
}
fn cpu_rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut ssq = 0.0f32;
    for &v in x {
        ssq += v * v;
    }
    let rnorm = 1.0f32 / (ssq / (n as f32) + eps).sqrt();
    (0..n).map(|i| x[i] * rnorm * weight[i]).collect()
}
fn cpu_gemv_f16(w: &[f16], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    assert_eq!(w.len(), rows * cols);
    assert_eq!(x.len(), cols);
    let mut out = vec![0.0f32; rows];
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let mut acc = 0.0f32;
        for c in 0..cols {
            acc += row[c].to_f32() * x[c];
        }
        out[r] = acc;
    }
    out
}
fn cpu_rope_inplace(x: &mut [f32], pos: u32, base: f32) {
    let head_dim = x.len();
    let half = head_dim / 2;
    for i in 0..half {
        let theta = (pos as f32) / base.powf(2.0 * i as f32 / head_dim as f32);
        let (sin, cos) = theta.sin_cos();
        let x0 = x[2 * i];
        let x1 = x[2 * i + 1];
        x[2 * i] = x0 * cos - x1 * sin;
        x[2 * i + 1] = x0 * sin + x1 * cos;
    }
}
fn max_violation_f16_vs_f32_tol(
    got: &[f16],
    want: &[f32],
    atol: f32,
    rtol: f32,
) -> (f32, usize, f32, f32) {
    assert!(
        got.len() >= want.len(),
        "shader probe output too short: got {} want {}",
        got.len(),
        want.len(),
    );
    let mut worst = f32::NEG_INFINITY;
    let mut argmax = 0usize;
    let mut got_v = 0.0f32;
    let mut want_v = 0.0f32;
    for i in 0..want.len() {
        let g = got[i].to_f32();
        let w = want[i];
        let allowed = atol + rtol * w.abs();
        let v = (g - w).abs() - allowed;
        if v > worst {
            worst = v;
            argmax = i;
            got_v = g;
            want_v = w;
        }
    }
    (worst, argmax, got_v, want_v)
}
fn max_violation_f16_vs_f32(got: &[f16], want: &[f32]) -> (f32, usize, f32, f32) {
    max_violation_f16_vs_f32_tol(got, want, ATOL, RTOL)
}
fn assert_layer_shapes(
    w: &MegakernelLayerWeightsF16,
    h: usize,
    q_dim: usize,
    kv_dim: usize,
    mid: usize,
    tag: &str,
) {
    assert_eq!(w.q_proj.len(), q_dim * h, "{tag}: q_proj shape");
    assert_eq!(w.k_proj.len(), kv_dim * h, "{tag}: k_proj shape");
    assert_eq!(w.v_proj.len(), kv_dim * h, "{tag}: v_proj shape");
    assert_eq!(w.o_proj.len(), h * q_dim, "{tag}: o_proj shape");
    assert_eq!(w.ffn_gate.len(), mid * h, "{tag}: ffn_gate shape");
    assert_eq!(w.ffn_up.len(), mid * h, "{tag}: ffn_up shape");
    assert_eq!(w.ffn_down.len(), h * mid, "{tag}: ffn_down shape");
    assert_eq!(w.attn_norm.len(), h, "{tag}: attn_norm shape");
    assert_eq!(w.ffn_norm.len(), h, "{tag}: ffn_norm shape");
    assert_eq!(w.q_bias.len(), q_dim, "{tag}: q_bias shape");
    assert_eq!(w.k_bias.len(), kv_dim, "{tag}: k_bias shape");
    assert_eq!(w.v_bias.len(), kv_dim, "{tag}: v_bias shape");
}
