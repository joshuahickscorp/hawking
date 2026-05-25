//! 2-layer megakernel parity test (2026-05-25 → 2026-05-26+).
//!
//! Grows stage-by-stage as the megakernel shader gains real compute.
//! Each landed stage extends the assertion forward via the shader's
//! `probe_stage` selector: the shader executes the full stage prefix
//! and copies the chosen intermediate to `x_out`; the test computes the
//! same intermediate in f32 from a synthetic residual and compares.
//!
//! When stage L of layer 1 lands, `probe_stage` collapses to
//! [`MK_PROBE_RESIDUAL`] and the dev-only escape hatch retires.
//!
//! Day-3 entry: stage A (layer-0 pre-attention rmsnorm), `MK_PROBE_XNORM_A`.
//! Day-4 entry: stages B/C/D (Q/K/V + biases + RoPE), `MK_PROBE_Q_ROT`.

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use dismantle_core::kernels::megakernel::{
    megakernel_2layer_dispatch, MK_PROBE_ATTN_OUT, MK_PROBE_Q_ROT, MK_PROBE_XNORM_A,
};
use dismantle_core::metal::MetalContext;
use dismantle_core::model::qwen_dense::{MegakernelLayerWeightsF16, QwenDense};
use dismantle_core::{Engine, EngineConfig};
use half::f16;

const TOKEN: u32 = 42;
const POS: usize = 0;
const LAST_LAYER: usize = 1;
const MAX_SEQ: u32 = 256;

// Qwen-3B shape constants (mirror shader header).
const HIDDEN: usize = 2048;
const N_HEADS: usize = 16;
const N_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 128;
const Q_DIM: usize = N_HEADS * HEAD_DIM; // 2048
const KV_DIM: usize = N_KV_HEADS * HEAD_DIM; // 256
const RMS_EPS: f32 = 1e-6;
const ROPE_THETA: f32 = 1_000_000.0;

const ATOL: f32 = 1e-3;
/// Relative tolerance for fp16 stores. The shader stores intermediates
/// (Q, K, V, attn_out, residual, …) as f16; the CPU reference is f32.
/// f16 carries ~10 mantissa bits → ~1e-3 RELATIVE precision, so for a
/// value of magnitude ~10 the absolute fp16 store noise is ~1e-2. The
/// effective gate is `|diff| ≤ ATOL + RTOL * |want|`, mirroring numpy's
/// `assert_allclose`. CLAUDE.md § "Verification rule" specifies atol=1e-3
/// fp16 for kernel parity with O(1) inputs; here the synthetic input
/// drives activations into O(10) range so the relative term takes over.
const RTOL: f32 = 2e-3;

fn weights_path() -> PathBuf {
    if let Ok(p) = std::env::var("DISMANTLE_QWEN_GGUF") {
        return PathBuf::from(p);
    }
    PathBuf::from("../../models/qwen2.5-3b-instruct-q4_k_m.gguf")
}

#[test]
#[ignore = "megakernel POC: requires Qwen-3B weights via DISMANTLE_QWEN_GGUF"]
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

    // Reference: existing CPU forward path (sanity-checks the model
    // loaded; not used directly in stage-by-stage probes).
    let ref_x = model
        .forward_layers_subset(TOKEN, POS, LAST_LAYER)
        .expect("forward_layers_subset");
    assert_eq!(ref_x.len(), h, "ref residual has wrong length");
    assert!(
        ref_x.iter().all(|v: &f32| v.is_finite()),
        "ref residual contains NaN/Inf"
    );

    // Weight prep — pre-dequantize layer 0 + layer 1 to f16.
    let layer0 = model
        .prep_megakernel_layer_f16(0)
        .expect("prep_megakernel_layer_f16(0)");
    let layer1 = model
        .prep_megakernel_layer_f16(1)
        .expect("prep_megakernel_layer_f16(1)");
    assert_layer_shapes(&layer0, h, q_dim, kv_dim, mid, "layer 0");
    assert_layer_shapes(&layer1, h, q_dim, kv_dim, mid, "layer 1");

    // Synthetic input residual — deterministic, distinct per element so
    // any harness bug (off-by-one stride, wrong gpu_address indexing,
    // etc.) surfaces in the readback.
    let x_in: Vec<f16> = (0..h)
        .map(|i| f16::from_f32((i as f32) * 0.001 - 1.0))
        .collect();
    let x_in_f32: Vec<f32> = x_in.iter().map(|v| v.to_f32()).collect();

    let ctx = MetalContext::new().expect("MetalContext::new");

    // ── Stage A: layer-0 pre-attention rmsnorm ──────────────────────────
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
        eprintln!(
            "stage-A rmsnorm parity OK (worst violation {worst:.3e} ≤ 0, atol={ATOL:.0e} rtol={RTOL:.0e})"
        );
    }

    // ── Stages B/C/D: Q/K/V GEMV + biases + RoPE on Q (probe = post-RoPE Q)
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
        // Shader emits Q_DIM = HIDDEN = 2048 f16 values into x_out.

        // CPU reference for post-RoPE Q:
        //   1. x_norm = rmsnorm(x_in, layer0.attn_norm)
        //   2. q = qw @ x_norm  (f16 weight × f32 activation, f32 acc)
        //   3. q += q_bias
        //   4. rope_inplace per head on q
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
        eprintln!(
            "stage-D Q (post-RoPE) parity OK (worst violation {worst:.3e} ≤ 0, atol={ATOL:.0e} rtol={RTOL:.0e})"
        );
    }

    // ── Stages E/F: KV write + MHA decode (probe = attn_out) ────────────
    //
    // At pos=0/seq_len=1 the softmax is degenerate (single position →
    // weight 1.0) so attn_out reduces to V replicated across grouped
    // heads. This still exercises:
    //   * KV write to DRAM at the correct (layer, slot, kv_head, dim) offset
    //   * the per-head loop and kv_h = h / group_size indexing
    //   * the (now-trivial) softmax max-reduce + sum-reduce paths
    //   * the V-weighted sum readback structure
    //
    // Non-trivial seq_len exercises (multi-position softmax) are queued
    // for a follow-up that exposes a persistent kv_cache buffer across
    // dispatches.
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
        eprintln!(
            "stage-F attn_out (pos=0, seq_len=1) parity OK (worst violation {worst:.3e} ≤ 0, atol={ATOL:.0e} rtol={RTOL:.0e})"
        );
    }
}

/// CPU reference for layer-0 stage-F attn_out at pos=0/seq_len=1.
/// Computes V (with bias, no rope) and replicates across grouped heads.
fn cpu_layer0_attn_out_pos0(
    x_in_f32: &[f32],
    layer0: &MegakernelLayerWeightsF16,
) -> Vec<f32> {
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

// ── CPU reference helpers ───────────────────────────────────────────────

/// Standard rmsnorm in f32: out[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps).
fn cpu_rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut ssq = 0.0f32;
    for &v in x {
        ssq += v * v;
    }
    let rnorm = 1.0f32 / (ssq / (n as f32) + eps).sqrt();
    (0..n).map(|i| x[i] * rnorm * weight[i]).collect()
}

/// Row-major f16-weight GEMV with f32 accumulation:
///   out[r] = Σ_c W[r, c] * x[c]
/// W is row-major (rows × cols). Mirrors the shader's per-row f32 acc.
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

/// In-place RoPE on one head_dim-vector at position `pos`. Interleaved
/// pair convention: rotate (x[2i], x[2i+1]) with θ = pos / base^(2i/dim).
/// Mirrors `crates/dismantle-core/src/kernels/mod.rs:rope_inplace`.
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

/// Returns the worst (atol-relative-violation, index) pair, where each
/// element's allowance is `ATOL + RTOL * |want|`. Caller asserts the
/// returned violation ≤ 0.
fn max_violation_f16_vs_f32(got: &[f16], want: &[f32]) -> (f32, usize, f32, f32) {
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
        let allowed = ATOL + RTOL * w.abs();
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
