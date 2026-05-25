//! 2-layer megakernel parity test (2026-05-25, day-3+).
//!
//! Day-3 (this commit) exercises the full dispatch harness — weight
//! upload via `gpu_address`, argbuf encode, `useResource`, dispatch,
//! readback — with the shader body still a pass-through (stages A..L
//! TODO). The correctness gate this lays down is the **pass-through
//! invariant** `x_out == x_in`, NOT a full CPU-vs-megakernel
//! comparison. That gate proves the harness moves bytes around
//! correctly; once stages A..L land, the comparison flips to
//! `atol=1e-3 fp16` vs the residual returned by
//! [`QwenDense::forward_layers_subset`] (already wired below).
//!
//! Infrastructure built in day-2:
//!   * [`QwenDense::forward_layers_subset`] — runs the first N layers
//!     of the existing CPU forward path and returns the residual
//!     stream after layer N (before final_norm + LM head).
//!   * [`QwenDense::prep_megakernel_layer_f16`] — pre-dequantizes one
//!     layer's Q4_K / Q6_K weights into f16 + f32 buffers in the
//!     layout the megakernel shader expects.
//!
//! See `memory/build_megakernel_day3_2026_05_25.md` for the day-3
//! closeout and stage-A entry point.

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use dismantle_core::kernels::megakernel::megakernel_2layer_dispatch;
use dismantle_core::metal::MetalContext;
use dismantle_core::model::qwen_dense::{MegakernelLayerWeightsF16, QwenDense};
use dismantle_core::{Engine, EngineConfig};
use half::f16;

const TOKEN: u32 = 42;
const POS: usize = 0;
const LAST_LAYER: usize = 1;
const MAX_SEQ: u32 = 256;

fn weights_path() -> PathBuf {
    if let Ok(p) = std::env::var("DISMANTLE_QWEN_GGUF") {
        return PathBuf::from(p);
    }
    PathBuf::from("../../models/qwen2.5-3b-instruct-q4_k_m.gguf")
}

#[test]
#[ignore = "megakernel POC day-3: harness-only (pass-through invariant); requires Qwen-3B weights via DISMANTLE_QWEN_GGUF"]
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

    // Reference: first 2 layers of the existing CPU forward path.
    // Day-3 doesn't compare against ref_x (shader is pass-through),
    // but the call still validates the CPU-side helper is healthy and
    // produces the residual that stage-A onwards will be measured
    // against in later days.
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

    let ctx = MetalContext::new().expect("MetalContext::new");
    let x_out = megakernel_2layer_dispatch(
        &ctx,
        &layer0,
        &layer1,
        &x_in,
        POS as u32,
        (POS + 1) as u32,
        MAX_SEQ,
    )
    .expect("megakernel_2layer_dispatch");

    assert_eq!(x_out.len(), h, "megakernel output residual length");

    // Day-3 invariant: shader body is still pass-through, so x_out
    // must equal x_in bit-identically. When stages A..L land, this
    // becomes a CPU-ref atol=1e-3 comparison.
    for (i, (got, want)) in x_out.iter().zip(x_in.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "pass-through mismatch at i={i}: got {} want {}",
            got.to_f32(),
            want.to_f32(),
        );
    }
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
    // Qwen2 carries Q/K/V biases; assert non-empty.
    assert_eq!(w.q_bias.len(), q_dim, "{tag}: q_bias shape");
    assert_eq!(w.k_bias.len(), kv_dim, "{tag}: k_bias shape");
    assert_eq!(w.v_bias.len(), kv_dim, "{tag}: v_bias shape");
}
