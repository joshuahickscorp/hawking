//! 2-layer megakernel parity test — POC scaffold (2026-05-25,
//! build/megakernel-day2 builds on 4591133).
//!
//! GATED OFF by default (`#[ignore]`'d) because the megakernel
//! dispatcher itself is still a stub
//! (`crate::kernels::megakernel::megakernel_2layer_dispatch` returns
//! `Err`). The reference-side infrastructure landed in day 2:
//!
//!   * [`QwenDense::forward_layers_subset`] — runs the first N layers
//!     of the existing CPU forward path and returns the residual
//!     stream after layer N (before final_norm + LM head).
//!   * [`QwenDense::prep_megakernel_layer_f16`] — pre-dequantizes one
//!     layer's Q4_K / Q6_K weights into f16 + f32 buffers in the
//!     layout the megakernel shader expects.
//!
//! When the dispatcher becomes functional (day 3+ of the megakernel
//! POC), drop the `#[ignore]`, plug `megakernel_2layer_dispatch` into
//! the marked TODO below, and compare its output residual to
//! `ref_x` with the `atol=1e-3` fp16 tolerance from
//! `~/.claude/projects/-Users-scammermike-Downloads-dismantle/memory/build_megakernel_design_2026_05_25.md`
//! § "Verification rule" (relaxed from bit-identical because the
//! megakernel runs GEMVs in f16 with f32 accumulators, so fp16 noise
//! at the residual is expected).

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use dismantle_core::model::qwen_dense::{MegakernelLayerWeightsF16, QwenDense};
use dismantle_core::{Engine, EngineConfig};

const TOKEN: u32 = 42;
const POS: usize = 0;
const LAST_LAYER: usize = 1;

fn weights_path() -> PathBuf {
    if let Ok(p) = std::env::var("DISMANTLE_QWEN_GGUF") {
        return PathBuf::from(p);
    }
    PathBuf::from("../../models/qwen2.5-3b-instruct-q4_k_m.gguf")
}

#[test]
#[ignore = "megakernel POC: dispatcher is a stub; see build_megakernel_design_2026_05_25.md § What attended work unblocks"]
fn megakernel_2layer_parity_qwen3b() {
    let weights = weights_path();
    if !weights.exists() {
        eprintln!("SKIP: model not at {}", weights.display());
        return;
    }

    let cfg = EngineConfig::default();
    let mut model = QwenDense::load(&weights, cfg).expect("load QwenDense");
    let h = model.config.hidden;
    let q_dim = model.config.n_heads * model.config.head_dim;
    let kv_dim = model.config.n_kv_heads * model.config.head_dim;
    let mid = model.config.intermediate;

    // Reference: first 2 layers of the existing CPU forward path.
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

    // TODO(megakernel-day3): when megakernel_2layer_dispatch is
    // functional, allocate Metal buffers for layer0/layer1 + KV cache
    // + ffn_scratch, dispatch the megakernel, read back the residual,
    // and compare to ref_x with atol=1e-3.
    //
    // The dispatcher currently lives at
    //   crates/dismantle-core/src/kernels/megakernel.rs
    // and is `pub(crate)`. Day 3 either (a) bumps it to `pub` so this
    // integration test can call it directly, or (b) exposes a thin
    // public adapter on QwenDense (e.g.
    // `forward_2layer_megakernel(token, pos) -> Vec<f32>`) that
    // wraps the dispatcher's argbuf assembly.
    let _ = (ref_x, layer0, layer1);
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
