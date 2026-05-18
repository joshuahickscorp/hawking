//! Path-to-90 — `Eagle4Head` skeleton for the trained EAGLE-4 draft head.
//!
//! EAGLE-4 is the routing-aware speculative-decoding head trained in
//! `eagle4`. Per its `bench_results.md` it ships two
//! checkpoints from one training run:
//!
//! - `best.npz` (v2-spec): **87.48% target-argmax acceptance** on V2-Lite
//!   held-out — the spec-decode-driver metric.
//! - `best_routing.npz` (v2-routing): 84.16% accept + 26% mask top-8
//!   recall — for prefetch-heavy runtimes.
//!
//! Both have the same architecture: 5-input fusion (prev_token embedding
//! + h_low + h_mid + h_high + h_shared) → in_proj → single transformer
//! block (RMSNorm/MHA/SwiGLU) → residual gate against post_norm(h_high)
//! → frozen V2-Lite LM head → token logits, plus two small heads for
//! 26×64 routing-mask logits and a P(accept) calibration scalar.
//!
//! ## Current state
//!
//! - `Eagle4Config` constants + `Eagle4Weights` struct describe the
//!   parameter layout the NPZ loader populates.
//! - `Eagle4Head::new_uninitialized()` produces an all-zero structural
//!   placeholder; `propose()` still errors via Unimplemented on it.
//! - `Eagle4Head::from_npz(path)` reads an `eagle4.py`-produced
//!   checkpoint and validates every required tensor's shape against
//!   the V2-Lite-specific constants. Forward pass still unimplemented.
//! - `DraftHead::propose` returns `Err(Unimplemented)`; the forward
//!   pass lands separately (see convergence doc § "Required dismantle
//!   changes" #4).
//!
//! See `reports/path_to_90/eagle4_convergence.md` for the integration
//! contract and order-of-work.

use crate::speculate::draft_head::{DraftHead, DraftInputs, DraftOutputs};
use crate::util::npz::{read_npz, NpyArray};
use crate::{Error, Result};
use std::collections::HashMap;
use std::path::Path;

/// V2-Lite-specific constants the head depends on. Mirrors
/// `eagle4.py:38-45` exactly.
pub mod cfg {
    pub const HIDDEN_DIM: usize = 2048;
    pub const VOCAB: usize = 102_400;
    pub const N_MOE_LAYERS: usize = 26;
    pub const N_ROUTED: usize = 64;
    pub const TOP_K_ROUTED: usize = 6;
    pub const N_HEADS: usize = 16;
    pub const HEAD_DIM: usize = HIDDEN_DIM / N_HEADS; // 128
    pub const INTERMEDIATE: usize = 5_632;
    pub const MASK_HIDDEN: usize = 512;
    pub const RMS_EPS: f32 = 1e-6;
    /// The head sees this many hidden vectors per token, in order:
    /// `[h_low, h_mid, h_high, h_shared]`.
    pub const N_HIDDENS: usize = 4;
    /// V2-Lite decoder layers the captures come from (0-indexed). Matches
    /// `eagle4/capture.py::FUSION_LAYERS = (2, 13, 25)`. The dismantle
    /// brief originally listed {2,14,24} — that was wrong; correct is
    /// {2,13,25}.
    pub const FUSION_LAYERS: [usize; 3] = [2, 13, 25];
    /// The MoE layer whose shared-expert output is captured as `h_shared`.
    /// V2-Lite has 27 layers (0..26); layer 0 is dense, layers 1..26 are
    /// MoE, so `h_shared` comes from layer 26.
    pub const SHARED_EXPERT_LAYER: usize = 26;
}

/// Flat-tensor view of the EAGLE-4 head's trainable parameters. The NPZ
/// loader populates these from the checkpoint file's key/value pairs.
///
/// Naming convention follows `eagle4.py::_flat_params` (dot-joined dict
/// walk over `head.trainable_parameters()`):
///
/// ```text
/// in_proj.weight              (HIDDEN, 5*HIDDEN)
/// block.attn_norm             (HIDDEN,)
/// block.attn.query_proj.weight  (HIDDEN, HIDDEN)
/// block.attn.key_proj.weight    (HIDDEN, HIDDEN)
/// block.attn.value_proj.weight  (HIDDEN, HIDDEN)
/// block.attn.out_proj.weight    (HIDDEN, HIDDEN)
/// block.mlp_norm              (HIDDEN,)
/// block.mlp.gate.weight       (INTERMEDIATE, HIDDEN)
/// block.mlp.up.weight         (INTERMEDIATE, HIDDEN)
/// block.mlp.down.weight       (HIDDEN, INTERMEDIATE)
/// residual_gate               (1,)
/// mask_proj_in.weight         (MASK_HIDDEN, HIDDEN)
/// mask_proj_out.weight        (N_MOE_LAYERS*N_ROUTED, MASK_HIDDEN)
/// calib_proj.weight           (1, HIDDEN)
/// calib_proj.bias             (1,)
/// ```
///
/// All weights stored row-major fp32. Dismantle's existing `gemv_f32` /
/// matmul kernels consume row-major fp32, so no in-loader transposition
/// is needed beyond the eagle4-side transpose-on-save in `extract_frozen`
/// (which writes lm_head transposed for direct `hidden @ lm_head` use).
pub struct Eagle4Weights {
    pub in_proj: Vec<f32>,            // (HIDDEN, 5*HIDDEN)
    pub block_attn_norm: Vec<f32>,    // (HIDDEN,)
    pub block_attn_q: Vec<f32>,       // (HIDDEN, HIDDEN)
    pub block_attn_k: Vec<f32>,       // (HIDDEN, HIDDEN)
    pub block_attn_v: Vec<f32>,       // (HIDDEN, HIDDEN)
    pub block_attn_o: Vec<f32>,       // (HIDDEN, HIDDEN)
    pub block_mlp_norm: Vec<f32>,     // (HIDDEN,)
    pub block_mlp_gate: Vec<f32>,     // (INTERMEDIATE, HIDDEN)
    pub block_mlp_up: Vec<f32>,       // (INTERMEDIATE, HIDDEN)
    pub block_mlp_down: Vec<f32>,     // (HIDDEN, INTERMEDIATE)
    pub residual_gate: f32,           // scalar
    pub mask_proj_in: Vec<f32>,       // (MASK_HIDDEN, HIDDEN)
    pub mask_proj_out: Vec<f32>,      // (N_MOE_LAYERS*N_ROUTED, MASK_HIDDEN)
    pub calib_proj_w: Vec<f32>,       // (1, HIDDEN)
    pub calib_proj_b: f32,            // scalar
}

impl Eagle4Weights {
    /// All zeros, residual_gate = 0.0. With this state the forward pass
    /// would produce `draft_hidden = post_norm(h_high) + 0.0 * <garbage>`
    /// which equals the V2-Lite identity baseline. Useful as a structural
    /// placeholder before the NPZ loader fills real weights.
    pub fn zeros() -> Self {
        let h = cfg::HIDDEN_DIM;
        let i = cfg::INTERMEDIATE;
        let m = cfg::MASK_HIDDEN;
        Self {
            in_proj: vec![0.0; h * (5 * h)],
            block_attn_norm: vec![0.0; h],
            block_attn_q: vec![0.0; h * h],
            block_attn_k: vec![0.0; h * h],
            block_attn_v: vec![0.0; h * h],
            block_attn_o: vec![0.0; h * h],
            block_mlp_norm: vec![0.0; h],
            block_mlp_gate: vec![0.0; i * h],
            block_mlp_up: vec![0.0; i * h],
            block_mlp_down: vec![0.0; h * i],
            residual_gate: 0.0,
            mask_proj_in: vec![0.0; m * h],
            mask_proj_out: vec![0.0; cfg::N_MOE_LAYERS * cfg::N_ROUTED * m],
            calib_proj_w: vec![0.0; h],
            calib_proj_b: 0.0,
        }
    }
}

/// Frozen-target parameters the head reads but does NOT train. In
/// integration, these are already loaded as part of dismantle's V2-Lite
/// GGUF — no need to load `v2lite_frozen.npz` separately at runtime.
/// The loader stub exists to validate the same model is in use.
pub struct Eagle4FrozenRefs<'a> {
    pub token_embd: &'a [f32], // (HIDDEN, VOCAB) transposed-on-save
    pub lm_head: &'a [f32],    // (HIDDEN, VOCAB) transposed-on-save
    pub output_norm: &'a [f32], // (HIDDEN,)
}

/// EAGLE-4 trained draft head.
pub struct Eagle4Head {
    weights: Eagle4Weights,
    checkpoint_id: String,
}

impl Eagle4Head {
    /// Structural placeholder with all-zero weights. `propose()` will
    /// still return `Err(Unimplemented)` because the forward pass itself
    /// is not implemented yet.
    pub fn new_uninitialized() -> Self {
        Self {
            weights: Eagle4Weights::zeros(),
            checkpoint_id: "eagle4-uninitialized".to_string(),
        }
    }

    /// Load from an NPZ file produced by `eagle4.py train`
    /// (`eagle4/eagle4.py:136 → np.savez(path, **flat)`).
    ///
    /// Key naming follows `eagle4.py::_flat_params`; see
    /// `Eagle4Weights`'s docstring. All weights are fp32 little-endian
    /// (uncompressed ZIP). `__step__` is parsed when present and folded
    /// into `checkpoint_id` so debugging across multiple checkpoints
    /// stays unambiguous.
    ///
    /// Returns `Error::Model` on shape/dtype/key mismatch; the head is
    /// only constructed when every required tensor is present at the
    /// expected shape.
    pub fn from_npz<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        let mut entries = read_npz(path_ref)?;

        let h = cfg::HIDDEN_DIM;
        let inter = cfg::INTERMEDIATE;
        let mhid = cfg::MASK_HIDDEN;
        let mask_out = cfg::N_MOE_LAYERS * cfg::N_ROUTED;

        let weights = Eagle4Weights {
            in_proj: take_f32(&mut entries, "in_proj.weight", &[h, 5 * h])?,
            block_attn_norm: take_f32(&mut entries, "block.attn_norm", &[h])?,
            block_attn_q: take_f32(&mut entries, "block.attn.query_proj.weight", &[h, h])?,
            block_attn_k: take_f32(&mut entries, "block.attn.key_proj.weight", &[h, h])?,
            block_attn_v: take_f32(&mut entries, "block.attn.value_proj.weight", &[h, h])?,
            block_attn_o: take_f32(&mut entries, "block.attn.out_proj.weight", &[h, h])?,
            block_mlp_norm: take_f32(&mut entries, "block.mlp_norm", &[h])?,
            block_mlp_gate: take_f32(&mut entries, "block.mlp.gate.weight", &[inter, h])?,
            block_mlp_up: take_f32(&mut entries, "block.mlp.up.weight", &[inter, h])?,
            block_mlp_down: take_f32(&mut entries, "block.mlp.down.weight", &[h, inter])?,
            residual_gate: take_f32_scalar(&mut entries, "residual_gate")?,
            mask_proj_in: take_f32(&mut entries, "mask_proj_in.weight", &[mhid, h])?,
            mask_proj_out: take_f32(&mut entries, "mask_proj_out.weight", &[mask_out, mhid])?,
            calib_proj_w: take_f32(&mut entries, "calib_proj.weight", &[1, h])?,
            calib_proj_b: take_f32_scalar(&mut entries, "calib_proj.bias")?,
        };

        // `__step__` is informational; absent on older checkpoints, so
        // missing-key is not an error.
        let step = entries
            .remove("__step__")
            .and_then(|a| a.as_i32_scalar().ok());

        // eagle4.py's _flat_params walker only emits the 14 trainable
        // tensors plus the scalar gate plus optional __step__; anything
        // else means the checkpoint format diverged.
        if !entries.is_empty() {
            let mut leftover: Vec<_> = entries.keys().cloned().collect();
            leftover.sort();
            return Err(Error::Model(format!(
                "Eagle4Head::from_npz: unexpected extra keys in {}: {:?}",
                path_ref.display(),
                leftover
            )));
        }

        let checkpoint_id = match step {
            Some(s) => format!("eagle4:{} (step={})", path_ref.display(), s),
            None => format!("eagle4:{}", path_ref.display()),
        };
        Ok(Self {
            weights,
            checkpoint_id,
        })
    }

    /// Read-only access to the loaded weights (useful for parity tests
    /// against eagle4's Python forward).
    pub fn weights(&self) -> &Eagle4Weights {
        &self.weights
    }
}

fn take_f32(
    entries: &mut HashMap<String, NpyArray>,
    key: &str,
    expected_shape: &[usize],
) -> Result<Vec<f32>> {
    let arr = entries
        .remove(key)
        .ok_or_else(|| Error::Model(format!("Eagle4Head::from_npz: missing key '{}'", key)))?;
    if arr.shape != expected_shape {
        return Err(Error::Model(format!(
            "Eagle4Head::from_npz: '{}' shape {:?}, expected {:?}",
            key, arr.shape, expected_shape
        )));
    }
    arr.as_f32()
}

fn take_f32_scalar(entries: &mut HashMap<String, NpyArray>, key: &str) -> Result<f32> {
    let arr = entries
        .remove(key)
        .ok_or_else(|| Error::Model(format!("Eagle4Head::from_npz: missing key '{}'", key)))?;
    arr.as_f32_scalar()
}

impl DraftHead for Eagle4Head {
    fn propose(&mut self, inputs: &DraftInputs<'_>, _k: usize) -> Result<DraftOutputs> {
        // Shape validation that WILL happen in the real impl. Useful here
        // to anchor the contract.
        if inputs.hiddens.len() != cfg::N_HIDDENS {
            return Err(Error::Model(format!(
                "Eagle4Head expects {} hiddens (h_low, h_mid, h_high, h_shared), got {}",
                cfg::N_HIDDENS,
                inputs.hiddens.len()
            )));
        }
        for (i, h) in inputs.hiddens.iter().enumerate() {
            if h.len() != cfg::HIDDEN_DIM {
                return Err(Error::Model(format!(
                    "Eagle4Head: hiddens[{}].len() = {}, expected {}",
                    i,
                    h.len(),
                    cfg::HIDDEN_DIM
                )));
            }
        }
        Err(Error::Unimplemented(
            "Eagle4Head::propose — forward pass not implemented yet; \
             see reports/path_to_90/eagle4_convergence.md § \
             'Required dismantle changes' #4",
        ))
    }

    fn hidden_dim(&self) -> usize {
        cfg::HIDDEN_DIM
    }

    fn n_hiddens(&self) -> usize {
        cfg::N_HIDDENS
    }

    fn id(&self) -> &str {
        &self.checkpoint_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uninitialized_head_has_right_shape() {
        let head = Eagle4Head::new_uninitialized();
        assert_eq!(head.hidden_dim(), cfg::HIDDEN_DIM);
        assert_eq!(head.n_hiddens(), 4);
        assert_eq!(head.id(), "eagle4-uninitialized");
        let w = head.weights();
        assert_eq!(w.in_proj.len(), cfg::HIDDEN_DIM * 5 * cfg::HIDDEN_DIM);
        assert_eq!(w.block_attn_q.len(), cfg::HIDDEN_DIM * cfg::HIDDEN_DIM);
        assert_eq!(w.block_mlp_gate.len(), cfg::INTERMEDIATE * cfg::HIDDEN_DIM);
        assert_eq!(
            w.mask_proj_out.len(),
            cfg::N_MOE_LAYERS * cfg::N_ROUTED * cfg::MASK_HIDDEN
        );
        assert_eq!(w.residual_gate, 0.0);
    }

    #[test]
    fn from_npz_missing_file_returns_io_error() {
        // Real impl: nonexistent path surfaces through std::io, not
        // Unimplemented (which was the skeleton's stub return).
        let result = Eagle4Head::from_npz("nonexistent.npz");
        assert!(matches!(result, Err(Error::Io(_))));
    }

    #[test]
    fn propose_validates_hidden_count() {
        let mut head = Eagle4Head::new_uninitialized();
        let hidden = vec![0.0f32; cfg::HIDDEN_DIM];
        let single: [&[f32]; 1] = [&hidden];
        let inputs = DraftInputs {
            prev_token: 0,
            hiddens: &single,
        };
        let result = head.propose(&inputs, 4);
        assert!(matches!(result, Err(Error::Model(_))));
    }

    #[test]
    fn propose_validates_hidden_dim() {
        let mut head = Eagle4Head::new_uninitialized();
        let too_small = vec![0.0f32; 64];
        let four: [&[f32]; 4] = [&too_small, &too_small, &too_small, &too_small];
        let inputs = DraftInputs {
            prev_token: 0,
            hiddens: &four,
        };
        let result = head.propose(&inputs, 4);
        assert!(matches!(result, Err(Error::Model(_))));
    }

    #[test]
    fn propose_with_valid_shapes_returns_unimplemented() {
        // Once shapes pass validation, the real forward kicks in. Until
        // it's implemented, that returns Unimplemented (not Model).
        let mut head = Eagle4Head::new_uninitialized();
        let h = vec![0.0f32; cfg::HIDDEN_DIM];
        let four: [&[f32]; 4] = [&h, &h, &h, &h];
        let inputs = DraftInputs {
            prev_token: 0,
            hiddens: &four,
        };
        let result = head.propose(&inputs, 4);
        assert!(matches!(result, Err(Error::Unimplemented(_))));
    }

    #[test]
    fn fusion_layer_constants_match_eagle4() {
        // Sanity guard against the brief's old {2,14,24} sneaking back in.
        assert_eq!(cfg::FUSION_LAYERS, [2, 13, 25]);
        assert_eq!(cfg::SHARED_EXPERT_LAYER, 26);
    }
}
