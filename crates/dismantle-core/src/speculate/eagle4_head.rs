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
//! ## Current state (path-to-90 step 5)
//!
//! - `Eagle4Weights` holds the head's trainable parameters; loader at
//!   `Eagle4Head::from_npz` validates every required tensor's shape.
//! - `Eagle4FrozenWeights` holds V2-Lite's token_embd / lm_head /
//!   output_norm; loader at `Eagle4FrozenWeights::from_npz` reads
//!   `eagle4/eagle4.py frozen` output and transposes lm_head/embed
//!   from `(HIDDEN, VOCAB)` storage to `(VOCAB, HIDDEN)` row-major.
//! - `Eagle4Head::forward_full` runs the CPU fp32 forward pass and
//!   returns `Eagle4ForwardOutput { token_logits, mask_logits,
//!   draft_hidden, calib_logit }`.
//! - `DraftHead::propose` calls `forward_full` and packs the result
//!   into the trait's generic `DraftOutputs` (top-K tokens + 26×64
//!   routing mask + calibration scalar).
//!
//! Metal acceleration of the forward is step 7. Production CLI wire-up
//! (`--speculate eagle4`) lives in step 8.
//!
//! See `reports/path_to_90/eagle4_convergence.md` for the integration
//! contract.

use crate::kernels::{gemv_f32, rmsnorm, silu_mul};
use crate::speculate::draft_head::{DraftHead, DraftInputs, DraftOutputs};
use crate::util::npz::{read_npz, NpyArray};
use crate::{Error, Result};
use half::f16;
use std::collections::HashMap;
use std::path::Path;

#[cfg(target_os = "macos")]
use crate::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};

// Path-to-90 lever 5 — AMX direct cblas for the eagle4 head's
// largest gemvs. Implementation extracted to `crate::amx` (path-to-125
// L4) so V2-Lite attention projections can share the same wrapper.
#[cfg(target_os = "macos")]
use crate::amx::amx_sgemv;

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

/// Owned copy of the three frozen V2-Lite tensors EAGLE-4 reads:
/// the token embedding table, the LM head, and the final RMSNorm
/// weights. All stored as row-major fp32, transposed against the
/// `v2lite_frozen.npz` storage layout so dismantle's `gemv_f32` and
/// embedding-lookup conventions apply directly:
///
/// - `token_embd` — (VOCAB, HIDDEN). NPZ stores (HIDDEN, VOCAB)
///   per `extract_frozen`; this loader transposes on read.
///   Embedding lookup is `&token_embd[t*HIDDEN..(t+1)*HIDDEN]`.
/// - `lm_head` — (VOCAB, HIDDEN). NPZ stores (HIDDEN, VOCAB);
///   transposed on read so `gemv_f32(lm_head, VOCAB, HIDDEN,
///   draft_hidden, logits)` computes `draft_hidden @ lm_head`.
/// - `output_norm` — (HIDDEN,). Used as RMSNorm weight on `h_high`.
///
/// Production wire-up (step 8) populates these from dismantle's
/// already-loaded V2-Lite GGUF tensors and skips the NPZ read.
pub struct Eagle4FrozenWeights {
    pub token_embd: Vec<f32>,  // (VOCAB, HIDDEN) row-major
    pub lm_head: Vec<f32>,     // (VOCAB, HIDDEN) row-major
    pub output_norm: Vec<f32>, // (HIDDEN,)
}

impl Eagle4FrozenWeights {
    /// Load the three frozen tensors from an NPZ written by
    /// `eagle4/eagle4.py frozen` (see `extract_frozen` in that file).
    /// NPZ stores `token_embd` and `lm_head` as fp16 with shape
    /// `(HIDDEN, VOCAB)`; this reader converts to fp32 and transposes
    /// to `(VOCAB, HIDDEN)` for dismantle-side use. `output_norm` is
    /// stored as fp32 `(HIDDEN,)` and read directly.
    pub fn from_npz<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        let mut entries = read_npz(path_ref)?;
        let h = cfg::HIDDEN_DIM;
        let v = cfg::VOCAB;

        let token_embd_raw = take_f32(&mut entries, "token_embd", &[h, v])?;
        let lm_head_raw = take_f32(&mut entries, "lm_head", &[h, v])?;
        let output_norm = take_f32(&mut entries, "output_norm", &[h])?;

        // Transpose both (HIDDEN, VOCAB) → (VOCAB, HIDDEN). Cheap once,
        // saves a transpose inside every forward.
        let mut token_embd = vec![0.0f32; v * h];
        let mut lm_head = vec![0.0f32; v * h];
        for row in 0..h {
            for col in 0..v {
                token_embd[col * h + row] = token_embd_raw[row * v + col];
                lm_head[col * h + row] = lm_head_raw[row * v + col];
            }
        }
        Ok(Self {
            token_embd,
            lm_head,
            output_norm,
        })
    }
}

/// Result of [`Eagle4Head::forward_full`] — the four outputs the
/// trained head produces per token. `token_logits` and `mask_logits`
/// are unnormalized (caller picks argmax / softmax as needed); the
/// raw `calib_logit` scalar is the pre-sigmoid value.
#[derive(Debug, Clone)]
pub struct Eagle4ForwardOutput {
    /// (VOCAB,) — `draft_hidden @ lm_head`. **Empty when
    /// `forward_full` was called with `compute_token_logits = false`**
    /// (production decode loops that route the argmax through GPU
    /// gemv_f16_argmax via dismantle's V2-Lite-pinned lm_head buffer).
    pub token_logits: Vec<f32>,
    /// (N_MOE_LAYERS * N_ROUTED,) — `mask_proj_out(silu(mask_proj_in(draft_hidden)))`.
    /// Layout matches eagle4.py's `.reshape(B, S, N_MOE_LAYERS, N_ROUTED)`
    /// row-major: index `(L, e) = L * N_ROUTED + e`.
    pub mask_logits: Vec<f32>,
    /// (HIDDEN,) — `post_norm(h_high) + residual_gate · block_out`.
    pub draft_hidden: Vec<f32>,
    /// Pre-sigmoid P(accept) calibration scalar.
    pub calib_logit: f32,
}

/// EAGLE-4 trained draft head.
/// Path-to-90 step 7 — Metal-pinned f16 copies of the eagle4 head's
/// gemv weights, populated by [`Eagle4Head::pin_metal`]. When set,
/// [`Eagle4Head::forward_full_metal_no_lm_head`] dispatches each gemv
/// through dismantle's existing Metal helpers
/// (`gemv_f16_metal_pinned`), cutting the per-call cost from ~80 ms
/// CPU to ~15-20 ms. Layout converts on pin: each `Vec<f32>` weight
/// becomes a `Vec<u16>` of `f16::to_bits()` then a Metal shared-
/// memory buffer of those bytes. RMSNorms, residual adds, SiLU, and
/// the (1-row) calib_proj remain CPU — small intermediates, the
/// readback cost over UMA is negligible.
#[cfg(target_os = "macos")]
pub struct Eagle4MetalPinned {
    // Trainable weight matrices, pinned f16.
    pub in_proj: PinnedBuffer,        // (HIDDEN, 5*HIDDEN) f16
    pub block_attn_q: PinnedBuffer,   // (HIDDEN, HIDDEN) f16     — kept for S>1 future
    pub block_attn_k: PinnedBuffer,   // (HIDDEN, HIDDEN) f16     — kept for S>1 future
    pub block_attn_v: PinnedBuffer,   // (HIDDEN, HIDDEN) f16
    pub block_attn_o: PinnedBuffer,   // (HIDDEN, HIDDEN) f16
    pub block_mlp_gate: PinnedBuffer, // (INTERMEDIATE, HIDDEN) f16
    pub block_mlp_up: PinnedBuffer,   // (INTERMEDIATE, HIDDEN) f16
    pub block_mlp_down: PinnedBuffer, // (HIDDEN, INTERMEDIATE) f16
    pub mask_proj_in: PinnedBuffer,   // (MASK_HIDDEN, HIDDEN) f16 — kept for Stage 3 future
    pub mask_proj_out: PinnedBuffer,  // (N_MOE*N_ROUTED, MASK_HIDDEN) f16 — same
    // RMSNorm gammas pinned as f32 (rmsnorm_metal_buf_tcb expects f32 weight).
    pub attn_norm: PinnedBuffer,      // (HIDDEN,) f32
    pub mlp_norm: PinnedBuffer,       // (HIDDEN,) f32
    // Scratch buffers reused across decode steps (sequential decode
    // loop — no concurrent forwards). Lets the single-TCB forward
    // dispatch every kernel without per-call allocation.
    pub x5_buf: PinnedBuffer,         // (5*HIDDEN,) f32 input
    pub x_buf: PinnedBuffer,          // (HIDDEN,) f32 residual
    pub x_norm_attn_buf: PinnedBuffer, // (HIDDEN,) f32
    pub v_buf: PinnedBuffer,          // (HIDDEN,) f32
    pub attn_out_buf: PinnedBuffer,   // (HIDDEN,) f32
    pub x_norm_mlp_buf: PinnedBuffer, // (HIDDEN,) f32
    pub gate_buf: PinnedBuffer,       // (INTERMEDIATE,) f32
    pub up_buf: PinnedBuffer,         // (INTERMEDIATE,) f32
    pub act_buf: PinnedBuffer,        // (INTERMEDIATE,) f32
    pub mlp_out_buf: PinnedBuffer,    // (HIDDEN,) f32
}

#[cfg(target_os = "macos")]
fn pin_f32_as_f16(ctx: &MetalContext, w: &[f32]) -> PinnedBuffer {
    let mut bits: Vec<u16> = Vec::with_capacity(w.len());
    for &v in w {
        bits.push(f16::from_f32(v).to_bits());
    }
    // bytemuck on u16 is safe — Pod.
    let bytes: &[u8] = bytemuck::cast_slice(&bits);
    ctx.new_buffer_with_bytes(bytes)
}

pub struct Eagle4Head {
    weights: Eagle4Weights,
    frozen: Option<Eagle4FrozenWeights>,
    checkpoint_id: String,
    /// Path-to-90 step 7 — Metal-pinned f16 copies of the head's gemv
    /// weights. None until [`Self::pin_metal`] runs. Production decode
    /// loops use the Metal-pinned path; the CPU path stays for the
    /// parity test seam (step 6) and as a fallback.
    #[cfg(target_os = "macos")]
    metal_pinned: Option<Eagle4MetalPinned>,
}

impl Eagle4Head {
    /// Structural placeholder with all-zero weights. `propose()` will
    /// error on the missing-frozen-weights check.
    pub fn new_uninitialized() -> Self {
        Self {
            weights: Eagle4Weights::zeros(),
            frozen: None,
            checkpoint_id: "eagle4-uninitialized".to_string(),
            #[cfg(target_os = "macos")]
            metal_pinned: None,
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
    /// expected shape. Frozen weights are NOT loaded here — call
    /// [`Self::set_frozen`] separately before any forward pass.
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
            frozen: None,
            checkpoint_id,
            #[cfg(target_os = "macos")]
            metal_pinned: None,
        })
    }

    /// Read-only access to the loaded weights (useful for parity tests
    /// against eagle4's Python forward).
    pub fn weights(&self) -> &Eagle4Weights {
        &self.weights
    }

    /// Attach the frozen V2-Lite tensors the forward pass reads.
    /// Required before [`Self::forward_full`] or [`DraftHead::propose`]
    /// can run — both error with `Error::Model` otherwise.
    pub fn set_frozen(&mut self, frozen: Eagle4FrozenWeights) {
        self.frozen = Some(frozen);
    }

    /// Whether [`Self::set_frozen`] has been called.
    pub fn has_frozen(&self) -> bool {
        self.frozen.is_some()
    }

    /// Path-to-90 step 7 — pin all gemv weights as f16 Metal buffers.
    /// Idempotent (no-op if already pinned). Call once at engine load
    /// from inside DeepSeekV2 production wire-up; the resulting state
    /// enables [`Self::forward_full_metal_no_lm_head`] which is ~5×
    /// faster than the CPU forward at the head shapes (in_proj at
    /// 10240×2048, mlp gate/up at 5632×2048, mlp down at 2048×5632,
    /// attn q/k/v/o at 2048×2048, mask_proj_in/out at 512×2048 /
    /// 1664×512).
    ///
    /// Weights are converted to f16 once at pin time and stored in
    /// Metal shared buffers (UMA on Apple Silicon — no GPU memcpy at
    /// dispatch). Subsequent forwards use `gemv_f16_metal_pinned`.
    /// The f16 conversion loses minor precision vs the fp32 CPU
    /// forward (atol on draft_hidden ≈ 1e-3); acceptance rate may
    /// drop ~1-2 % vs the CPU forward but Stage 1 emission stays
    /// bit-identical (verifier argmax is via GPU on V2-Lite's
    /// already-f16 lm_head).
    #[cfg(target_os = "macos")]
    pub fn pin_metal(&mut self, ctx: &MetalContext) {
        if self.metal_pinned.is_some() {
            return;
        }
        let w = &self.weights;
        let h = cfg::HIDDEN_DIM;
        let inter = cfg::INTERMEDIATE;
        // RMSNorm gammas: kept as f32 on Metal (rmsnorm_metal_buf_tcb's
        // weight_buf is f32 — matches V2-Lite production rmsnorm path).
        let attn_norm_bytes: &[u8] = bytemuck::cast_slice(&w.block_attn_norm);
        let mlp_norm_bytes: &[u8] = bytemuck::cast_slice(&w.block_mlp_norm);
        let f32_bytes_for = |n: usize| n * std::mem::size_of::<f32>();
        self.metal_pinned = Some(Eagle4MetalPinned {
            in_proj:        pin_f32_as_f16(ctx, &w.in_proj),
            block_attn_q:   pin_f32_as_f16(ctx, &w.block_attn_q),
            block_attn_k:   pin_f32_as_f16(ctx, &w.block_attn_k),
            block_attn_v:   pin_f32_as_f16(ctx, &w.block_attn_v),
            block_attn_o:   pin_f32_as_f16(ctx, &w.block_attn_o),
            block_mlp_gate: pin_f32_as_f16(ctx, &w.block_mlp_gate),
            block_mlp_up:   pin_f32_as_f16(ctx, &w.block_mlp_up),
            block_mlp_down: pin_f32_as_f16(ctx, &w.block_mlp_down),
            mask_proj_in:   pin_f32_as_f16(ctx, &w.mask_proj_in),
            mask_proj_out:  pin_f32_as_f16(ctx, &w.mask_proj_out),
            attn_norm:      ctx.new_buffer_with_bytes(attn_norm_bytes),
            mlp_norm:       ctx.new_buffer_with_bytes(mlp_norm_bytes),
            // Scratch buffers — zero-init via new_buffer; subsequent
            // writes via write_buffer_bytes from the per-step forward.
            x5_buf:           ctx.new_buffer(f32_bytes_for(5 * h)),
            x_buf:            ctx.new_buffer(f32_bytes_for(h)),
            x_norm_attn_buf:  ctx.new_buffer(f32_bytes_for(h)),
            v_buf:            ctx.new_buffer(f32_bytes_for(h)),
            attn_out_buf:     ctx.new_buffer(f32_bytes_for(h)),
            x_norm_mlp_buf:   ctx.new_buffer(f32_bytes_for(h)),
            gate_buf:         ctx.new_buffer(f32_bytes_for(inter)),
            up_buf:           ctx.new_buffer(f32_bytes_for(inter)),
            act_buf:          ctx.new_buffer(f32_bytes_for(inter)),
            mlp_out_buf:      ctx.new_buffer(f32_bytes_for(h)),
        });
    }

    /// Whether [`Self::pin_metal`] has been called.
    #[cfg(target_os = "macos")]
    pub fn has_metal_pinned(&self) -> bool {
        self.metal_pinned.is_some()
    }

    /// Path-to-90 step 7 — Metal-accelerated forward, same semantics
    /// as [`Self::forward_full_no_lm_head`] but uses Metal-pinned f16
    /// gemvs for the 9 large weight matrices (in_proj, attn q/k/v/o,
    /// mlp gate/up/down, mask_proj_in/out). RMSNorms, SiLU, residual
    /// adds, and the tiny (1-row) calib_proj remain CPU — small
    /// intermediates, the readback over UMA is negligible.
    ///
    /// Requires both [`Self::set_frozen`] and [`Self::pin_metal`] to
    /// have run. Errors otherwise.
    #[cfg(target_os = "macos")]
    /// Path-to-90 lever 5 — AMX-accelerated head forward via direct
    /// cblas_sgemv from Accelerate.framework. Same outputs as
    /// `forward_full_no_lm_head` but routes the 6 large gemvs
    /// (in_proj, V, O, gate, up, down) through AMX. CPU intermediates
    /// (rmsnorm, silu_mul, residual adds) stay CPU — they're 2048-
    /// element ops and the AMX dispatch overhead would dominate.
    ///
    /// Why AMX vs Metal: per deep_research § Apple Silicon, AMX hits
    /// ~1790 GFLOPS at batch=1 for cblas_sgemv vs Metal f16 gemv's
    /// effective ~225 GFLOPS for the head's matrix shapes. Net per-
    /// gemv cost: ~10-50 µs vs Metal's ~1-5 ms.
    ///
    /// Does NOT need frozen weights pinned to Metal (eagle4 head's
    /// f32 Vec<f32> is the AMX input directly). Does not need a
    /// MetalContext. Q + K still skipped (S=1 dead computation),
    /// mask still skipped (Stage 3 only).
    #[cfg(target_os = "macos")]
    pub fn forward_full_amx_no_lm_head(
        &self,
        prev_token: u32,
        h_low: &[f32],
        h_mid: &[f32],
        h_high: &[f32],
        h_shared: &[f32],
    ) -> Result<Eagle4ForwardOutput> {
        let frozen = self.frozen.as_ref().ok_or_else(|| {
            Error::Model(
                "Eagle4Head::forward_full_amx_no_lm_head: frozen weights not loaded".into(),
            )
        })?;
        let h = cfg::HIDDEN_DIM;
        let inter = cfg::INTERMEDIATE;
        let vocab = cfg::VOCAB;
        let eps = cfg::RMS_EPS;

        if (prev_token as usize) >= vocab {
            return Err(Error::Model(format!(
                "Eagle4Head::forward_full_amx_no_lm_head: prev_token={} >= VOCAB={}",
                prev_token, vocab
            )));
        }
        for (name, vsl) in [
            ("h_low", h_low),
            ("h_mid", h_mid),
            ("h_high", h_high),
            ("h_shared", h_shared),
        ] {
            if vsl.len() != h {
                return Err(Error::Model(format!(
                    "Eagle4Head::forward_full_amx_no_lm_head: {name}.len()={} expected {h}",
                    vsl.len()
                )));
            }
        }

        let w = &self.weights;
        let pe_off = (prev_token as usize) * h;
        let prev_embed = &frozen.token_embd[pe_off..pe_off + h];

        // 1. Embed + concat into x5
        let mut x5 = Vec::with_capacity(5 * h);
        x5.extend_from_slice(prev_embed);
        x5.extend_from_slice(h_low);
        x5.extend_from_slice(h_mid);
        x5.extend_from_slice(h_high);
        x5.extend_from_slice(h_shared);

        // 2. in_proj via AMX: weight (HIDDEN, 5*HIDDEN), x5 → x (HIDDEN)
        let mut x = vec![0.0f32; h];
        amx_sgemv(h, 5 * h, &w.in_proj, &x5, &mut x);

        // 3. Attention block — RMSNorm → V → O → residual.
        //    Q + K skipped at S=1 (diagonal-mask softmax collapses to v).
        let mut x_normed = vec![0.0f32; h];
        rmsnorm(&x, &w.block_attn_norm, eps, &mut x_normed);
        let mut v = vec![0.0f32; h];
        amx_sgemv(h, h, &w.block_attn_v, &x_normed, &mut v);
        let mut attn_out = vec![0.0f32; h];
        amx_sgemv(h, h, &w.block_attn_o, &v, &mut attn_out);
        for i in 0..h {
            x[i] += attn_out[i];
        }

        // 4. MLP block — RMSNorm → SwiGLU → residual.
        rmsnorm(&x, &w.block_mlp_norm, eps, &mut x_normed);
        let mut gate_out = vec![0.0f32; inter];
        let mut up_out = vec![0.0f32; inter];
        amx_sgemv(inter, h, &w.block_mlp_gate, &x_normed, &mut gate_out);
        amx_sgemv(inter, h, &w.block_mlp_up, &x_normed, &mut up_out);
        let mut act = vec![0.0f32; inter];
        silu_mul(&gate_out, &up_out, &mut act);
        let mut mlp_out = vec![0.0f32; h];
        amx_sgemv(h, inter, &w.block_mlp_down, &act, &mut mlp_out);
        for i in 0..h {
            x[i] += mlp_out[i];
        }

        // 5. Baseline + residual gate
        let mut baseline = vec![0.0f32; h];
        rmsnorm(h_high, &frozen.output_norm, eps, &mut baseline);
        let alpha = w.residual_gate;
        let mut draft_hidden = vec![0.0f32; h];
        for i in 0..h {
            draft_hidden[i] = baseline[i] + alpha * x[i];
        }

        // 6. mask_proj SKIPPED (Stage 3 only).
        let mask_logits: Vec<f32> = Vec::new();

        // 7. calib_proj — tiny (1×HIDDEN); CPU is fine.
        let mut calib_buf = [0.0f32; 1];
        gemv_f32(&w.calib_proj_w, 1, h, &draft_hidden, &mut calib_buf);
        let calib_logit = calib_buf[0] + w.calib_proj_b;

        Ok(Eagle4ForwardOutput {
            token_logits: Vec::new(),
            mask_logits,
            draft_hidden,
            calib_logit,
        })
    }

    pub fn forward_full_metal_no_lm_head(
        &self,
        ctx: &MetalContext,
        prev_token: u32,
        h_low: &[f32],
        h_mid: &[f32],
        h_high: &[f32],
        h_shared: &[f32],
    ) -> Result<Eagle4ForwardOutput> {
        self.forward_full_metal_no_lm_head_on(ctx, prev_token, h_low, h_mid, h_high, h_shared, false)
    }

    /// path-to-125 L5w — variant that routes the Metal head TCB through
    /// the secondary command queue when `use_secondary=true`. The
    /// caller is responsible for setting up a `SharedEventBarrier` if
    /// the head's output is consumed by a kernel on the primary queue
    /// (otherwise CPU readback in this function already provides the
    /// ordering — every Metal kernel in the head completes before
    /// `commit_and_wait` returns).
    pub fn forward_full_metal_no_lm_head_on(
        &self,
        ctx: &MetalContext,
        prev_token: u32,
        h_low: &[f32],
        h_mid: &[f32],
        h_high: &[f32],
        h_shared: &[f32],
        use_secondary: bool,
    ) -> Result<Eagle4ForwardOutput> {
        let frozen = self.frozen.as_ref().ok_or_else(|| {
            Error::Model(
                "Eagle4Head::forward_full_metal_no_lm_head: frozen weights not loaded".into(),
            )
        })?;
        let pinned = self.metal_pinned.as_ref().ok_or_else(|| {
            Error::Model(
                "Eagle4Head::forward_full_metal_no_lm_head: weights not pinned to Metal — \
                 call pin_metal() first"
                    .into(),
            )
        })?;
        let h = cfg::HIDDEN_DIM;
        let inter = cfg::INTERMEDIATE;
        let mhid = cfg::MASK_HIDDEN;
        let mask_out = cfg::N_MOE_LAYERS * cfg::N_ROUTED;
        let vocab = cfg::VOCAB;
        let eps = cfg::RMS_EPS;

        if (prev_token as usize) >= vocab {
            return Err(Error::Model(format!(
                "Eagle4Head::forward_full_metal_no_lm_head: prev_token={} >= VOCAB={}",
                prev_token, vocab
            )));
        }
        for (name, vsl) in [
            ("h_low", h_low),
            ("h_mid", h_mid),
            ("h_high", h_high),
            ("h_shared", h_shared),
        ] {
            if vsl.len() != h {
                return Err(Error::Model(format!(
                    "Eagle4Head::forward_full_metal_no_lm_head: {name}.len()={} expected {h}",
                    vsl.len()
                )));
            }
        }

        let w = &self.weights;
        let pe_off = (prev_token as usize) * h;
        let prev_embed = &frozen.token_embd[pe_off..pe_off + h];

        // 1. Embed + concat into the persistent x5 scratch buffer (CPU).
        //    Write directly into the pinned buffer's memory — UMA, no
        //    GPU copy.
        {
            let mut x5_tmp = Vec::with_capacity(5 * h);
            x5_tmp.extend_from_slice(prev_embed);
            x5_tmp.extend_from_slice(h_low);
            x5_tmp.extend_from_slice(h_mid);
            x5_tmp.extend_from_slice(h_high);
            x5_tmp.extend_from_slice(h_shared);
            let bytes: &[u8] = bytemuck::cast_slice(&x5_tmp);
            crate::metal::MetalContext::write_buffer_bytes(&pinned.x5_buf, bytes);
        }

        // 2. Single-TCB encoding: all the head's gemv + rmsnorm +
        //    silu_mul + add_inplace ops go into ONE Metal command
        //    buffer. One commit per head forward instead of 5-9
        //    separate dispatches. Q and K stay skipped (at S=1 they
        //    contribute nothing — see commit 5b17428's rationale).
        //    mask_proj stays skipped (consumed only by Stage 3
        //    prefetch — see commit 0d6a2a3).
        {
            let mut tcb = if use_secondary {
                TokenCommandBuffer::new_on_secondary(ctx)
            } else {
                TokenCommandBuffer::new(ctx)
            };

            // in_proj: x5 → x_buf
            crate::kernels::gemv_f16_metal_buf_tcb(
                &mut tcb, &pinned.in_proj, h, 5 * h, &pinned.x5_buf, &pinned.x_buf,
            )?;
            // attn_norm: x_buf → x_norm_attn_buf
            crate::kernels::rmsnorm_metal_buf_tcb(
                &mut tcb, &pinned.x_buf, &pinned.attn_norm, eps, h, &pinned.x_norm_attn_buf,
            )?;
            // V: x_norm_attn_buf → v_buf
            crate::kernels::gemv_f16_metal_buf_tcb(
                &mut tcb, &pinned.block_attn_v, h, h, &pinned.x_norm_attn_buf, &pinned.v_buf,
            )?;
            // O: v_buf → attn_out_buf
            crate::kernels::gemv_f16_metal_buf_tcb(
                &mut tcb, &pinned.block_attn_o, h, h, &pinned.v_buf, &pinned.attn_out_buf,
            )?;
            // x_buf += attn_out_buf
            crate::kernels::add_inplace_metal_tcb(
                &mut tcb, &pinned.x_buf, &pinned.attn_out_buf, h,
            )?;
            // mlp_norm: x_buf → x_norm_mlp_buf
            crate::kernels::rmsnorm_metal_buf_tcb(
                &mut tcb, &pinned.x_buf, &pinned.mlp_norm, eps, h, &pinned.x_norm_mlp_buf,
            )?;
            // gate, up: x_norm_mlp_buf → gate_buf, up_buf
            crate::kernels::gemv_f16_metal_buf_tcb(
                &mut tcb, &pinned.block_mlp_gate, inter, h, &pinned.x_norm_mlp_buf, &pinned.gate_buf,
            )?;
            crate::kernels::gemv_f16_metal_buf_tcb(
                &mut tcb, &pinned.block_mlp_up, inter, h, &pinned.x_norm_mlp_buf, &pinned.up_buf,
            )?;
            // silu_mul(gate, up) → act
            crate::kernels::silu_mul_tcb(&mut tcb, &pinned.gate_buf, &pinned.up_buf, &pinned.act_buf, inter)?;
            // down: act_buf → mlp_out_buf
            crate::kernels::gemv_f16_metal_buf_tcb(
                &mut tcb, &pinned.block_mlp_down, h, inter, &pinned.act_buf, &pinned.mlp_out_buf,
            )?;
            // x_buf += mlp_out_buf
            crate::kernels::add_inplace_metal_tcb(
                &mut tcb, &pinned.x_buf, &pinned.mlp_out_buf, h,
            )?;

            tcb.commit_and_wait()?;
        }

        // 3. Read x_buf back to CPU (single readback after the TCB).
        let mut x = vec![0.0f32; h];
        let ptr = pinned.x_buf.contents() as *const f32;
        let src = unsafe { std::slice::from_raw_parts(ptr, h) };
        x.copy_from_slice(src);

        // 4. baseline + residual gate (CPU — 2048-element ops; Metal
        //    launch overhead exceeds compute).
        let mut baseline = vec![0.0f32; h];
        rmsnorm(h_high, &frozen.output_norm, eps, &mut baseline);
        let alpha = w.residual_gate;
        let mut draft_hidden = vec![0.0f32; h];
        for i in 0..h {
            draft_hidden[i] = baseline[i] + alpha * x[i];
        }

        // 6. mask_proj is SKIPPED in the Metal path. mask_logits is
        //    consumed only by Stage 3's mask-driven prefetch
        //    (execution_plan.md step 18), which hasn't shipped yet.
        //    At K=1 spec decode the accept/reject logic doesn't read
        //    it. Saves ~8 ms/token (mask_proj_in 512×2048 +
        //    mask_proj_out 1664×512). When Stage 3 lands, add a
        //    `compute_mask: bool` flag (the pinned weights are kept;
        //    only the dispatch is gated).
        let _ = mhid;
        let _ = mask_out;
        let mask_logits: Vec<f32> = Vec::new();

        // 7. calib_proj (CPU — single-row dot product; Metal launch
        //    overhead exceeds the compute)
        let mut calib_buf = [0.0f32; 1];
        gemv_f32(&w.calib_proj_w, 1, h, &draft_hidden, &mut calib_buf);
        let calib_logit = calib_buf[0] + w.calib_proj_b;

        Ok(Eagle4ForwardOutput {
            token_logits: Vec::new(), // skipped — caller uses GPU argmax via dismantle's lm_head
            mask_logits,
            draft_hidden,
            calib_logit,
        })
    }

    /// Path-to-90 step 5 — CPU fp32 forward of the full EAGLE-4 head.
    ///
    /// Implements the architecture documented in
    /// `reports/path_to_90/eagle4_convergence.md § EAGLE-4 forward`:
    ///
    /// ```text
    /// prev_embed   = token_embd[prev_token]                  (HIDDEN,)
    /// x            = concat(prev_embed, h_low, h_mid, h_high, h_shared)  (5*HIDDEN,)
    /// x            = in_proj @ x                              (HIDDEN,)
    /// x            = TransformerBlock(x)                      single position, diagonal-mask
    /// baseline     = rmsnorm(h_high, output_norm)             (HIDDEN,)
    /// draft_hidden = baseline + residual_gate · x             (HIDDEN,)
    /// token_logits = draft_hidden @ lm_head                   (VOCAB,)
    /// mask_logits  = mask_proj_out(silu(mask_proj_in(draft_hidden)))   (N_MOE*N_ROUTED,)
    /// calib_logit  = calib_proj_w @ draft_hidden + calib_proj_b        scalar
    /// ```
    ///
    /// Per-call sequence length is 1, so the transformer block's self-
    /// attention reduces to `o_proj(v_proj(x_normed))`: q·k^T is a
    /// single scalar per head, softmax of a single value is 1.0, and
    /// the diagonal mask is identically zero at S=1. Q and K are still
    /// computed for clarity and to match eagle4.py's eval-mode flow
    /// (where the batched (1, take, HIDDEN) call runs with a diagonal
    /// mask that makes each record independent).
    ///
    /// Errors if [`Self::set_frozen`] hasn't been called, if
    /// `prev_token >= VOCAB`, or if any hidden vector is mis-sized.
    pub fn forward_full(
        &self,
        prev_token: u32,
        h_low: &[f32],
        h_mid: &[f32],
        h_high: &[f32],
        h_shared: &[f32],
    ) -> Result<Eagle4ForwardOutput> {
        self.forward_full_inner(prev_token, h_low, h_mid, h_high, h_shared, true)
    }

    /// Path-to-90 step 10 follow-up — forward without the CPU LM head
    /// gemv. Production decode loops use this and route the token
    /// argmax through dismantle's `gemv_f16_argmax_dispatch` against
    /// the V2-Lite GGUF's already-pinned Metal lm_head buffer
    /// (~10 ms vs the CPU gemv's ~165 ms on 819 MB of f32 weights).
    /// Returns the same struct with `token_logits` as an empty Vec —
    /// caller must compute argmax via the dismantle-side GPU dispatch.
    pub fn forward_full_no_lm_head(
        &self,
        prev_token: u32,
        h_low: &[f32],
        h_mid: &[f32],
        h_high: &[f32],
        h_shared: &[f32],
    ) -> Result<Eagle4ForwardOutput> {
        self.forward_full_inner(prev_token, h_low, h_mid, h_high, h_shared, false)
    }

    fn forward_full_inner(
        &self,
        prev_token: u32,
        h_low: &[f32],
        h_mid: &[f32],
        h_high: &[f32],
        h_shared: &[f32],
        compute_token_logits: bool,
    ) -> Result<Eagle4ForwardOutput> {
        let frozen = self.frozen.as_ref().ok_or_else(|| {
            Error::Model(
                "Eagle4Head::forward_full: frozen weights not loaded — \
                 call set_frozen() with Eagle4FrozenWeights first"
                    .into(),
            )
        })?;
        let h = cfg::HIDDEN_DIM;
        let inter = cfg::INTERMEDIATE;
        let mhid = cfg::MASK_HIDDEN;
        let mask_out = cfg::N_MOE_LAYERS * cfg::N_ROUTED;
        let vocab = cfg::VOCAB;
        let eps = cfg::RMS_EPS;

        if (prev_token as usize) >= vocab {
            return Err(Error::Model(format!(
                "Eagle4Head::forward_full: prev_token={} >= VOCAB={}",
                prev_token, vocab
            )));
        }
        for (name, vsl) in [
            ("h_low", h_low),
            ("h_mid", h_mid),
            ("h_high", h_high),
            ("h_shared", h_shared),
        ] {
            if vsl.len() != h {
                return Err(Error::Model(format!(
                    "Eagle4Head::forward_full: {name}.len()={} expected {h}",
                    vsl.len()
                )));
            }
        }

        let w = &self.weights;

        // 1. Embedding lookup. token_embd loaded as (VOCAB, HIDDEN) row-major.
        let pe_off = (prev_token as usize) * h;
        let prev_embed = &frozen.token_embd[pe_off..pe_off + h];

        // 2. Concatenate the five HIDDEN-dim vectors into 5*HIDDEN.
        let mut x5 = Vec::with_capacity(5 * h);
        x5.extend_from_slice(prev_embed);
        x5.extend_from_slice(h_low);
        x5.extend_from_slice(h_mid);
        x5.extend_from_slice(h_high);
        x5.extend_from_slice(h_shared);

        // 3. in_proj: (HIDDEN, 5*HIDDEN) @ (5*HIDDEN,) → (HIDDEN,).
        //    gemv_f32: out[r] = sum_c W[r*cols+c] * x[c].
        //    Equivalent to Python nn.Linear(5H→H) at batch=1.
        let mut x = vec![0.0f32; h];
        gemv_f32(&w.in_proj, h, 5 * h, &x5, &mut x);

        // 4. Transformer block — RMSNorm → MHA → residual → RMSNorm → SwiGLU → residual.

        // 4a. Pre-attn RMSNorm.
        let mut x_normed = vec![0.0f32; h];
        rmsnorm(&x, &w.block_attn_norm, eps, &mut x_normed);

        // 4b. Q, K, V projections (all HIDDEN×HIDDEN).
        let mut q = vec![0.0f32; h];
        let mut k = vec![0.0f32; h];
        let mut v = vec![0.0f32; h];
        gemv_f32(&w.block_attn_q, h, h, &x_normed, &mut q);
        gemv_f32(&w.block_attn_k, h, h, &x_normed, &mut k);
        gemv_f32(&w.block_attn_v, h, h, &x_normed, &mut v);

        // 4c. Self-attention at S=1, per-head.
        //     scores_h = (q_h · k_h^T) / sqrt(head_dim)  — single scalar.
        //     softmax([s]) = [1.0].
        //     attn_h    = 1.0 · v_h  =  v_h.
        // So the MHA-internal output equals `v` element-for-element. Q and K
        // are computed above for fidelity but contribute nothing at S=1.
        // This is identical to Python eval's per-record behavior with the
        // (mx.eye(S)-1)*1e9 diagonal mask: off-diagonal scores get -1e9,
        // softmax collapses to a one-hot on the diagonal, attention = v[i].
        let _ = (q, k);

        // 4d. Output projection.
        let mut attn_out = vec![0.0f32; h];
        gemv_f32(&w.block_attn_o, h, h, &v, &mut attn_out);

        // 4e. Attention residual.
        for i in 0..h {
            x[i] += attn_out[i];
        }

        // 4f. Pre-MLP RMSNorm.
        rmsnorm(&x, &w.block_mlp_norm, eps, &mut x_normed);

        // 4g. SwiGLU.
        let mut gate_out = vec![0.0f32; inter];
        let mut up_out = vec![0.0f32; inter];
        let mut act = vec![0.0f32; inter];
        gemv_f32(&w.block_mlp_gate, inter, h, &x_normed, &mut gate_out);
        gemv_f32(&w.block_mlp_up, inter, h, &x_normed, &mut up_out);
        silu_mul(&gate_out, &up_out, &mut act);
        let mut mlp_out = vec![0.0f32; h];
        gemv_f32(&w.block_mlp_down, h, inter, &act, &mut mlp_out);

        // 4h. MLP residual.
        for i in 0..h {
            x[i] += mlp_out[i];
        }

        // 5. Baseline = rmsnorm(h_high, output_norm).
        let mut baseline = vec![0.0f32; h];
        rmsnorm(h_high, &frozen.output_norm, eps, &mut baseline);

        // 6. Residual-gate fusion. draft_hidden = baseline + α · x.
        let alpha = w.residual_gate;
        let mut draft_hidden = vec![0.0f32; h];
        for i in 0..h {
            draft_hidden[i] = baseline[i] + alpha * x[i];
        }

        // 7. token_logits = draft_hidden @ lm_head; lm_head is (VOCAB, HIDDEN).
        // CPU LM head gemv is the dominant cost in this forward
        // (~165 ms on 819 MB f32). Production decode loops skip it
        // and recompute argmax via dismantle's GPU
        // `gemv_f16_argmax_dispatch` against the V2-Lite GGUF's
        // already-pinned Metal lm_head buffer. Parity tests (step 6)
        // still compute it because they verify the full logit vector.
        let token_logits = if compute_token_logits {
            let mut tl = vec![0.0f32; vocab];
            gemv_f32(&frozen.lm_head, vocab, h, &draft_hidden, &mut tl);
            tl
        } else {
            Vec::new()
        };

        // 8. mask_logits = mask_proj_out(silu(mask_proj_in(draft_hidden))).
        let mut mp_in = vec![0.0f32; mhid];
        gemv_f32(&w.mask_proj_in, mhid, h, &draft_hidden, &mut mp_in);
        let mut mp_silu = vec![0.0f32; mhid];
        for i in 0..mhid {
            let s = mp_in[i];
            mp_silu[i] = s / (1.0 + (-s).exp()); // SiLU(s) = s · sigmoid(s)
        }
        let mut mask_logits = vec![0.0f32; mask_out];
        gemv_f32(&w.mask_proj_out, mask_out, mhid, &mp_silu, &mut mask_logits);

        // 9. calib_logit = calib_proj_w · draft_hidden + calib_proj_b.
        let mut calib_buf = [0.0f32; 1];
        gemv_f32(&w.calib_proj_w, 1, h, &draft_hidden, &mut calib_buf);
        let calib_logit = calib_buf[0] + w.calib_proj_b;

        Ok(Eagle4ForwardOutput {
            token_logits,
            mask_logits,
            draft_hidden,
            calib_logit,
        })
    }
}

/// Return the indices of the top-`k` values in `logits` (highest first).
/// Ties broken by index ascending. Truncates to `logits.len()` if k > len.
fn top_k_indices(logits: &[f32], k: usize) -> Vec<usize> {
    let want = k.min(logits.len());
    if want == 0 {
        return Vec::new();
    }
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    // Partial sort: pull the top `want` to the front by value descending.
    idx.select_nth_unstable_by(want - 1, |&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut top: Vec<usize> = idx.into_iter().take(want).collect();
    top.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    top
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
    fn propose(&mut self, inputs: &DraftInputs<'_>, k: usize) -> Result<DraftOutputs> {
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
        // hiddens order is fixed by the contract: [h_low, h_mid, h_high, h_shared].
        let h_low = inputs.hiddens[0];
        let h_mid = inputs.hiddens[1];
        let h_high = inputs.hiddens[2];
        let h_shared = inputs.hiddens[3];

        let out = self.forward_full(inputs.prev_token, h_low, h_mid, h_high, h_shared)?;

        // Top-K tokens from token_logits. K=0 → 1 (caller always wants
        // at least the argmax — returning empty would silently break the
        // verify path).
        let want = k.max(1);
        let topk = top_k_indices(&out.token_logits, want);

        // Predicted routing mask: per-MoE-layer top-`TOP_K_ROUTED` from
        // `mask_logits`. Each layer's row is `mask_logits[L*N_ROUTED..(L+1)*N_ROUTED]`.
        let mut routing_mask = vec![0u8; cfg::N_MOE_LAYERS * cfg::N_ROUTED];
        for layer in 0..cfg::N_MOE_LAYERS {
            let row = &out.mask_logits[layer * cfg::N_ROUTED..(layer + 1) * cfg::N_ROUTED];
            for idx in top_k_indices(row, cfg::TOP_K_ROUTED) {
                routing_mask[layer * cfg::N_ROUTED + idx] = 1;
            }
        }

        Ok(DraftOutputs {
            tokens: topk.into_iter().map(|i| i as u32).collect(),
            routing_mask: Some(routing_mask),
            calib: Some(out.calib_logit),
        })
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
        assert!(!head.has_frozen());
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
    fn propose_without_frozen_errors() {
        // Frozen weights are required before any forward — the old
        // "Unimplemented" stub return is now replaced by a clear
        // Error::Model from forward_full.
        let mut head = Eagle4Head::new_uninitialized();
        let h = vec![0.0f32; cfg::HIDDEN_DIM];
        let four: [&[f32]; 4] = [&h, &h, &h, &h];
        let inputs = DraftInputs {
            prev_token: 0,
            hiddens: &four,
        };
        let result = head.propose(&inputs, 4);
        match result {
            Err(Error::Model(msg)) => assert!(
                msg.contains("frozen weights not loaded"),
                "expected frozen-weights error, got: {msg}"
            ),
            other => panic!("expected Err(Model), got {other:?}"),
        }
    }

    #[test]
    fn propose_with_frozen_runs_forward() {
        // All-zero weights + all-zero hiddens → token_logits all zero
        // (every weight is 0). top_k still returns indices, calib=0,
        // routing_mask all zeros.
        let mut head = Eagle4Head::new_uninitialized();
        let frozen = Eagle4FrozenWeights {
            token_embd: vec![0.0; cfg::VOCAB * cfg::HIDDEN_DIM],
            lm_head: vec![0.0; cfg::VOCAB * cfg::HIDDEN_DIM],
            output_norm: vec![1.0; cfg::HIDDEN_DIM], // identity-ish RMSNorm scale
        };
        head.set_frozen(frozen);
        assert!(head.has_frozen());
        let hh = vec![0.0f32; cfg::HIDDEN_DIM];
        let four: [&[f32]; 4] = [&hh, &hh, &hh, &hh];
        let inputs = DraftInputs {
            prev_token: 0,
            hiddens: &four,
        };
        let out = head.propose(&inputs, 4).expect("propose with zero weights");
        assert_eq!(out.tokens.len(), 4);
        assert_eq!(
            out.routing_mask.as_ref().unwrap().len(),
            cfg::N_MOE_LAYERS * cfg::N_ROUTED
        );
        assert_eq!(out.calib, Some(0.0));
    }

    #[test]
    fn top_k_indices_orders_by_value_desc() {
        let v = vec![0.1, 0.5, 0.3, 0.7, 0.2];
        let top3 = top_k_indices(&v, 3);
        assert_eq!(top3, vec![3, 1, 2]); // 0.7, 0.5, 0.3
    }

    #[test]
    fn fusion_layer_constants_match_eagle4() {
        // Sanity guard against the brief's old {2,14,24} sneaking back in.
        assert_eq!(cfg::FUSION_LAYERS, [2, 13, 25]);
        assert_eq!(cfg::SHARED_EXPERT_LAYER, 26);
    }
}
