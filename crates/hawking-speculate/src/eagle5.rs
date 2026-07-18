//! Eagle5 v2 draft head for speculative decoding.
//!
//! The trained head (see `tools/training/eagle5_train.py` and
//! `reports/eagle5_v2_wiring_handoff.md`) is a ~25M-param transformer
//! block that consumes
//!
//! - prev_token embedding (2048),
//! - residual stream at the capture layer (2048),
//! - intermediate stream at the capture layer (2048),
//!
//! and produces (a) a draft token logit vector over the LM-head vocab and
//! (b) an auxiliary channel-sparsity logit vector. For runtime spec-decode
//! purposes only the token logit matters: we argmax it, feed the resulting
//! id back into the head, and emit K draft tokens. The verifier (the full
//! V2-Lite model) consumes the K draft ids serially with early-mismatch
//! exit, so greedy output at temperature=0 is bit-identical to no-spec
//! greedy regardless of the head's accept rate.
//!
//! ## What this module ships today
//!
//! 1. `Eagle5Head` — the dispatchable draft-head state, holding either a
//!    `Mock` (deterministic random weights) or a `Trained` (loaded from
//!    safetensors) variant. The runtime calls `propose(prev_token, k)`
//!    to get K draft ids per step and `note_token` to record verified
//!    tokens for future calls. `reset()` clears any per-sequence state
//!    between generation requests.
//! 2. `Eagle5Head::mock(seed, hidden, vocab)` — the deterministic
//!    runtime-validation fallback. Uses a single linear projection
//!    `W ∈ R^{vocab × hidden}` and a constant embed table; produces
//!    repeatable draft ids without touching disk. The mock head is NOT
//!    a quality model — its accept rate is near 1/vocab. It exists
//!    purely so the runtime path can be measured before the trained
//!    checkpoint lands.
//! 3. `Eagle5Head::load_from_safetensors(path, …)` — the loader stub.
//!    Currently returns `Err(Error::Unimplemented(...))`; the trained
//!    head's tensor layout (per the eagle4 reference q4-export recipe)
//!    will plug into this method without changing any other call site.
//!    Wiring the trained head into the runtime is then a one-file
//!    change.
//!
//! ## How a trained head will swap in
//!
//! - `tools/training/eagle5_quantize.py` writes a safetensors file with
//!   keys `in_proj`, `block.{attn,mlp}`, `out_lm_head`, plus shared
//!   frozen references to `token_embd` and `output_norm` from
//!   `eagle4/v2lite_frozen.npz` (per the design doc §3).
//! - `Eagle5Head::load_from_safetensors` deserializes those weights,
//!   stores a `Trained` variant, and the runtime call paths
//!   (`propose`, `note_token`, `reset`) dispatch identically.
//! - Hidden-state input (`residual_in`, `intermediate`) is optional in
//!   this module's contract today; the trained path will require the
//!   decode loop to cache the verifier's per-layer hidden via a small
//!   sidecar buffer. That wiring is documented in the call-site comment
//!   in `model/deepseek_v2.rs::generate()`'s Eagle5 branch.

use crate::safetensors_io::SafeTensors;
use crate::Error;
use crate::Result;
use half::f16;
use std::path::Path;

/// Eagle5 v2 head state. Holds either a mock-random-weights variant
/// (for runtime-wiring validation) or a trained variant loaded from a
/// safetensors checkpoint. The two variants share a single
/// `propose/note_token/reset` API so the decode loop never branches on
/// which one is live.
pub struct Eagle5Head {
    inner: Inner,
    /// Vocab size of the LM-head argmax output. Mock and trained must
    /// agree with the verifier's vocab; mismatch is an error at load.
    vocab: usize,
    /// Hidden dim of the head; matches V2-Lite's `cfg.hidden`. For the
    /// mock this only sizes the deterministic embed table.
    hidden: usize,
    /// Most-recently-accepted token id, used as the starting prev_token
    /// for the next draft window. `None` until the first `note_token`.
    last_token: Option<u32>,
    /// Optional vocab-pruned LM head for the propose hot path:
    /// `(pruned_lm_head [hidden, n_pruned] f16, remap[pruned_idx] = real id)`.
    /// Built by `set_vocab_prune` from the verifier's prune mapping. When
    /// present, `propose_rollout_chained` sizes its lm_head matmul to the
    /// pruned vocab (the dominant propose cost, ~4.7× smaller at q3b) and
    /// remaps draft ids back to real ids. **Parity-safe:** drafts only
    /// affect speed, and the verifier emits only pruned-vocab tokens, so a
    /// draft outside the pruned set could never have been accepted anyway.
    lm_head_pruned: Option<(Vec<f16>, Vec<u32>)>,
}

enum Inner {
    /// Deterministic random-weights mock. `embed` is a `vocab × hidden`
    /// f32 table (xorshift-seeded); `out_w` is a `vocab × hidden` f32
    /// projection from the previous-token embedding to a per-vocab
    /// logit. Argmax over `out_w @ embed[prev]` yields the next draft.
    ///
    /// This is intentionally a tiny CPU-only path so the mock never
    /// allocates Metal buffers or holds GPU state. Accept rate is
    /// effectively 0% — the only purpose is to drive the spec-decode
    /// runtime branch end-to-end without depending on training.
    Mock { embed: Vec<f32>, out_w: Vec<f32> },
    /// Trained head loaded from a safetensors checkpoint produced by
    /// `colab/finish_q3b_reconciliation.ipynb` (or the predecessor
    /// `colab/eagle5_train_pytorch.py`). Holds the full Eagle6 head:
    /// `in_proj` + N transformer blocks + the frozen lm_head and
    /// token embedding shared with the verifier model.
    ///
    /// Memory footprint is dominated by the frozen `_token_embd` and
    /// `_lm_head` (kept in f16 to halve RAM): for Qwen-3B that's
    /// 2 × (2048 × 151936) × 2 bytes ≈ 1.25 GB; trainable weights add
    /// another ~96M params × 4 = 384 MB. q1p5 head is ~700 MB total.
    ///
    /// **Forward pass not yet implemented.** The propose() dispatcher
    /// currently treats Trained as if it were Mock (linear projection
    /// of token_embd[prev] through lm_head). The real Eagle6 forward
    /// (rmsnorm + in_proj + transformer-block attn/mlp + final norm +
    /// lm_head) lands in a follow-up commit; this commit ships only
    /// the loader + struct.
    Trained {
        config: TrainedConfig,
        /// [hidden, 3 * hidden] f32 — projects [prev_embd | residual |
        /// intermediate] from the verifier's capture layer down to the
        /// head's hidden_dim.
        in_proj: Vec<f32>,
        /// Length = num_blocks. Each block is one transformer block
        /// (norm → attn → norm → mlp with gated SiLU).
        blocks: Vec<TrainedBlock>,
        /// Scalar gate on the residual stream merge (0..1-ish).
        residual_gate: f32,
        /// [hidden] f32 — final RMSNorm gain. Frozen, shared with verifier.
        output_norm: Vec<f32>,
        /// [hidden, vocab] f16 — frozen token embedding shared with verifier.
        token_embd_f16: Vec<f16>,
        /// [hidden, vocab] f16 — frozen LM head shared with verifier.
        lm_head_f16: Vec<f16>,
    },
}

/// Architecture metadata for a loaded Eagle6 trained head.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TrainedConfig {
    pub hidden_dim: usize,
    pub vocab_size: usize,
    pub n_heads: usize,
    pub ff_mult: f32,
    pub num_blocks: usize,
    /// hidden_dim * ff_mult, the FFN inner dim (gate/up cols, down rows).
    pub ff_dim: usize,
}

/// One transformer block of an Eagle6 head. All tensors f32 to match
/// the trainer's safetensors output; quantization (e.g. Q4_K of the
/// projections) is a Phase A.2+ optimization.
#[allow(dead_code)]
pub struct TrainedBlock {
    /// [hidden] f32 — pre-attn RMSNorm gain.
    pub attn_norm: Vec<f32>,
    /// [hidden, hidden] f32 — Q projection (row-major: rows × cols).
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub out_proj: Vec<f32>,
    /// [hidden] f32 — pre-mlp RMSNorm gain.
    pub mlp_norm: Vec<f32>,
    /// [ff_dim, hidden] f32 — gated SiLU gate.
    pub mlp_gate: Vec<f32>,
    /// [ff_dim, hidden] f32 — gated SiLU up.
    pub mlp_up: Vec<f32>,
    /// [hidden, ff_dim] f32 — gated SiLU down.
    pub mlp_down: Vec<f32>,
}

impl Eagle5Head {
    /// Construct a deterministic mock head for runtime-wiring tests.
    ///
    /// `seed` controls the xorshift PRNG used to fill the weight
    /// tables. The same `(seed, hidden, vocab)` triple always
    /// reproduces bit-identical draft ids, which is what the parity
    /// test relies on to assert that spec-decode greedy matches
    /// no-spec greedy regardless of which (deterministic) drafts the
    /// head proposes.
    pub fn mock(seed: u64, hidden: usize, vocab: usize) -> Self {
        // Two tables: a per-token embedding lookup, and a per-vocab
        // output projection. Both are tiny and CPU-only so the mock
        // never touches Metal. Filling order is fixed so test runs
        // reproduce.
        let mut rng = XorShift64::new(seed.wrapping_add(0xa5a5_a5a5));
        let mut embed = vec![0.0f32; vocab * hidden];
        // Centered in [-1/sqrt(hidden), +1/sqrt(hidden)] so the
        // projection output magnitude stays sane regardless of dim.
        let scale = 1.0 / (hidden as f32).sqrt();
        for slot in embed.iter_mut() {
            *slot = rng.next_uniform() * scale;
        }
        let mut out_w = vec![0.0f32; vocab * hidden];
        for slot in out_w.iter_mut() {
            *slot = rng.next_uniform() * scale;
        }
        Self {
            inner: Inner::Mock { embed, out_w },
            vocab,
            hidden,
            last_token: None,
            lm_head_pruned: None,
        }
    }

    /// Build the vocab-pruned LM head used by the propose hot path from the
    /// verifier's prune mapping (`remap[pruned_idx] = real_token_id`, the
    /// same `vocab_prune_remap` the verifier slices its own LM head with).
    /// No-op for Mock heads. Idempotent — safe to call once after load.
    ///
    /// Cost: builds a `[hidden, n_pruned]` f16 copy of the relevant LM-head
    /// columns (~131 MB at q3b 32K). Pays back by shrinking the per-draft
    /// argmax matmul from full vocab (151936) to `n_pruned` (32000).
    pub fn set_vocab_prune(&mut self, remap: &[u32]) {
        if self.lm_head_pruned.is_some() {
            return; // idempotent — already built.
        }
        let (h, v) = (self.hidden, self.vocab);
        if let Inner::Trained { lm_head_f16, .. } = &self.inner {
            let n = remap.len();
            let mut pruned = vec![f16::from_f32(0.0); h * n];
            for i in 0..h {
                let src = &lm_head_f16[i * v..(i + 1) * v];
                let dst = &mut pruned[i * n..(i + 1) * n];
                for (j, &rid) in remap.iter().enumerate() {
                    dst[j] = src[(rid as usize).min(v - 1)];
                }
            }
            self.lm_head_pruned = Some((pruned, remap.to_vec()));
            eprintln!(
                "[eagle5] built vocab-pruned propose LM head: {} -> {} cols",
                v,
                remap.len()
            );
        }
    }

    /// True for a Trained head that has not yet had its pruned LM head
    /// built. Used to gate the one-time `set_vocab_prune` build (and the
    /// remap clone it needs) so it runs once, not per decode step.
    pub fn needs_vocab_prune(&self) -> bool {
        matches!(self.inner, Inner::Trained { .. }) && self.lm_head_pruned.is_none()
    }

    /// Load a trained Eagle6 head from a safetensors checkpoint
    /// produced by `colab/finish_q3b_reconciliation.ipynb`.
    ///
    /// Validates the file's `__metadata__` matches `(hidden, vocab)`
    /// arguments — mismatch is a hard error to catch q3b head being
    /// loaded against a Qwen-1.5B verifier or vice versa. Reads all
    /// runtime tensors into RAM (trainable f32 + frozen f16). The
    /// training-only `calib_proj.{weight,bias}` is intentionally not
    /// loaded.
    ///
    /// File-format expectations (see safetensors_io.rs comment):
    /// - 1-block head: keys under `block.*`
    /// - N-block head (N > 1): `block.*` + `extra_blocks.{0..N-2}.*`
    ///
    /// **The Trained variant's `propose()` dispatch is still a
    /// simplified linear projection (lm_head @ token_embd[prev]).
    /// The real Eagle6 forward (in_proj + transformer block + final
    /// norm + lm_head) lands in a follow-up commit.** This loader
    /// stages the weights so that follow-up is a single-file change.
    pub fn load_from_safetensors(
        path: &Path,
        expected_hidden: usize,
        expected_vocab: usize,
    ) -> Result<Self> {
        let st = SafeTensors::open(path)?;
        let meta = st.metadata();
        let parse_usize = |k: &str| -> Result<Option<usize>> {
            meta.get(k)
                .map(|v| {
                    v.parse::<usize>()
                        .map_err(|e| Error::Model(format!("eagle5: '{k}' parse failed: {e}")))
                })
                .transpose()
        };
        let parse_f32 = |k: &str| -> Result<Option<f32>> {
            meta.get(k)
                .map(|v| {
                    v.parse::<f32>()
                        .map_err(|e| Error::Model(format!("eagle5: '{k}' parse failed: {e}")))
                })
                .transpose()
        };
        let in_proj_shape = st.shape("in_proj.weight")?;
        if in_proj_shape.len() != 2 || in_proj_shape[1] != 3 * in_proj_shape[0] {
            return Err(Error::Model(format!(
                "eagle5: in_proj.weight shape is {in_proj_shape:?}; expected [hidden, 3*hidden]"
            )));
        }
        let inferred_hidden = in_proj_shape[0];
        let emb_shape = st.shape("_token_embd")?;
        if emb_shape.len() != 2 {
            return Err(Error::Model(format!(
                "eagle5: _token_embd shape is {emb_shape:?}; expected [hidden, vocab]"
            )));
        }
        let inferred_vocab = emb_shape[1];
        let gate_shape = st.shape("block.mlp.gate.weight")?;
        if gate_shape.len() != 2 || gate_shape[1] != inferred_hidden {
            return Err(Error::Model(format!(
                "eagle5: block.mlp.gate.weight shape is {gate_shape:?}; expected [ff_dim, hidden]"
            )));
        }
        let inferred_ff_dim = gate_shape[0];
        let inferred_ff_mult = inferred_ff_dim as f32 / inferred_hidden as f32;
        let inferred_num_blocks = {
            let mut n = 1usize;
            while st.has(&format!("extra_blocks.{}.attn_norm", n - 1)) {
                n += 1;
            }
            n
        };
        let inferred_n_heads = if inferred_hidden % 16 == 0 {
            16
        } else {
            [12usize, 8, 4, 2, 1]
                .into_iter()
                .find(|h| inferred_hidden % h == 0)
                .unwrap_or(1)
        };

        let hidden_dim = parse_usize("hidden_dim")?.unwrap_or(inferred_hidden);
        let vocab_size = parse_usize("vocab_size")?.unwrap_or(inferred_vocab);
        let n_heads = parse_usize("n_heads")?.unwrap_or(inferred_n_heads);
        let num_blocks = parse_usize("num_blocks")?.unwrap_or(inferred_num_blocks);
        let ff_mult = parse_f32("ff_mult")?.unwrap_or(inferred_ff_mult);
        let ff_dim = ((hidden_dim as f32) * ff_mult).round() as usize;
        if hidden_dim != expected_hidden {
            return Err(Error::Model(format!(
                "eagle5: head hidden_dim={hidden_dim} but verifier expects {expected_hidden}"
            )));
        }
        if vocab_size != expected_vocab {
            return Err(Error::Model(format!(
                "eagle5: head vocab_size={vocab_size} but verifier expects {expected_vocab}"
            )));
        }
        if num_blocks == 0 {
            return Err(Error::Model("eagle5: num_blocks must be ≥1".to_string()));
        }
        if n_heads == 0 || hidden_dim % n_heads != 0 {
            return Err(Error::Model(format!(
                "eagle5: invalid n_heads={n_heads} for hidden={hidden_dim}"
            )));
        }
        if inferred_ff_dim != ff_dim {
            return Err(Error::Model(format!(
                "eagle5: inferred ff_dim={inferred_ff_dim} but metadata implies {ff_dim}"
            )));
        }

        let config = TrainedConfig {
            hidden_dim,
            vocab_size,
            n_heads,
            ff_mult,
            num_blocks,
            ff_dim,
        };

        // in_proj is shape [hidden, 3 * hidden]: takes the concatenated
        // [prev_token_embd | residual_in | intermediate] (each `hidden`
        // wide) and projects to hidden. Match the trainer's PyTorch
        // Linear weight layout: rows = out_features, cols = in_features.
        let in_proj = st.read_f32("in_proj.weight", &[hidden_dim, 3 * hidden_dim])?;

        // First block keys live under `block.`; subsequent blocks under
        // `extra_blocks.{N-2}.` (where N is the 1-based block index).
        // This mirrors the trainer's nn.Module naming.
        let load_block = |prefix: &str| -> Result<TrainedBlock> {
            Ok(TrainedBlock {
                attn_norm: st.read_f32(&format!("{prefix}attn_norm"), &[hidden_dim])?,
                q_proj: st
                    .read_f32(&format!("{prefix}q_proj.weight"), &[hidden_dim, hidden_dim])?,
                k_proj: st
                    .read_f32(&format!("{prefix}k_proj.weight"), &[hidden_dim, hidden_dim])?,
                v_proj: st
                    .read_f32(&format!("{prefix}v_proj.weight"), &[hidden_dim, hidden_dim])?,
                out_proj: st.read_f32(
                    &format!("{prefix}out_proj.weight"),
                    &[hidden_dim, hidden_dim],
                )?,
                mlp_norm: st.read_f32(&format!("{prefix}mlp_norm"), &[hidden_dim])?,
                mlp_gate: st
                    .read_f32(&format!("{prefix}mlp.gate.weight"), &[ff_dim, hidden_dim])?,
                mlp_up: st.read_f32(&format!("{prefix}mlp.up.weight"), &[ff_dim, hidden_dim])?,
                mlp_down: st
                    .read_f32(&format!("{prefix}mlp.down.weight"), &[hidden_dim, ff_dim])?,
            })
        };

        let mut blocks = Vec::with_capacity(num_blocks);
        blocks.push(load_block("block.")?);
        for i in 0..num_blocks.saturating_sub(1) {
            blocks.push(load_block(&format!("extra_blocks.{i}."))?);
        }

        // residual_gate is a 1-element f32 vector; unwrap to scalar.
        let gate_vec = st.read_f32("residual_gate", &[1])?;
        let residual_gate = gate_vec[0];

        // Frozen tensors shared with the verifier. f16 to halve RAM.
        let output_norm = st.read_f32("_output_norm", &[hidden_dim])?;
        let token_embd_f16 = st.read_f16("_token_embd", &[hidden_dim, vocab_size])?;
        let lm_head_f16 = st.read_f16("_lm_head", &[hidden_dim, vocab_size])?;

        Ok(Self {
            inner: Inner::Trained {
                config,
                in_proj,
                blocks,
                residual_gate,
                output_norm,
                token_embd_f16,
                lm_head_f16,
            },
            vocab: vocab_size,
            hidden: hidden_dim,
            last_token: None,
            lm_head_pruned: None,
        })
    }

    /// Propose up to `k` draft token ids for the next decode step.
    ///
    /// Uses the most-recently-noted token (or `prev_token` if no
    /// `note_token` has been called yet) as the starting point and
    /// auto-regressively feeds each proposed id back through the head.
    /// Returns the K draft ids the verifier will check.
    pub fn propose(&mut self, prev_token: u32, k: usize) -> Vec<u32> {
        if k == 0 || self.vocab == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(k);
        let mut cur = self.last_token.unwrap_or(prev_token);
        for _ in 0..k {
            let next = self.argmax_step(cur);
            out.push(next);
            cur = next;
        }
        out
    }

    /// Real Eagle6 forward propose: like `propose()` but supplies the
    /// verifier's residual + intermediate streams (captured from the
    /// head's capture layer at the verifier's current decode position)
    /// for use by the Trained variant's `in_proj`.
    ///
    /// For Mock heads, the captured streams are ignored (Mock has no
    /// `in_proj`). For Trained heads, they're concatenated with the
    /// previous-token embedding to form the (3 * hidden,) `in_proj`
    /// input.
    ///
    /// Auto-regressive chain at S=1: each step's prev_token is the
    /// previous draft. residual + intermediate stay constant across
    /// all K steps within a single verify cycle — they're the
    /// snapshot from the last verifier forward.
    pub fn propose_with_capture(
        &mut self,
        prev_token: u32,
        residual_in: &[f32],
        intermediate: &[f32],
        k: usize,
    ) -> Vec<u32> {
        if k == 0 || self.vocab == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(k);
        let mut cur = self.last_token.unwrap_or(prev_token);
        for _ in 0..k {
            let next = self.argmax_step_full(cur, residual_in, intermediate);
            out.push(next);
            cur = next;
        }
        out
    }

    /// Rollout propose from an EXPLICIT start token (ignores `self.last_token`).
    ///
    /// The head is trained so that, given `residual_T` (the capture-layer
    /// residual of the forward that consumed token T) and an advancing token,
    /// it predicts the continuation T+1, T+2, … with the residual held FIXED
    /// across depths (matching the rollout training objective). The runtime
    /// must therefore start the chain at T — the token whose residual was
    /// captured — NOT at the bonus token T+1. `out[0]` is the head's
    /// prediction of T+1 (≈ the verifier's bonus); `out[1..]` are the genuine
    /// look-ahead drafts for T+2, T+3, … that the verifier checks.
    pub fn propose_rollout(
        &self,
        start_token: u32,
        residual_in: &[f32],
        intermediate: &[f32],
        k: usize,
    ) -> Vec<u32> {
        if k == 0 || self.vocab == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(k);
        let mut cur = start_token;
        for _ in 0..k {
            let next = self.argmax_step_full(cur, residual_in, intermediate);
            out.push(next);
            cur = next;
        }
        out
    }

    /// EAGLE-style chained-hidden rollout propose. Matches `--rollout-chain-
    /// hidden` training: depth 0 uses the captured `residual_in`; each deeper
    /// depth feeds the head's OWN `draft_hidden` from the previous step as the
    /// residual (intermediate=0), so the head advances its own hidden state.
    /// This is what makes depth-2+ drafts usable (q3b depth-2 16%→47%).
    /// `out[0]` ≈ T+1 (the verifier's free token); `out[1..]` are the genuine
    /// look-ahead drafts. Trained heads only; Mock falls back to fixed-residual.
    pub fn propose_rollout_chained(
        &self,
        start_token: u32,
        residual_in: &[f32],
        intermediate: &[f32],
        k: usize,
    ) -> Vec<u32> {
        if k == 0 || self.vocab == 0 {
            return Vec::new();
        }
        match &self.inner {
            Inner::Mock { .. } => self.propose_rollout(start_token, residual_in, intermediate, k),
            Inner::Trained {
                config,
                in_proj,
                blocks,
                residual_gate,
                output_norm,
                token_embd_f16,
                lm_head_f16,
            } => {
                use crate::eagle5_forward::{compute_draft_hidden, lm_head_logits};
                let h = config.hidden_dim;
                let v = config.vocab_size;
                let zeros = vec![0.0f32; h];
                // Pruned LM head for the per-draft argmax — the dominant
                // propose cost. When present, the matmul is sized to the
                // verifier's pruned vocab (~32K) instead of the full ~152K.
                // Parity-safe: the verifier emits only pruned-vocab tokens,
                // so a draft outside the pruned set could never be accepted.
                let pruned = self.lm_head_pruned.as_ref();
                let mut out = Vec::with_capacity(k);
                let mut cur = start_token;
                let mut res: Vec<f32> = residual_in.to_vec();
                let mut inter: &[f32] = intermediate;
                for _ in 0..k {
                    let draft_hidden = compute_draft_hidden(
                        config,
                        in_proj,
                        blocks,
                        *residual_gate,
                        output_norm,
                        token_embd_f16,
                        cur,
                        &res,
                        inter,
                    );
                    let next = match pruned {
                        Some((lm_pruned, remap)) => {
                            let logits = lm_head_logits(&draft_hidden, lm_pruned, h, remap.len());
                            remap[crate::argmax_f32(&logits) as usize]
                        }
                        None => {
                            let logits = lm_head_logits(&draft_hidden, lm_head_f16, h, v);
                            crate::argmax_f32(&logits) as u32
                        }
                    };
                    out.push(next);
                    cur = next;
                    // Chain: next depth's residual = this depth's draft_hidden,
                    // intermediate carries no chained signal.
                    res = draft_hidden;
                    inter = &zeros;
                }
                out
            }
        }
    }

    /// Full-forward argmax step. For Trained heads invokes the real
    /// Eagle6 forward pass via `eagle5_forward::forward_single_step`.
    /// For Mock heads falls back to the simple linear projection.
    fn argmax_step_full(&self, prev: u32, residual_in: &[f32], intermediate: &[f32]) -> u32 {
        match &self.inner {
            Inner::Mock { .. } => self.argmax_step(prev),
            Inner::Trained { .. } => {
                let logits = self
                    .forward_logits(prev, residual_in, intermediate)
                    .expect("Trained variant must return Some(logits)");
                crate::argmax_f32(&logits) as u32
            }
        }
    }

    /// Run a single Eagle6 forward step and return the full vocab-length
    /// logits vector. Only meaningful for `Inner::Trained`; returns
    /// `None` for `Inner::Mock` (Mock has no transformer-block forward).
    ///
    /// Exposed for parity-testing the Rust forward against the PyTorch
    /// reference. Not on the runtime hot path — runtime uses
    /// `propose_with_capture` which argmaxes internally.
    pub fn forward_logits(
        &self,
        prev_token: u32,
        residual_in: &[f32],
        intermediate: &[f32],
    ) -> Option<Vec<f32>> {
        match &self.inner {
            Inner::Mock { .. } => None,
            Inner::Trained {
                config,
                in_proj,
                blocks,
                residual_gate,
                output_norm,
                token_embd_f16,
                lm_head_f16,
            } => Some(crate::eagle5_forward::forward_single_step(
                config,
                in_proj,
                blocks,
                *residual_gate,
                output_norm,
                token_embd_f16,
                lm_head_f16,
                prev_token,
                residual_in,
                intermediate,
            )),
        }
    }

    /// Pure logit-lens argmax: `argmax(RMSNorm(residual, output_norm) @ lm_head)`.
    /// This is the head's `baseline` term with NO transformer-block / gate
    /// contribution — i.e. "what does the captured residual predict on its
    /// own through the unembedding." Comparing this to the model's real
    /// next token measures the layer-K logit-lens ceiling (how viable the
    /// capture layer is for speculation), independent of head training.
    /// Returns `None` for Mock heads.
    pub fn lens_argmax(&self, residual_in: &[f32]) -> Option<u32> {
        match &self.inner {
            Inner::Mock { .. } => None,
            Inner::Trained {
                config,
                output_norm,
                lm_head_f16,
                ..
            } => {
                let h = config.hidden_dim;
                let v = config.vocab_size;
                // RMSNorm(residual, output_norm), fp32.
                let mut ss = 0.0f32;
                for &x in &residual_in[..h] {
                    ss += x * x;
                }
                let inv = 1.0f32 / ((ss / h as f32) + 1e-6).sqrt();
                let mut baseline = vec![0.0f32; h];
                for i in 0..h {
                    baseline[i] = residual_in[i] * inv * output_norm[i];
                }
                // argmax over lm_head: logits[k] = sum_i baseline[i]*lm_head[i*v+k]
                let mut best_k = 0u32;
                let mut best_v = f32::NEG_INFINITY;
                let mut acc = vec![0.0f32; v];
                for i in 0..h {
                    let bi = baseline[i];
                    let row = &lm_head_f16[i * v..(i + 1) * v];
                    for k in 0..v {
                        acc[k] += bi * row[k].to_f32();
                    }
                }
                for (k, &val) in acc.iter().enumerate() {
                    if val > best_v {
                        best_v = val;
                        best_k = k as u32;
                    }
                }
                Some(best_k)
            }
        }
    }

    /// Record a token that was emitted (either an accepted draft or a
    /// verifier correction). The next `propose` call will seed from
    /// this token.
    pub fn note_token(&mut self, token: u32) {
        self.last_token = Some(token);
    }

    /// Reset per-sequence state between generation requests.
    pub fn reset(&mut self) {
        self.last_token = None;
    }

    /// Vocab size the head was constructed for. Callers use this to
    /// validate that the verifier's LM-head agrees.
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Hidden dim the head was constructed for.
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// One head forward step: prev_token → next_token via argmax of
    /// `out_w @ embed[prev_token]`. Inlined so the per-token cost is a
    /// single `hidden`-vector dot product times `vocab` rows; well
    /// under 1 ms for V2-Lite's (2048, 102400) shape on CPU.
    fn argmax_step(&self, prev: u32) -> u32 {
        let prev = (prev as usize).min(self.vocab.saturating_sub(1));
        match &self.inner {
            Inner::Mock { embed, out_w } => {
                let h = self.hidden;
                let row = &embed[prev * h..(prev + 1) * h];
                // Compute logit per output vocab id and track argmax in
                // one pass — vocab * hidden FMAs, no allocation.
                let mut best_id = 0u32;
                let mut best_score = f32::NEG_INFINITY;
                for v in 0..self.vocab {
                    let w_row = &out_w[v * h..(v + 1) * h];
                    let mut acc = 0.0f32;
                    for k in 0..h {
                        acc += w_row[k] * row[k];
                    }
                    if acc > best_score {
                        best_score = acc;
                        best_id = v as u32;
                    }
                }
                best_id
            }
            Inner::Trained {
                token_embd_f16,
                lm_head_f16,
                ..
            } => {
                // SIMPLIFIED placeholder: real Eagle6 forward (in_proj +
                // transformer block + final norm + lm_head) is Phase A.2.
                // For now we do a single linear-projection-of-embedding
                // argmax against the frozen lm_head — accept rate will be
                // near zero, but it proves the loader + dispatch wire
                // end-to-end. Storage is [hidden, vocab] (transpose of
                // Mock's [vocab, hidden] layout), so the access pattern
                // strides through the vocab axis.
                let h = self.hidden;
                let v_total = self.vocab;
                // Extract column `prev` of token_embd: embd_prev[i] = token_embd[i, prev].
                let mut embd_prev = vec![0.0f32; h];
                for i in 0..h {
                    embd_prev[i] = token_embd_f16[i * v_total + prev].to_f32();
                }
                let mut best_id = 0u32;
                let mut best_score = f32::NEG_INFINITY;
                for v in 0..v_total {
                    // logit[v] = sum_i lm_head[i, v] * embd_prev[i]
                    let mut acc = 0.0f32;
                    for i in 0..h {
                        acc += lm_head_f16[i * v_total + v].to_f32() * embd_prev[i];
                    }
                    if acc > best_score {
                        best_score = acc;
                        best_id = v as u32;
                    }
                }
                best_id
            }
        }
    }
}

/// Minimal xorshift64 PRNG for deterministic mock-head weights.
/// Reusing this avoids pulling in `rand` for the lib crate.
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0xdead_beef_cafe_babe
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform-ish f32 in [-1.0, +1.0).
    fn next_uniform(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32; // 24 bits of entropy
        let unit = bits as f32 / ((1u32 << 24) as f32); // [0, 1)
        unit * 2.0 - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_head_propose_is_deterministic() {
        let mut h1 = Eagle5Head::mock(42, 64, 1000);
        let mut h2 = Eagle5Head::mock(42, 64, 1000);
        let p1 = h1.propose(7, 4);
        let p2 = h2.propose(7, 4);
        assert_eq!(p1, p2, "same seed must reproduce draft ids");
        assert_eq!(p1.len(), 4);
    }

    #[test]
    fn mock_head_propose_respects_k() {
        let mut h = Eagle5Head::mock(1, 32, 64);
        assert!(h.propose(0, 0).is_empty());
        assert_eq!(h.propose(0, 1).len(), 1);
        assert_eq!(h.propose(0, 8).len(), 8);
    }

    #[test]
    fn mock_head_propose_returns_in_vocab() {
        let mut h = Eagle5Head::mock(99, 16, 50);
        let drafts = h.propose(3, 16);
        for d in drafts {
            assert!((d as usize) < 50, "draft id {d} out of vocab");
        }
    }

    #[test]
    fn mock_head_note_token_seeds_next_propose() {
        let mut h = Eagle5Head::mock(7, 32, 100);
        let from_default = h.propose(5, 1);
        // Note a different token then propose 1 — should differ from
        // the from_default proposal because the head's prev_token is
        // now the noted id, not 5.
        h.reset();
        h.note_token(11);
        let from_noted = h.propose(5, 1);
        assert_ne!(
            from_default, from_noted,
            "note_token must seed next propose"
        );
    }

    #[test]
    fn mock_head_reset_clears_state() {
        let mut h = Eagle5Head::mock(3, 32, 100);
        h.note_token(42);
        let after_note = h.propose(0, 1);
        h.reset();
        let after_reset = h.propose(0, 1);
        // After reset, propose(prev) should use prev=0 not the noted 42.
        // We don't assert equality with a precomputed value (cost: extra
        // fixture) — just that reset changes the result.
        assert_ne!(after_note, after_reset, "reset must clear last_token");
    }

    #[test]
    fn mock_head_vocab_and_hidden_match_ctor() {
        let h = Eagle5Head::mock(0, 128, 5000);
        assert_eq!(h.vocab(), 5000);
        assert_eq!(h.hidden(), 128);
    }

    #[test]
    fn trained_loader_rejects_nonexistent_file() {
        let result = Eagle5Head::load_from_safetensors(
            Path::new("/nonexistent/path.safetensors"),
            128,
            5000,
        );
        // io::NotFound bubbles up as Error::Io, not Unimplemented anymore.
        assert!(matches!(result, Err(Error::Io(_))));
    }

    /// Builds a tiny synthetic Eagle6 safetensors file in a temp dir and
    /// asserts the loader reads all tensors with the expected shapes
    /// and the metadata-validation guards fire on mismatches.
    #[test]
    fn trained_loader_reads_synthetic_head() {
        use std::io::Write;
        let hidden = 8;
        let vocab = 16;
        let n_heads = 2;
        let ff_mult = 2.0_f32;
        let ff_dim = (hidden as f32 * ff_mult) as usize;
        let num_blocks = 1;

        // Build the safetensors header JSON + concatenated tensor bytes.
        // Layout order matches what the loader expects to find by name.
        let mut tensor_bytes: Vec<u8> = Vec::new();
        let mut entries: Vec<(String, &'static str, Vec<usize>, usize, usize)> = Vec::new();
        let push_f32 =
            |name: &str, shape: Vec<usize>, entries: &mut Vec<_>, bytes: &mut Vec<u8>| {
                let n = shape.iter().product::<usize>();
                let start = bytes.len();
                for i in 0..n {
                    bytes.extend_from_slice(&((i as f32) * 0.01_f32).to_le_bytes());
                }
                let end = bytes.len();
                entries.push((name.to_string(), "F32", shape, start, end));
            };
        let push_f16 =
            |name: &str, shape: Vec<usize>, entries: &mut Vec<_>, bytes: &mut Vec<u8>| {
                let n = shape.iter().product::<usize>();
                let start = bytes.len();
                for i in 0..n {
                    let v = f16::from_f32((i as f32) * 0.01_f32);
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                let end = bytes.len();
                entries.push((name.to_string(), "F16", shape, start, end));
            };
        push_f32(
            "in_proj.weight",
            vec![hidden, 3 * hidden],
            &mut entries,
            &mut tensor_bytes,
        );
        push_f32(
            "block.attn_norm",
            vec![hidden],
            &mut entries,
            &mut tensor_bytes,
        );
        for k in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            push_f32(
                &format!("block.{k}.weight"),
                vec![hidden, hidden],
                &mut entries,
                &mut tensor_bytes,
            );
        }
        push_f32(
            "block.mlp_norm",
            vec![hidden],
            &mut entries,
            &mut tensor_bytes,
        );
        push_f32(
            "block.mlp.gate.weight",
            vec![ff_dim, hidden],
            &mut entries,
            &mut tensor_bytes,
        );
        push_f32(
            "block.mlp.up.weight",
            vec![ff_dim, hidden],
            &mut entries,
            &mut tensor_bytes,
        );
        push_f32(
            "block.mlp.down.weight",
            vec![hidden, ff_dim],
            &mut entries,
            &mut tensor_bytes,
        );
        push_f32("residual_gate", vec![1], &mut entries, &mut tensor_bytes);
        push_f32(
            "_output_norm",
            vec![hidden],
            &mut entries,
            &mut tensor_bytes,
        );
        push_f16(
            "_token_embd",
            vec![hidden, vocab],
            &mut entries,
            &mut tensor_bytes,
        );
        push_f16(
            "_lm_head",
            vec![hidden, vocab],
            &mut entries,
            &mut tensor_bytes,
        );

        // Construct the JSON header. Use a manual builder so the field
        // order is stable and the test doesn't pull serde_json::json! in.
        let mut header = String::from("{");
        header.push_str(&format!(
            "\"__metadata__\":{{\"hidden_dim\":\"{hidden}\",\"vocab_size\":\"{vocab}\",\"n_heads\":\"{n_heads}\",\"ff_mult\":\"{ff_mult}\",\"num_blocks\":\"{num_blocks}\"}}"
        ));
        for (name, dtype, shape, start, end) in &entries {
            let shape_str = shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            header.push_str(&format!(
                ",\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{shape_str}],\"data_offsets\":[{start},{end}]}}"
            ));
        }
        header.push('}');
        let header_bytes = header.as_bytes();
        let header_len = header_bytes.len() as u64;

        let dir = std::env::temp_dir().join(format!("eagle5_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("synthetic.safetensors");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&header_len.to_le_bytes()).unwrap();
        f.write_all(header_bytes).unwrap();
        f.write_all(&tensor_bytes).unwrap();
        drop(f);

        // Happy path: load + dispatch.
        let head = Eagle5Head::load_from_safetensors(&path, hidden, vocab).unwrap();
        assert_eq!(head.hidden(), hidden);
        assert_eq!(head.vocab(), vocab);
        // propose() should not crash for the Trained variant; result is
        // a deterministic argmax over the synthetic weights.
        let mut h = head;
        let drafts = h.propose(0, 3);
        assert_eq!(drafts.len(), 3);
        for d in &drafts {
            assert!(((*d) as usize) < vocab);
        }

        // Mismatched hidden/vocab must fail loudly.
        let err_hidden = Eagle5Head::load_from_safetensors(&path, hidden + 1, vocab);
        assert!(matches!(err_hidden, Err(Error::Model(_))));
        let err_vocab = Eagle5Head::load_from_safetensors(&path, hidden, vocab + 1);
        assert!(matches!(err_vocab, Err(Error::Model(_))));

        // Legacy Colab checkpoints before May 2026 omitted __metadata__.
        // The loader should infer hidden/vocab/ff shape metadata from tensor
        // shapes while still validating against the verifier dimensions.
        let mut legacy_header = String::from("{");
        for (idx, (name, dtype, shape, start, end)) in entries.iter().enumerate() {
            if idx > 0 {
                legacy_header.push(',');
            }
            let shape_str = shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            legacy_header.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{shape_str}],\"data_offsets\":[{start},{end}]}}"
            ));
        }
        legacy_header.push('}');
        let legacy_path = dir.join("synthetic_legacy_no_metadata.safetensors");
        let mut f = std::fs::File::create(&legacy_path).unwrap();
        f.write_all(&(legacy_header.len() as u64).to_le_bytes())
            .unwrap();
        f.write_all(legacy_header.as_bytes()).unwrap();
        f.write_all(&tensor_bytes).unwrap();
        drop(f);
        let legacy = Eagle5Head::load_from_safetensors(&legacy_path, hidden, vocab).unwrap();
        assert_eq!(legacy.hidden(), hidden);
        assert_eq!(legacy.vocab(), vocab);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&legacy_path);
        let _ = std::fs::remove_dir(&dir);
    }
}
