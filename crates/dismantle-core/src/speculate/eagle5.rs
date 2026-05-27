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

use crate::speculate::safetensors_io::SafeTensors;
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
    Mock {
        embed: Vec<f32>,
        out_w: Vec<f32>,
    },
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
        }
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
        let parse_usize = |k: &str| -> Result<usize> {
            meta.get(k)
                .ok_or_else(|| Error::Model(format!("eagle5: safetensors missing '{k}' metadata")))?
                .parse::<usize>()
                .map_err(|e| Error::Model(format!("eagle5: '{k}' parse failed: {e}")))
        };
        let parse_f32 = |k: &str| -> Result<f32> {
            meta.get(k)
                .ok_or_else(|| Error::Model(format!("eagle5: safetensors missing '{k}' metadata")))?
                .parse::<f32>()
                .map_err(|e| Error::Model(format!("eagle5: '{k}' parse failed: {e}")))
        };
        let hidden_dim = parse_usize("hidden_dim")?;
        let vocab_size = parse_usize("vocab_size")?;
        let n_heads = parse_usize("n_heads")?;
        let num_blocks = parse_usize("num_blocks")?;
        let ff_mult = parse_f32("ff_mult")?;
        let ff_dim = ((hidden_dim as f32) * ff_mult) as usize;
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
                q_proj: st.read_f32(
                    &format!("{prefix}q_proj.weight"),
                    &[hidden_dim, hidden_dim],
                )?,
                k_proj: st.read_f32(
                    &format!("{prefix}k_proj.weight"),
                    &[hidden_dim, hidden_dim],
                )?,
                v_proj: st.read_f32(
                    &format!("{prefix}v_proj.weight"),
                    &[hidden_dim, hidden_dim],
                )?,
                out_proj: st.read_f32(
                    &format!("{prefix}out_proj.weight"),
                    &[hidden_dim, hidden_dim],
                )?,
                mlp_norm: st.read_f32(&format!("{prefix}mlp_norm"), &[hidden_dim])?,
                mlp_gate: st.read_f32(
                    &format!("{prefix}mlp.gate.weight"),
                    &[ff_dim, hidden_dim],
                )?,
                mlp_up: st.read_f32(&format!("{prefix}mlp.up.weight"), &[ff_dim, hidden_dim])?,
                mlp_down: st.read_f32(
                    &format!("{prefix}mlp.down.weight"),
                    &[hidden_dim, ff_dim],
                )?,
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
            state: if seed == 0 { 0xdead_beef_cafe_babe } else { seed },
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
        assert_ne!(from_default, from_noted, "note_token must seed next propose");
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
        let mut push_f32 =
            |name: &str, shape: Vec<usize>, entries: &mut Vec<_>, bytes: &mut Vec<u8>| {
                let n = shape.iter().product::<usize>();
                let start = bytes.len();
                for i in 0..n {
                    bytes.extend_from_slice(&((i as f32) * 0.01_f32).to_le_bytes());
                }
                let end = bytes.len();
                entries.push((name.to_string(), "F32", shape, start, end));
            };
        let mut push_f16 =
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

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }
}
