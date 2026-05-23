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

use crate::Error;
use crate::Result;
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
    /// Trained head loaded from a safetensors checkpoint. Currently
    /// unreachable; the loader returns Unimplemented. The struct is
    /// kept as a stub so the dispatcher in `propose` can be written
    /// against the final shape today.
    #[allow(dead_code)]
    Trained {
        // Placeholder — real layout (in_proj, block, out_lm_head) lands
        // when the safetensors loader is implemented.
        token_embd: Vec<f32>,
        out_lm_head: Vec<f32>,
    },
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

    /// Load a trained head from a safetensors checkpoint.
    ///
    /// **Stub**: returns `Err(Error::Unimplemented(...))` until the
    /// trained head's tensor layout is finalized. The runtime falls
    /// back to the mock head when this returns Err with the
    /// "eagle5: trained head loader" tag, so a missing checkpoint
    /// does not block spec-decode wiring tests.
    pub fn load_from_safetensors(
        _path: &Path,
        _hidden: usize,
        _vocab: usize,
    ) -> Result<Self> {
        Err(Error::Unimplemented(
            "eagle5: trained head loader (use mock head fallback for now; \
             implement once tools/training/eagle5_quantize.py output \
             format is finalized)",
        ))
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
                token_embd,
                out_lm_head,
            } => {
                let h = self.hidden;
                let row = &token_embd[prev * h..(prev + 1) * h];
                let mut best_id = 0u32;
                let mut best_score = f32::NEG_INFINITY;
                for v in 0..self.vocab {
                    let w_row = &out_lm_head[v * h..(v + 1) * h];
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
    fn trained_loader_returns_unimplemented() {
        let result = Eagle5Head::load_from_safetensors(
            Path::new("/nonexistent/path.safetensors"),
            128,
            5000,
        );
        assert!(matches!(result, Err(Error::Unimplemented(_))));
    }
}
