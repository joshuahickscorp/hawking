//! Event Horizon — unified proposal-market contracts.
//!
//! A *proposal market*: many cheap proposers compete each decode step under ONE
//! exact verifier. Losslessness is structural — the verifier accepts the longest
//! argmax-confirmed prefix and emits the target's own correction, so a proposer
//! may be arbitrarily wrong with ZERO quality cost. Greedy-only (temperature==0)
//! in Phase 0/1. This module defines contracts only; concrete impls live beside
//! their state (user_ngram.rs, eagle5.rs).

use crate::speculate::shared::DraftToken;

/// Wall-clock currency of the router's expected_gain rule. Everything is ns;
/// convert to ms only at the human-readable edge.
pub type CostNs = u64;

/// Trailing-context view handed to a proposer each step. Borrowed, never owned.
pub struct Ctx<'a> {
    /// Emitted token ids so far, **oldest-first** (matches UserNgramDraft::propose,
    /// which reads ctx.last() as `cur`, ctx[len-2] as `prev`).
    pub tokens: &'a [u32],
    /// Position of the next token to predict (== verifier seq_len).
    pub pos: usize,
    /// Optional target hidden tap; `None` on the base path / cycle-1 / capture-off.
    pub hidden: Option<HiddenTap<'a>>,
}

impl<'a> Ctx<'a> {
    /// The `[.., prev, cur]` slice UserNgramDraft wants (it reads only last two).
    #[inline]
    pub fn last_two(&self) -> &[u32] {
        let n = self.tokens.len();
        &self.tokens[n.saturating_sub(2)..]
    }
}

/// Borrowed target hidden state for hidden-coupled proposers (Phase 4+).
/// Base proposers never read it.
#[allow(dead_code)]
pub struct HiddenTap<'a> {
    pub residual: &'a [f32],
    pub intermediate: &'a [f32],
    pub start_token: u32,
}

/// Per-step spending limit. `k` is hard-capped at 7 by the caller because
/// forward_tokens_verify batches bonus + ≤7 drafts into its B≤8 fast path.
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    pub k: usize,
}
impl Budget {
    pub const VERIFY_BATCH: usize = 8;
    pub const MAX_DRAFT_LEN: usize = 7;
    pub fn line(k: usize) -> Self {
        Budget {
            k: k.min(Self::MAX_DRAFT_LEN),
        }
    }
}

/// What a proposer hands the verifier. Phase 0/1 constructs only `TokenLine`.
pub enum Proposal {
    /// Linear AR / n-gram draft of bare ids → shared::verify_draft_ids_until_mismatch.
    /// May be SHORTER than budget.k (chain miss) or empty (verifier degenerates
    /// an empty line to a plain 1-token decode). THE LIVE BASE.
    TokenLine(Vec<u32>),
    /// Linear draft with per-token logits → shared::verify_window. n-gram cannot
    /// fill this; reserved for future lossless sampling.
    TokenLineWithLogits(Vec<DraftToken>),
    /// SpecInfer/DDTree token tree. NO ENGINE SUPPORT yet (forward_tokens_verify
    /// is contiguous/causal only) — Phase-6 Metal build. Defined now so the
    /// verifier dispatch signature is stable.
    TokenTree {
        nodes: Vec<u32>,
        ancestor_mask: Vec<u64>,
        position_ids: Vec<usize>,
    },
}

impl Proposal {
    pub fn is_empty(&self) -> bool {
        match self {
            Proposal::TokenLine(v) => v.is_empty(),
            Proposal::TokenLineWithLogits(v) => v.is_empty(),
            Proposal::TokenTree { nodes, .. } => nodes.is_empty(),
        }
    }
    pub fn draft_len(&self) -> usize {
        match self {
            Proposal::TokenLine(v) => v.len(),
            Proposal::TokenLineWithLogits(v) => v.len(),
            Proposal::TokenTree { nodes, .. } => nodes.len(),
        }
    }
}

/// Read-only telemetry snapshot a proposer may consult. Phase-1 proposers ignore
/// it; field set mirrors the router's StepObservation so signatures stay stable.
#[derive(Debug, Clone, Copy, Default)]
pub struct Telemetry {
    pub accepted: u64,
    pub rejected: u64,
    pub draft_ns: CostNs,
    pub verify_ns: CostNs,
    pub sync_ns: CostNs,
}

/// The unified proposer interface. Extends the spec sketch with the two hooks the
/// recon found missing: `observe` (mandatory online feedback) and `warm`/`reset`
/// (per-request lifecycle without discarding a persistent per-user index).
pub trait Proposer {
    /// Stable identity, the router's telemetry/hysteresis key ("user_ngram", ...).
    fn name(&self) -> &'static str;

    /// Taps target hidden states (EAGLE-family)? Base proposers: false. The
    /// router uses this to turn the target's 2 capture dispatches on only when a
    /// hidden proposer is live.
    fn requires_hidden(&self) -> bool {
        false
    }

    /// Needs detok→retok text bridge (cross-tokenizer)? Base proposers: false.
    fn requires_text_bridge(&self) -> bool {
        false
    }

    /// Predicted draft cost (ns) — the router's draft_ns input. Base proposers ~0.
    fn cost_estimate(&self, _ctx: &Ctx<'_>, _budget: Budget) -> CostNs {
        0
    }

    /// Produce a draft. MUST respect budget.k (≤7) and never mutate emitted
    /// output — only proposes; the verifier decides. May return empty/short.
    /// `&mut self` because stateful proposers may advance cursors; pure ones ignore it.
    fn propose(&mut self, ctx: &Ctx<'_>, budget: Budget, tel: &Telemetry) -> Proposal;

    /// MANDATORY online feedback (the spec sketch's biggest gap). The engine calls
    /// this with EVERY emitted (verifier) token, in order, even while the router
    /// has this proposer disabled — the n-gram index decays otherwise. Stateless
    /// proposers no-op.
    fn observe(&mut self, _emitted: &[u32]) {}

    /// Per-request warm-start (prompt/session/repo). n-gram → warm_start (seed
    /// grams, reset cursor); the learned index persists across turns. No-op default.
    fn warm(&mut self, _history: &[u32]) {}

    /// Per-request teardown: clear rolling cursor but KEEP the learned index
    /// (n-gram reset_context, never clear). No-op default.
    fn reset(&mut self) {}
}
