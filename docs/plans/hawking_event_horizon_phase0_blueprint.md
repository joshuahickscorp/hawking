I have everything grounded against the real source. The skeletons in the designs match the actual signatures (`verify_draft_ids_until_mismatch`, `SpecGovernor::new/step/is_enabled`, the `'ud_loop` accept block at 2632, `forward_tokens_verify` returning `(preds, _resids)`). Now I'll write the synthesized blueprint.

This is a synthesis-and-return task — my output is the deliverable, not a file. Here is the consolidated Event Horizon Phase-0/1 implementation blueprint.

---

# Event Horizon — Phase-0/1 Implementation Blueprint

A unified speculative-decoding **proposal market**: many cheap proposers compete each decode step under ONE exact target verifier (lossless by construction), with a Metal-aware **router** that enables/disables proposers by expected *wall-clock* gain. This blueprint collapses the four env-selected QwenDense spec loops into one `Proposer → Verifier → Router` flow that reuses the existing primitives unchanged, and is ordered so every step compiles.

**Grounding note:** all signatures below are verified against source — `verify_draft_ids_until_mismatch` (`shared.rs:66`), `SpecGovernor::{new,with_thresholds,step,is_enabled,accept_rate}` (`governor.rs:100-220`), `UserNgramDraft::{new,propose,note_token,warm_start,reset_context}` (`user_ngram.rs:57-155`), the live `'ud_loop` accept block (`qwen_dense.rs:2632-2706`), `forward_tokens_verify → (Vec<u32>, Vec<Vec<f32>>)` (`qwen_dense.rs:9004`, called at `:2631`), `forward_token_greedy_tcb` (`qwen_dense.rs:4456`, called at `:2569`).

---

## 1. Module layout

```
crates/hawking-core/src/speculate/
├── mod.rs              (EDIT: +3 lines — pub mod {proposal, router, verifier};)
├── shared.rs           (UNCHANGED — verify_draft_ids_until_mismatch, DraftStats, VerifyResult)
├── governor.rs         (UNCHANGED — SpecGovernor, wrapped N× by the router)
├── replay_oracle.rs    (UNCHANGED — τ≥2.5 GO gate, the neural-promotion precondition)
├── user_ngram.rs       (EDIT: + impl Proposer for NgramProposer adapter at bottom)
├── eagle5.rs           (Phase 4+: + impl Proposer for Eagle5Head — NOT in Phase 0/1)
│
├── proposal.rs    (NEW — the contracts: Ctx, Budget, Proposal, Telemetry, CostNs, Proposer trait)
├── verifier.rs    (NEW — ExactTarget trait + Verifier + VerifyOutcome; QwenDense impl lives in qwen_dense.rs)
└── router.rs      (NEW — ProposalRouter, ProposerId, RouterCtx, RouterPlan, StepObservation, CostModel)
```

**Placement rationale (de-duplicated from the four designs):**
- **`proposal.rs`** holds the *contracts only* — the `Proposer` trait plus `Ctx`/`Budget`/`Proposal`/`Telemetry`. Concrete `impl Proposer` blocks live **beside their state** (`user_ngram.rs`, `eagle5.rs`) so they reach private fields without orphan-rule trouble. The two designs disagreed on filename (`proposal.rs` vs `proposer.rs`) — **use `proposal.rs`** for the contracts; the `NgramProposer` adapter goes *in `user_ngram.rs`*, not a separate file.
- **`verifier.rs`** is pure CPU/std (compiles on every platform, unit-testable without Metal). Only the `impl ExactTarget for QwenDense` is `#[cfg(target_os = "macos")]` and lives in `qwen_dense.rs`.
- **`router.rs`** depends only on `governor::SpecGovernor` + `std` — no Metal, no tensors, like `governor.rs`.
- KV bookkeeping (`self.kv.seq_len = …`) stays in the **engine**, never in `Verifier` — it is engine-specific (QwenDense write-forward vs DeepSeek rollback).

---

## 2. Consolidated, mutually-consistent Rust skeletons

The four designs had three reconcilable inconsistencies, resolved here:
1. **`Telemetry` split.** Spec `Telemetry` had both inputs (ns timings) and router-owned state. **Resolution:** the loop hands the router a per-cycle `StepObservation` (measured inputs) + a `RouterCtx` (target/context signals); the router *owns* the EWMA `CostModel`. `Telemetry` in `proposal.rs` is the read-only snapshot a proposer may consult (Phase-1 proposers ignore it).
2. **`Ctx` hidden fields.** Two variants (`residual`/`intermediate` as separate `Option`s vs a `HiddenTap` sub-struct). **Resolution:** use `HiddenTap<'a>` — keeps the two streams paired (they are `None`/`Some` together) and carries `start_token`.
3. **`Proposal` width.** One design had 6 variants (`#[non_exhaustive]`), another only `TokenLine`. **Resolution:** define `TokenLine` + `TokenLineWithLogits` + the reserved `TokenTree` now (so verifier dispatch is written once), comment out the text/retrieval variants until their phases. No `#[non_exhaustive]` — let the compiler enforce total matches during Phase 0/1.

### 2a. `proposal.rs` — contracts

```rust
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
        Budget { k: k.min(Self::MAX_DRAFT_LEN) }
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
    // --- reserved (uncomment with their phase; each adds a match arm everywhere) ---
    // TextSpan(String),
    // CrossTokenizerSpan { text: String, src_vocab: String },
    // RetrievalSpan { tokens: Vec<u32>, source: String },
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
    fn requires_hidden(&self) -> bool { false }

    /// Needs detok→retok text bridge (cross-tokenizer)? Base proposers: false.
    fn requires_text_bridge(&self) -> bool { false }

    /// Predicted draft cost (ns) — the router's draft_ns input. Base proposers ~0.
    fn cost_estimate(&self, _ctx: &Ctx<'_>, _budget: Budget) -> CostNs { 0 }

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
```

### 2b. `user_ngram.rs` — base adapter (appended; inherent API untouched)

```rust
use crate::speculate::proposal::{Budget, Ctx, Proposal, Proposer, Telemetry};

/// Default-on, lossless, tokenizer-native base proposer: a thin Proposer adapter
/// over UserNgramDraft (the live τ=1.43 base that beat the trained head). Newtype
/// so adapter-only state (cost prior) can be added later without touching the
/// inherent API (its determinism/tie-break is load-bearing for parity).
#[derive(Debug, Default)]
pub struct NgramProposer {
    inner: UserNgramDraft,
}

impl NgramProposer {
    pub fn new() -> Self { Self { inner: UserNgramDraft::new() } }
    pub fn index(&self) -> &UserNgramDraft { &self.inner }
}

impl Proposer for NgramProposer {
    fn name(&self) -> &'static str { "user_ngram" }

    fn propose(&mut self, ctx: &Ctx<'_>, budget: Budget, _tel: &Telemetry) -> Proposal {
        // UserNgramDraft::propose is &self, reads only the last two ids, may
        // return < k on a chain miss. Caller already clamped budget.k to ≤7.
        Proposal::TokenLine(self.inner.propose(ctx.last_two(), budget.k))
    }

    fn observe(&mut self, emitted: &[u32]) {
        // MANDATORY: grow the index from the emitted (verifier) stream. One
        // note_token per token, in order — identical to the live 'ud_loop feed.
        for &t in emitted { self.inner.note_token(t); }
    }

    fn warm(&mut self, history: &[u32]) { self.inner.warm_start(history); }
    fn reset(&mut self) { self.inner.reset_context(); } // cursor only — keep grams
}
```

### 2c. `verifier.rs` — exact target path (the lossless core)

```rust
//! Event Horizon — the exact target verifier. Every token returned is an argmax
//! of the target model; worst case a proposer is wrong and we fall back to one
//! greedy token. Output is bit-identical to plain greedy (Phase 0, temp==0).
//! Adds NO model math — wraps forward_tokens_verify + forward_token_greedy_tcb.
//! KV bookkeeping stays with the caller (returns the position math via next_seq_len).

use crate::speculate::shared::{verify_draft_ids_until_mismatch, VerifyResult};
use crate::Result;

/// The only thing the verifier needs from a model. QwenDense is the Phase-0
/// implementor; DeepSeek-V2 can impl the same over its rollback KV later.
pub trait ExactTarget {
    /// Batched linear verify: `tokens` at contiguous `positions` in one TCB →
    /// (argmax_per_pos, residual_per_pos). b == tokens.len() must be 1..=8 for the
    /// fast path. Wraps QwenDense::forward_tokens_verify (qwen_dense.rs:9004).
    fn forward_tokens_verify(
        &mut self, tokens: &[u32], positions: &[usize],
    ) -> Result<(Vec<u32>, Vec<Vec<f32>>)>;

    /// Single greedy/bonus token; writes KV[pos], returns argmax. Wraps
    /// QwenDense::forward_token_greedy_tcb (qwen_dense.rs:4456).
    fn forward_token_greedy(&mut self, token: u32, pos: usize) -> Result<u32>;

    /// Phase-6: ancestor-mask tree verify? False until the Metal build lands.
    fn supports_tree_verify(&self) -> bool { false }
}

/// Result of one exact verify pass. accepted ids are argmax-confirmed;
/// correction is the target's argmax at the first divergence (None ⇒ full accept).
#[derive(Debug, Clone, Default)]
pub struct VerifyOutcome {
    pub accepted: Vec<u32>,
    pub correction: Option<u32>,
    /// KV length the caller sets seq_len to before the next cycle:
    /// reject  ⇒ bonus_pos + accepted.len() + 1 (correction slot)
    /// accept  ⇒ bonus_pos + draft.len()
    pub next_seq_len: usize,
    /// Per-position residuals (EAGLE hidden tap). Empty unless want_residuals,
    /// so the n-gram base pays zero copy cost.
    pub residuals: Vec<Vec<f32>>,
}

/// Stateless-per-call verifier. Configured once per request.
#[derive(Debug, Clone)]
pub struct Verifier {
    pub max_batch: usize,      // forward_tokens_verify fast-path cap (8)
    pub want_residuals: bool,  // fill VerifyOutcome::residuals (hidden tap); off for n-gram
}
impl Default for Verifier {
    fn default() -> Self { Self { max_batch: 8, want_residuals: false } }
}

impl Verifier {
    pub fn new(max_batch: usize, want_residuals: bool) -> Self {
        Self { max_batch: max_batch.clamp(1, 8), want_residuals }
    }

    /// THE single home for the accept rule (retires the inline copy at
    /// qwen_dense.rs:2632). Bit-identical to the inline loop by construction:
    /// same vtoks = [bonus, draft[0..k-1]], same preds[i]==draft[i] test.
    pub fn verify_line<T: ExactTarget>(
        &self, target: &mut T, bonus: u32, bonus_pos: usize, draft: &[u32],
    ) -> Result<VerifyOutcome> {
        // Degenerate: empty draft → one plain greedy bonus step (still lossless).
        if draft.is_empty() {
            let corr = target.forward_token_greedy(bonus, bonus_pos)?;
            return Ok(VerifyOutcome {
                accepted: Vec::new(), correction: Some(corr),
                next_seq_len: bonus_pos + 1, residuals: Vec::new(),
            });
        }
        // Clamp bonus + draft ≤ max_batch.
        let k = draft.len().min(self.max_batch.saturating_sub(1));
        let draft = &draft[..k];

        let mut vtoks = Vec::with_capacity(k);
        vtoks.push(bonus);
        if k > 1 { vtoks.extend_from_slice(&draft[..k - 1]); }
        let vpos: Vec<usize> = (0..k).map(|j| bonus_pos + j).collect();

        let (preds, residuals) = target.forward_tokens_verify(&vtoks, &vpos)?;
        debug_assert_eq!(preds.len(), k);

        let VerifyResult { accepted_count, first_divergent_token } =
            verify_draft_ids_until_mismatch(draft, |i| Ok(preds[i]))?;

        let accepted = draft[..accepted_count].to_vec();
        let next_seq_len = if first_divergent_token.is_some() {
            bonus_pos + accepted_count + 1
        } else {
            bonus_pos + k
        };
        Ok(VerifyOutcome {
            accepted,
            correction: first_divergent_token,
            next_seq_len,
            residuals: if self.want_residuals { residuals } else { Vec::new() },
        })
    }
}
```

`impl ExactTarget for QwenDense` (lives in `qwen_dense.rs`, macOS-only — UFCS to avoid self-recursion):

```rust
#[cfg(target_os = "macos")]
impl crate::speculate::verifier::ExactTarget for QwenDense {
    fn forward_tokens_verify(&mut self, tokens: &[u32], positions: &[usize])
        -> Result<(Vec<u32>, Vec<Vec<f32>>)>
    { QwenDense::forward_tokens_verify(self, tokens, positions) }      // :9004
    fn forward_token_greedy(&mut self, token: u32, pos: usize) -> Result<u32>
    { QwenDense::forward_token_greedy_tcb(self, token, pos) }          // :4456
    // supports_tree_verify() stays default false until Phase 6.
}
```

### 2d. `router.rs` — wall-clock arbiter (v1)

```rust
//! Wall-clock-optimizing proposal router. Generalizes SpecGovernor from one
//! optional accept-rate gate to N per-proposer hysteresis machines under a
//! wall-clock expected_gain arbiter. Pure CPU logic — the loop feeds it measured
//! ns and it returns a plan. Losslessness is independent of the router; it only
//! chooses whether/how much to propose.

use super::governor::SpecGovernor;

pub const MAX_VERIFY_BATCH: usize = 8;
pub const MAX_DRAFT_LEN: usize = MAX_VERIFY_BATCH - 1; // 7, matches k_la cap

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProposerId { UserNgram, SuffixArray, Eagle5, Rest, CrossTokenizer }
impl ProposerId {
    pub fn as_str(self) -> &'static str {
        match self {
            ProposerId::UserNgram => "user_ngram",
            ProposerId::SuffixArray => "suffix_array",
            ProposerId::Eagle5 => "eagle5",
            ProposerId::Rest => "rest",
            ProposerId::CrossTokenizer => "cross_tokenizer",
        }
    }
}

/// Per-cycle measurements the loop hands back after each verify.
#[derive(Debug, Clone, Copy, Default)]
pub struct StepObservation {
    pub accepted: usize,         // na / first_reject (qwen_dense.rs:2632)
    pub drafted: usize,          // draft_len
    pub draft_ns: u64,           // wrap propose()
    pub verify_extra_ns: u64,    // B-token verify minus the 1 fwd you'd run anyway
    pub retokenize_ns: u64,      // 0 for token-native proposers
    pub sync_ns: u64,            // GPU submit/commit/wait
}

/// Target/context signals the loop fills each step.
#[derive(Debug, Clone, Copy)]
pub struct RouterCtx {
    pub target_ns_per_token: f32, // value an accepted draft token SAVES (small on
                                  // fast Qwen-3B → auto-kills neural spec there)
    pub context_confidence: f32,  // [0,1]; higher ⇒ longer draft (EAGLE-2 length)
    pub hidden_available: bool,   // gates any requires_hidden proposer this step
}

#[derive(Debug, Clone, PartialEq)]
pub enum RouterPlan {
    NoSpec,                                          // plain single-token greedy
    Spec { id: ProposerId, draft_len: usize, tree_width: usize },
}

#[derive(Debug, Clone, Copy)]
struct CostModel {
    ewma_accept_len: f32, ewma_draft_ns: f32, ewma_verify_extra_ns: f32,
    ewma_retok_ns: f32, ewma_sync_ns: f32, ewma_hit_frac: f32, seen: u64,
}
impl CostModel {
    fn new() -> Self {
        Self { ewma_accept_len: 1.0, ewma_draft_ns: 0.0, ewma_verify_extra_ns: 0.0,
               ewma_retok_ns: 0.0, ewma_sync_ns: 0.0, ewma_hit_frac: 1.0, seen: 0 }
    }
    fn update(&mut self, o: &StepObservation, alpha: f32) {
        let mix = |old: f32, new: f32| old + alpha * (new - old);
        self.ewma_accept_len = mix(self.ewma_accept_len, o.accepted as f32);
        self.ewma_draft_ns = mix(self.ewma_draft_ns, o.draft_ns as f32);
        self.ewma_verify_extra_ns = mix(self.ewma_verify_extra_ns, o.verify_extra_ns as f32);
        self.ewma_retok_ns = mix(self.ewma_retok_ns, o.retokenize_ns as f32);
        self.ewma_sync_ns = mix(self.ewma_sync_ns, o.sync_ns as f32);
        let hit = if o.drafted > 0 { o.accepted as f32 / o.drafted as f32 } else { 0.0 };
        self.ewma_hit_frac = mix(self.ewma_hit_frac, hit);
        self.seen += 1;
    }
}

struct Slot {
    id: ProposerId,
    gov: SpecGovernor,
    cost: CostModel,
    requires_hidden: bool,
    requires_text_bridge: bool,
    /// true only after replay_oracle verdict == "GO" (τ≥2.5) on the target
    /// workload. n-gram base = true unconditionally; any requires_hidden slot
    /// stays false until gated. THE KILL-LEDGER RULE IN CODE.
    oracle_cleared: bool,
}

pub struct ProposalRouter { slots: Vec<Slot>, alpha: f32, margin_ns: f32 }

impl ProposalRouter {
    /// Build with the always-on n-gram base. Neural/cross slots via enable_neural_slot.
    pub fn new(window: usize, min_accept_rate: f32, margin_ns: f32) -> Self {
        let base = Slot {
            id: ProposerId::UserNgram, gov: SpecGovernor::new(window, min_accept_rate),
            cost: CostModel::new(), requires_hidden: false, requires_text_bridge: false,
            oracle_cleared: true,
        };
        Self { slots: vec![base], alpha: 0.10, margin_ns }
    }

    /// Register a gated proposer. REFUSES any hidden/text-bridge slot whose
    /// offline oracle verdict is not "GO". oracle_verdict = ReplayReport::verdict().
    pub fn enable_neural_slot(
        &mut self, id: ProposerId, window: usize, min_accept_rate: f32,
        requires_hidden: bool, requires_text_bridge: bool, oracle_verdict: &str,
    ) -> crate::Result<()> {
        if (requires_hidden || requires_text_bridge) && oracle_verdict != "GO" {
            return Err(crate::Error::Model(
                "gated proposer denied: oracle verdict not GO (tau<2.5)".into()));
        }
        self.slots.push(Slot {
            id, gov: SpecGovernor::new(window, min_accept_rate), cost: CostModel::new(),
            requires_hidden, requires_text_bridge, oracle_cleared: true,
        });
        Ok(())
    }

    fn expected_gain_ns(&self, slot: &Slot, ctx: &RouterCtx, planned_len: usize) -> f32 {
        let c = &slot.cost;
        let e_accepted = c.ewma_accept_len.min(planned_len as f32);
        let benefit = e_accepted * ctx.target_ns_per_token;
        let cost = c.ewma_draft_ns + c.ewma_verify_extra_ns + c.ewma_retok_ns + c.ewma_sync_ns;
        benefit - cost
    }

    fn plan_shape(&self, slot: &Slot, ctx: &RouterCtx) -> (usize, usize) {
        let conf = (ctx.context_confidence * slot.cost.ewma_hit_frac).clamp(0.0, 1.0);
        let len = 1 + ((MAX_DRAFT_LEN - 1) as f32 * conf).round() as usize;
        (len.clamp(1, MAX_DRAFT_LEN), 1) // tree_width=1 until Phase 6
    }

    /// Two-tier: (1) governor health gate, (2) wall-clock arbiter — max positive
    /// expected_gain - margin among healthy slots. None clears ⇒ NoSpec.
    pub fn plan(&self, ctx: &RouterCtx) -> RouterPlan {
        let mut best: Option<(ProposerId, usize, usize, f32)> = None;
        for slot in &self.slots {
            if !slot.oracle_cleared { continue; }
            if slot.requires_hidden && !ctx.hidden_available { continue; }
            if !slot.gov.is_enabled() { continue; }
            let (draft_len, tree_width) = self.plan_shape(slot, ctx);
            let gain = self.expected_gain_ns(slot, ctx, draft_len);
            if gain <= self.margin_ns { continue; }
            let score = gain - self.margin_ns;
            if best.map_or(true, |(_, _, _, bs)| score > bs) {
                best = Some((slot.id, draft_len, tree_width, score));
            }
        }
        match best {
            Some((id, draft_len, tree_width, _)) =>
                RouterPlan::Spec { id, draft_len, tree_width },
            None => RouterPlan::NoSpec,
        }
    }

    /// Feed back the cycle that ran: update EWMA + step the slot's governor
    /// (the existing g.step(na>0) contract).
    pub fn record(&mut self, id: ProposerId, o: &StepObservation) {
        let alpha = self.alpha;
        if let Some(slot) = self.slots.iter_mut().find(|s| s.id == id) {
            slot.cost.update(o, alpha);
            slot.gov.step(o.accepted > 0);
        }
    }

    /// Preserve the "keep observing while disabled" contract (qwen_dense.rs:2609):
    /// feed a skipped slot a pessimistic false so cooldown counts down.
    pub fn observe_disabled(&mut self, id: ProposerId) {
        if let Some(slot) = self.slots.iter_mut().find(|s| s.id == id) {
            slot.gov.step(false);
        }
    }

    pub fn accept_rate(&self, id: ProposerId) -> Option<f32> {
        self.slots.iter().find(|s| s.id == id).map(|s| s.gov.accept_rate())
    }
}
```

**Ownership boundary (the key consolidation):** the router holds **no proposer handle** — `Slot` carries only the governor + cost model + flags. The concrete `NgramProposer`/`Eagle5Head` instances live on the engine (Eagle5 is Metal-coupled, `#[cfg(macOS)]`, pinned buffers — it fights `dyn`). The loop indexes its own proposers by `ProposerId`. This keeps the router Metal-free and unit-testable.

---

## 3. Exact engine `generate()` seam (file:line from recon)

**Target:** `crates/hawking-core/src/model/qwen_dense.rs`, the `'ud_loop` bonus-first n-gram loop — the bit-identical parity reference. The seam replaces the inline accept block, keeping Stage-1 (bonus emit) and KV bookkeeping in the engine.

| Live site | What it does today | Phase-0 replacement |
|---|---|---|
| `:2563` `step_start = Instant::now()` | loop timing | keep; also wrap `propose` + `verify_line` for `draft_ns`/`verify_extra_ns` |
| `:2569` `forward_token_greedy_tcb(last_id, pos)` | Stage-1 bonus | **unchanged** (engine-owned) |
| `:2576` `draft_index.note_token(bonus)` | feed bonus | becomes `proposer.observe(&[bonus])` |
| `:2604-2612` `gov_propose` consult + `draft_index.propose` | governor + propose | `router.plan(&rctx)` → `proposer.propose(&ctx, Budget::line(k_avail), &tel)` |
| **`:2625-2640`** inline `vtoks`/`forward_tokens_verify`/accept scan | **two-copy accept rule** | **`verifier.verify_line(self, bonus, bonus_pos, &draft)?`** |
| `:2641-2647` `stats.draft_*` + `g.step` | counters + governor | `router.record(id, &obs)` + same `stats.draft_*` from `outcome` |
| `:2650-2655` `usage_capture::record_draft` | L3.1 capture | **unchanged** |
| `:2660-2678` emit accepted loop | per accepted draft | iterate `outcome.accepted`; `proposer.observe(&[id])` per token |
| `:2688-2706` correction + `self.kv.seq_len = pos` | correction + KV | use `outcome.correction` + `outcome.next_seq_len`; **`self.kv.seq_len` stays engine-owned** |

Borrow-safety: copy out `bonus`/`bonus_pos`/`draft` (all `Copy`/owned slice) *before* the `verifier.verify_line(self, …)` `&mut self` borrow; the call returns the owned `VerifyOutcome` before `self.tokenizer`/`self.sampler`/`self.kv` are touched. The skeleton already copies `preds`/`residuals` into owned `Vec`s so nothing borrows the model past the call.

**The four loops collapse to one body:** `'udpf_loop` / `'pf_loop` are the `draft.len()==1` case of `verify_line`; the eagle paths differ only in the *proposer* (`propose_rollout_chained`) and in `verifier.want_residuals = true`. Phase 0 unifies `'ud_loop`/`'udpf_loop`; eagle paths fold in at Phase 4.

---

## 4. Ordered, compilable implementation checklist (smallest compiling step first)

Each step compiles + passes `cargo test -p hawking-core` before the next. Everything is behind `HAWKING_QWEN_EVENT_HORIZON` (default OFF) until Step P0.7 — the live `'ud_loop` is the untouched fallback and parity reference throughout.

### Phase 0 — unify, behind a flag, n-gram only

- **P0.1 — Types compile standalone.** Add `proposal.rs` (§2a) + `pub mod proposal;` to `mod.rs`. Depends only on `shared::DraftToken`. `cargo build -p hawking-core` green; no behavior change.
- **P0.2 — n-gram adapter.** Append `NgramProposer` + `impl Proposer` (§2b) to `user_ngram.rs`. Unit test: `warm` → `propose` → `observe` reproduces a known draft from `replay_oracle`'s corpus. Inherent `UserNgramDraft` API untouched.
- **P0.3 — Linear verifier.** Add `verifier.rs` (§2c) + `pub mod verifier;`. Pure CPU. Unit test with a **mock `ExactTarget`** (a `Vec<u32>` of canned preds): assert `verify_line` reproduces accept/correction/`next_seq_len` for full-accept, mid-reject, and empty-draft cases. No Metal.
- **P0.4 — QwenDense ExactTarget impl.** Add the `#[cfg(macOS)] impl ExactTarget for QwenDense` (§2c) in `qwen_dense.rs`. UFCS-safe. Compiles on macOS; non-macOS unaffected (module is cfg-free).
- **P0.5 — Router v1.** Add `router.rs` (§2d) + `pub mod router;`. Unit tests: governor gate generalizes (N slots), `expected_gain` goes negative when `target_ns_per_token` is small (the Qwen-3B auto-kill), `enable_neural_slot` **rejects** a `requires_hidden` slot with verdict ≠ "GO". No Metal.
- **P0.6 — Wire verifier into `'ud_loop` (flag-gated).** Behind `HAWKING_QWEN_EVENT_HORIZON`, replace the inline accept block (`:2625-2640`) with `verifier.verify_line`. Keep Stage-1 bonus, `usage_capture`, `stats.draft_*`, and `self.kv.seq_len` engine-owned. **Parity gate:** with the flag ON, 16/64-token greedy output must be **bit-identical** to the flag-OFF `'ud_loop` (this is the whole losslessness claim — verify in `main` yourself, never trust an agent's "parity passed").
- **P0.7 — Wire router into the loop (flag-gated).** Replace the `gov_propose` consult with `router.plan(&rctx)` → `NoSpec`/`Spec{draft_len}`; feed `router.record`/`observe_disabled`. Add the `Instant::now()` spans for `draft_ns`/`verify_extra_ns`/`sync_ns` (promote the `HAWKING_QWEN_VERIFY_TIMING` probe at `:9020`/`:9056` from `eprintln` to a returned number). Parity still bit-identical (router only gates *whether* to propose). Bench: confirm no regression vs `'ud_loop` at `target_ns_per_token` typical of Qwen-3B.

### Phase 1 — base proposer market (still no neural)

- **P1.1 — Telemetry → GenStats.** Roll per-proposer accept/reject + `accept_rate(id)` into `engine.rs::GenStats::stats_json()` (`:259`). Diagnostic only; no behavior change.
- **P1.2 — Warm-start sources.** Route prompt/session/repo through `Proposer::warm` (repo → session → prompt order). Engine holds `NgramProposer` across requests for session persistence; `reset()` (cursor only) between requests, **never `clear()`**.
- **P1.3 — SuffixArrayDraft (second base proposer).** Add the rolling-window exact-match copier (`impl Proposer`, `name()=="suffix_array"`, `CostNs` free) for long exact recurrences the bigram misses. Register as a second always-on `oracle_cleared` slot. **Must** add the hash-collision guard (`stream[prior_start..+h] == tail`) before copying — still lossless without it, but wastes cycles.
- **P1.4 — Two-proposer arbitration.** Router picks max `expected_gain` between n-gram and suffix each step (both `CostNs` free; suffix-first, n-gram fallback is the natural composition). Bench the market vs single n-gram on code/JSON/agent-loop corpora.
- **P1.5 — Flip the flag default** once P0.6 parity + P1.4 bench both hold on the target workload. The legacy `'ud_loop` stays compiled as the parity oracle.

---

## 5. Kill-ledger guardrails (hard constraints, enforced in code)

Per `docs/dead_levers.md`: the trained EAGLE-3-like head is **NET-NEGATIVE** on Qwen-3B+code (0.40×/0.30×/0.21× at K=2/4/8; device accept 6.5% vs 52% offline ≈ 8× forward-parity gap; τ=0.877 vs gate τ≥2.5). The free n-gram (τ=1.43) **beat it**.

1. **n-gram (+ suffix) base only through Phase 1.** `ProposalRouter::new` seeds exactly the `UserNgram` slot with `oracle_cleared: true`. `SuffixArray` is the only Phase-1 addition (also model-free, `oracle_cleared: true`). **No `Eagle5Head` is constructed or wired in Phase 0/1.**
2. **Trained head stays behind the τ≥2.5 oracle, enforced at construction.** `enable_neural_slot` returns `Err(Error::Model("...tau<2.5"))` for any `requires_hidden`/`requires_text_bridge` slot whose `oracle_verdict != "GO"`. The verdict comes from `replay_oracle::ReplayReport::verdict()` (`replay_oracle.rs:96`, GO≥2.5) run **offline on the target workload first**. This is the kill-ledger rule as a compile-enforced precondition — there is no code path to enable a neural proposer without a GO.
3. **The router IS the novelty / the safety.** On a fast target (small `target_ns_per_token`), `expected_gain_ns` goes negative for any proposer whose `draft_ns` is non-trivial → `plan()` returns `NoSpec` for that slot. This is exactly the behavior that *would have disabled the old head* on Qwen-3B. The router makes a neural slot **safe to even wire** (default-OFF, gain-gated, hidden-coupled) — never a reason to enable it.
4. **Losslessness is structural and flag-independent.** Every emitted token is a verifier argmax token. The router/governor only choose *whether/how much to propose*, never *what* is emitted (the invariant the live `'ud_loop` already guarantees, `:2657`). The P0.6 bit-identical parity gate is the proof obligation — non-negotiable before flipping any default.
5. **Greedy-only invariant preserved.** All spec paths hard-require `temperature==0` + `repetition_penalty==1.0` (`qwen_dense.rs:1375-1387`). `verify_line` argmaxes; no sampling. Lossless speculative *sampling* (resample-on-reject) is **not** Phase 0/1.

---

## 6. Compile-risk + open-question list

**Compile risks (code-grounded):**
1. **`crate::Result`/`crate::Error::Model(String)`** — confirmed shape at `shared.rs:9,43`. Router's `enable_neural_slot` must return `crate::Result<()>` (not `&'static str`) to compose with `?` at the call site. ✅ resolved in §2d.
2. **`UserNgramDraft::propose` is `&self`, trait is `&mut self`** — calling `&self` through `&mut self` compiles (mutates nothing). No inherent-API change. ✅ low.
3. **UFCS in `impl ExactTarget for QwenDense`** — `QwenDense::forward_tokens_verify(self, …)` (qualified) or it self-recurses on the same-named trait method. ✅ resolved.
4. **Borrow conflict at the call site** — `verify_line(self, …)` takes `&mut self` while the loop also touches `self.tokenizer`/`self.sampler`/`self.kv`. Sequential is fine; copy `bonus`/`bonus_pos`/`draft` out first (all `Copy`/owned). ⚠️ care, not a type error.
5. **`preds[i]` OOB** — closure indexes `preds` over `0..draft.len()`; `forward_tokens_verify` returns exactly `b` preds (`:9068`). `debug_assert_eq!(preds.len(), k)` added. ✅ low.
6. **`SpecGovernor` is `Clone` not `Copy`** (`VecDeque`) — fine inside owned `Slot`; don't require `Copy` on `Slot`/`ProposalRouter`. ✅.
7. **`Proposal::TokenTree` has no verifier** — constructing one hits no engine path in Phase 0/1 (router only emits `draft_len` lines, `tree_width==1`). Inert until Phase 6. ✅.
8. **`HiddenTap`/`Telemetry` fields unused in Phase 1** → `unused` warnings. `#[allow(dead_code)]` on the structs or leave until Eagle5. ⚠️ warning-only.
9. **`Budget.k` ceiling is the caller's contract** — `propose` doesn't re-clamp; the wiring site must pass `k_avail = user_draft_k.min(remaining).min(8)` as today (`:2592`), and `Budget::line` clamps to 7. ⚠️ caller-contract.
10. **No `serde` on the new types** — if `Telemetry`/router stats need to land in `stats_json()`, add `#[derive(Serialize)]` (crate has `serde_json`). ⚠️ Phase 1.1.

**Open questions (need a measurement or a decision):**
1. **`context_confidence` source.** `forward_tokens_verify` returns residuals, not a logit vector — an entropy term forces an extra LM-head pass. **Proposal:** start with a cheap proxy (recent accepted-run length, already tracked via `ewma_hit_frac`); don't wire entropy until measured to pay off.
2. **`target_ns_per_token` measurement.** The auto-kill lever. Needs a per-step measured target-forward cost; the `HAWKING_QWEN_VERIFY_TIMING` probe (`:9020`) is the seam, but it's currently debug-only. Promote it to a returned number (P0.7). **Open:** smoothing window for a stable estimate.
3. **`alpha=0.10` / `margin_ns` values** — placeholders, must be tuned on the Metal bench matrix. The skeleton compiles; the constants are guesses.
4. **`sync_ns` attribution** — the research flags GPU/CPU sync as the likely gain-eroder, but no clean per-step sync timer exists today. **Open:** whether `verify_extra_ns` already absorbs it or it needs a separate `enc.commit→wait` span.
5. **DeepSeek `ExactTarget`** — uses explicit KV rollback, not write-forward (`deepseek_v2.rs:~1238`). The `ExactTarget` trait admits it (additive impl), but `next_seq_len` semantics differ (rollback to `draft_start + first_reject + 1`). Phase 0/1 is QwenDense-only; flag for the cross-engine phase.
6. **Serve-side dead `SpecGovernor`** (`hawking-serve/src/spec_gov.rs`, 0 consult sites, worse flapping semantics). Recommend **delete** and route serve through the core governor/router — but that's a separate cleanup, not a Phase 0/1 blocker.
7. **Suffix-array `H` (window length)** — default 3 for code/JSON is a guess; the precision/recall tradeoff (longer H ⇒ fewer hits, more precise) needs the same corpus bench as P1.4.

**Files to create:** `crates/hawking-core/src/speculate/{proposal.rs, verifier.rs, router.rs}`.
**Files to edit:** `crates/hawking-core/src/speculate/mod.rs` (+3 `pub mod`), `crates/hawking-core/src/speculate/user_ngram.rs` (append `NgramProposer`), `crates/hawking-core/src/model/qwen_dense.rs` (`impl ExactTarget` + flag-gated `'ud_loop` seam at `:2563-2706`).
**Reused unchanged:** `shared.rs` (`verify_draft_ids_until_mismatch`), `governor.rs` (`SpecGovernor`), `replay_oracle.rs` (`verdict()` τ≥2.5 gate).