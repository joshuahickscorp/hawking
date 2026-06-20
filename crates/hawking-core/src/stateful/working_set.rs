//! L1.1 — KV cache as a living working set.
//!
//! Bible §8.1 L1.1. Attention is sparse — most cached tokens contribute
//! almost nothing to the current token. Rank cached tokens by importance
//! and **evict or compress** the low-value ones, keeping the KV at a
//! **bounded working-set size** instead of a linearly-growing blob. This
//! turns whole-file/codebase context from choking at ~32K into running
//! at 200K+ in the same 18 GB, and cuts KV-read bandwidth at long
//! context. Energy verdict: **GENUINE at long context**, NEUTRAL short.
//!
//! Exactness: **Q** (approximate attention; bounded, tunable by
//! working-set size) — **with a lossless-mode escape hatch**
//! ([`WorkingSetMode::Lossless`] / [`LosslessPolicy`]) that keeps all
//! positions for correctness-critical / greedy-exact runs.
//!
//! # Status: INTERFACES ONLY — all policy/working-set bodies are
//! `todo!()`.
//!
//! Build the bodies only after the **attention-mass concentration**
//! oracle shows that a small bounded position set captures ≥99% of
//! attention mass per layer on Qwen2.5-3B, with a tolerable
//! quality-vs-budget curve. If mass is spread broadly the lever dies on
//! the oracle (same discipline that killed block-256 FFN sparsity). The
//! [`LosslessPolicy`] ships regardless — it is the no-op escape hatch and
//! needs no oracle. See `plans/stateful_core_design_2026_05_30.md` §2.5.
//!
//! # Where this hooks in (described, not wired)
//!
//! The working set sits at the KV append site in
//! `QwenDense::forward_token` (`crates/hawking-core/src/model/qwen_dense.rs`
//! ~line 2316, where `kv_off = kv.seq_len * stride` and `seq_len` bumps
//! after the layer loop). A future wiring would feed attention scores to
//! [`KvEvictionPolicy::observe_attention`] during the MHA compute and
//! apply the [`EvictionPlan`] to compact the [`crate::cache::KvCache`]
//! arenas. This module does **not** touch that path.

/// Per-(layer, query) attention signal handed to a policy so it can
/// accumulate its importance statistic. Borrowed for the duration of the
/// call; the policy copies out only what it needs (a running sum, a
/// pooled window), never the whole map.
///
/// `scores[j]` is the attention weight the current query position placed
/// on cached key position `j` (post-softmax, one head or head-averaged —
/// an implementation choice the body fixes). `len()` == the current
/// retained-position count for this layer.
#[derive(Debug, Clone, Copy)]
pub struct AttentionScores<'a> {
    /// Post-softmax attention weights over cached positions, in
    /// retained-position order.
    pub weights: &'a [f32],
}

impl<'a> AttentionScores<'a> {
    pub fn len(&self) -> usize {
        self.weights.len()
    }
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }
}

/// What to do with one retained KV position when the budget is exceeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionAction {
    /// Drop the position entirely (pure bookkeeping; GPU-favorable).
    Drop,
    // FUSED-QKV KERNEL COUPLING (later) — design doc §2.4.
    /// Keep the position but re-encode its K/V at lower precision using
    /// an existing quant codec, to be read inline by the future fused
    /// quantized-KV attention kernel (the mlx-qsdpa pattern). The codec
    /// choice is carried out-of-band by the working set's
    /// `KvCompressionCodec` association point (not in this stub).
    Compress,
}

/// A policy's decision for one layer: which retained positions to act on.
/// Positions are in retained-position order (the same order
/// [`AttentionScores`] uses). An empty plan = keep everything (what
/// [`LosslessPolicy`] always returns).
#[derive(Debug, Clone, Default)]
pub struct EvictionPlan {
    /// `(position_index, action)` pairs. **Invariants the body must
    /// uphold:** never includes a protected sink/recent-window position;
    /// applying the plan leaves the retained count within budget.
    pub actions: Vec<(usize, EvictionAction)>,
}

impl EvictionPlan {
    /// The keep-everything plan.
    pub fn keep_all() -> Self {
        Self {
            actions: Vec::new(),
        }
    }
    pub fn is_keep_all(&self) -> bool {
        self.actions.is_empty()
    }
}

/// Read-only context a policy sees when computing an [`EvictionPlan`].
#[derive(Debug, Clone, Copy)]
pub struct WorkingSetCtx {
    /// Current retained-position count for this layer.
    pub retained: usize,
    /// Target maximum retained positions per layer (the working-set size).
    pub budget_positions: usize,
    /// Absolute model position of the token currently being admitted
    /// (== pre-bump `kv.seq_len` at the append site).
    pub current_pos: usize,
}

/// The KV eviction policy — the core L1.1 abstraction. Swappable so
/// StreamingLLM / H2O / SnapKV (and the lossless no-op) all live behind
/// one interface.
///
/// Lifecycle per token, per layer (described; wiring is future work):
/// `on_admit` → `observe_attention` (during MHA) → `select_evictions`
/// (only when admitting would exceed the budget).
pub trait KvEvictionPolicy {
    /// Stable name for diagnostics / config (e.g. `"streaming-llm"`).
    fn name(&self) -> &'static str;

    /// Called when a new token position is admitted into the working set.
    /// Lets positional policies (StreamingLLM) record sink/recent status.
    fn on_admit(&mut self, pos: usize, layer: usize);

    /// Called with the post-softmax attention the current query placed
    /// over cached positions, so cumulative/pooled policies (H2O, SnapKV)
    /// can update their statistic. Positional policies ignore it.
    fn observe_attention(&mut self, layer: usize, query_pos: usize, scores: &AttentionScores);

    /// Decide which retained positions to drop/compress for `layer` to
    /// stay within `ctx.budget_positions`. Returns [`EvictionPlan::keep_all`]
    /// when nothing should be evicted.
    fn select_evictions(&mut self, layer: usize, ctx: &WorkingSetCtx) -> EvictionPlan;

    /// `true` only for the keep-everything escape hatch. The working set
    /// uses this to assert the **E (greedy-lossless)** guarantee when
    /// running in [`WorkingSetMode::Lossless`].
    fn is_lossless(&self) -> bool {
        false
    }
}

/// StreamingLLM: keep the first `sinks` **attention-sink** positions plus
/// a `recent` trailing window; evict the middle. Purely positional —
/// ignores [`KvEvictionPolicy::observe_attention`].
#[derive(Debug, Clone, Copy)]
pub struct StreamingLlmPolicy {
    /// Number of leading attention-sink positions to pin.
    pub sinks: usize,
    /// Size of the trailing recent window to pin.
    pub recent: usize,
}

impl StreamingLlmPolicy {
    pub fn new(sinks: usize, recent: usize) -> Self {
        Self { sinks, recent }
    }
}

impl KvEvictionPolicy for StreamingLlmPolicy {
    fn name(&self) -> &'static str {
        "streaming-llm"
    }
    fn on_admit(&mut self, _pos: usize, _layer: usize) {
        todo!("record recent-window membership")
    }
    fn observe_attention(&mut self, _layer: usize, _query_pos: usize, _scores: &AttentionScores) {
        // Positional policy: attention signal is irrelevant.
    }
    fn select_evictions(&mut self, _layer: usize, _ctx: &WorkingSetCtx) -> EvictionPlan {
        todo!("drop positions outside [0..sinks) ∪ recent-window")
    }
}

/// H2O (Heavy-Hitter Oracle): keep the recent window plus the
/// **heavy hitters** ranked by *cumulative attention mass* (a running
/// per-position sum updated in [`KvEvictionPolicy::observe_attention`]).
#[derive(Debug, Clone, Copy)]
pub struct H2OPolicy {
    /// Size of the always-kept recent window.
    pub recent: usize,
    /// Number of heavy-hitter positions to keep beyond the recent window.
    pub heavy: usize,
}

impl H2OPolicy {
    pub fn new(recent: usize, heavy: usize) -> Self {
        Self { recent, heavy }
    }
}

impl KvEvictionPolicy for H2OPolicy {
    fn name(&self) -> &'static str {
        "h2o"
    }
    fn on_admit(&mut self, _pos: usize, _layer: usize) {
        todo!("initialize cumulative-score accumulator slot for the new position")
    }
    fn observe_attention(&mut self, _layer: usize, _query_pos: usize, _scores: &AttentionScores) {
        todo!("add scores into the per-position cumulative-mass accumulator")
    }
    fn select_evictions(&mut self, _layer: usize, _ctx: &WorkingSetCtx) -> EvictionPlan {
        todo!("keep recent window + top-`heavy` by cumulative mass; drop the rest")
    }
}

/// SnapKV: keep a recent window plus positions selected by **pooled
/// importance** — max/avg-pool the attention map over a recent
/// observation window and keep the top-k. Compresses the prompt KV down
/// to its load-bearing positions.
#[derive(Debug, Clone, Copy)]
pub struct SnapKvPolicy {
    /// Size of the recent observation window used to compute importance.
    pub window: usize,
    /// Number of pooled-importance positions to keep.
    pub keep: usize,
    /// Pooling kernel width for smoothing the importance map.
    pub pool_kernel: usize,
}

impl SnapKvPolicy {
    pub fn new(window: usize, keep: usize, pool_kernel: usize) -> Self {
        Self {
            window,
            keep,
            pool_kernel,
        }
    }
}

impl KvEvictionPolicy for SnapKvPolicy {
    fn name(&self) -> &'static str {
        "snapkv"
    }
    fn on_admit(&mut self, _pos: usize, _layer: usize) {
        todo!("track recent observation-window membership")
    }
    fn observe_attention(&mut self, _layer: usize, _query_pos: usize, _scores: &AttentionScores) {
        todo!("accumulate the recent-window attention map for pooling")
    }
    fn select_evictions(&mut self, _layer: usize, _ctx: &WorkingSetCtx) -> EvictionPlan {
        todo!("pool importance over window; keep recent + top-`keep`; drop the rest")
    }
}

/// The lossless escape hatch: keep everything, evict nothing. Used when
/// [`WorkingSetMode::Lossless`] is selected for correctness-critical or
/// greedy-exact runs. Ships regardless of the L1.1 oracle (it is a no-op
/// over today's behavior). `is_lossless()` is `true`.
#[derive(Debug, Clone, Copy, Default)]
pub struct LosslessPolicy;

impl KvEvictionPolicy for LosslessPolicy {
    fn name(&self) -> &'static str {
        "lossless"
    }
    fn on_admit(&mut self, _pos: usize, _layer: usize) {
        // No bookkeeping: every position is kept forever.
    }
    fn observe_attention(&mut self, _layer: usize, _query_pos: usize, _scores: &AttentionScores) {
        // No statistic needed.
    }
    fn select_evictions(&mut self, _layer: usize, _ctx: &WorkingSetCtx) -> EvictionPlan {
        EvictionPlan::keep_all()
    }
    fn is_lossless(&self) -> bool {
        true
    }
}

/// Per-layer bound on retained KV positions — the working-set size. When
/// admitting a new token would exceed `max_positions`, the policy's
/// [`KvEvictionPolicy::select_evictions`] runs and the plan is applied
/// *before* the append, so steady-state cost is `O(max_positions)`, not
/// `O(seq_len)`.
#[derive(Debug, Clone, Copy)]
pub struct WorkingSetBudget {
    pub max_positions: usize,
}

impl WorkingSetBudget {
    pub fn new(max_positions: usize) -> Self {
        Self { max_positions }
    }
}

/// Whether the working set evicts (`Bounded`) or keeps everything
/// (`Lossless`, the **E** escape hatch). `Lossless` forces keep-all
/// semantics regardless of the configured policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkingSetMode {
    /// Eviction active — bounded RAM, Q (bounded by budget).
    Bounded,
    /// Keep all, no eviction — bit-identical to today's behavior (E).
    Lossless,
}

impl Default for WorkingSetMode {
    fn default() -> Self {
        // Default to the safe, lossless behavior; opting into eviction is
        // explicit and oracle-gated.
        Self::Lossless
    }
}

/// Owns the working-set budget + eviction policy and drives them over the
/// live KV. Wraps (does not replace) [`crate::cache::KvCache`]; a future
/// wiring applies the [`EvictionPlan`] to compact that cache's arenas.
///
/// Generic over the policy `P` so the policy is a zero-cost,
/// statically-dispatched choice; `KvWorkingSet<LosslessPolicy>` is the
/// no-op default.
#[derive(Debug)]
pub struct KvWorkingSet<P: KvEvictionPolicy> {
    policy: P,
    budget: WorkingSetBudget,
    mode: WorkingSetMode,
    // Retained-position bookkeeping (per-layer block tables / live-index
    // maps) is omitted from the interface stub — a body decision.
}

impl<P: KvEvictionPolicy> KvWorkingSet<P> {
    /// Build a working set with an explicit policy, budget, and mode.
    pub fn new(policy: P, budget: WorkingSetBudget, mode: WorkingSetMode) -> Self {
        Self {
            policy,
            budget,
            mode,
        }
    }

    pub fn policy(&self) -> &P {
        &self.policy
    }
    pub fn budget(&self) -> WorkingSetBudget {
        self.budget
    }
    pub fn mode(&self) -> WorkingSetMode {
        self.mode
    }

    /// `true` when this working set is guaranteed greedy-lossless — either
    /// the mode is [`WorkingSetMode::Lossless`] or the policy reports
    /// itself lossless. The future KV-append wiring asserts this for
    /// correctness-critical runs.
    pub fn is_lossless(&self) -> bool {
        self.mode == WorkingSetMode::Lossless || self.policy.is_lossless()
    }

    /// Called at the KV-append site once a new token's K/V have been
    /// computed for every layer (positionally, just before `seq_len`
    /// bumps). In `Bounded` mode this consults the policy and applies any
    /// [`EvictionPlan`]; in `Lossless` mode it is a no-op. Returns the
    /// plan that was applied per layer for diagnostics (empty in lossless
    /// mode).
    ///
    /// `layer_scores[l]` is the attention the current query placed over
    /// layer `l`'s retained positions (already computed by MHA).
    pub fn on_token_appended(
        &mut self,
        _current_pos: usize,
        _layer_scores: &[AttentionScores],
    ) -> Vec<EvictionPlan> {
        todo!(
            "lossless: no-op. bounded: per layer, observe_attention then \
             select_evictions when retained would exceed budget; \
             return the applied plans"
        )
    }
}

impl Default for KvWorkingSet<LosslessPolicy> {
    /// The safe default: lossless, effectively unbounded (the budget is
    /// inert in lossless mode). Equivalent to today's keep-all KV.
    fn default() -> Self {
        Self::new(
            LosslessPolicy,
            WorkingSetBudget::new(usize::MAX),
            WorkingSetMode::Lossless,
        )
    }
}
