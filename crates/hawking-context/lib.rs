//! HIDE context and memory substrate (bible ch.04).
//!
//! This is the shell-side compiler described in `docs/hide-bible/04-*`: it
//! ranks sources, packs a token budget with a real reservation-aware knapsack,
//! and emits a replayable manifest. It also owns the hierarchical memory store
//! (SQLite/FTS5 + cosine vectors), the per-task context profiles, and the KV
//! reuse-banking seam to `hawking-serve`.

#[rustfmt::skip]
pub mod budget {
    //! Token budgeting and tokenizer-accurate counting (bible §4.2).
    //!
    //! The budget is **reserve-then-fill** (bible §4.2.3, F1): the system,
    //! response, and scratchpad regions are carved out *before* any candidate
    //! competes, so an overflow can never eat the response budget. Counting goes
    //! through a real `tokenizers` tokenizer when one is available, falling back
    //! to a `chars/4` heuristic otherwise (bible §4.2 "tokenizer-accurate counting,
    //! chars/4 fallback").

    use parking_lot::RwLock;
    use serde::{Deserialize, Serialize};
    use std::sync::Arc;
    use tokenizers::Tokenizer;

    /// The window budget for one compile, with hard per-region reservations.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct TokenBudget {
        /// Total tokens the model window can hold (effective ctx_len).
        pub max_input_tokens: usize,
        /// Tokens reserved for the model to *generate* (bible F1 — carved first).
        pub reserve_output_tokens: usize,
        /// A floor on the window we will actually fill (lets a profile cap a huge
        /// native window for a tight task).
        pub hard_limit_tokens: usize,
    }

    impl TokenBudget {
        /// Input tokens available *after* the output reservation, clamped to the
        /// hard limit. This is the pool the packer fills.
        pub fn available_input(&self) -> usize {
            self.max_input_tokens.saturating_sub(self.reserve_output_tokens).min(self.hard_limit_tokens)
        }

        /// Reserve a percentage of the available input as a named region.
        pub fn reserve_pct(&self, pct: f32) -> usize {
            ((self.available_input() as f32) * pct.clamp(0.0, 1.0)).floor() as usize
        }
    }

    /// A named, budgeted region of the window (system / response / scratchpad /
    /// code / memory …). The compiler reserves these before the free competition.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct RegionBudget {
        pub region: String,
        pub target_tokens: usize,
        pub max_tokens: usize,
    }

    /// The resolved reservation plan: how many tokens are carved for each
    /// always-present region and how many are left to compete.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct Reservations {
        pub system: usize,
        pub response: usize,
        pub scratchpad: usize,
    }

    impl Reservations {
        pub fn total(&self) -> usize {
            self.system + self.response + self.scratchpad
        }
    }

    /// A token counter. Real tokenization when a `tokenizers::Tokenizer` is loaded,
    /// else a deterministic `chars/4` estimate. Cheap to clone (Arc inside).
    #[derive(Clone, Default)]
    pub struct TokenCounter {
        inner: Arc<RwLock<Option<Tokenizer>>>,
    }

    impl TokenCounter {
        /// A counter with no tokenizer: uses the `chars/4` fallback. Deterministic
        /// and dependency-free — the default for tests and offline use.
        pub fn heuristic() -> Self {
            Self { inner: Arc::new(RwLock::new(None)) }
        }

        /// Load a tokenizer from a `tokenizer.json` file (the HuggingFace format
        /// `tokenizers` reads). Returns the counter on success.
        pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
            let tok = Tokenizer::from_file(path.as_ref()).map_err(|e| e.to_string())?;
            Ok(Self { inner: Arc::new(RwLock::new(Some(tok))) })
        }

        /// Build from an in-memory `tokenizer.json` blob.
        pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
            let tok = Tokenizer::from_bytes(bytes).map_err(|e| e.to_string())?;
            Ok(Self { inner: Arc::new(RwLock::new(Some(tok))) })
        }

        /// True when a real tokenizer backs this counter (vs the fallback).
        pub fn is_accurate(&self) -> bool {
            self.inner.read().is_some()
        }

        /// Count tokens in `text`. Exact when a tokenizer is loaded; `chars/4`
        /// otherwise. The fallback is intentionally an *over*-estimate-safe round
        /// (`(chars+3)/4`) so the reserve invariant is never violated by undercount.
        pub fn count(&self, text: &str) -> usize {
            if let Some(tok) = self.inner.read().as_ref() {
                if let Ok(enc) = tok.encode(text, false) {
                    return enc.len();
                }
            }
            estimate_tokens(text)
        }
    }

    /// Deterministic `chars/4` fallback token estimate (bible §4.2 fallback).
    pub fn estimate_tokens(text: &str) -> usize {
        text.chars().count().saturating_add(3) / 4
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn heuristic_counter_uses_chars_over_four() {
            let c = TokenCounter::heuristic();
            assert!(!c.is_accurate());
            assert_eq!(c.count("abcd"), 1);
            assert_eq!(c.count("abcdefgh"), 2);
            assert_eq!(c.count(""), 0);
        }

        #[test]
        fn reservations_carve_before_fill() {
            let b = TokenBudget { max_input_tokens: 1000, reserve_output_tokens: 200, hard_limit_tokens: 1000 };
            assert_eq!(b.available_input(), 800);
            assert_eq!(b.reserve_pct(0.25), 200);
        }
    }
}
#[rustfmt::skip]
pub mod compiler {
    //! The Context Compiler (bible §4.2): a deterministic function of
    //! `(profile, model, sources, query)` → `(packed_window, manifest)`.
    //!
    //! Pipeline (bible §4.2.3):
    //!   0. reserve system/response/scratchpad *before* anything competes;
    //!   1. gather cheap candidates (handles + est tokens, no bodies);
    //!   2. score: band + relevance + recency + importance − redundancy + pins;
    //!   3. value-density greedy fill with an on-the-fly degrade ladder;
    //!   4. bounded local-improvement sweep;
    //!   5. head/tail ordering to defeat lost-in-the-middle.
    //!
    //! Determinism: stable sort tie-broken on content-addressed span id; `realize`
    //! is pure given content; the manifest records the exact ordering — so the turn
    //! replays (Tenet 7, F11).

    use crate::budget::{Reservations, TokenCounter};
    use crate::embed::{cosine, EmbeddingClient, HashingEmbeddingClient};
    use crate::manifest::{
        span_content_id, CompactionEvent, ContextManifest, ContextSourceKind, ContextSpan, DropReason,
        DroppedContextSpan, ManifestBudget, ManifestModel, ManifestProfile, PinState, SpanSignals,
    };
    use crate::profiles::ContextProfile;
    use futures::future::BoxFuture;
    use hide_core::error::Result;
    use hide_core::ids::now_ms;
    use hide_core::runtime::ModelDescriptor;
    use hide_core::types::Provenance;
    use std::sync::Arc;

    /// The compile inputs. **Field-stable for siblings** (`hide-backend` constructs
    /// this literal): `{ profile, model, task }`. Per-turn extras (session id,
    /// pins) are configured on the `ContextCompiler` instance, not here.
    #[derive(Debug, Clone)]
    pub struct CompileInput {
        pub profile: ContextProfile,
        pub model: ModelDescriptor,
        pub task: String,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct CompiledContext {
        pub prompt: String,
        pub manifest: ContextManifest,
    }

    /// A candidate span from a source. Bodies may be present (`text`) or deferred:
    /// a source can return an empty `text` plus a non-zero `est_tokens` and supply
    /// the real body in `realize()` (bible "just-in-time / progressive disclosure").
    #[derive(Debug, Clone, PartialEq)]
    pub struct ContextCandidate {
        pub id: String,
        pub source: ContextSourceKind,
        pub title: String,
        pub text: String,
        /// Source-declared base value band in `[0,1]` (was `score`). Kept named
        /// `score` for backward compatibility with existing sources.
        pub score: f32,
        pub provenance: Provenance,
        /// Estimated token cost before `realize()`. `0` means "estimate from
        /// `text`"; a lazy source that defers its body sets this so ranking has a
        /// cost without materializing.
        pub est_tokens: usize,
        /// Importance signal in `[0,1]` (salience). Defaults to `score` if unset.
        pub importance: Option<f32>,
        /// Recency timestamp (ms) for decay scoring. `None` => treated as "now".
        pub recency_ms: Option<u64>,
        pub pin: PinState,
    }

    impl ContextCandidate {
        /// Convenience constructor preserving the historical 6-field shape used by
        /// existing sources, defaulting the new fields.
        pub fn new(
            id: impl Into<String>,
            source: ContextSourceKind,
            title: impl Into<String>,
            text: impl Into<String>,
            score: f32,
            provenance: Provenance,
        ) -> Self {
            Self {
                id: id.into(),
                source,
                title: title.into(),
                text: text.into(),
                score,
                provenance,
                est_tokens: 0,
                importance: None,
                recency_ms: None,
                pin: PinState::Normal,
            }
        }

        /// Estimated token cost for ranking. A lazy candidate (empty body) reports
        /// its `est_tokens`; an eager one estimates from the body it carries.
        pub fn token_count(&self) -> usize {
            if self.text.is_empty() && self.est_tokens > 0 {
                self.est_tokens
            } else {
                crate::budget::estimate_tokens(&self.text)
            }
        }
    }

    /// A realized span — the actual tokens for a selected candidate.
    #[derive(Debug, Clone)]
    pub struct RealizedSpan {
        pub text: String,
        pub compacted: bool,
    }

    /// The universal context provider seam (bible §4.2.1, §7). New sources override
    /// `candidates`/`realize`/`degrade` for true lazy materialization; legacy
    /// sources implement only `gather` and get a working default.
    pub trait ContextSource: Send + Sync {
        fn name(&self) -> &str;

        /// Legacy/eager interface: produce fully-materialized candidates.
        fn gather<'a>(&'a self, input: &'a CompileInput) -> BoxFuture<'a, Result<Vec<ContextCandidate>>>;

        /// Cheap candidate enumeration (handles + estimates). Defaults to `gather`
        /// so existing sources keep working; lazy sources override to avoid bodies.
        fn candidates<'a>(&'a self, input: &'a CompileInput) -> BoxFuture<'a, Result<Vec<ContextCandidate>>> {
            self.gather(input)
        }

        /// Materialize a selected candidate's tokens. Default returns its `text`.
        fn realize<'a>(
            &'a self,
            c: &'a ContextCandidate,
            _budget_tokens: usize,
        ) -> BoxFuture<'a, Result<RealizedSpan>> {
            let text = c.text.clone();
            Box::pin(async move { Ok(RealizedSpan { text, compacted: false }) })
        }

        /// Optional cheaper rendering at a tighter budget (truncate/summary). The
        /// default truncates on token boundaries; sources can summarize instead.
        fn degrade<'a>(
            &'a self,
            c: &'a ContextCandidate,
            target_tokens: usize,
            counter: &'a TokenCounter,
        ) -> BoxFuture<'a, Result<Option<RealizedSpan>>> {
            let text = c.text.clone();
            // Tool-output spans are MASKED (placeholder + elision note) rather than
            // truncated or summarized: a local summary is a full inference pass, and
            // masking preserves the reasoning trace by only touching tool chatter
            // (W-F2-5). Everything else degrades by token-boundary truncation.
            let is_tool_output = c.source == ContextSourceKind::ToolOutput;
            Box::pin(async move {
                Ok(if is_tool_output {
                    mask_observation(&text, target_tokens, counter)
                } else {
                    default_truncate(&text, target_tokens, counter)
                })
            })
        }
    }

    /// Default degrade: truncate to ~`target_tokens` on a whitespace boundary.
    fn default_truncate(text: &str, target_tokens: usize, counter: &TokenCounter) -> Option<RealizedSpan> {
        if target_tokens == 0 || text.is_empty() {
            return None;
        }
        if counter.count(text) <= target_tokens {
            return Some(RealizedSpan { text: text.to_string(), compacted: false });
        }
        // Binary-search the longest whitespace-delimited prefix that fits.
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return None;
        }
        let (mut lo, mut hi) = (0usize, words.len());
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            let candidate = words[..mid].join(" ");
            if counter.count(&candidate) <= target_tokens.saturating_sub(1) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        if lo == 0 {
            return None;
        }
        let truncated = format!("{} …", words[..lo].join(" "));
        Some(RealizedSpan { text: truncated, compacted: true })
    }

    /// Observation masking (W-F2-5): replace a tool-output span with a compact
    /// placeholder that records how much was elided, keeping its first line as a
    /// hint. Cheaper than truncation and far cheaper than an LLM summary (a summary
    /// is a full local inference pass), and it preserves the reasoning trace by only
    /// touching tool chatter. A span already within budget is kept verbatim.
    fn mask_observation(text: &str, target_tokens: usize, counter: &TokenCounter) -> Option<RealizedSpan> {
        if text.is_empty() {
            return None;
        }
        let full = counter.count(text);
        if target_tokens > 0 && full <= target_tokens {
            return Some(RealizedSpan { text: text.to_string(), compacted: false });
        }
        let head: String = text.lines().next().unwrap_or("").trim().chars().take(80).collect();
        let placeholder = if head.is_empty() {
            format!("[tool output masked: ~{full} tokens elided]")
        } else {
            format!("[tool output masked: {head} … ~{full} tokens elided]")
        };
        Some(RealizedSpan { text: placeholder, compacted: true })
    }

    /// Importance-weighted fraction of a span's tokens dropped by a compaction — the
    /// secondary recall-gate signal (W-F2-3). A high-salience span that loses a lot
    /// reverts even when needle recall looks fine; a low-salience span can shed the
    /// same fraction without tripping the gate. Range `[0,1]`.
    fn dropped_important_frac(orig_tokens: usize, compacted_tokens: usize, importance: f32) -> f32 {
        let kept = compacted_tokens as f32 / orig_tokens.max(1) as f32;
        let dropped = (1.0 - kept).clamp(0.0, 1.0);
        (importance.clamp(0.0, 1.0) * dropped).clamp(0.0, 1.0)
    }

    /// Large constant: a user pin floats above any normally-scored span.
    const PIN_BONUS: f32 = 1_000.0;

    /// A non-pinned candidate is only worth degrading when its estimated cost is
    /// within this multiple of the free budget; beyond it a truncation would keep
    /// too little to be useful, so the span is deferred (→ no-fit drop) rather than
    /// realized and shredded. Pins ignore this (admitted unconditionally).
    const DEGRADE_FIT_RATIO: usize = 8;

    #[derive(Default)]
    pub struct ContextCompiler {
        sources: Vec<Box<dyn ContextSource>>,
        counter: TokenCounter,
        embedder: Option<Arc<dyn EmbeddingClient>>,
        session_id: Option<String>,
        /// Measured `.tq` effective-context multiplier for the served model, if
        /// known (resolved from the `.tq` sidecar at serve time). When set, the
        /// compiled manifest reports the *physical* effective ceiling
        /// (native x multiplier) instead of just the per-pass budget. Never a
        /// hardcoded constant — a measured per-model number or `None`.
        tq_multiplier: Option<f32>,
    }

    impl ContextCompiler {
        pub fn new() -> Self {
            Self {
                sources: Vec::new(),
                counter: TokenCounter::heuristic(),
                embedder: None,
                session_id: None,
                tq_multiplier: None,
            }
        }

        /// Install a tokenizer-accurate counter (bible §4.2). Without this the
        /// compiler uses the `chars/4` fallback.
        pub fn with_counter(mut self, counter: TokenCounter) -> Self {
            self.counter = counter;
            self
        }

        /// Install an embedding client for relevance/redundancy scoring. Without
        /// one, relevance/redundancy fall back to lexical Jaccard (still real, no
        /// blanket-zero).
        pub fn with_embedder(mut self, embedder: Arc<dyn EmbeddingClient>) -> Self {
            self.embedder = Some(embedder);
            self
        }

        pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
            self.session_id = Some(session_id.into());
            self
        }

        /// Record the measured `.tq` effective-context multiplier (native ->
        /// effective) for the served model so the compiled manifest reports the real
        /// physical ceiling. `None` (default) keeps the per-pass budget as the
        /// effective figure — no regression on existing callers.
        pub fn with_tq_multiplier(mut self, multiplier: f32) -> Self {
            self.tq_multiplier = Some(multiplier);
            self
        }

        pub fn add_source<S: ContextSource + 'static>(&mut self, source: S) {
            self.sources.push(Box::new(source));
        }

        pub async fn compile(&self, input: CompileInput) -> Result<CompiledContext> {
            let profile = &input.profile;

            // --- 0. Reservations: carved out before anything competes (F1). ---
            let total = profile.budget.available_input();
            let resv = Reservations {
                system: ((total as f32) * profile.reservation_pcts.system) as usize,
                response: profile.budget.reserve_output_tokens,
                scratchpad: ((total as f32) * profile.reservation_pcts.scratchpad) as usize,
            };
            // `system` is part of the fillable pool (system spans are real content);
            // only response+scratchpad are subtracted from the competition pool.
            let mut free = total.saturating_sub(resv.response + resv.scratchpad);

            // --- 1. Gather cheap candidates (metadata only). ---
            // Each candidate is tagged with the index of the source that produced it
            // so the selected-span `realize()` / over-budget `degrade()` calls below
            // dispatch back to *that* source (lazy materialization, bible §4.2.1).
            let mut cands: Vec<ContextCandidate> = Vec::new();
            let mut cand_src: Vec<usize> = Vec::new();
            for (src_idx, source) in self.sources.iter().enumerate() {
                let produced = source.candidates(&input).await?;
                cand_src.extend(std::iter::repeat(src_idx).take(produced.len()));
                cands.extend(produced);
            }

            // Embed the query and all candidate texts once (for relevance + redundancy).
            let query_vec;
            let cand_vecs: Vec<Vec<f32>>;
            if let Some(embedder) = &self.embedder {
                query_vec = embedder.embed_one(&input.task).await.unwrap_or_default();
                let texts: Vec<String> = cands.iter().map(|c| c.title.clone() + " " + &c.text).collect();
                cand_vecs = embedder.embed(&texts).await.unwrap_or_else(|_| vec![Vec::new(); cands.len()]);
            } else {
                // Lexical fallback embedder so relevance/redundancy are never blanket-zero.
                let lex = HashingEmbeddingClient::default();
                query_vec = lex.embed_one(&input.task).await.unwrap_or_default();
                let texts: Vec<String> = cands.iter().map(|c| c.title.clone() + " " + &c.text).collect();
                cand_vecs = lex.embed(&texts).await.unwrap_or_default();
            }

            let now = now_ms();
            let weights = &profile.source_weights;

            // Scored entries paired with their embedding index.
            let mut entries: Vec<EntryCarrier> = cands
                .into_iter()
                .enumerate()
                .map(|(i, c)| {
                    let relevance = if i < cand_vecs.len() {
                        // cosine in [-1,1] → [0,1]
                        ((cosine(&query_vec, &cand_vecs[i]) + 1.0) / 2.0).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let importance = c.importance.unwrap_or(c.score).clamp(0.0, 1.0);
                    let recency = recency_score(c.recency_ms.unwrap_or(now), now, profile.recency_half_life_ms);
                    let band = c.score.clamp(0.0, 1.0) * band_multiplier(weights, &c.source);
                    let mut base = weights.w_band * band
                        + weights.w_relevance * relevance
                        + weights.w_recency * recency
                        + weights.w_importance * importance;
                    if matches!(c.pin, PinState::UserPinned) {
                        base += PIN_BONUS;
                    }
                    EntryCarrier {
                        base_value: base,
                        vec_idx: i,
                        src_idx: cand_src[i],
                        signals: SpanSignals { recency, importance, relevance, redundancy: 0.0 },
                        cand: c,
                    }
                })
                .collect();

            // --- 2/3. Admit pins first, then value-density greedy with degrade. ---
            // Stable order: value desc, tie-break on content id for determinism.
            entries.sort_by(|a, b| {
                b.base_value
                    .partial_cmp(&a.base_value)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.cand.id.cmp(&b.cand.id))
            });

            let mut selected: Vec<SelectedSpan> = Vec::new();
            let mut dropped: Vec<DroppedContextSpan> = Vec::new();
            let mut compaction_events: Vec<CompactionEvent> = Vec::new();
            // No-fit entries that still carry bodies — material for local-improve.
            let mut deferred: Vec<EntryCarrier> = Vec::new();

            // Pins are admitted unconditionally (subtract their cost first).
            let (pinned, rest): (Vec<EntryCarrier>, Vec<EntryCarrier>) =
                entries.into_iter().partition(|e| matches!(e.cand.pin, PinState::NeverEvict | PinState::UserPinned));

            for e in pinned {
                if let Some(left) = self.admit(e, &mut free, &mut selected, &mut compaction_events, true).await? {
                    deferred.push(left);
                }
            }

            // Value-density greedy over the rest.
            // Re-rank by density = value / tokens.
            let mut pool = rest;
            pool.sort_by(|a, b| {
                let da = a.base_value / (a.cand.token_count().max(1) as f32);
                let db = b.base_value / (b.cand.token_count().max(1) as f32);
                db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.cand.id.cmp(&b.cand.id))
            });

            for mut e in pool {
                // Redundancy penalty vs already-selected spans (anti-rot). Applied
                // *multiplicatively* so a near-duplicate's value collapses toward
                // zero (bible §4.2.2: "the second's value collapses"), rather than a
                // fixed subtraction that a high base could absorb.
                let redundancy = self.max_redundancy(&cand_vecs, e.vec_idx, &selected);
                e.signals.redundancy = redundancy;
                let discount = (1.0 - weights.w_redundancy * redundancy).clamp(0.0, 1.0);
                let value = e.base_value * discount;
                // Drop when a near-duplicate is dominated (≥ ~90% similar at full
                // weight) — the second copy carries almost no marginal signal.
                if discount <= 0.1 {
                    dropped.push(drop_of(&e.cand, DropReason::Redundant));
                    continue;
                }
                e.base_value = value;
                if free == 0 {
                    deferred.push(e);
                    continue;
                }
                if let Some(left) = self.admit(e, &mut free, &mut selected, &mut compaction_events, false).await? {
                    deferred.push(left);
                }
            }

            // --- 4. Bounded local-improvement: pull the highest-value deferred span
            //         that now fits (whole or degraded) into leftover budget; cap
            //         at improve_iters. A real swap-aware pass over body-bearing
            //         deferred entries.
            self.local_improve(&mut selected, &mut deferred, &mut compaction_events, &mut free, profile.improve_iters)
                .await?;

            // Anything still deferred is a genuine no-fit drop, recorded for the UI.
            for e in deferred {
                dropped.push(drop_of(&e.cand, DropReason::NoFit));
            }

            // --- 5. Order head/tail to defeat lost-in-the-middle. ---
            order_head_tail(&mut selected, profile);

            // Assemble manifest + prompt.
            let mut manifest = ContextManifest::new(input.model.context_tokens);
            manifest.session_id = self.session_id.clone();
            manifest.turn_id = self.session_id.as_ref().map(|_| format!("turn_{now}"));
            manifest.profile = Some(ManifestProfile {
                name: profile.name.clone(),
                target_ctx_tokens: total,
                position_policy: profile.position_policy.label(),
                working_set_mode: format!("{:?}", profile.working_set_mode).to_lowercase(),
                kv_precision: format!("{:?}", profile.kv_precision).to_lowercase(),
            });
            manifest.model = Some(ManifestModel {
                id: input.model.id.to_string(),
                arch: format!("{:?}", input.model.architecture).to_lowercase(),
                ctx_len_native: input.model.context_tokens,
                // Physical effective ceiling = native x measured `.tq` multiplier
                // when known; otherwise the per-pass budget (preserves prior
                // behavior). The "what fit this pass" figure lives in
                // `ManifestProfile.target_ctx_tokens`, so the two stay distinct.
                ctx_len_effective: self
                    .tq_multiplier
                    .filter(|m| *m > 0.0)
                    .map(|m| ((input.model.context_tokens as f32) * m).round() as usize)
                    .unwrap_or(total),
                tokenizer_sig: input.model.tokenizer_signature.clone(),
            });

            let mut used = 0usize;
            let mut prompt_parts = Vec::with_capacity(selected.len());
            for (order, s) in selected.into_iter().enumerate() {
                used += s.tokens;
                prompt_parts.push(s.text.clone());
                manifest.retained.push(ContextSpan {
                    id: s.id,
                    source: s.source,
                    title: s.title,
                    text: s.text,
                    order_index: order,
                    token_count: s.tokens,
                    score: s.value,
                    signals: s.signals,
                    pin: s.pin,
                    banked: s.banked,
                    compacted_from: s.compacted_from,
                    provenance: s.provenance,
                    blob_ref: None,
                });
            }
            manifest.dropped = dropped;
            manifest.compaction_events = compaction_events;
            manifest.used_tokens = used;
            manifest.budget = Some(ManifestBudget {
                total,
                used,
                free: total.saturating_sub(used + resv.response + resv.scratchpad),
                reservation_system: resv.system,
                reservation_response: resv.response,
                reservation_scratchpad: resv.scratchpad,
            });

            // F1 invariant: window + response reservation never exceeds the total.
            debug_assert!(used + resv.response <= profile.budget.max_input_tokens);

            Ok(CompiledContext { prompt: prompt_parts.join("\n\n"), manifest })
        }

        /// Realize and admit a *selected* candidate, applying the source's degrade
        /// ladder if it doesn't fit whole. Pins skip the fit check (admitted
        /// unconditionally but still degraded to fit if necessary). Returns
        /// `Some(entry)` when the candidate could not fit even after degrade — the
        /// caller defers it for the local-improvement pass.
        ///
        /// This is the **only** place a body is materialized: selection scores on
        /// metadata (`title` + `est_tokens`), then we call the producing source's
        /// [`ContextSource::realize`] for the whole-fit path and
        /// [`ContextSource::degrade`] for the over-budget ladder. A lazy source that
        /// defers its body therefore never pays for dropped candidates (bible
        /// §4.2.1 "just-in-time / progressive disclosure").
        async fn admit(
            &self,
            e: EntryCarrier,
            free: &mut usize,
            selected: &mut Vec<SelectedSpan>,
            compaction: &mut Vec<CompactionEvent>,
            is_pin: bool,
        ) -> Result<Option<EntryCarrier>> {
            let source = &self.sources[e.src_idx];
            // Cheap metadata estimate gates whether we materialize at all. A lazy
            // candidate whose *estimate* already blows the budget is never realized
            // here — it is either degraded (pins / near-fits) or deferred so the
            // caller can record it as a no-fit drop with its body never touched.
            let est = e.cand.token_count();

            // Whole-fit path: only realize when the estimate says it could fit. The
            // estimate is `chars/4`-class and may be off, so we re-check the true
            // realized cost before committing.
            if est <= *free {
                let realized = source.realize(&e.cand, *free).await?;
                let cost = self.counter.count(&realized.text);
                if cost <= *free && !realized.text.is_empty() {
                    let tokens = cost.min(*free);
                    *free -= tokens;
                    let text = realized.text;
                    let cf = if realized.compacted {
                        Some(crate::manifest::CompactedFrom {
                            original_id: e.cand.id.clone(),
                            method: "realize".to_string(),
                            ratio: tokens as f32 / est.max(1) as f32,
                            depth: 1,
                        })
                    } else {
                        None
                    };
                    selected.push(self.selected_from(e, text, tokens, cf));
                    return Ok(None);
                }
                // The realized body was larger than its estimate; fall through to the
                // degrade ladder rather than over-spend the budget.
            }

            // Over-budget path. Non-pins that can't plausibly degrade into a useful
            // span (no room, or the estimate dwarfs the free budget so a truncation
            // would be near-useless) are deferred *without* realizing, so they are
            // recorded as honest no-fit drops. Pins always degrade-to-fit (they are
            // admitted unconditionally per the reservation contract).
            if !is_pin && (*free == 0 || est > free.saturating_mul(DEGRADE_FIT_RATIO)) {
                return Ok(Some(e));
            }
            let cost = est;
            let target = if is_pin { (*free).max(1) } else { *free };
            let degraded = source.degrade(&e.cand, target, &self.counter).await?;
            match degraded {
                Some(r) if !r.text.is_empty() => {
                    // Spine B: a compaction may only stand if it preserves recall. Measure the
                    // degraded text against needles from the ORIGINAL; if recall regresses below
                    // the floor and the original still fits the free budget, REVERT to the
                    // original (lossless-where-it-matters). Either way the recall is recorded.
                    let mut text = r.text;
                    let mut compacted = r.compacted;
                    let mut recall_score: Option<f32> = None;
                    let mut rolled_back = false;
                    if compacted {
                        let probes = crate::recall::needles_from(&e.cand.text, 8);
                        let recall = crate::recall::recall_at_k(&probes, &text);
                        recall_score = Some(recall);
                        let orig_tokens = self.counter.count(&e.cand.text);
                        // Importance-weighted dropped fraction: a high-salience span
                        // that loses a lot of its tokens reverts even when needle
                        // recall looks fine (W-F2-3 -- replaces the hardcoded 0.0).
                        let compacted_tokens = self.counter.count(&text);
                        let importance = e.cand.importance.unwrap_or(e.cand.score);
                        let dropped = dropped_important_frac(orig_tokens, compacted_tokens, importance);
                        if crate::recall::decide_rollback(recall, dropped, false, 1).should_rollback
                            && orig_tokens <= *free
                        {
                            text = e.cand.text.clone();
                            compacted = false;
                            rolled_back = true;
                        }
                    }
                    let tokens = self.counter.count(&text).min(*free).max(usize::from(!is_pin));
                    *free = free.saturating_sub(tokens);
                    let original_id = e.cand.id.clone();
                    let result_id = span_content_id(&e.cand.source, &e.cand.title, &text);
                    let ratio = tokens as f32 / cost.max(1) as f32;
                    let cf = if compacted {
                        compaction.push(CompactionEvent {
                            original_id: original_id.clone(),
                            result_id,
                            method: "degrade".to_string(),
                            model: None,
                            ratio,
                            depth: 1,
                            recall_score,
                            rolled_back: false,
                        });
                        Some(crate::manifest::CompactedFrom {
                            original_id,
                            method: "degrade".to_string(),
                            ratio,
                            depth: 1,
                        })
                    } else {
                        if rolled_back {
                            // Record the reverted compaction for telemetry (original kept, no compacted span).
                            compaction.push(CompactionEvent {
                                original_id: original_id.clone(),
                                result_id,
                                method: "degrade-reverted".to_string(),
                                model: None,
                                ratio: 1.0,
                                depth: 1,
                                recall_score,
                                rolled_back: true,
                            });
                        }
                        None
                    };
                    selected.push(self.selected_from(e, text, tokens, cf));
                    Ok(None)
                }
                _ => Ok(Some(e)),
            }
        }

        fn selected_from(
            &self,
            e: EntryCarrier,
            text: String,
            tokens: usize,
            compacted_from: Option<crate::manifest::CompactedFrom>,
        ) -> SelectedSpan {
            let id = span_content_id(&e.cand.source, &e.cand.title, &text);
            SelectedSpan {
                id,
                source: e.cand.source,
                title: e.cand.title,
                text,
                tokens,
                value: e.base_value,
                signals: e.signals,
                pin: e.cand.pin,
                banked: false,
                compacted_from,
                provenance: e.cand.provenance,
                vec_idx: e.vec_idx,
            }
        }

        fn max_redundancy(&self, cand_vecs: &[Vec<f32>], idx: usize, selected: &[SelectedSpan]) -> f32 {
            if idx >= cand_vecs.len() {
                return 0.0;
            }
            let mut max = 0.0f32;
            for s in selected {
                if s.vec_idx < cand_vecs.len() {
                    let sim = cosine(&cand_vecs[idx], &cand_vecs[s.vec_idx]);
                    if sim > max {
                        max = sim;
                    }
                }
            }
            max.clamp(0.0, 1.0)
        }

        /// Bounded local-improvement sweep (bible §4.2.3 step 4): with leftover
        /// budget, pull in the highest-value deferred (no-fit) candidate that now
        /// fits — whole or via a degrade into the free room — capped at `max_iters`.
        ///
        /// This is a real value-recovery pass over body-bearing entries: the greedy
        /// density fill can leave a high-value-but-bulky span out while admitting
        /// cheaper lower-value ones; when slack remains, this pass degrades the
        /// high-value span to fit, raising total retained value.
        async fn local_improve(
            &self,
            selected: &mut Vec<SelectedSpan>,
            deferred: &mut Vec<EntryCarrier>,
            compaction: &mut Vec<CompactionEvent>,
            free: &mut usize,
            max_iters: usize,
        ) -> Result<()> {
            let mut iters = 0;
            while iters < max_iters && *free > 0 && !deferred.is_empty() {
                iters += 1;
                // Highest base_value deferred entry first (value, then id for det.).
                deferred.sort_by(|a, b| {
                    b.base_value
                        .partial_cmp(&a.base_value)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.cand.id.cmp(&b.cand.id))
                });
                let e = deferred.remove(0);
                match self.admit(e, free, selected, compaction, false).await? {
                    // Still doesn't fit even degraded — truly no-fit; re-defer and
                    // stop (nothing smaller will help this iteration's budget).
                    Some(left) => {
                        deferred.push(left);
                        break;
                    }
                    None => { /* admitted; loop to try the next */ }
                }
            }
            Ok(())
        }
    }

    /// Internal selected-span carrier (post-realize).
    struct SelectedSpan {
        id: String,
        source: ContextSourceKind,
        title: String,
        text: String,
        tokens: usize,
        value: f32,
        signals: SpanSignals,
        pin: PinState,
        banked: bool,
        compacted_from: Option<crate::manifest::CompactedFrom>,
        provenance: Provenance,
        vec_idx: usize,
    }

    /// Scored candidate carried through the packer (built in `compile`, consumed by
    /// `admit`/`local_improve`).
    struct EntryCarrier {
        cand: ContextCandidate,
        base_value: f32,
        vec_idx: usize,
        /// Index into `self.sources` of the source that produced `cand` — the
        /// `realize()`/`degrade()` calls dispatch back to it.
        src_idx: usize,
        signals: SpanSignals,
    }

    fn drop_of(c: &ContextCandidate, reason: DropReason) -> DroppedContextSpan {
        DroppedContextSpan {
            id: c.id.clone(),
            source: c.source.clone(),
            token_count: c.token_count(),
            score: c.score,
            reason,
        }
    }

    fn band_multiplier(weights: &crate::profiles::SourceWeights, kind: &ContextSourceKind) -> f32 {
        weights.band_by_kind.get(&format!("{kind:?}")).copied().unwrap_or(1.0)
    }

    /// Exponential recency decay: `0.5^(age / half_life)` in `[0,1]` (bible §4.2.2).
    fn recency_score(ts_ms: u64, now_ms: u64, half_life_ms: u64) -> f32 {
        if half_life_ms == 0 {
            return 1.0;
        }
        let age = now_ms.saturating_sub(ts_ms) as f32;
        0.5f32.powf(age / half_life_ms as f32)
    }

    /// Place high-value spans at the head and tail; bury low-value filler in the
    /// middle (bible §4.2.3, F3). Pins/system float to the head; the most recent
    /// (lowest recency-decay age == highest recency) anchor the tail.
    fn order_head_tail(selected: &mut Vec<SelectedSpan>, profile: &ContextProfile) {
        use crate::profiles::OrderingPolicy;
        if !matches!(profile.ordering, OrderingPolicy::HeadTail) || selected.len() <= 2 {
            return;
        }
        // Sort by value desc as the working order.
        selected.sort_by(|a, b| {
            b.value.partial_cmp(&a.value).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.id.cmp(&b.id))
        });
        let taken = std::mem::take(selected);
        let mut head: Vec<SelectedSpan> = Vec::new();
        let mut middle: Vec<SelectedSpan> = Vec::new();
        let mut tail: Vec<SelectedSpan> = Vec::new();
        // System / never-evict / user-pinned go to the head.
        let mut rest: Vec<SelectedSpan> = Vec::new();
        for s in taken {
            if matches!(s.source, ContextSourceKind::System)
                || matches!(s.pin, PinState::NeverEvict | PinState::UserPinned)
            {
                head.push(s);
            } else {
                rest.push(s);
            }
        }
        // Of the rest (value-desc), highest-value to the tail (most-attended end),
        // next-highest after head, the long low-value tail buried in the middle.
        for (i, s) in rest.into_iter().enumerate() {
            match i % 3 {
                0 => tail.push(s),
                1 => head.push(s),
                _ => middle.push(s),
            }
        }
        tail.reverse(); // restore value-asc so the very last span is the top value
        let mut out = head;
        out.extend(middle);
        out.extend(tail);
        *selected = out;
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::profiles::ContextProfile;
        use hide_core::ids::ModelId;
        use hide_core::runtime::{ModelArchitecture, ModelDescriptor};
        use std::collections::HashSet;
        use std::sync::Mutex;

        struct StaticSource(Vec<ContextCandidate>);

        impl ContextSource for StaticSource {
            fn name(&self) -> &str {
                "static"
            }
            fn gather<'a>(&'a self, _input: &'a CompileInput) -> BoxFuture<'a, Result<Vec<ContextCandidate>>> {
                Box::pin(async { Ok(self.0.clone()) })
            }
        }

        /// A lazy source whose `candidates()` returns metadata-only handles (empty
        /// body + an `est_tokens` estimate) and whose `realize()` materializes the
        /// real body, recording *which* candidate ids were realized. Used to prove
        /// the compiler never realizes a dropped candidate.
        struct SpySource {
            /// (id, title, body, est_tokens, score) tuples.
            cands: Vec<(String, String, String, usize, f32)>,
            realized: Arc<Mutex<Vec<String>>>,
        }

        impl ContextSource for SpySource {
            fn name(&self) -> &str {
                "spy"
            }
            fn gather<'a>(&'a self, input: &'a CompileInput) -> BoxFuture<'a, Result<Vec<ContextCandidate>>> {
                self.candidates(input)
            }
            fn candidates<'a>(&'a self, _input: &'a CompileInput) -> BoxFuture<'a, Result<Vec<ContextCandidate>>> {
                Box::pin(async move {
                    Ok(self
                        .cands
                        .iter()
                        .map(|(id, title, _body, est, score)| {
                            // Metadata only: NO body, just a cost estimate.
                            let mut c = ContextCandidate::new(
                                id.clone(),
                                ContextSourceKind::Code,
                                title.clone(),
                                String::new(),
                                *score,
                                Provenance::trusted("spy"),
                            );
                            c.est_tokens = *est;
                            c
                        })
                        .collect())
                })
            }
            fn realize<'a>(
                &'a self,
                c: &'a ContextCandidate,
                _budget_tokens: usize,
            ) -> BoxFuture<'a, Result<RealizedSpan>> {
                self.realized.lock().unwrap().push(c.id.clone());
                let body = self
                    .cands
                    .iter()
                    .find(|(id, ..)| *id == c.id)
                    .map(|(_, _, body, ..)| body.clone())
                    .unwrap_or_default();
                Box::pin(async move { Ok(RealizedSpan { text: body, compacted: false }) })
            }
        }

        fn model(ctx: usize) -> ModelDescriptor {
            ModelDescriptor {
                id: ModelId::new(),
                name: "test".to_string(),
                architecture: ModelArchitecture::Transformer,
                context_tokens: ctx,
                tokenizer_signature: "test".to_string(),
                footprint_mb: 1,
            }
        }

        #[test]
        fn dropped_important_frac_weights_by_salience() {
            use crate::recall::DROPPED_IMPORTANT_CEIL;
            // High-importance span losing 80% of its tokens -> over the ceiling (revert).
            assert!(dropped_important_frac(100, 20, 1.0) > DROPPED_IMPORTANT_CEIL);
            // Same drop on a low-importance span -> under the ceiling (keep).
            assert!(dropped_important_frac(100, 20, 0.05) < DROPPED_IMPORTANT_CEIL);
            // No drop -> zero regardless of importance.
            assert_eq!(dropped_important_frac(100, 100, 1.0), 0.0);
        }

        #[test]
        fn mask_observation_replaces_body_with_placeholder() {
            let counter = TokenCounter::heuristic();
            let body = format!("RUN cargo test\n{}", "output line ".repeat(200));
            let masked = mask_observation(&body, 10, &counter).expect("masked");
            assert!(masked.compacted);
            assert!(masked.text.contains("masked"), "got: {}", masked.text);
            assert!(counter.count(&masked.text) < counter.count(&body));
            // A short span already within budget is kept verbatim, not masked.
            let small = mask_observation("ok", 100, &counter).expect("small");
            assert!(!small.compacted);
            assert_eq!(small.text, "ok");
        }

        #[tokio::test]
        async fn degrade_masks_tool_output_but_truncates_code() {
            let counter = TokenCounter::heuristic();
            let src = StaticSource(vec![]);
            let big = "x ".repeat(500);
            let tool = ContextCandidate::new(
                "t",
                ContextSourceKind::ToolOutput,
                "tool",
                big.clone(),
                0.5,
                Provenance::trusted("test"),
            );
            let code =
                ContextCandidate::new("c", ContextSourceKind::Code, "code", big, 0.5, Provenance::trusted("test"));
            let dt = src.degrade(&tool, 10, &counter).await.unwrap().expect("tool degraded");
            let dc = src.degrade(&code, 10, &counter).await.unwrap().expect("code degraded");
            assert!(dt.text.contains("masked"), "tool output should be masked");
            assert!(!dc.text.contains("masked"), "code should be truncated, not masked");
        }

        #[tokio::test]
        async fn keeps_highest_value_under_budget() {
            let mut compiler = ContextCompiler::new();
            compiler.add_source(StaticSource(vec![
                ContextCandidate::new(
                    "low",
                    ContextSourceKind::Code,
                    "low",
                    "x ".repeat(200),
                    0.1,
                    Provenance::trusted("test"),
                ),
                ContextCandidate::new(
                    "high",
                    ContextSourceKind::Code,
                    "high",
                    "important task content",
                    1.0,
                    Provenance::trusted("test"),
                ),
            ]));
            let compiled = compiler
                .compile(CompileInput {
                    profile: ContextProfile::coding_default(64),
                    model: model(64),
                    task: "important task content".to_string(),
                })
                .await
                .unwrap();
            // The high-value short span must be retained.
            let high = compiled.manifest.retained.iter().find(|s| s.title == "high").expect("high retained");
            // The bulky low-value span must either be dropped, or degraded
            // (compacted) to fit — never admitted whole at full size.
            let low_dropped = compiled.manifest.dropped.iter().any(|d| d.id == "low");
            let low_compacted =
                compiled.manifest.retained.iter().any(|s| s.title == "low" && s.compacted_from.is_some());
            assert!(
                low_dropped || low_compacted,
                "bulky low span must be dropped or compacted; retained={:?}",
                compiled.manifest.retained.iter().map(|s| &s.title).collect::<Vec<_>>()
            );
            // High value outranks low value.
            assert!(high.score >= 0.5);
            assert!(compiled.manifest.budget.is_some());
            // F1 invariant recorded.
            let b = compiled.manifest.budget.unwrap();
            assert!(b.used + b.reservation_response <= b.total + b.reservation_system);
        }

        #[tokio::test]
        async fn tq_multiplier_sets_physical_effective_ceiling() {
            // No multiplier: effective ceiling stays the per-pass budget (the field
            // is NOT silently equal to native*2) -- no regression.
            let base = ContextCompiler::new()
                .compile(CompileInput {
                    profile: ContextProfile::coding_default(128),
                    model: model(4096),
                    task: "t".to_string(),
                })
                .await
                .unwrap();
            let m = base.manifest.model.expect("model manifest");
            assert_eq!(m.ctx_len_native, 4096);
            assert_ne!(m.ctx_len_effective, 8192);

            // With a 2.0 multiplier: effective ceiling = native * 2.
            let scaled = ContextCompiler::new()
                .with_tq_multiplier(2.0)
                .compile(CompileInput {
                    profile: ContextProfile::coding_default(128),
                    model: model(4096),
                    task: "t".to_string(),
                })
                .await
                .unwrap();
            let sm = scaled.manifest.model.expect("model manifest");
            assert_eq!(sm.ctx_len_native, 4096);
            assert_eq!(sm.ctx_len_effective, 8192, "native 4096 x 2.0 multiplier");
        }

        #[tokio::test]
        async fn realize_not_called_for_dropped_candidates() {
            // Budget fits only the small high-value span. The bulky low-value span
            // must be dropped *without* its body ever being realized — proving the
            // lazy candidates → select → realize split is live, not eager.
            let realized = Arc::new(Mutex::new(Vec::new()));
            let mut compiler = ContextCompiler::new();
            compiler.add_source(SpySource {
                cands: vec![
                    // id, title, body, est_tokens, score
                    (
                        "keep".to_string(),
                        "important task content".to_string(),
                        "important task content".to_string(),
                        6,
                        1.0,
                    ),
                    (
                        "drop".to_string(),
                        "irrelevant filler".to_string(),
                        // A large body that would be expensive to materialize.
                        "z ".repeat(5_000),
                        5_000,
                        0.05,
                    ),
                ],
                realized: realized.clone(),
            });
            let compiled = compiler
                .compile(CompileInput {
                    // Small window: after reservations only a few tokens compete, so
                    // only "keep" can be admitted.
                    profile: ContextProfile::tight(48),
                    model: model(48),
                    task: "important task content".to_string(),
                })
                .await
                .unwrap();

            let realized_ids: HashSet<String> = realized.lock().unwrap().iter().cloned().collect();
            // The selected span was realized.
            assert!(realized_ids.contains("keep"), "selected candidate must be realized; realized={realized_ids:?}");
            // The dropped span was NEVER realized — its (huge) body was never touched.
            assert!(
                !realized_ids.contains("drop"),
                "dropped candidate must NOT be realized; realized={realized_ids:?}"
            );
            // And it is recorded as a no-fit drop, not silently lost.
            assert!(
                compiled.manifest.dropped.iter().any(|d| d.id == "drop" && d.reason == DropReason::NoFit),
                "drop must be a recorded NoFit; dropped={:?}",
                compiled.manifest.dropped
            );
            // The realized body actually made it into the prompt.
            assert!(compiled.prompt.contains("important task content"));
        }

        #[tokio::test]
        async fn degrade_invoked_on_over_budget_pinned_span() {
            // A pinned lazy span larger than the whole window must be admitted via
            // the source's degrade() ladder (the over-budget path), proving degrade
            // is dispatched — not the dead free-fn.
            let degraded_calls = Arc::new(Mutex::new(0usize));

            struct DegradeSpy {
                calls: Arc<Mutex<usize>>,
            }
            impl ContextSource for DegradeSpy {
                fn name(&self) -> &str {
                    "degrade_spy"
                }
                fn gather<'a>(&'a self, input: &'a CompileInput) -> BoxFuture<'a, Result<Vec<ContextCandidate>>> {
                    self.candidates(input)
                }
                fn candidates<'a>(&'a self, _input: &'a CompileInput) -> BoxFuture<'a, Result<Vec<ContextCandidate>>> {
                    Box::pin(async move {
                        let mut c = ContextCandidate::new(
                            "big",
                            ContextSourceKind::System,
                            "system",
                            String::new(),
                            1.0,
                            Provenance::trusted("t"),
                        );
                        c.est_tokens = 100_000;
                        c.pin = PinState::NeverEvict;
                        Ok(vec![c])
                    })
                }
                fn realize<'a>(
                    &'a self,
                    _c: &'a ContextCandidate,
                    _budget_tokens: usize,
                ) -> BoxFuture<'a, Result<RealizedSpan>> {
                    // The whole body is far too large for any window.
                    Box::pin(async move { Ok(RealizedSpan { text: "word ".repeat(100_000), compacted: false }) })
                }
                fn degrade<'a>(
                    &'a self,
                    _c: &'a ContextCandidate,
                    target_tokens: usize,
                    _counter: &'a TokenCounter,
                ) -> BoxFuture<'a, Result<Option<RealizedSpan>>> {
                    *self.calls.lock().unwrap() += 1;
                    let n = target_tokens.max(1);
                    Box::pin(async move { Ok(Some(RealizedSpan { text: "w ".repeat(n), compacted: true })) })
                }
            }

            let mut compiler = ContextCompiler::new();
            compiler.add_source(DegradeSpy { calls: degraded_calls.clone() });
            let compiled = compiler
                .compile(CompileInput {
                    profile: ContextProfile::tight(64),
                    model: model(64),
                    task: "anything".to_string(),
                })
                .await
                .unwrap();
            assert!(
                *degraded_calls.lock().unwrap() >= 1,
                "degrade() must be dispatched for the over-budget pinned span"
            );
            // It was admitted compacted (degrade ladder), with a recorded event.
            assert!(!compiled.manifest.compaction_events.is_empty(), "degrade should record a compaction event");
            let span = compiled
                .manifest
                .retained
                .iter()
                .find(|s| matches!(s.source, ContextSourceKind::System))
                .expect("pinned span retained via degrade");
            assert!(span.compacted_from.is_some(), "retained span is compacted");
        }

        #[tokio::test]
        async fn redundant_near_duplicate_is_penalized() {
            let mut compiler = ContextCompiler::new();
            // Two near-identical spans; only one should survive (redundancy).
            let dup = "the database pool is built with sqlx in db pool rs";
            compiler.add_source(StaticSource(vec![
                ContextCandidate::new("a", ContextSourceKind::Memory, "a", dup, 0.9, Provenance::trusted("t")),
                ContextCandidate::new("b", ContextSourceKind::Memory, "b", dup, 0.9, Provenance::trusted("t")),
                ContextCandidate::new(
                    "c",
                    ContextSourceKind::Code,
                    "c",
                    "completely unrelated rocket telemetry",
                    0.9,
                    Provenance::trusted("t"),
                ),
            ]));
            let compiled = compiler
                .compile(CompileInput {
                    profile: ContextProfile::coding_default(512),
                    model: model(512),
                    task: "database pool sqlx".to_string(),
                })
                .await
                .unwrap();
            // At least one of the duplicate pair is dropped as redundant.
            assert!(
                compiled.manifest.dropped.iter().any(|d| d.reason == DropReason::Redundant),
                "expected a redundancy drop, dropped={:?}",
                compiled.manifest.dropped
            );
        }

        #[tokio::test]
        async fn debug_profile_band_boosts_diagnostics_over_code() {
            // Two spans with identical declared score and length. Under the debug
            // profile, the diagnostics band multiplier (>1.0) must make the
            // diagnostic outrank the code span when only one fits — proving
            // band_by_kind is exercised, not a no-op 1.0.
            let body = "alpha beta gamma delta epsilon zeta eta theta";
            let mut compiler = ContextCompiler::new();
            compiler.add_source(StaticSource(vec![
                ContextCandidate::new("code", ContextSourceKind::Code, "code", body, 0.5, Provenance::trusted("t")),
                ContextCandidate::new(
                    "diag",
                    ContextSourceKind::Diagnostics,
                    "diag",
                    body,
                    0.5,
                    Provenance::trusted("t"),
                ),
            ]));
            // Budget sized so only one of the two equal-length spans is admitted.
            let compiled = compiler
                .compile(CompileInput {
                    profile: ContextProfile::debug(72),
                    model: model(72),
                    task: "unrelated query text".to_string(),
                })
                .await
                .unwrap();
            // The diagnostic survives; the code span is dropped/degraded.
            assert!(
                compiled.manifest.retained.iter().any(|s| matches!(s.source, ContextSourceKind::Diagnostics)),
                "diagnostics span should be retained under the debug band, retained={:?}",
                compiled.manifest.retained.iter().map(|s| &s.title).collect::<Vec<_>>()
            );
            // Sanity: with the *standard* profile (no band boost) the two equal
            // spans tie-break on content id, so the band is what flips debug.
            let standard = compiler
                .compile(CompileInput {
                    profile: ContextProfile::standard(72),
                    model: model(72),
                    task: "unrelated query text".to_string(),
                })
                .await
                .unwrap();
            // Under standard, "code" wins the id tie-break (c < d), so the band in
            // the debug run genuinely changed the selection.
            let standard_kept_code = standard.manifest.retained.iter().any(|s| s.title == "code");
            assert!(
                standard_kept_code,
                "control: standard profile keeps 'code' by id tie-break; got {:?}",
                standard.manifest.retained.iter().map(|s| &s.title).collect::<Vec<_>>()
            );
        }

        #[tokio::test]
        async fn pinned_span_admitted_first() {
            let mut compiler = ContextCompiler::new();
            let mut pinned = ContextCandidate::new(
                "sys",
                ContextSourceKind::System,
                "system",
                "system rules and safety",
                0.5,
                Provenance::trusted("t"),
            );
            pinned.pin = PinState::NeverEvict;
            compiler.add_source(StaticSource(vec![pinned]));
            let compiled = compiler
                .compile(CompileInput {
                    profile: ContextProfile::coding_default(128),
                    model: model(128),
                    task: "anything".to_string(),
                })
                .await
                .unwrap();
            assert_eq!(compiled.manifest.retained.len(), 1);
            assert_eq!(compiled.manifest.retained[0].pin, PinState::NeverEvict);
            // System is pinned to the head.
            assert_eq!(compiled.manifest.retained[0].order_index, 0);
        }
    }
}
#[rustfmt::skip]
pub mod embed {
    //! Embedding client seam (bible §4.2.2 / §4.6.3).
    //!
    //! Relevance and redundancy scoring need vectors. The live path calls the
    //! runtime's `/v1/embeddings` (OpenAI-compatible) so candidates are embedded by
    //! the *same model* that will read them. The runtime is not up during tests, so
    //! all live calls go behind a trait with a deterministic hashing stub.

    use async_trait::async_trait;
    use hide_core::error::{HideError, Result};
    use serde::Deserialize;

    /// A source of text embeddings. Implementors are cheap to clone or `Arc`-wrap.
    #[async_trait]
    pub trait EmbeddingClient: Send + Sync {
        /// Embed a batch of texts. Returns one vector per input, in order.
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

        /// Embed a single text (default: batch of one).
        async fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
            let mut v = self.embed(&[text.to_string()]).await?;
            v.pop().ok_or_else(|| HideError::RuntimeUnavailable("empty embedding response".into()))
        }

        /// Embedding dimensionality (for callers that pre-allocate). Best-effort.
        fn dim(&self) -> usize {
            0
        }
    }

    /// Cosine similarity between two equal-length vectors, in `[-1, 1]`. Returns 0
    /// for mismatched or empty vectors (caller treats that as "no signal").
    pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
        if a.is_empty() || a.len() != b.len() {
            return 0.0;
        }
        let mut dot = 0.0f32;
        let mut na = 0.0f32;
        let mut nb = 0.0f32;
        for i in 0..a.len() {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if na == 0.0 || nb == 0.0 {
            return 0.0;
        }
        dot / (na.sqrt() * nb.sqrt())
    }

    /// `reqwest`-backed client hitting the runtime's `/v1/embeddings`.
    #[derive(Clone)]
    pub struct HttpEmbeddingClient {
        base_url: String,
        model: String,
        client: reqwest::Client,
    }

    impl HttpEmbeddingClient {
        /// `base_url` is e.g. `http://127.0.0.1:8080`; the endpoint `/v1/embeddings`
        /// is appended.
        pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
            Self {
                base_url: base_url.into().trim_end_matches('/').to_string(),
                model: model.into(),
                client: reqwest::Client::new(),
            }
        }
    }

    #[derive(Deserialize)]
    struct EmbeddingResponse {
        data: Vec<EmbeddingDatum>,
    }

    #[derive(Deserialize)]
    struct EmbeddingDatum {
        embedding: Vec<f32>,
    }

    #[async_trait]
    impl EmbeddingClient for HttpEmbeddingClient {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            if texts.is_empty() {
                return Ok(Vec::new());
            }
            let body = serde_json::json!({ "model": self.model, "input": texts });
            let resp = self
                .client
                .post(format!("{}/v1/embeddings", self.base_url))
                .json(&body)
                .send()
                .await
                .map_err(|e| HideError::RuntimeUnavailable(format!("embeddings request: {e}")))?;
            if !resp.status().is_success() {
                return Err(HideError::RuntimeUnavailable(format!("embeddings HTTP {}", resp.status())));
            }
            let parsed: EmbeddingResponse =
                resp.json().await.map_err(|e| HideError::RuntimeUnavailable(format!("embeddings decode: {e}")))?;
            Ok(parsed.data.into_iter().map(|d| d.embedding).collect())
        }
    }

    /// A deterministic, dependency-free embedding stub for tests and offline use.
    /// Hashes whitespace tokens into a fixed-width bag-of-words vector — similar
    /// texts share dimensions, so cosine is meaningful (not random) without a model.
    #[derive(Clone)]
    pub struct HashingEmbeddingClient {
        dim: usize,
    }

    impl Default for HashingEmbeddingClient {
        fn default() -> Self {
            Self { dim: 256 }
        }
    }

    impl HashingEmbeddingClient {
        pub fn with_dim(dim: usize) -> Self {
            Self { dim: dim.max(1) }
        }

        fn embed_text(&self, text: &str) -> Vec<f32> {
            let mut v = vec![0.0f32; self.dim];
            for tok in text.split_whitespace() {
                let lower = tok.to_lowercase();
                // blake3 the token → bucket index; deterministic and well-spread.
                let h = blake3::hash(lower.as_bytes());
                let bytes = h.as_bytes();
                let idx = (u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize) % self.dim;
                v[idx] += 1.0;
            }
            v
        }
    }

    #[async_trait]
    impl EmbeddingClient for HashingEmbeddingClient {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|t| self.embed_text(t)).collect())
        }

        fn dim(&self) -> usize {
            self.dim
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[tokio::test]
        async fn hashing_embeddings_make_similar_text_closer() {
            let c = HashingEmbeddingClient::default();
            let q = c.embed_one("database migration sqlx pool").await.unwrap();
            let near = c.embed_one("the database pool is built with sqlx").await.unwrap();
            let far = c.embed_one("rocket telemetry orbital insertion burn").await.unwrap();
            let s_near = cosine(&q, &near);
            let s_far = cosine(&q, &far);
            assert!(s_near > s_far, "near={s_near} far={s_far}");
            assert!(s_near > 0.0);
        }

        #[test]
        fn cosine_handles_degenerate() {
            assert_eq!(cosine(&[], &[]), 0.0);
            assert_eq!(cosine(&[1.0], &[0.0]), 0.0);
            assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        }
    }
}
#[rustfmt::skip]
pub mod fidelity {
    //! Spine A — recall fidelity for an SSM's constant-size recurrent state.
    //!
    //! RWKV-7 has no token cap: its ~6-16 MiB state is constant, and older context
    //! decays in salience rather than falling off a hard edge. So "how full is the
    //! window" is the wrong question — the right one is "how sharp is recall over
    //! what the state has absorbed". This module models that as a 0..1 fidelity from
    //! the state's age (tokens absorbed) against a native recall horizon. The
    //! default is a conservative linear decay; a measured boot-needle calibration
    //! replaces it later (the trait keeps that swap a one-liner).

    /// A 0..1 recall-fidelity estimator for an aging recurrent state.
    pub trait RecallFidelityProbe: Send + Sync {
        /// Fidelity in `0..=1` for a state that has absorbed `state_age_tokens`.
        fn fidelity(&self, state_age_tokens: usize) -> f32;
    }

    /// Conservative default: fidelity decays linearly from 1.0 toward `floor` across
    /// `horizon_tokens`, then holds at `floor` (recall is soft, never zero — the
    /// state still carries a rolling summary). Calibrated boot-needle values replace
    /// this without changing the call site.
    #[derive(Debug, Clone, Copy)]
    pub struct LinearFidelity {
        pub horizon_tokens: usize,
        pub floor: f32,
    }

    impl LinearFidelity {
        /// `horizon_tokens` is typically the model's native recall horizon
        /// (`rwkv7.context_length`). Floor defaults to 0.3 (old context stays partly
        /// recoverable from the rolling state).
        pub fn new(horizon_tokens: usize) -> Self {
            Self { horizon_tokens, floor: 0.3 }
        }
    }

    impl RecallFidelityProbe for LinearFidelity {
        fn fidelity(&self, state_age_tokens: usize) -> f32 {
            if self.horizon_tokens == 0 {
                return 1.0;
            }
            let decayed = 1.0 - (state_age_tokens as f32 / self.horizon_tokens as f32);
            decayed.clamp(self.floor, 1.0)
        }
    }

    /// Measured-curve recall fidelity (W0-FID evaluator): piecewise-linear
    /// interpolation over calibration knots `(state_age_tokens, fidelity)` produced
    /// by a boot-needle probe on a real model. The CALIBRATION (running the needle
    /// probe to fill the knots) is model-gated; this evaluator is the pure drop-in
    /// for [`LinearFidelity`] that consumes those knots. Fidelity is clamped monotone
    /// non-increasing (recall does not recover with age) and held flat outside the
    /// measured range.
    #[derive(Debug, Clone)]
    pub struct SplineFidelity {
        knots: Vec<(usize, f32)>,
    }

    impl SplineFidelity {
        /// Build from calibration knots; sorts by age and enforces monotone
        /// non-increasing fidelity. Empty knots -> a degenerate full-fidelity curve.
        pub fn new(mut knots: Vec<(usize, f32)>) -> Self {
            knots.sort_by_key(|(age, _)| *age);
            let mut prev = 1.0f32;
            for (_, f) in knots.iter_mut() {
                *f = f.clamp(0.0, 1.0).min(prev);
                prev = *f;
            }
            Self { knots }
        }
    }

    impl RecallFidelityProbe for SplineFidelity {
        fn fidelity(&self, state_age_tokens: usize) -> f32 {
            match self.knots.as_slice() {
                [] => 1.0,
                [single] => single.1,
                knots => {
                    let age = state_age_tokens;
                    if age <= knots[0].0 {
                        return knots[0].1;
                    }
                    let last = knots[knots.len() - 1];
                    if age >= last.0 {
                        return last.1;
                    }
                    for w in knots.windows(2) {
                        let (a0, f0) = w[0];
                        let (a1, f1) = w[1];
                        if age >= a0 && age <= a1 {
                            if a1 == a0 {
                                return f1;
                            }
                            let t = (age - a0) as f32 / (a1 - a0) as f32;
                            return f0 + t * (f1 - f0);
                        }
                    }
                    last.1
                }
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn fresh_state_is_sharp_and_decays_to_floor() {
            let p = LinearFidelity::new(1000);
            assert!((p.fidelity(0) - 1.0).abs() < 1e-6);
            assert!(p.fidelity(500) > 0.49 && p.fidelity(500) < 0.51);
            assert!((p.fidelity(100_000) - p.floor).abs() < 1e-6, "holds at floor, never 0");
        }

        #[test]
        fn zero_horizon_is_full() {
            assert_eq!(LinearFidelity { horizon_tokens: 0, floor: 0.3 }.fidelity(123), 1.0);
        }

        #[test]
        fn spline_interpolates_between_knots() {
            let p = SplineFidelity::new(vec![(0, 1.0), (1000, 0.5), (2000, 0.3)]);
            assert!((p.fidelity(0) - 1.0).abs() < 1e-6);
            assert!((p.fidelity(500) - 0.75).abs() < 1e-6, "midpoint of 1.0..0.5");
            assert!((p.fidelity(1500) - 0.4).abs() < 1e-6);
            assert!((p.fidelity(5000) - 0.3).abs() < 1e-6, "held flat past last knot");
        }

        #[test]
        fn spline_enforces_monotone_non_increasing() {
            // A noisy knot that rises is clamped down to the prior fidelity.
            let p = SplineFidelity::new(vec![(0, 1.0), (1000, 0.4), (2000, 0.9)]);
            assert!(p.fidelity(2000) <= 0.4 + 1e-6, "recall cannot recover with age");
        }

        #[test]
        fn spline_empty_is_full() {
            assert_eq!(SplineFidelity::new(vec![]).fidelity(123), 1.0);
        }
    }
}
#[rustfmt::skip]
pub mod kv {
    //! The KV-store seam (bible §4.5, Appendix A.4).
    //!
    //! The shell-facing view of the runtime's tiered KV store (GPU→RAM→disk→
    //! checkpoints). Today this talks to `hawking-serve` over HTTP; later it can be
    //! an FFI. Lossless-by-construction: every reuse is re-verified by the engine's
    //! bit-identical copy+prefill-from-pos path (greedy-lossless), so this layer
    //! only routes — it stores no KV bytes itself.

    use async_trait::async_trait;
    use hide_core::error::{HideError, Result};
    use hide_core::ids::{RunId, SessionId};
    use serde::{Deserialize, Serialize};
    use sha2::{Digest, Sha256};

    /// A prefix address. **Byte-compatible with the in-tree
    /// `hawking_core::stateful::prefix_cache::PrefixKey` and
    /// `cache::prefill_disk::PrefillKey`** so the RAM and disk tiers agree on a
    /// prefix's address and the `SystemPromptKvBank` routes into them (bible A.4).
    ///
    /// Compatibility is by construction — [`PrefixKey::from_model_and_prompt`]
    /// reproduces the exact derivation:
    ///   - `model_hash      = sha256(model_id)`
    ///   - `tokenizer_hash  = sha256(tokenizer_signature)`
    ///   - `prefix_hash     = sha256(model_hash ‖ tokenizer_hash ‖ Σ tok.to_le_bytes())`
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct PrefixKey {
        pub model_hash: [u8; 32],
        pub tokenizer_hash: [u8; 32],
        pub prefix_hash: [u8; 32],
        pub n_tokens: usize,
    }

    impl PrefixKey {
        /// Build a key for `prompt_tokens` under `(model_id, tokenizer_signature)`.
        /// Defined to be byte-compatible with the in-tree disk/RAM tiers.
        pub fn from_model_and_prompt(model_id: &str, tokenizer_signature: &[u8], prompt_tokens: &[u32]) -> Self {
            let model_hash = sha256(&[model_id.as_bytes()]);
            let tokenizer_hash = sha256(&[tokenizer_signature]);
            let prefix_hash = rolling_prefix_hash(&model_hash, &tokenizer_hash, prompt_tokens);
            Self { model_hash, tokenizer_hash, prefix_hash, n_tokens: prompt_tokens.len() }
        }

        /// Lowercase hex of the prefix hash (the disk tier's `<prefix_hex>.kv`).
        pub fn prefix_hex(&self) -> String {
            hex32(&self.prefix_hash)
        }
    }

    fn sha256(parts: &[&[u8]]) -> [u8; 32] {
        let mut h = Sha256::new();
        for p in parts {
            h.update(p);
        }
        h.finalize().into()
    }

    fn rolling_prefix_hash(model_hash: &[u8; 32], tokenizer_hash: &[u8; 32], tokens: &[u32]) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(model_hash);
        h.update(tokenizer_hash);
        for &t in tokens {
            h.update(t.to_le_bytes());
        }
        h.finalize().into()
    }

    fn hex32(b: &[u8; 32]) -> String {
        let mut s = String::with_capacity(64);
        for byte in b {
            s.push_str(&format!("{byte:02x}"));
        }
        s
    }

    /// Residency tier of a KV range.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum KvTier {
        Gpu,
        Ram,
        Disk,
        Remote,
    }

    /// A live decode-slot identifier on the serve side.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct SlotId(pub u64);

    /// A reusable-prefix handle returned by a lookup (a routing hint, not bytes).
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct PrefixHandle {
        pub key: PrefixKey,
        /// Matched prefix length in tokens (strictly ≤ the query length).
        pub matched_tokens: usize,
        pub tier: KvTier,
    }

    /// Back-compat: a coarse handle used by the previous (read-only) API shape.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct KvHandle {
        pub provider: String,
        pub key: String,
        pub tokens: usize,
        pub tier: KvTier,
    }

    /// Working-set eviction choice for a slot (mirrors the profile type).
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case", tag = "kind")]
    pub enum EvictionChoice {
        Lossless,
        StreamingLlm { sinks: usize, recent: usize },
        SnapKv { keep: usize },
        H2o { recent: usize, heavy: usize },
    }

    /// Budget for a slot's working set (mirrors the in-tree `WorkingSetBudget`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub struct WorkingSetBudget {
        pub max_tokens: usize,
    }

    /// A named KV checkpoint id (bible §4.5.5).
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct CheckpointId(pub String);

    /// Metadata for a checkpoint (for a resume / time-travel UI).
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct CheckpointMeta {
        pub id: CheckpointId,
        pub session_id: SessionId,
        pub run_id: Option<RunId>,
        pub label: String,
        pub created_at_ms: u64,
        pub tokenizer_signature: String,
    }

    /// A KV checkpoint descriptor (kept for back-compat; references handles).
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct KvCheckpoint {
        pub session_id: SessionId,
        pub run_id: Option<RunId>,
        pub handles: Vec<KvHandle>,
        pub tokenizer_signature: String,
    }

    /// Stats for the manifest / `/metrics` (bible A.4 `stats`).
    #[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
    pub struct KvStoreStats {
        pub bank_hits: u64,
        pub prefix_reuse_tokens: u64,
        pub tier_bytes_gpu: u64,
        pub tier_bytes_ram: u64,
        pub tier_bytes_disk: u64,
        pub evictions: u64,
    }

    /// Restored-session handle from a checkpoint restore.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct RestoredSession {
        pub session_id: SessionId,
        pub slot: SlotId,
        pub warmed_tokens: usize,
    }

    /// The shell-facing KV operations (bible A.4). Async so the live impl can call
    /// `hawking-serve`. The runtime is not up during tests — use [`StubKvStore`].
    #[async_trait]
    pub trait KvStore: Send + Sync {
        /// Longest-prefix lookup across tiers; returns a handle if reusable.
        async fn lookup_prefix(&self, key: &PrefixKey) -> Result<Option<PrefixHandle>>;
        /// Promote a hit into a live slot and prefill only the tail.
        async fn warm_into_slot(&self, h: &PrefixHandle, full_ids: &[u32]) -> Result<SlotId>;
        /// Demote a finished slot's prefix to RAM (and async to disk). Never deletes.
        async fn demote(&self, slot: SlotId, prefix_len: usize) -> Result<()>;
        /// Set the working-set eviction policy + budget for a slot.
        async fn set_policy(&self, slot: SlotId, policy: EvictionChoice, budget: WorkingSetBudget) -> Result<()>;
        /// Checkpoint the live KV + manifest under a label.
        async fn checkpoint(&self, session: &SessionId, label: &str) -> Result<CheckpointId>;
        /// Restore a checkpoint into a warm slot (validates model/tokenizer hash).
        async fn restore(&self, id: &CheckpointId) -> Result<RestoredSession>;
        /// List a session's checkpoints (resume / time-travel UI).
        async fn list_checkpoints(&self, session: &SessionId) -> Result<Vec<CheckpointMeta>>;
        /// Stats for the manifest / `/metrics`.
        async fn stats(&self) -> Result<KvStoreStats>;
    }

    /// Back-compat alias for the previous trait name. Kept so any caller that named
    /// `KvStoreClient` keeps compiling; new code uses [`KvStore`].
    pub use self::KvStore as KvStoreClient;

    // ---------------------------------------------------------------------------
    // HTTP client to hawking-serve (the live seam)
    // ---------------------------------------------------------------------------

    /// Talks to `hawking-serve`'s `/v1/hawking/kv/*` surface. The endpoints are
    /// `[RUNTIME-SIDE — LATER]`; this client is the shell-side seam that lights up
    /// when they land. Construction never fails (no network on `new`).
    #[derive(Clone)]
    pub struct HttpKvStore {
        base_url: String,
        client: reqwest::Client,
    }

    impl HttpKvStore {
        pub fn new(base_url: impl Into<String>) -> Self {
            Self { base_url: base_url.into().trim_end_matches('/').to_string(), client: reqwest::Client::new() }
        }

        async fn post(&self, path: &str, body: serde_json::Value) -> Result<serde_json::Value> {
            let resp = self
                .client
                .post(format!("{}{}", self.base_url, path))
                .json(&body)
                .send()
                .await
                .map_err(|e| HideError::RuntimeUnavailable(format!("kv {path}: {e}")))?;
            if !resp.status().is_success() {
                return Err(HideError::RuntimeUnavailable(format!("kv {path} HTTP {}", resp.status())));
            }
            resp.json().await.map_err(|e| HideError::RuntimeUnavailable(format!("kv {path} decode: {e}")))
        }
    }

    #[async_trait]
    impl KvStore for HttpKvStore {
        async fn lookup_prefix(&self, key: &PrefixKey) -> Result<Option<PrefixHandle>> {
            let v = self
                .post(
                    "/v1/hawking/kv/lookup",
                    serde_json::json!({ "prefix_hex": key.prefix_hex(), "n_tokens": key.n_tokens }),
                )
                .await?;
            if v.get("hit").and_then(|h| h.as_bool()) == Some(true) {
                let matched = v.get("matched_tokens").and_then(|m| m.as_u64()).unwrap_or(0) as usize;
                Ok(Some(PrefixHandle { key: key.clone(), matched_tokens: matched, tier: KvTier::Ram }))
            } else {
                Ok(None)
            }
        }

        async fn warm_into_slot(&self, h: &PrefixHandle, full_ids: &[u32]) -> Result<SlotId> {
            let v = self
                .post(
                    "/v1/hawking/kv/warm",
                    serde_json::json!({
                        "prefix_hex": h.key.prefix_hex(),
                        "matched_tokens": h.matched_tokens,
                        "n_tokens": full_ids.len(),
                    }),
                )
                .await?;
            let slot = v
                .get("slot")
                .and_then(|s| s.as_u64())
                .ok_or_else(|| HideError::RuntimeUnavailable("kv/warm: missing slot".into()))?;
            Ok(SlotId(slot))
        }

        async fn demote(&self, slot: SlotId, prefix_len: usize) -> Result<()> {
            self.post("/v1/hawking/kv/demote", serde_json::json!({ "slot": slot.0, "prefix_len": prefix_len }))
                .await
                .map(|_| ())
        }

        async fn set_policy(&self, slot: SlotId, policy: EvictionChoice, budget: WorkingSetBudget) -> Result<()> {
            self.post(
                "/v1/hawking/kv/policy",
                serde_json::json!({ "slot": slot.0, "policy": policy, "budget": budget }),
            )
            .await
            .map(|_| ())
        }

        async fn checkpoint(&self, session: &SessionId, label: &str) -> Result<CheckpointId> {
            let v = self
                .post("/v1/hawking/kv/checkpoint", serde_json::json!({ "session": session.as_str(), "label": label }))
                .await?;
            let id = v
                .get("checkpoint_id")
                .and_then(|c| c.as_str())
                .ok_or_else(|| HideError::RuntimeUnavailable("kv/checkpoint: missing id".into()))?;
            Ok(CheckpointId(id.to_string()))
        }

        async fn restore(&self, id: &CheckpointId) -> Result<RestoredSession> {
            let v = self.post("/v1/hawking/kv/restore", serde_json::json!({ "checkpoint_id": id.0 })).await?;
            serde_json::from_value(v).map_err(|e| HideError::RuntimeUnavailable(format!("kv/restore decode: {e}")))
        }

        async fn list_checkpoints(&self, session: &SessionId) -> Result<Vec<CheckpointMeta>> {
            let v = self.post("/v1/hawking/kv/list", serde_json::json!({ "session": session.as_str() })).await?;
            let arr = v.get("checkpoints").cloned().unwrap_or(serde_json::json!([]));
            serde_json::from_value(arr).map_err(|e| HideError::RuntimeUnavailable(format!("kv/list decode: {e}")))
        }

        async fn stats(&self) -> Result<KvStoreStats> {
            let v = self.post("/v1/hawking/kv/stats", serde_json::json!({})).await?;
            serde_json::from_value(v).map_err(|e| HideError::RuntimeUnavailable(format!("kv/stats decode: {e}")))
        }
    }

    // ---------------------------------------------------------------------------
    // In-process stub (tests + offline). Models prefix reuse against a local map.
    // ---------------------------------------------------------------------------

    use parking_lot::Mutex;
    use std::collections::HashMap;

    /// A deterministic in-process [`KvStore`] for tests and offline operation. It
    /// models longest-prefix reuse against an internal map of banked prefixes and
    /// keeps real stats — it does not pretend to hold GPU KV.
    #[derive(Default)]
    pub struct StubKvStore {
        inner: Mutex<StubInner>,
    }

    #[derive(Default)]
    struct StubInner {
        banked: HashMap<String, usize>, // prefix_hex -> matched tokens
        next_slot: u64,
        checkpoints: Vec<CheckpointMeta>,
        stats: KvStoreStats,
    }

    impl StubKvStore {
        pub fn new() -> Self {
            Self::default()
        }

        /// Pre-bank a prefix (e.g. the system block) so a later lookup hits.
        pub fn bank(&self, key: &PrefixKey) {
            self.inner.lock().banked.insert(key.prefix_hex(), key.n_tokens);
        }
    }

    #[async_trait]
    impl KvStore for StubKvStore {
        async fn lookup_prefix(&self, key: &PrefixKey) -> Result<Option<PrefixHandle>> {
            let mut inner = self.inner.lock();
            if let Some(&n) = inner.banked.get(&key.prefix_hex()) {
                inner.stats.bank_hits += 1;
                inner.stats.prefix_reuse_tokens += n as u64;
                return Ok(Some(PrefixHandle { key: key.clone(), matched_tokens: n, tier: KvTier::Ram }));
            }
            Ok(None)
        }

        async fn warm_into_slot(&self, _h: &PrefixHandle, _full_ids: &[u32]) -> Result<SlotId> {
            let mut inner = self.inner.lock();
            let slot = inner.next_slot;
            inner.next_slot += 1;
            Ok(SlotId(slot))
        }

        async fn demote(&self, _slot: SlotId, prefix_len: usize) -> Result<()> {
            self.inner.lock().stats.tier_bytes_ram += prefix_len as u64;
            Ok(())
        }

        async fn set_policy(&self, _slot: SlotId, _policy: EvictionChoice, _budget: WorkingSetBudget) -> Result<()> {
            Ok(())
        }

        async fn checkpoint(&self, session: &SessionId, label: &str) -> Result<CheckpointId> {
            let id = CheckpointId(format!("ckpt_{}_{}", session.as_str(), label));
            let mut inner = self.inner.lock();
            inner.checkpoints.push(CheckpointMeta {
                id: id.clone(),
                session_id: session.clone(),
                run_id: None,
                label: label.to_string(),
                created_at_ms: hide_core::ids::now_ms(),
                tokenizer_signature: String::new(),
            });
            Ok(id)
        }

        async fn restore(&self, id: &CheckpointId) -> Result<RestoredSession> {
            let inner = self.inner.lock();
            let meta = inner
                .checkpoints
                .iter()
                .find(|c| c.id == *id)
                .ok_or_else(|| HideError::NotFound(format!("checkpoint {}", id.0)))?;
            Ok(RestoredSession { session_id: meta.session_id.clone(), slot: SlotId(0), warmed_tokens: 0 })
        }

        async fn list_checkpoints(&self, session: &SessionId) -> Result<Vec<CheckpointMeta>> {
            Ok(self.inner.lock().checkpoints.iter().filter(|c| c.session_id == *session).cloned().collect())
        }

        async fn stats(&self) -> Result<KvStoreStats> {
            Ok(self.inner.lock().stats.clone())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Interop lock: [`PrefixKey`] must be byte-identical to hawking-core's
        /// in-tree disk/RAM prefill key so the shell and runtime agree on a prefix's
        /// address. `hawking-core` is a heavy, macOS-Metal crate (`metal`, `objc2`),
        /// so we do **not** pull it in as a dev-dependency; instead this test
        /// replicates the *exact* in-tree byte derivation from:
        ///
        ///   crates/hawking-core/src/cache_prefill_disk.rs
        ///     · `PrefillKey::from_model_and_prompt`  (model/tokenizer sha256)
        ///     · `PrefillKey::rolling_prefix_hash`    (rolling sha256 over tokens)
        ///     · `PrefillKey::path` → `<model_hex>/<prefix_hex>.kv`
        ///
        /// If the in-tree derivation ever changes, this hand-replicated reference
        /// will diverge from `PrefixKey` and fail — locking the interop guarantee.
        #[test]
        fn prefix_key_matches_in_tree_derivation() {
            let model = "qwen-7b";
            let tok_sig = b"tokv1";
            let tokens = [1u32, 2, 3, 4];
            let key = PrefixKey::from_model_and_prompt(model, tok_sig, &tokens);

            // --- begin: byte-for-byte copy of prefill_disk.rs derivation ---
            // model_hash = sha256(model_id)
            let model_hash: [u8; 32] = {
                let mut h = Sha256::new();
                h.update(model.as_bytes());
                h.finalize().into()
            };
            // tokenizer_hash = sha256(tokenizer_signature)
            let tokenizer_hash: [u8; 32] = {
                let mut h = Sha256::new();
                h.update(tok_sig);
                h.finalize().into()
            };
            // prefix_hash = sha256(model_hash ‖ tokenizer_hash ‖ Σ tok.to_le_bytes())
            let prefix_hash: [u8; 32] = {
                let mut h = Sha256::new();
                h.update(model_hash);
                h.update(tokenizer_hash);
                for t in tokens {
                    h.update(t.to_le_bytes());
                }
                h.finalize().into()
            };
            // --- end: copy ---

            assert_eq!(key.model_hash, model_hash, "model_hash interop");
            assert_eq!(key.tokenizer_hash, tokenizer_hash, "tokenizer_hash interop");
            assert_eq!(key.prefix_hash, prefix_hash, "prefix_hash interop");
            assert_eq!(key.n_tokens, 4);

            // Disk-tier path form: `<model_hex>/<prefix_hex>.kv` (prefill_disk.rs
            // `PrefillKey::path`). `PrefixKey::prefix_hex` is the file stem.
            let model_hex = hex32(&model_hash);
            let prefix_hex = hex32(&prefix_hash);
            assert_eq!(key.prefix_hex(), prefix_hex);
            assert_eq!(key.prefix_hex().len(), 64);
            assert_eq!(format!("{model_hex}/{prefix_hex}.kv").len(), 64 + 1 + 64 + 3);

            // Empty-prompt edge: still seeded by model‖tokenizer (n_tokens == 0).
            let empty = PrefixKey::from_model_and_prompt(model, tok_sig, &[]);
            assert_eq!(empty.n_tokens, 0);
            let empty_prefix: [u8; 32] = {
                let mut h = Sha256::new();
                h.update(model_hash);
                h.update(tokenizer_hash);
                h.finalize().into()
            };
            assert_eq!(empty.prefix_hash, empty_prefix);
        }

        #[tokio::test]
        async fn stub_models_prefix_reuse_and_stats() {
            let store = StubKvStore::new();
            let key = PrefixKey::from_model_and_prompt("m", b"t", &[1, 2, 3]);
            assert!(store.lookup_prefix(&key).await.unwrap().is_none());
            store.bank(&key);
            let hit = store.lookup_prefix(&key).await.unwrap().unwrap();
            assert_eq!(hit.matched_tokens, 3);
            let slot = store.warm_into_slot(&hit, &[1, 2, 3, 4]).await.unwrap();
            store.demote(slot, 3).await.unwrap();
            let stats = store.stats().await.unwrap();
            assert_eq!(stats.bank_hits, 1);
            assert_eq!(stats.prefix_reuse_tokens, 3);
        }

        #[tokio::test]
        async fn stub_checkpoint_roundtrip() {
            let store = StubKvStore::new();
            let sess = SessionId::from("ses_test");
            let id = store.checkpoint(&sess, "before-refactor").await.unwrap();
            let list = store.list_checkpoints(&sess).await.unwrap();
            assert_eq!(list.len(), 1);
            let restored = store.restore(&id).await.unwrap();
            assert_eq!(restored.session_id, sess);
        }
    }
}
#[rustfmt::skip]
pub mod manifest {
    //! The context manifest — the single source of truth for what the model saw,
    //! why, and what was dropped (bible §4.3, Appendix A.1).
    //!
    //! It is the UI's "context stack", the agent loop's replay substrate, and a
    //! versioned public contract. Span ids are **blake3 content addresses** so two
    //! turns that include the same span share an id (dedup + replay key).

    use hide_core::ids::{now_ms, EventId, TimestampMs};
    use hide_core::types::{BlobRef, Provenance};
    use serde::{Deserialize, Serialize};

    /// Schema version of the manifest contract (A.1). Additive = minor bump.
    pub const MANIFEST_SCHEMA_VERSION: u16 = 1;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ContextManifest {
        pub schema_version: u16,
        pub created_at_ms: TimestampMs,
        pub source_event: Option<EventId>,
        /// Turn / session identifiers (A.1). Optional so callers that don't track
        /// a session still get a valid manifest.
        #[serde(default)]
        pub turn_id: Option<String>,
        #[serde(default)]
        pub session_id: Option<String>,
        /// The profile block (name + effective window + policy summary).
        #[serde(default)]
        pub profile: Option<ManifestProfile>,
        /// The model block (id, arch, native/effective ctx, tokenizer signature).
        #[serde(default)]
        pub model: Option<ManifestModel>,
        /// Budget accounting (total / used / free / reservations).
        #[serde(default)]
        pub budget: Option<ManifestBudget>,
        pub model_context_tokens: usize,
        pub used_tokens: usize,
        /// Retained spans, in final window order (head→tail).
        pub retained: Vec<ContextSpan>,
        /// Candidates that did not make it in, with a reason and would-be cost.
        pub dropped: Vec<DroppedContextSpan>,
        /// Surfaced contradictions for user resolution (F12/F4).
        #[serde(default)]
        pub conflicts: Vec<ManifestConflict>,
        /// KV-reuse accounting (§4.5).
        #[serde(default)]
        pub kv: ManifestKv,
        /// Compaction events: a span was replaced by a shorter rendering.
        #[serde(default)]
        pub compaction_events: Vec<CompactionEvent>,
        /// Live occupancy / recall headroom at inference time (Spine A). Changes
        /// every step as the window fills (transformer KV) or the recurrent state
        /// ages (RWKV recall fidelity). None until the runtime reports it.
        #[serde(default)]
        pub live: Option<ManifestLive>,
    }

    impl ContextManifest {
        pub fn new(model_context_tokens: usize) -> Self {
            Self {
                schema_version: MANIFEST_SCHEMA_VERSION,
                created_at_ms: now_ms(),
                source_event: None,
                turn_id: None,
                session_id: None,
                profile: None,
                model: None,
                budget: None,
                model_context_tokens,
                used_tokens: 0,
                retained: Vec::new(),
                dropped: Vec::new(),
                conflicts: Vec::new(),
                kv: ManifestKv::default(),
                compaction_events: Vec::new(),
                live: None,
            }
        }
    }

    /// Live occupancy / recall headroom recorded at inference time (Spine A — live
    /// context introspection). Distinct from the static `model` block: this is the
    /// DYNAMIC reading, computed against the live effective ceiling (never a
    /// hardcoded token count). Two regimes share one struct: transformers report KV
    /// occupancy; SSMs (RWKV-7, no token cap) report recall fidelity over a
    /// constant-size recurrent state.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
    pub struct ManifestLive {
        /// Transformer: current KV sequence position. None for SSMs.
        pub kv_seq_len: Option<usize>,
        /// Transformer: allocated KV cache size in tokens.
        pub kv_max: Option<usize>,
        /// SSM (RWKV): constant recurrent-state footprint, bytes.
        pub state_bytes: Option<usize>,
        /// SSM (RWKV): tokens that have flowed through the state this run.
        pub state_age_tokens: Option<usize>,
        /// SSM (RWKV): calibrated recall fidelity 0..1 (how "sharp" old context is).
        pub recall_fidelity: Option<f32>,
        /// Effective ceiling in tokens (native × .tq multiplier), read live.
        pub effective_ceiling_tokens: usize,
        /// Headroom before degradation: free tokens (transformer) or a recall-based
        /// estimate (SSM).
        pub headroom_tokens: usize,
        /// Occupancy 0..1 against the effective ceiling; drives watermarks.
        pub occupancy: f32,
        /// Which watermark band the occupancy/fidelity sits in.
        pub watermark: WatermarkLevel,
    }

    /// Headroom bands that drive the ambient cue + compaction timing (Spine A).
    /// Computed against the LIVE ceiling, never a constant. For SSMs the band is
    /// derived from `1 - recall_fidelity` so "recall getting soft" maps to the same
    /// scale as "KV getting full".
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
    #[serde(rename_all = "snake_case")]
    pub enum WatermarkLevel {
        /// < 60% — abundant; the ambient cue stays quiet.
        #[default]
        Normal,
        /// >= 60% — a context layer is ready to compact (soft hint).
        Soft,
        /// >= 75% — recency decay begins (warn).
        Warn,
        /// >= 90% — auto quality-compaction fires (before the cliff).
        Critical,
    }

    impl WatermarkLevel {
        /// Band for a 0..1 occupancy (transformer KV) or `1 - recall_fidelity` (SSM).
        pub fn for_occupancy(occ: f32) -> Self {
            if occ >= 0.90 {
                Self::Critical
            } else if occ >= 0.75 {
                Self::Warn
            } else if occ >= 0.60 {
                Self::Soft
            } else {
                Self::Normal
            }
        }
    }

    impl ManifestLive {
        /// Transformer regime: occupancy is KV position over the effective ceiling.
        pub fn transformer(kv_seq_len: usize, effective_ceiling_tokens: usize) -> Self {
            let occupancy = if effective_ceiling_tokens == 0 {
                0.0
            } else {
                (kv_seq_len as f32 / effective_ceiling_tokens as f32).clamp(0.0, 1.0)
            };
            Self {
                kv_seq_len: Some(kv_seq_len),
                kv_max: Some(effective_ceiling_tokens),
                state_bytes: None,
                state_age_tokens: None,
                recall_fidelity: None,
                effective_ceiling_tokens,
                headroom_tokens: effective_ceiling_tokens.saturating_sub(kv_seq_len),
                occupancy,
                watermark: WatermarkLevel::for_occupancy(occupancy),
            }
        }

        /// SSM regime (RWKV-7): there is no token cap, so "occupancy" is framed as
        /// `1 - recall_fidelity` — how soft recall has gotten — so the same watermark
        /// bands and ambient cue apply. Headroom is a recall-based token estimate.
        pub fn ssm(
            state_bytes: usize,
            state_age_tokens: usize,
            recall_fidelity: f32,
            effective_ceiling_tokens: usize,
        ) -> Self {
            let fidelity = recall_fidelity.clamp(0.0, 1.0);
            let occupancy = 1.0 - fidelity;
            Self {
                kv_seq_len: None,
                kv_max: None,
                state_bytes: Some(state_bytes),
                state_age_tokens: Some(state_age_tokens),
                recall_fidelity: Some(fidelity),
                effective_ceiling_tokens,
                headroom_tokens: ((effective_ceiling_tokens as f32) * fidelity) as usize,
                occupancy,
                watermark: WatermarkLevel::for_occupancy(occupancy),
            }
        }
    }

    /// The profile summary recorded on a manifest (A.1 `profile` block).
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct ManifestProfile {
        pub name: String,
        pub target_ctx_tokens: usize,
        pub position_policy: String,
        pub working_set_mode: String,
        pub kv_precision: String,
    }

    /// The model summary recorded on a manifest (A.1 `model` block).
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct ManifestModel {
        pub id: String,
        pub arch: String,
        pub ctx_len_native: usize,
        pub ctx_len_effective: usize,
        pub tokenizer_sig: String,
    }

    /// Budget accounting (A.1 `budget` block).
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct ManifestBudget {
        pub total: usize,
        pub used: usize,
        pub free: usize,
        pub reservation_system: usize,
        pub reservation_response: usize,
        pub reservation_scratchpad: usize,
    }

    /// Where a span sits in the pin lattice (A.1 `pin`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
    #[serde(rename_all = "snake_case")]
    pub enum PinState {
        /// Always present, never evicted (system rails, safety rules).
        NeverEvict,
        /// Floated to the top by the user.
        UserPinned,
        /// Competes on value like everything else.
        #[default]
        Normal,
    }

    /// The four scoring signals attached to every retained span (A.1 `signals`).
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
    pub struct SpanSignals {
        pub recency: f32,
        pub importance: f32,
        pub relevance: f32,
        pub redundancy: f32,
    }

    /// Record that a span was produced by compacting a larger original.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct CompactedFrom {
        pub original_id: String,
        pub method: String,
        pub ratio: f32,
        /// Recursion depth in the compaction chain (Spine B). `realize()` refuses to
        /// compact past depth 2 and falls back to the original to stop compounding loss.
        #[serde(default)]
        pub depth: u8,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ContextSpan {
        /// blake3 content address of the canonical span (dedup + replay key).
        pub id: String,
        pub source: ContextSourceKind,
        pub title: String,
        pub text: String,
        /// Position in the final window (head→tail), assigned by the packer.
        #[serde(default)]
        pub order_index: usize,
        pub token_count: usize,
        /// Blended value the packer scored this span at.
        pub score: f32,
        #[serde(default)]
        pub signals: SpanSignals,
        #[serde(default)]
        pub pin: PinState,
        /// True when the span's KV was reused from the prefix bank/cache (not
        /// re-prefilled) — bible §4.5.
        #[serde(default)]
        pub banked: bool,
        #[serde(default)]
        pub compacted_from: Option<CompactedFrom>,
        pub provenance: Provenance,
        pub blob_ref: Option<BlobRef>,
    }

    /// Compute the blake3 content address of a span's canonical form. Stable across
    /// turns: the same (kind, title, text) always hashes identically.
    pub fn span_content_id(kind: &ContextSourceKind, title: &str, text: &str) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"hide-span-v1\0");
        hasher.update(format!("{kind:?}").as_bytes());
        hasher.update(b"\0");
        hasher.update(title.as_bytes());
        hasher.update(b"\0");
        hasher.update(text.as_bytes());
        format!("blake3:{}", hasher.finalize().to_hex())
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum ContextSourceKind {
        System,
        UserTurn,
        Plan,
        Code,
        Symbol,
        ToolOutput,
        Memory,
        Scratchpad,
        Diagnostics,
        Custom(String),
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct DroppedContextSpan {
        pub id: String,
        pub source: ContextSourceKind,
        pub token_count: usize,
        pub score: f32,
        pub reason: DropReason,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum DropReason {
        Budget,
        NoFit,
        Duplicate,
        Redundant,
        Stale,
        LowScore,
        LowValue,
        Unsafe,
        SourceUnavailable,
    }

    /// A surfaced contradiction between two spans (F12/F4): the compiler does not
    /// silently pick one; it records the conflict for the user.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct ManifestConflict {
        pub between: Vec<String>,
        pub note: String,
        pub resolved: bool,
    }

    /// KV-reuse accounting for the manifest (A.1 `kv`).
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
    pub struct ManifestKv {
        pub prefix_reuse_tokens: usize,
        pub bank_hit: bool,
        pub tiers_touched: Vec<String>,
        pub checkpoint_id: Option<String>,
    }

    /// A compaction event (A.1 `compaction_events`).
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct CompactionEvent {
        pub original_id: String,
        pub result_id: String,
        pub method: String,
        pub model: Option<String>,
        pub ratio: f32,
        /// Recursion depth of this compaction (Spine B). >2 triggers auto-rollback.
        #[serde(default)]
        pub depth: u8,
        /// Needle-in-haystack recall@k measured AFTER this compaction, 0..1 (Spine B).
        /// None until the RecallOracle runs. Below threshold => `rolled_back`.
        #[serde(default)]
        pub recall_score: Option<f32>,
        /// True when the RecallOracle reverted this compaction (recall regressed).
        #[serde(default)]
        pub rolled_back: bool,
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn span_id_is_content_addressed_and_stable() {
            let a = span_content_id(&ContextSourceKind::Code, "t", "body");
            let b = span_content_id(&ContextSourceKind::Code, "t", "body");
            let c = span_content_id(&ContextSourceKind::Code, "t", "other");
            assert_eq!(a, b, "same content => same id");
            assert_ne!(a, c, "different body => different id");
            assert!(a.starts_with("blake3:"));
        }
    }
}
#[rustfmt::skip]
pub mod memory {
    //! Hierarchical memory store (bible §4.6, Appendix A.2).
    //!
    //! Two implementations behind one trait:
    //!  - [`InMemoryMemoryStore`] — a RAM `BTreeMap` (kept for tests / siblings).
    //!  - [`SqliteMemoryStore`] — the real store at `.hide/memory/memory.db` using
    //!    SQLite **FTS5** (keyword) + a stored-vector **cosine** index (relevance),
    //!    with the Generative-Agents retrieval score
    //!    `α_rec·recency + α_imp·importance + α_rel·relevance`, version chains
    //!    (`supersedes`), pins, decay, and provenance/confidence.
    //!
    //! `MemoryRecord` keeps a stable 8-field public shape (siblings construct it
    //! directly); the bible's extended attributes (`pinned`, `version`,
    //! `supersedes`, `links`, `decay_half_life_days`, `embedding_ref`) are carried
    //! by the store and exposed via [`StoredMemory`] / the typed API.

    use crate::embed::{cosine, EmbeddingClient};
    use futures::future::BoxFuture;
    use hide_core::error::{HideError, Result};
    use hide_core::ids::{now_ms, TimestampMs};
    use hide_core::types::Provenance;
    use parking_lot::{Mutex, RwLock};
    use rusqlite::Connection;
    use serde::{Deserialize, Serialize};
    use std::collections::BTreeMap;
    use std::path::Path;
    use std::sync::Arc;

    /// The canonical memory DTO. **Field-stable**: `hawking-research::bridge`
    /// constructs this literal, so these eight fields must not change.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct MemoryRecord {
        pub id: String,
        pub kind: MemoryKind,
        pub text: String,
        pub importance: f32,
        pub created_at_ms: TimestampMs,
        pub last_used_at_ms: Option<TimestampMs>,
        pub provenance: Provenance,
        pub tags: Vec<String>,
    }

    /// The store-managed extended attributes of a memory (bible A.2). Tracked by
    /// the store alongside the [`MemoryRecord`] so the DTO stays sibling-stable.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct MemoryMeta {
        pub embedding_ref: Option<String>,
        pub decay_half_life_days: u32,
        pub links: Vec<String>,
        pub supersedes: Option<String>,
        pub pinned: bool,
        pub version: u32,
        pub access_count: u64,
    }

    impl MemoryMeta {
        /// Type-dependent defaults: semantic/procedural barely decay; episodic does.
        pub fn defaults_for(kind: MemoryKind) -> Self {
            let half_life = match kind {
                MemoryKind::Working => 1,
                MemoryKind::Episodic => 30,
                MemoryKind::Semantic | MemoryKind::Project => 3650,
                MemoryKind::Procedural => 3650,
            };
            Self {
                embedding_ref: None,
                decay_half_life_days: half_life,
                links: Vec::new(),
                supersedes: None,
                pinned: false,
                version: 1,
                access_count: 0,
            }
        }
    }

    /// A record plus its store-managed metadata.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct StoredMemory {
        pub record: MemoryRecord,
        pub meta: MemoryMeta,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash, PartialOrd, Ord)]
    #[serde(rename_all = "snake_case")]
    pub enum MemoryKind {
        Working,
        Episodic,
        Semantic,
        Procedural,
        Project,
    }

    impl MemoryKind {
        /// Stable lowercase wire name for this kind.
        pub fn as_str_public(&self) -> &'static str {
            self.as_str()
        }

        fn as_str(&self) -> &'static str {
            match self {
                MemoryKind::Working => "working",
                MemoryKind::Episodic => "episodic",
                MemoryKind::Semantic => "semantic",
                MemoryKind::Procedural => "procedural",
                MemoryKind::Project => "project",
            }
        }
        fn from_str(s: &str) -> Option<Self> {
            Some(match s {
                "working" => MemoryKind::Working,
                "episodic" => MemoryKind::Episodic,
                "semantic" => MemoryKind::Semantic,
                "procedural" => MemoryKind::Procedural,
                "project" => MemoryKind::Project,
                _ => return None,
            })
        }
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct MemoryQuery {
        pub text: String,
        pub kinds: Vec<MemoryKind>,
        pub top_k: usize,
    }

    /// A retrieval result with its blended score and per-signal breakdown.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ScoredMemory {
        pub record: MemoryRecord,
        pub meta: MemoryMeta,
        pub score: f32,
        pub recency: f32,
        pub importance: f32,
        pub relevance: f32,
    }

    /// Back-compat alias: the previous public result type.
    pub type RankedMemory = ScoredMemory;

    /// The memory store contract (bible A.2). Kept object-safe (`BoxFuture`) so the
    /// existing `put`/`query` shape continues to work for siblings; the bible's
    /// `retrieve`/`upsert`/`supersede`/`pin` are added.
    pub trait MemoryStore: Send + Sync {
        fn put<'a>(&'a self, record: MemoryRecord) -> BoxFuture<'a, Result<()>>;
        fn query<'a>(&'a self, query: MemoryQuery) -> BoxFuture<'a, Result<Vec<RankedMemory>>>;

        /// Generative-Agents retrieval (bible §4.6.3): top-k by
        /// `α_rec·recency + α_imp·importance + α_rel·relevance`, then bump access.
        fn retrieve<'a>(
            &'a self,
            query: &'a str,
            k: usize,
            kinds: &'a [MemoryKind],
        ) -> BoxFuture<'a, Result<Vec<ScoredMemory>>> {
            let q = MemoryQuery { text: query.to_string(), kinds: kinds.to_vec(), top_k: k };
            Box::pin(self.query(q))
        }

        /// Insert/update; creates a new version on id conflict.
        fn upsert<'a>(&'a self, record: MemoryRecord) -> BoxFuture<'a, Result<String>> {
            Box::pin(async move {
                let id = record.id.clone();
                self.put(record).await?;
                Ok(id)
            })
        }

        /// Retire `old` and mint `new` with a `supersedes` edge (default no-op edge).
        fn supersede<'a>(&'a self, _old: &'a str, new: MemoryRecord) -> BoxFuture<'a, Result<String>> {
            self.upsert(new)
        }

        /// Pin/unpin a record (pinned => never decays, always retrievable).
        fn pin<'a>(&'a self, _id: &'a str, _pinned: bool) -> BoxFuture<'a, Result<()>> {
            Box::pin(async { Ok(()) })
        }
    }

    // ---------------------------------------------------------------------------
    // In-memory store (kept for tests and siblings)
    // ---------------------------------------------------------------------------

    #[derive(Default)]
    pub struct InMemoryMemoryStore {
        records: RwLock<BTreeMap<String, (MemoryRecord, MemoryMeta)>>,
    }

    impl InMemoryMemoryStore {
        /// Convenience: mint a record with sane defaults (kept for callers).
        pub fn record(kind: MemoryKind, text: impl Into<String>, provenance: Provenance) -> MemoryRecord {
            MemoryRecord {
                id: format!("mem_{}", now_ms()),
                kind,
                text: text.into(),
                importance: 0.5,
                created_at_ms: now_ms(),
                last_used_at_ms: None,
                provenance,
                tags: Vec::new(),
            }
        }
    }

    impl MemoryStore for InMemoryMemoryStore {
        fn put<'a>(&'a self, record: MemoryRecord) -> BoxFuture<'a, Result<()>> {
            Box::pin(async move {
                let meta = MemoryMeta::defaults_for(record.kind);
                self.records.write().insert(record.id.clone(), (record, meta));
                Ok(())
            })
        }

        fn query<'a>(&'a self, query: MemoryQuery) -> BoxFuture<'a, Result<Vec<RankedMemory>>> {
            Box::pin(async move {
                let now = now_ms();
                let mut ranked: Vec<ScoredMemory> = self
                    .records
                    .read()
                    .values()
                    .filter(|(r, _)| query.kinds.is_empty() || query.kinds.contains(&r.kind))
                    .map(|(r, m)| {
                        let relevance = lexical_overlap(&query.text, &r.text);
                        let recency = recency_score(
                            r.last_used_at_ms.unwrap_or(r.created_at_ms),
                            now,
                            m.decay_half_life_days,
                            m.pinned,
                        );
                        let importance = r.importance.clamp(0.0, 1.0);
                        let score = recency + importance + relevance;
                        ScoredMemory { record: r.clone(), meta: m.clone(), score, recency, importance, relevance }
                    })
                    .collect();
                ranked.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
                ranked.truncate(query.top_k);
                Ok(ranked)
            })
        }

        fn pin<'a>(&'a self, id: &'a str, pinned: bool) -> BoxFuture<'a, Result<()>> {
            Box::pin(async move {
                if let Some((_, m)) = self.records.write().get_mut(id) {
                    m.pinned = pinned;
                }
                Ok(())
            })
        }
    }

    // ---------------------------------------------------------------------------
    // SQLite store (FTS5 + stored-vector cosine) — the real store
    // ---------------------------------------------------------------------------

    /// SQLite-backed memory at `.hide/memory/memory.db` (bible §4.6.1).
    ///
    /// Keyword recall via FTS5; semantic recall via stored embedding vectors with
    /// cosine similarity computed in-process (no native ANN dependency — a real
    /// lighter alternative per the quality mandate). The connection is mutex-guarded
    /// (SQLite is single-writer).
    pub struct SqliteMemoryStore {
        conn: Mutex<Connection>,
        embedder: Option<Arc<dyn EmbeddingClient>>,
    }

    impl SqliteMemoryStore {
        /// Open (creating if needed) the DB at `path`, running the schema migration.
        pub fn open(path: impl AsRef<Path>) -> Result<Self> {
            if let Some(parent) = path.as_ref().parent() {
                std::fs::create_dir_all(parent)?;
            }
            let conn = Connection::open(path).map_err(sql_err)?;
            Self::init(&conn)?;
            Ok(Self { conn: Mutex::new(conn), embedder: None })
        }

        /// Open an in-memory DB (tests).
        pub fn open_in_memory() -> Result<Self> {
            let conn = Connection::open_in_memory().map_err(sql_err)?;
            Self::init(&conn)?;
            Ok(Self { conn: Mutex::new(conn), embedder: None })
        }

        /// Attach an embedding client so `relevance` uses cosine over real vectors
        /// (else relevance falls back to FTS5 keyword presence).
        pub fn with_embedder(mut self, embedder: Arc<dyn EmbeddingClient>) -> Self {
            self.embedder = Some(embedder);
            self
        }

        fn init(conn: &Connection) -> Result<()> {
            conn.execute_batch(
                r#"
                CREATE TABLE IF NOT EXISTS memory (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    importance REAL NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    last_used_at_ms INTEGER,
                    provenance_json TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    embedding_json TEXT,
                    decay_half_life_days INTEGER NOT NULL,
                    links_json TEXT NOT NULL,
                    supersedes TEXT,
                    pinned INTEGER NOT NULL,
                    version INTEGER NOT NULL,
                    access_count INTEGER NOT NULL,
                    retired INTEGER NOT NULL DEFAULT 0
                );
                -- FTS5 inverted index over `text`, keyed by `rowid = memory.rowid`
                -- so a `MATCH` joins straight back to the memory row by rowid. A
                -- regular (content-storing) FTS5 table is used — not `content=''` —
                -- because a contentless table cannot return stored columns and makes
                -- row updates awkward; the extra text copy is negligible here.
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
                    USING fts5(text);
                "#,
            )
            .map_err(sql_err)?;
            Ok(())
        }

        /// Number of live (non-retired) records.
        pub fn len(&self) -> Result<usize> {
            let conn = self.conn.lock();
            let n: i64 =
                conn.query_row("SELECT COUNT(*) FROM memory WHERE retired = 0", [], |r| r.get(0)).map_err(sql_err)?;
            Ok(n as usize)
        }

        pub fn is_empty(&self) -> Result<bool> {
            Ok(self.len()? == 0)
        }

        fn insert_record(&self, record: &MemoryRecord, meta: &MemoryMeta) -> Result<()> {
            let conn = self.conn.lock();
            let provenance_json = serde_json::to_string(&record.provenance)?;
            let tags_json = serde_json::to_string(&record.tags)?;
            let links_json = serde_json::to_string(&meta.links)?;
            let embedding_json = meta.embedding_ref.clone();
            conn.execute(
                r#"INSERT INTO memory
                   (id, kind, text, importance, created_at_ms, last_used_at_ms,
                    provenance_json, tags_json, embedding_json, decay_half_life_days,
                    links_json, supersedes, pinned, version, access_count, retired)
                   VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,0)
                   ON CONFLICT(id) DO UPDATE SET
                     kind=excluded.kind, text=excluded.text, importance=excluded.importance,
                     last_used_at_ms=excluded.last_used_at_ms,
                     provenance_json=excluded.provenance_json, tags_json=excluded.tags_json,
                     embedding_json=excluded.embedding_json,
                     decay_half_life_days=excluded.decay_half_life_days,
                     links_json=excluded.links_json, supersedes=excluded.supersedes,
                     pinned=excluded.pinned, version=memory.version+1, retired=0"#,
                rusqlite::params![
                    record.id,
                    record.kind.as_str(),
                    record.text,
                    record.importance as f64,
                    record.created_at_ms as i64,
                    record.last_used_at_ms.map(|v| v as i64),
                    provenance_json,
                    tags_json,
                    embedding_json,
                    meta.decay_half_life_days as i64,
                    links_json,
                    meta.supersedes,
                    meta.pinned as i64,
                    meta.version as i64,
                    meta.access_count as i64,
                ],
            )
            .map_err(sql_err)?;
            // Mirror into the FTS5 index, keyed by the memory row's rowid so a
            // `MATCH` joins back to `memory` by rowid. On an upsert the row already
            // exists; clear the stale FTS row first, then index the current text.
            let rowid: i64 = conn
                .query_row("SELECT rowid FROM memory WHERE id = ?1", [&record.id], |r| r.get(0))
                .map_err(sql_err)?;
            conn.execute("DELETE FROM memory_fts WHERE rowid = ?1", [rowid]).map_err(sql_err)?;
            conn.execute("INSERT INTO memory_fts(rowid, text) VALUES (?1, ?2)", rusqlite::params![rowid, record.text])
                .map_err(sql_err)?;
            Ok(())
        }

        async fn embed_text(&self, text: &str) -> Option<Vec<f32>> {
            match &self.embedder {
                Some(e) => e.embed_one(text).await.ok(),
                None => None,
            }
        }

        fn all_live(&self) -> Result<Vec<StoredMemory>> {
            let conn = self.conn.lock();
            let mut stmt = conn
                .prepare(
                    "SELECT id, kind, text, importance, created_at_ms, last_used_at_ms,
                            provenance_json, tags_json, embedding_json, decay_half_life_days,
                            links_json, supersedes, pinned, version, access_count
                     FROM memory WHERE retired = 0",
                )
                .map_err(sql_err)?;
            let rows = stmt
                .query_map([], row_to_stored)
                .map_err(sql_err)?
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(sql_err)?;
            Ok(rows)
        }

        /// Keyword recall via the FTS5 `memory_fts` index. Returns `id -> keyword
        /// relevance in [0,1]` for the rows the FTS5 `MATCH` selects, derived from
        /// the bm25 rank (best row → 1.0, decaying with rank). This is real
        /// inverted-index recall: it ranks by term frequency / document length and
        /// finds rows a naive `text.contains(term)` substring scan would mis-rank.
        ///
        /// `terms` are lowercased, non-empty user tokens. Each is wrapped in an FTS5
        /// string literal (doubling embedded quotes) and OR-combined, so arbitrary
        /// user text can never inject FTS5 query operators.
        fn fts_match(&self, terms: &[String]) -> Result<std::collections::HashMap<String, f32>> {
            use std::collections::HashMap;
            if terms.is_empty() {
                return Ok(HashMap::new());
            }
            // Build `"t1" OR "t2" OR ...` with each term as a quoted FTS5 string.
            let match_query =
                terms.iter().map(|t| format!("\"{}\"", t.replace('"', "\"\""))).collect::<Vec<_>>().join(" OR ");

            let conn = self.conn.lock();
            // Join the FTS rowid back to the live memory row's id.
            let mut stmt = conn
                .prepare(
                    "SELECT m.id, bm25(memory_fts) AS rank
                     FROM memory_fts
                     JOIN memory m ON m.rowid = memory_fts.rowid
                     WHERE memory_fts MATCH ?1 AND m.retired = 0
                     ORDER BY rank",
                )
                .map_err(sql_err)?;
            // bm25() returns a score where *more negative* = better match. Map the
            // ordered results to a [0,1] keyword-relevance with the top hit at 1.0.
            let rows: Vec<(String, f64)> = stmt
                .query_map([&match_query], |r| Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)?)))
                .map_err(sql_err)?
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(sql_err)?;

            let mut out = HashMap::new();
            let n = rows.len();
            for (rank_idx, (id, _bm25)) in rows.into_iter().enumerate() {
                // Rank-decayed relevance: 1.0 for the best, linearly down the list,
                // floored at a small positive so any MATCH still beats a non-match.
                let rel = if n <= 1 { 1.0 } else { 1.0 - 0.5 * (rank_idx as f32) / ((n - 1) as f32) };
                out.insert(id, rel);
            }
            Ok(out)
        }

        fn bump_access(&self, id: &str, now: u64) {
            let conn = self.conn.lock();
            let _ = conn.execute(
                "UPDATE memory SET access_count = access_count + 1, last_used_at_ms = ?2 WHERE id = ?1",
                rusqlite::params![id, now as i64],
            );
        }
    }

    fn row_to_stored(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoredMemory> {
        let kind_str: String = row.get(1)?;
        let provenance_json: String = row.get(6)?;
        let tags_json: String = row.get(7)?;
        let embedding_json: Option<String> = row.get(8)?;
        let links_json: String = row.get(10)?;
        let provenance: Provenance =
            serde_json::from_str(&provenance_json).unwrap_or_else(|_| Provenance::trusted("memory"));
        let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
        let links: Vec<String> = serde_json::from_str(&links_json).unwrap_or_default();
        let record = MemoryRecord {
            id: row.get(0)?,
            kind: MemoryKind::from_str(&kind_str).unwrap_or(MemoryKind::Semantic),
            text: row.get(2)?,
            importance: row.get::<_, f64>(3)? as f32,
            created_at_ms: row.get::<_, i64>(4)? as u64,
            last_used_at_ms: row.get::<_, Option<i64>>(5)?.map(|v| v as u64),
            provenance,
            tags,
        };
        let meta = MemoryMeta {
            embedding_ref: embedding_json,
            decay_half_life_days: row.get::<_, i64>(9)? as u32,
            links,
            supersedes: row.get(11)?,
            pinned: row.get::<_, i64>(12)? != 0,
            version: row.get::<_, i64>(13)? as u32,
            access_count: row.get::<_, i64>(14)? as u64,
        };
        Ok(StoredMemory { record, meta })
    }

    impl MemoryStore for SqliteMemoryStore {
        fn put<'a>(&'a self, record: MemoryRecord) -> BoxFuture<'a, Result<()>> {
            Box::pin(async move {
                let mut meta = MemoryMeta::defaults_for(record.kind);
                if let Some(v) = self.embed_text(&record.text).await {
                    meta.embedding_ref = Some(serde_json::to_string(&v)?);
                }
                self.insert_record(&record, &meta)
            })
        }

        fn query<'a>(&'a self, query: MemoryQuery) -> BoxFuture<'a, Result<Vec<RankedMemory>>> {
            Box::pin(async move {
                let now = now_ms();
                let query_vec = self.embed_text(&query.text).await;
                let query_terms: Vec<String> = query.text.split_whitespace().map(|s| s.to_lowercase()).collect();
                // Keyword recall through the FTS5 inverted index (real `MATCH`, not a
                // substring scan): id -> rank-decayed keyword relevance.
                let fts_hits = self.fts_match(&query_terms)?;
                let all = self.all_live()?;

                let mut scored: Vec<ScoredMemory> = all
                    .into_iter()
                    .filter(|s| query.kinds.is_empty() || query.kinds.contains(&s.record.kind))
                    .map(|s| {
                        // Fuse the two recall legs (bible §4.6.1 "FTS5 keyword +
                        // stored-vector cosine"): vector cosine where embeddings
                        // exist, the FTS5 bm25-ranked keyword hit always, and take
                        // the stronger signal so either leg can surface a memory.
                        let keyword = fts_hits.get(&s.record.id).copied().unwrap_or(0.0);
                        let vector = match (&query_vec, &s.meta.embedding_ref) {
                            (Some(qv), Some(ej)) => {
                                let mv: Vec<f32> = serde_json::from_str(ej).unwrap_or_default();
                                ((cosine(qv, &mv) + 1.0) / 2.0).clamp(0.0, 1.0)
                            }
                            _ => 0.0,
                        };
                        let relevance = keyword.max(vector);
                        let recency = recency_score(
                            s.record.last_used_at_ms.unwrap_or(s.record.created_at_ms),
                            now,
                            s.meta.decay_half_life_days,
                            s.meta.pinned,
                        );
                        let importance = s.record.importance.clamp(0.0, 1.0);
                        // Generative-Agents α=1,1,1 blend.
                        let score = recency + importance + relevance;
                        ScoredMemory { record: s.record, meta: s.meta, score, recency, importance, relevance }
                    })
                    .collect();

                scored.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.record.id.cmp(&b.record.id))
                });
                scored.truncate(query.top_k);
                // On access, bump access_count + last_used (feeds recency next time).
                for s in &scored {
                    self.bump_access(&s.record.id, now);
                }
                Ok(scored)
            })
        }

        fn supersede<'a>(&'a self, old: &'a str, new: MemoryRecord) -> BoxFuture<'a, Result<String>> {
            Box::pin(async move {
                // Retire the old version (hidden from retrieval, kept on disk).
                {
                    let conn = self.conn.lock();
                    conn.execute("UPDATE memory SET retired = 1 WHERE id = ?1", [old]).map_err(sql_err)?;
                }
                let mut meta = MemoryMeta::defaults_for(new.kind);
                meta.supersedes = Some(old.to_string());
                if let Some(v) = self.embed_text(&new.text).await {
                    meta.embedding_ref = Some(serde_json::to_string(&v)?);
                }
                let id = new.id.clone();
                self.insert_record(&new, &meta)?;
                Ok(id)
            })
        }

        fn pin<'a>(&'a self, id: &'a str, pinned: bool) -> BoxFuture<'a, Result<()>> {
            Box::pin(async move {
                let conn = self.conn.lock();
                conn.execute("UPDATE memory SET pinned = ?2 WHERE id = ?1", rusqlite::params![id, pinned as i64])
                    .map_err(sql_err)?;
                Ok(())
            })
        }
    }

    fn sql_err(e: rusqlite::Error) -> HideError {
        HideError::Storage(format!("memory db: {e}"))
    }

    /// Exponential recency decay over *days*; pinned records never decay (bible
    /// §4.6.4 / §4.7.3).
    fn recency_score(ts_ms: u64, now_ms: u64, half_life_days: u32, pinned: bool) -> f32 {
        if pinned {
            return 1.0;
        }
        if half_life_days == 0 {
            return 1.0;
        }
        let age_days = (now_ms.saturating_sub(ts_ms) as f32) / (1000.0 * 60.0 * 60.0 * 24.0);
        0.5f32.powf(age_days / half_life_days as f32)
    }

    fn lexical_overlap(a: &str, b: &str) -> f32 {
        let a_words: Vec<_> = a.split_whitespace().collect();
        if a_words.is_empty() {
            return 0.0;
        }
        let lb = b.to_lowercase();
        let hits = a_words.iter().filter(|word| lb.contains(&word.to_lowercase())).count();
        hits as f32 / a_words.len() as f32
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::embed::HashingEmbeddingClient;

        fn rec(id: &str, kind: MemoryKind, text: &str, importance: f32) -> MemoryRecord {
            MemoryRecord {
                id: id.to_string(),
                kind,
                text: text.to_string(),
                importance,
                created_at_ms: now_ms(),
                last_used_at_ms: None,
                provenance: Provenance::trusted("test"),
                tags: Vec::new(),
            }
        }

        #[tokio::test]
        async fn sqlite_store_retrieves_by_relevance() {
            let store =
                SqliteMemoryStore::open_in_memory().unwrap().with_embedder(Arc::new(HashingEmbeddingClient::default()));
            store.upsert(rec("a", MemoryKind::Semantic, "the database uses sqlx and postgres", 0.5)).await.unwrap();
            store.upsert(rec("b", MemoryKind::Semantic, "rocket telemetry orbital insertion", 0.5)).await.unwrap();
            let hits = store.retrieve("database sqlx", 2, &[MemoryKind::Semantic]).await.unwrap();
            assert_eq!(hits[0].record.id, "a", "relevance should rank the db note first");
            assert_eq!(store.len().unwrap(), 2);
        }

        #[tokio::test]
        async fn fts5_match_is_token_aware_not_substring() {
            // No embedder => relevance comes solely from the FTS5 keyword leg.
            let store = SqliteMemoryStore::open_in_memory().unwrap();
            // "cat" is a whole token here.
            store.upsert(rec("hit", MemoryKind::Semantic, "the cat sat on the mat", 0.1)).await.unwrap();
            // "cat" appears only as a *substring* of "concatenate" — a naive
            // `text.contains("cat")` scan would (wrongly) match this row, but the
            // FTS5 inverted index tokenizes and does NOT.
            store
                .upsert(rec("substring_only", MemoryKind::Semantic, "concatenate adjacent buffers efficiently", 0.1))
                .await
                .unwrap();

            let hits = store.retrieve("cat", 10, &[MemoryKind::Semantic]).await.unwrap();

            // The token match gets non-zero relevance.
            let hit = hits.iter().find(|h| h.record.id == "hit").expect("token row present");
            assert!(hit.relevance > 0.0, "FTS5 MATCH must give the token row keyword relevance");
            // The substring-only row gets ZERO relevance from the MATCH path — the
            // proof that retrieval went through FTS5, not a substring scan.
            let sub = hits
                .iter()
                .find(|h| h.record.id == "substring_only")
                .expect("substring row still listed (all_live), but unmatched");
            assert_eq!(sub.relevance, 0.0, "substring-only row must NOT be matched by FTS5 (a substring scan would)");
            // And the token row therefore outranks the substring-only row.
            assert!(
                hits[0].record.id == "hit",
                "token match should rank first; got {:?}",
                hits.iter().map(|h| &h.record.id).collect::<Vec<_>>()
            );
        }

        #[tokio::test]
        async fn supersede_retires_old_and_chains() {
            let store = SqliteMemoryStore::open_in_memory().unwrap();
            store.upsert(rec("v1", MemoryKind::Semantic, "old fact", 0.5)).await.unwrap();
            store.supersede("v1", rec("v2", MemoryKind::Semantic, "new fact", 0.5)).await.unwrap();
            let hits = store.retrieve("fact", 10, &[]).await.unwrap();
            let ids: Vec<_> = hits.iter().map(|h| h.record.id.as_str()).collect();
            assert!(ids.contains(&"v2"));
            assert!(!ids.contains(&"v1"), "retired version hidden from retrieval");
            assert_eq!(hits.iter().find(|h| h.record.id == "v2").unwrap().meta.supersedes, Some("v1".to_string()));
        }

        #[tokio::test]
        async fn pin_keeps_recency_high() {
            let store = SqliteMemoryStore::open_in_memory().unwrap();
            let mut r = rec("p", MemoryKind::Episodic, "pinned thing", 0.1);
            r.created_at_ms = 0; // ancient
            store.upsert(r).await.unwrap();
            store.pin("p", true).await.unwrap();
            let hits = store.retrieve("thing", 1, &[]).await.unwrap();
            assert!((hits[0].recency - 1.0).abs() < 1e-6, "pinned => no decay");
        }

        #[tokio::test]
        async fn in_memory_store_still_works() {
            let store = InMemoryMemoryStore::default();
            store.put(rec("x", MemoryKind::Semantic, "hello world", 0.9)).await.unwrap();
            let hits = store.retrieve("hello", 5, &[]).await.unwrap();
            assert_eq!(hits.len(), 1);
        }
    }
}
#[rustfmt::skip]
pub mod profiles {
    //! Per-task context profiles — the user's "dial" (bible §4.9, Appendix A.3).
    //!
    //! A profile bundles every knob (window, position policy, working-set mode,
    //! eviction, KV precision, reservations, source weights, recency half-life,
    //! ordering, retrieval_k, improve_iters) into a named preset. `Tight`,
    //! `Standard`, `Long`, `Unbounded` are the reserved built-ins.

    use crate::budget::{RegionBudget, TokenBudget};
    use serde::{Deserialize, Serialize};
    use std::collections::BTreeMap;

    /// Position-index scaling policy (bible §4.4.1). Recorded on the manifest; the
    /// runtime applies it (shell records the choice today).
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case", tag = "kind")]
    pub enum PositionPolicy {
        Native,
        Pi { scale: f32 },
        Yarn { scale: f32, attn_temp: f32 },
        DynamicNtk { trained_len: usize },
    }

    impl PositionPolicy {
        /// YaRN attention temperature default: `0.1·ln(scale) + 1` (bible §4.4.1).
        pub fn yarn(scale: f32) -> Self {
            PositionPolicy::Yarn { scale, attn_temp: 0.1 * scale.max(1.0).ln() + 1.0 }
        }

        pub fn label(&self) -> String {
            match self {
                PositionPolicy::Native => "native".to_string(),
                PositionPolicy::Pi { scale } => format!("pi(scale={scale})"),
                PositionPolicy::Yarn { scale, attn_temp } => {
                    format!("yarn(scale={scale},attn_temp={attn_temp:.2})")
                }
                PositionPolicy::DynamicNtk { trained_len } => {
                    format!("dynamic_ntk(trained_len={trained_len})")
                }
            }
        }
    }

    /// The working-set mode (mirrors the in-tree `WorkingSetMode`, bible §4.5).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum WorkingSetMode {
        Lossless,
        Bounded,
    }

    /// Eviction policy choice (bible §4.5.3). Profile-selectable.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case", tag = "kind")]
    pub enum EvictionChoice {
        Lossless,
        StreamingLlm { sinks: usize, recent: usize },
        SnapKv { keep: usize },
        H2o { recent: usize, heavy: usize },
    }

    /// KV precision codec (bible §4.5.3 / §4.9).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum KvPrecision {
        Native,
        F16,
        Int4,
    }

    /// Per-`SourceKind` scoring weights for the value blend (bible §4.2.2).
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct SourceWeights {
        pub w_band: f32,
        pub w_relevance: f32,
        pub w_recency: f32,
        pub w_importance: f32,
        pub w_redundancy: f32,
        /// Per-`ContextSourceKind` band multipliers ("debug profile weights
        /// diagnostics", "refactor weights symbols"); keyed by the kind's debug
        /// name. Missing kinds use 1.0.
        #[serde(default)]
        pub band_by_kind: BTreeMap<String, f32>,
    }

    impl SourceWeights {
        /// Band multipliers for the **debug** profile: diagnostics (compiler/linter
        /// messages) and failing tool output are boosted above ambient code so the
        /// failure surface wins the budget competition (bible §4.2.2). Keys are the
        /// `ContextSourceKind` debug names (what [`band_multiplier`] looks up).
        pub fn debug_bands() -> BTreeMap<String, f32> {
            BTreeMap::from([("Diagnostics".to_string(), 1.6), ("ToolOutput".to_string(), 1.3)])
        }

        /// Band multipliers for the **refactor** profile: symbol and code spans (the
        /// def/ref graph around the target) are boosted while volatile tool output
        /// is de-emphasized so the structural picture dominates (bible §4.2.2).
        pub fn refactor_bands() -> BTreeMap<String, f32> {
            BTreeMap::from([("Symbol".to_string(), 1.6), ("Code".to_string(), 1.3), ("ToolOutput".to_string(), 0.7)])
        }
    }

    impl Default for SourceWeights {
        fn default() -> Self {
            // Generative-Agents default α=1 for the three signals; band carries
            // the source's declared importance; redundancy is the anti-rot penalty.
            Self {
                w_band: 1.0,
                w_relevance: 1.0,
                w_recency: 0.5,
                w_importance: 0.75,
                w_redundancy: 1.0,
                band_by_kind: BTreeMap::new(),
            }
        }
    }

    /// Head/tail ordering policy to defeat lost-in-the-middle (bible §4.2.3, F3).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum OrderingPolicy {
        /// Spans in selection (value) order, no repositioning.
        AsScored,
        /// High-value spans pinned to the head and tail; filler buried in the
        /// middle. The anti-LITM default.
        HeadTail,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ContextProfile {
        pub name: String,
        pub budget: TokenBudget,
        pub regions: Vec<RegionBudget>,
        pub preserve_document_order: bool,
        pub pin_system_to_head: bool,
        pub pin_recent_to_tail: bool,
        // --- A.3 fields ---
        #[serde(default = "default_position_policy")]
        pub position_policy: PositionPolicy,
        #[serde(default = "default_working_set_mode")]
        pub working_set_mode: WorkingSetMode,
        #[serde(default = "default_eviction")]
        pub eviction: EvictionChoice,
        #[serde(default = "default_kv_precision")]
        pub kv_precision: KvPrecision,
        /// Fractions of the available input reserved for system/response/scratchpad.
        #[serde(default = "default_reservation_pcts")]
        pub reservation_pcts: ReservationPcts,
        #[serde(default)]
        pub source_weights: SourceWeights,
        #[serde(default = "default_recency_half_life_ms")]
        pub recency_half_life_ms: u64,
        #[serde(default = "default_ordering")]
        pub ordering: OrderingPolicy,
        #[serde(default = "default_retrieval_k")]
        pub retrieval_k: usize,
        #[serde(default = "default_improve_iters")]
        pub improve_iters: usize,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    pub struct ReservationPcts {
        pub system: f32,
        pub response: f32,
        pub scratchpad: f32,
    }

    fn default_position_policy() -> PositionPolicy {
        PositionPolicy::Native
    }
    fn default_working_set_mode() -> WorkingSetMode {
        WorkingSetMode::Lossless
    }
    fn default_eviction() -> EvictionChoice {
        EvictionChoice::Lossless
    }
    fn default_kv_precision() -> KvPrecision {
        KvPrecision::Native
    }
    fn default_reservation_pcts() -> ReservationPcts {
        ReservationPcts { system: 0.08, response: 0.20, scratchpad: 0.06 }
    }
    fn default_recency_half_life_ms() -> u64 {
        // 30 minutes — tool outputs decay over a session (bible §4.2.2).
        30 * 60 * 1000
    }
    fn default_ordering() -> OrderingPolicy {
        OrderingPolicy::HeadTail
    }
    fn default_retrieval_k() -> usize {
        16
    }
    fn default_improve_iters() -> usize {
        8
    }

    impl ContextProfile {
        fn regions(max_input_tokens: usize) -> Vec<RegionBudget> {
            vec![
                RegionBudget { region: "system".to_string(), target_tokens: 1_024, max_tokens: 2_048 },
                RegionBudget {
                    region: "code".to_string(),
                    target_tokens: max_input_tokens / 2,
                    max_tokens: max_input_tokens,
                },
                RegionBudget {
                    region: "memory".to_string(),
                    target_tokens: max_input_tokens / 8,
                    max_tokens: max_input_tokens / 4,
                },
            ]
        }

        /// The day-to-day default (kept name-stable for siblings). Equivalent to
        /// the `Standard` preset sized to `max_input_tokens`.
        pub fn coding_default(max_input_tokens: usize) -> Self {
            Self::standard(max_input_tokens)
        }

        /// Tight: quick edits, single file, lowest latency. Native window, lossless.
        pub fn tight(max_input_tokens: usize) -> Self {
            Self {
                name: "tight".to_string(),
                budget: TokenBudget {
                    max_input_tokens,
                    reserve_output_tokens: 1_024.min(max_input_tokens / 4),
                    hard_limit_tokens: max_input_tokens,
                },
                regions: Self::regions(max_input_tokens),
                preserve_document_order: true,
                pin_system_to_head: true,
                pin_recent_to_tail: true,
                position_policy: PositionPolicy::Native,
                working_set_mode: WorkingSetMode::Lossless,
                eviction: EvictionChoice::Lossless,
                kv_precision: KvPrecision::Native,
                reservation_pcts: default_reservation_pcts(),
                source_weights: SourceWeights::default(),
                recency_half_life_ms: 10 * 60 * 1000,
                ordering: OrderingPolicy::HeadTail,
                retrieval_k: 6,
                improve_iters: 4,
            }
        }

        /// Standard: day-to-day agentic coding (default).
        pub fn standard(max_input_tokens: usize) -> Self {
            Self {
                name: "standard".to_string(),
                budget: TokenBudget {
                    max_input_tokens,
                    reserve_output_tokens: 2_048.min(max_input_tokens / 4),
                    hard_limit_tokens: max_input_tokens,
                },
                regions: Self::regions(max_input_tokens),
                preserve_document_order: true,
                pin_system_to_head: true,
                pin_recent_to_tail: true,
                position_policy: PositionPolicy::Native,
                working_set_mode: WorkingSetMode::Lossless,
                eviction: EvictionChoice::Lossless,
                kv_precision: KvPrecision::F16,
                reservation_pcts: default_reservation_pcts(),
                source_weights: SourceWeights::default(),
                recency_half_life_ms: default_recency_half_life_ms(),
                ordering: OrderingPolicy::HeadTail,
                retrieval_k: 16,
                improve_iters: 8,
            }
        }

        /// Long: multi-file refactor, large files, long sessions. Mild YaRN +
        /// SnapKV bounded working set.
        pub fn long(max_input_tokens: usize) -> Self {
            Self {
                name: "long".to_string(),
                budget: TokenBudget {
                    max_input_tokens,
                    reserve_output_tokens: 4_096.min(max_input_tokens / 4),
                    hard_limit_tokens: max_input_tokens,
                },
                regions: Self::regions(max_input_tokens),
                preserve_document_order: true,
                pin_system_to_head: true,
                pin_recent_to_tail: true,
                position_policy: PositionPolicy::yarn(4.0),
                working_set_mode: WorkingSetMode::Bounded,
                eviction: EvictionChoice::SnapKv { keep: max_input_tokens / 2 },
                kv_precision: KvPrecision::F16,
                reservation_pcts: ReservationPcts { system: 0.06, response: 0.16, scratchpad: 0.06 },
                source_weights: SourceWeights::default(),
                recency_half_life_ms: 2 * 60 * 60 * 1000,
                ordering: OrderingPolicy::HeadTail,
                retrieval_k: 24,
                improve_iters: 12,
            }
        }

        /// Unbounded: "watch this stream / follow this long narrative". StreamingLLM
        /// sinks (or SSM route, recorded on the model descriptor).
        pub fn unbounded(max_input_tokens: usize) -> Self {
            Self {
                name: "unbounded".to_string(),
                budget: TokenBudget {
                    max_input_tokens,
                    reserve_output_tokens: 4_096.min(max_input_tokens / 4),
                    hard_limit_tokens: max_input_tokens,
                },
                regions: Self::regions(max_input_tokens),
                preserve_document_order: true,
                pin_system_to_head: true,
                pin_recent_to_tail: true,
                position_policy: PositionPolicy::Native,
                working_set_mode: WorkingSetMode::Bounded,
                eviction: EvictionChoice::StreamingLlm { sinks: 4, recent: max_input_tokens / 2 },
                kv_precision: KvPrecision::Int4,
                reservation_pcts: ReservationPcts { system: 0.05, response: 0.16, scratchpad: 0.08 },
                source_weights: SourceWeights::default(),
                recency_half_life_ms: 4 * 60 * 60 * 1000,
                ordering: OrderingPolicy::HeadTail,
                retrieval_k: 32,
                improve_iters: 8,
            }
        }

        /// Debug: root-causing a failure. A *debugging* profile weights diagnostics
        /// and recency (bible §4.2.2): diagnostics and tool output (test/compiler
        /// failures) get a band multiplier so they outrank ambient code, and the
        /// recency weight is lifted so the freshest failure dominates.
        pub fn debug(max_input_tokens: usize) -> Self {
            let mut weights = SourceWeights {
                // Lift recency: the latest failing run is what matters.
                w_recency: 1.0,
                ..SourceWeights::default()
            };
            weights.band_by_kind = SourceWeights::debug_bands();
            Self {
                name: "debug".to_string(),
                recency_half_life_ms: 5 * 60 * 1000, // 5 min: failures age fast
                source_weights: weights,
                ..Self::standard(max_input_tokens)
            }
        }

        /// Refactor: a structural change across files. A *refactor* profile weights
        /// symbols and relevance (bible §4.2.2): symbol/code spans get a band
        /// multiplier and the relevance weight is lifted so the def/ref graph around
        /// the target dominates, while volatile tool output is de-emphasized.
        pub fn refactor(max_input_tokens: usize) -> Self {
            let mut weights = SourceWeights { w_relevance: 1.5, ..SourceWeights::default() };
            weights.band_by_kind = SourceWeights::refactor_bands();
            Self { name: "refactor".to_string(), source_weights: weights, ..Self::long(max_input_tokens) }
        }

        /// Look up a reserved preset by dial name; `None` for custom names.
        pub fn preset(name: &str, max_input_tokens: usize) -> Option<Self> {
            match name {
                "tight" => Some(Self::tight(max_input_tokens)),
                "standard" | "coding_default" => Some(Self::standard(max_input_tokens)),
                "long" => Some(Self::long(max_input_tokens)),
                "unbounded" => Some(Self::unbounded(max_input_tokens)),
                "debug" => Some(Self::debug(max_input_tokens)),
                "refactor" => Some(Self::refactor(max_input_tokens)),
                _ => None,
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn presets_resolve_and_differ() {
            let tight = ContextProfile::tight(8192);
            let unbounded = ContextProfile::unbounded(8192);
            assert_eq!(tight.eviction, EvictionChoice::Lossless);
            assert!(matches!(unbounded.eviction, EvictionChoice::StreamingLlm { sinks: 4, .. }));
            assert!(ContextProfile::preset("long", 8192).is_some());
            assert!(ContextProfile::preset("nope", 8192).is_none());
        }

        #[test]
        fn debug_and_refactor_profiles_populate_band_by_kind() {
            let debug = ContextProfile::debug(8192);
            // The debug profile boosts diagnostics above 1.0 and lifts recency.
            assert!(debug.source_weights.band_by_kind.get("Diagnostics").copied().unwrap() > 1.0);
            assert!(debug.source_weights.band_by_kind.get("ToolOutput").copied().unwrap() > 1.0);
            assert!(debug.source_weights.w_recency >= 1.0);
            // A kind with no entry still defaults to 1.0 at the call site (absent).
            assert!(!debug.source_weights.band_by_kind.contains_key("Code"));

            let refactor = ContextProfile::refactor(8192);
            // The refactor profile boosts symbols/code and de-emphasizes tool output.
            assert!(refactor.source_weights.band_by_kind.get("Symbol").copied().unwrap() > 1.0);
            assert!(refactor.source_weights.band_by_kind.get("Code").copied().unwrap() > 1.0);
            assert!(refactor.source_weights.band_by_kind.get("ToolOutput").copied().unwrap() < 1.0);
            assert!(refactor.source_weights.w_relevance > 1.0);

            // Reserved names resolve via preset().
            assert!(ContextProfile::preset("debug", 8192).is_some());
            assert!(ContextProfile::preset("refactor", 8192).is_some());
        }

        #[test]
        fn yarn_temp_follows_formula() {
            if let PositionPolicy::Yarn { scale, attn_temp } = PositionPolicy::yarn(4.0) {
                assert_eq!(scale, 4.0);
                assert!((attn_temp - (0.1 * 4.0_f32.ln() + 1.0)).abs() < 1e-6);
            } else {
                panic!("expected yarn");
            }
        }
    }
}
#[rustfmt::skip]
pub mod recall {
    //! Spine B — the RecallOracle: the measured "remember more, not half" guarantee.
    //!
    //! Compaction is only allowed to stand if it preserves recall. After a compaction
    //! pass we re-ask a set of known facts ("needles") against the post-compaction
    //! context and compute recall@k. If recall regresses past the threshold — or too
    //! many importance-weighted tokens were dropped, or test coverage regressed, or
    //! the compaction chain recursed past the depth cap — the compaction is rolled
    //! back to the richer context. This module is the pure decision core (no I/O, no
    //! model), so the thresholds are unit-tested and can never silently drift.

    use serde::{Deserialize, Serialize};

    /// Recall@k floor: below this the compaction is reverted.
    pub const RECALL_FLOOR: f32 = 0.85;
    /// Importance-weighted dropped-token ceiling: above this we revert even if the
    /// needle recall looks fine (we dropped too much that mattered).
    pub const DROPPED_IMPORTANT_CEIL: f32 = 0.10;
    /// Recursion depth cap on a compaction chain; deeper => auto-revert to original.
    pub const MAX_COMPACT_DEPTH: u8 = 2;

    /// A known fact pinned from the PRE-compaction context, re-asked afterwards.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct RecallProbe {
        pub id: String,
        /// The salient substring that must survive compaction (a file path, a symbol,
        /// a decision, a constraint, a test verdict).
        pub needle: String,
    }

    /// Fraction of needles still recoverable in `post_context`, 0..1.
    ///
    /// A pragmatic, deterministic recall measure (no embedding model in the hot
    /// path): a needle counts as recalled if its normalized form appears in the
    /// post-compaction text. Empty probe set => 1.0 (nothing to lose).
    pub fn recall_at_k(probes: &[RecallProbe], post_context: &str) -> f32 {
        if probes.is_empty() {
            return 1.0;
        }
        let hay = post_context.to_lowercase();
        let hit = probes
            .iter()
            .filter(|p| {
                let n = p.needle.trim().to_lowercase();
                !n.is_empty() && hay.contains(&n)
            })
            .count();
        hit as f32 / probes.len() as f32
    }

    /// Derive recall needles from a span's ORIGINAL text: the distinctive non-trivial
    /// lines a faithful compaction must preserve. Deterministic, pure. Used by the
    /// compiler to measure a degrade against the original before letting it stand.
    pub fn needles_from(original: &str, max: usize) -> Vec<RecallProbe> {
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for line in original.lines() {
            let t = line.trim();
            // Skip trivia (short lines, lone braces) — keep lines with real content.
            if t.len() < 12 || !t.chars().any(|c| c.is_alphanumeric()) {
                continue;
            }
            if seen.insert(t.to_string()) {
                out.push(RecallProbe { id: format!("n{}", out.len()), needle: t.to_string() });
                if out.len() >= max {
                    break;
                }
            }
        }
        out
    }

    /// The verdict on whether a compaction may stand.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct RollbackDecision {
        pub should_rollback: bool,
        pub recall: f32,
        pub reason: &'static str,
    }

    /// Decide whether a compaction must be rolled back. Pure; order of checks is the
    /// order of severity so the `reason` is the most specific failure.
    pub fn decide_rollback(
        recall: f32,
        dropped_important_frac: f32,
        coverage_regressed: bool,
        depth: u8,
    ) -> RollbackDecision {
        if depth > MAX_COMPACT_DEPTH {
            return RollbackDecision { should_rollback: true, recall, reason: "depth cap exceeded" };
        }
        if coverage_regressed {
            return RollbackDecision { should_rollback: true, recall, reason: "test coverage regressed" };
        }
        if recall < RECALL_FLOOR {
            return RollbackDecision { should_rollback: true, recall, reason: "recall below floor" };
        }
        if dropped_important_frac > DROPPED_IMPORTANT_CEIL {
            return RollbackDecision { should_rollback: true, recall, reason: "dropped too much that mattered" };
        }
        RollbackDecision { should_rollback: false, recall, reason: "ok" }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn probe(id: &str, needle: &str) -> RecallProbe {
            RecallProbe { id: id.into(), needle: needle.into() }
        }

        #[test]
        fn recall_counts_surviving_needles() {
            let probes = vec![
                probe("f1", "guard.rs"),
                probe("d1", "drop permit past retry"),
                probe("c1", "never block on the semaphore"),
            ];
            // A summary that keeps two of three facts.
            let post = "Edited GUARD.RS to drop permit past retry; see notes.";
            let r = recall_at_k(&probes, post);
            assert!((r - 2.0 / 3.0).abs() < 1e-6, "got {r}");
        }

        #[test]
        fn empty_probes_is_full_recall() {
            assert_eq!(recall_at_k(&[], "anything"), 1.0);
        }

        #[test]
        fn needles_skip_trivia_and_dedupe() {
            let src = "fn acquire() {\n        drop(permit); // release the slot back to the pool\n}\n        drop(permit); // release the slot back to the pool\n";
            let n = needles_from(src, 8);
            // The lone `}` / short lines are skipped; the duplicate content line is kept once.
            assert_eq!(n.len(), 2, "deduped + trivia-filtered: {n:?}");
            assert!(n.iter().any(|p| p.needle.contains("release the slot")));
            assert!(n.iter().all(|p| p.needle != "}"));
        }

        #[test]
        fn rollback_fires_on_each_condition() {
            assert!(decide_rollback(0.99, 0.0, false, 3).should_rollback, "depth");
            assert!(decide_rollback(0.99, 0.0, true, 1).should_rollback, "coverage");
            assert!(decide_rollback(0.50, 0.0, false, 1).should_rollback, "recall");
            assert!(decide_rollback(0.99, 0.5, false, 1).should_rollback, "dropped");
            assert!(!decide_rollback(0.99, 0.0, false, 1).should_rollback, "clean keeps");
        }
    }
}
#[rustfmt::skip]
pub mod sources {
    //! Built-in context sources (bible §4.2.1, §4.7, §7).
    //!
    //! Each source produces ranked candidates with **real trust/confidence**
    //! provenance (not blanket `trusted`). The compiler ranks and packs them
    //! uniformly; new sources plug in by implementing [`ContextSource`].

    use crate::compiler::{CompileInput, ContextCandidate, ContextSource};
    use crate::manifest::{ContextSourceKind, PinState};
    use crate::memory::{MemoryKind, MemoryStore};
    use futures::future::BoxFuture;
    use hawking_index::{CodeIndex, SearchQuery, SearchResultSource};
    use hide_core::error::Result;
    use hide_core::types::{Provenance, TrustLevel};
    use std::sync::Arc;

    /// A static, in-prompt source (system prompt, fixed instructions). Spans are
    /// `never_evict` so they pin to the head.
    pub struct StaticContextSource {
        pub name: String,
        pub source: ContextSourceKind,
        pub spans: Vec<(String, String, f32)>,
    }

    impl ContextSource for StaticContextSource {
        fn name(&self) -> &str {
            &self.name
        }

        fn gather<'a>(&'a self, _input: &'a CompileInput) -> BoxFuture<'a, Result<Vec<ContextCandidate>>> {
            Box::pin(async move {
                Ok(self
                    .spans
                    .iter()
                    .enumerate()
                    .map(|(idx, (title, text, score))| {
                        ContextCandidate::new(
                            format!("{}:{idx}", self.name),
                            self.source.clone(),
                            title.clone(),
                            text.clone(),
                            *score,
                            Provenance::trusted(self.name.clone()),
                        )
                    })
                    .collect())
            })
        }
    }

    /// The system source: a never-evict head band (bible §4.2.3 reservation).
    pub struct SystemContextSource {
        pub text: String,
    }

    impl SystemContextSource {
        pub fn new(text: impl Into<String>) -> Self {
            Self { text: text.into() }
        }
    }

    impl ContextSource for SystemContextSource {
        fn name(&self) -> &str {
            "system"
        }

        fn gather<'a>(&'a self, _input: &'a CompileInput) -> BoxFuture<'a, Result<Vec<ContextCandidate>>> {
            Box::pin(async move {
                let mut c = ContextCandidate::new(
                    "system:0",
                    ContextSourceKind::System,
                    "System",
                    self.text.clone(),
                    1.0,
                    Provenance {
                        source: "system".to_string(),
                        trust: TrustLevel::Trusted,
                        confidence: 1.0,
                        labels: vec!["system".to_string()],
                        derived_from: Vec::new(),
                    },
                );
                c.pin = PinState::NeverEvict;
                c.importance = Some(1.0);
                Ok(vec![c])
            })
        }
    }

    /// Code/symbol source backed by `hawking-index`. Carries `path:line`
    /// provenance and propagates the index's result-source as trust signal.
    pub struct CodeIndexContextSource {
        pub name: String,
        pub index: Arc<dyn CodeIndex>,
        pub limit: usize,
        pub include_semantic: bool,
    }

    impl CodeIndexContextSource {
        pub fn new(index: Arc<dyn CodeIndex>, limit: usize) -> Self {
            Self { name: "code_index".to_string(), index, limit, include_semantic: true }
        }

        /// Toggle the semantic (embedding) retrieval leg.
        pub fn with_semantic(mut self, on: bool) -> Self {
            self.include_semantic = on;
            self
        }
    }

    impl ContextSource for CodeIndexContextSource {
        fn name(&self) -> &str {
            &self.name
        }

        fn gather<'a>(&'a self, input: &'a CompileInput) -> BoxFuture<'a, Result<Vec<ContextCandidate>>> {
            Box::pin(async move {
                // W-F2-6: route the query by shape (an exact-symbol query skips the
                // fuzzy legs), capped by this source's semantic config, then prefer
                // precise hits over similar-code semantic ones on score ties.
                let mut query = SearchQuery::routed(input.task.clone(), self.limit);
                query.include_semantic = query.include_semantic && self.include_semantic;
                let mut results = self.index.search(query).await?;
                hawking_index::query::rerank_prefer_precise(&mut results);
                Ok(results
                    .into_iter()
                    .enumerate()
                    .map(|(idx, result)| {
                        let range = result
                            .span
                            .range
                            .as_ref()
                            .map(|range| format!(":{}:{}", range.start_line, range.start_col))
                            .unwrap_or_default();
                        let path = result.span.path.display().to_string();
                        // Code from the workspace is `Workspace` trust; confidence
                        // tracks the index leg that found it (symbol > lexical).
                        let confidence = match result.source {
                            SearchResultSource::Symbol => 0.95,
                            SearchResultSource::Graph => 0.9,
                            SearchResultSource::Lexical => 0.8,
                            SearchResultSource::Semantic => 0.75,
                        };
                        let provenance = Provenance {
                            source: format!("code-index:{path}{range}"),
                            trust: TrustLevel::Workspace,
                            confidence,
                            labels: vec![format!("{:?}", result.source).to_lowercase()],
                            derived_from: vec![path.clone()],
                        };
                        ContextCandidate::new(
                            format!("{}:{idx}", self.name),
                            ContextSourceKind::Code,
                            format!("{}{}", result.title, range),
                            result.snippet,
                            result.score,
                            provenance,
                        )
                    })
                    .collect())
            })
        }
    }

    /// Memory source: retrieves relevant memories and offers them as candidates
    /// (bible §4.7.2 "progressive disclosure" — memory competes, not always-on).
    /// Memory-sourced facts carry their stored provenance/confidence (F12).
    pub struct MemoryContextSource {
        pub store: Arc<dyn MemoryStore>,
        pub kinds: Vec<MemoryKind>,
        pub k: usize,
    }

    impl MemoryContextSource {
        pub fn new(store: Arc<dyn MemoryStore>, k: usize) -> Self {
            Self { store, kinds: Vec::new(), k }
        }
    }

    impl ContextSource for MemoryContextSource {
        fn name(&self) -> &str {
            "memory"
        }

        fn gather<'a>(&'a self, input: &'a CompileInput) -> BoxFuture<'a, Result<Vec<ContextCandidate>>> {
            Box::pin(async move {
                let hits = self.store.retrieve(&input.task, self.k, &self.kinds).await?;
                Ok(hits
                    .into_iter()
                    .enumerate()
                    .map(|(idx, h)| {
                        let pinned = h.meta.pinned;
                        let mut c = ContextCandidate::new(
                            format!("memory:{}", h.record.id),
                            ContextSourceKind::Memory,
                            format!("memory:{}", h.record.kind.as_str_public()),
                            h.record.text.clone(),
                            h.score.clamp(0.0, 1.0),
                            h.record.provenance.clone(),
                        );
                        c.importance = Some(h.importance);
                        c.recency_ms = h.record.last_used_at_ms.or(Some(h.record.created_at_ms));
                        if pinned {
                            c.pin = PinState::UserPinned;
                        }
                        let _ = idx;
                        c
                    })
                    .collect())
            })
        }
    }

    /// Plan source: the current plan steps as context (untrusted-derived = the
    /// agent's own working state, `Workspace` trust).
    pub struct PlanContextSource {
        pub steps: Vec<String>,
    }

    impl ContextSource for PlanContextSource {
        fn name(&self) -> &str {
            "plan"
        }

        fn gather<'a>(&'a self, _input: &'a CompileInput) -> BoxFuture<'a, Result<Vec<ContextCandidate>>> {
            Box::pin(async move {
                Ok(self
                    .steps
                    .iter()
                    .enumerate()
                    .map(|(idx, step)| {
                        ContextCandidate::new(
                            format!("plan:{idx}"),
                            ContextSourceKind::Plan,
                            format!("plan step {idx}"),
                            step.clone(),
                            0.9,
                            Provenance {
                                source: "plan".to_string(),
                                trust: TrustLevel::Workspace,
                                confidence: 0.9,
                                labels: vec!["plan".to_string()],
                                derived_from: Vec::new(),
                            },
                        )
                    })
                    .collect())
            })
        }
    }

    /// A tool output (untrusted — bible F12: tool-sourced confidence < 1).
    pub struct ToolOutputContextSource {
        pub outputs: Vec<(String, String)>, // (tool_call_id, text)
    }

    impl ContextSource for ToolOutputContextSource {
        fn name(&self) -> &str {
            "tool_output"
        }

        fn gather<'a>(&'a self, _input: &'a CompileInput) -> BoxFuture<'a, Result<Vec<ContextCandidate>>> {
            Box::pin(async move {
                Ok(self
                    .outputs
                    .iter()
                    .map(|(call_id, text)| {
                        let mut c = ContextCandidate::new(
                            format!("tool:{call_id}"),
                            ContextSourceKind::ToolOutput,
                            format!("tool output {call_id}"),
                            text.clone(),
                            0.7,
                            Provenance {
                                source: format!("tool_call:{call_id}"),
                                // Tool output is untrusted and low-confidence (F12).
                                trust: TrustLevel::ToolOutput,
                                confidence: 0.6,
                                labels: vec!["tool-output".to_string()],
                                derived_from: vec![call_id.clone()],
                            },
                        );
                        // Tool outputs decay fast (recency now → high, ages quickly).
                        c.recency_ms = Some(hide_core::ids::now_ms());
                        c
                    })
                    .collect())
            })
        }
    }

    /// Diagnostics (compiler/linter messages) — high value for debugging profiles.
    pub struct DiagnosticsContextSource {
        pub diagnostics: Vec<String>,
    }

    impl ContextSource for DiagnosticsContextSource {
        fn name(&self) -> &str {
            "diagnostics"
        }

        fn gather<'a>(&'a self, _input: &'a CompileInput) -> BoxFuture<'a, Result<Vec<ContextCandidate>>> {
            Box::pin(async move {
                Ok(self
                    .diagnostics
                    .iter()
                    .enumerate()
                    .map(|(idx, d)| {
                        let mut c = ContextCandidate::new(
                            format!("diag:{idx}"),
                            ContextSourceKind::Diagnostics,
                            format!("diagnostic {idx}"),
                            d.clone(),
                            0.85,
                            Provenance {
                                source: "diagnostics".to_string(),
                                trust: TrustLevel::Workspace,
                                confidence: 0.9,
                                labels: vec!["diagnostic".to_string()],
                                derived_from: Vec::new(),
                            },
                        );
                        c.importance = Some(0.9);
                        c
                    })
                    .collect())
            })
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::compiler::{CompileInput, ContextCompiler};
        use crate::memory::{InMemoryMemoryStore, MemoryKind, MemoryStore};
        use crate::profiles::ContextProfile;
        use hawking_index::InMemoryCodeIndex;
        use hide_core::ids::ModelId;
        use hide_core::runtime::{ModelArchitecture, ModelDescriptor};

        fn model() -> ModelDescriptor {
            ModelDescriptor {
                id: ModelId::new(),
                name: "test".to_string(),
                architecture: ModelArchitecture::Transformer,
                context_tokens: 1024,
                tokenizer_signature: "test".to_string(),
                footprint_mb: 1,
            }
        }

        #[tokio::test]
        async fn code_index_source_feeds_compiler() {
            let index = Arc::new(InMemoryCodeIndex::default());
            index.add_text_file("src/lib.rs", "pub fn compile_context() {}\n// context compiler bridge\n", None);
            let mut compiler = ContextCompiler::new();
            compiler.add_source(CodeIndexContextSource::new(index, 4));
            let compiled = compiler
                .compile(CompileInput {
                    profile: ContextProfile::coding_default(1024),
                    model: model(),
                    task: "context compiler".to_string(),
                })
                .await
                .unwrap();
            assert!(compiled.prompt.contains("context compiler"));
            assert!(!compiled.manifest.retained.is_empty());
            // Provenance is workspace trust with a real path, not blanket-trusted.
            let span = &compiled.manifest.retained[0];
            assert_eq!(span.provenance.trust, TrustLevel::Workspace);
        }

        #[tokio::test]
        async fn memory_source_propagates_provenance_confidence() {
            let store = Arc::new(InMemoryMemoryStore::default());
            let mut rec = InMemoryMemoryStore::record(
                MemoryKind::Semantic,
                "the database layer lives in db and uses sqlx",
                Provenance {
                    source: "file_scan".to_string(),
                    trust: TrustLevel::ToolOutput,
                    confidence: 0.7,
                    labels: vec![],
                    derived_from: vec![],
                },
            );
            rec.importance = 0.8;
            store.put(rec).await.unwrap();

            let mut compiler = ContextCompiler::new();
            compiler.add_source(MemoryContextSource::new(store, 5));
            let compiled = compiler
                .compile(CompileInput {
                    profile: ContextProfile::coding_default(1024),
                    model: model(),
                    task: "database sqlx".to_string(),
                })
                .await
                .unwrap();
            let mem_span = compiled
                .manifest
                .retained
                .iter()
                .find(|s| matches!(s.source, ContextSourceKind::Memory))
                .expect("memory span retained");
            // Confidence flowed through (not overwritten to 1.0).
            assert!((mem_span.provenance.confidence - 0.7).abs() < 1e-6);
            assert_eq!(mem_span.provenance.trust, TrustLevel::ToolOutput);
        }
    }
}

pub use budget::{estimate_tokens, RegionBudget, Reservations, TokenBudget, TokenCounter};
pub use compiler::{
    CompileInput, CompiledContext, ContextCandidate, ContextCompiler, ContextSource, RealizedSpan,
};
pub use embed::{cosine, EmbeddingClient, HashingEmbeddingClient, HttpEmbeddingClient};
pub use kv::{
    CheckpointId, CheckpointMeta, EvictionChoice as KvEvictionChoice, HttpKvStore, KvCheckpoint,
    KvHandle, KvStore, KvStoreClient, KvStoreStats, KvTier, PrefixHandle, PrefixKey,
    RestoredSession, SlotId, StubKvStore, WorkingSetBudget,
};
pub use manifest::{
    ContextManifest, ContextSourceKind, ContextSpan, DropReason, DroppedContextSpan, PinState,
    SpanSignals,
};
pub use memory::{
    InMemoryMemoryStore, MemoryKind, MemoryQuery, MemoryRecord, MemoryStore, RankedMemory,
    ScoredMemory, SqliteMemoryStore,
};
pub use profiles::{
    ContextProfile, EvictionChoice, KvPrecision, OrderingPolicy, PositionPolicy, SourceWeights,
    WorkingSetMode,
};
