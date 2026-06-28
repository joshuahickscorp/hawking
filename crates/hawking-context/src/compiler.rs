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
    span_content_id, CompactionEvent, ContextManifest, ContextSourceKind, ContextSpan,
    DropReason, DroppedContextSpan, ManifestBudget, ManifestModel, ManifestProfile, PinState,
    SpanSignals,
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
    fn gather<'a>(
        &'a self,
        input: &'a CompileInput,
    ) -> BoxFuture<'a, Result<Vec<ContextCandidate>>>;

    /// Cheap candidate enumeration (handles + estimates). Defaults to `gather`
    /// so existing sources keep working; lazy sources override to avoid bodies.
    fn candidates<'a>(
        &'a self,
        input: &'a CompileInput,
    ) -> BoxFuture<'a, Result<Vec<ContextCandidate>>> {
        self.gather(input)
    }

    /// Materialize a selected candidate's tokens. Default returns its `text`.
    fn realize<'a>(
        &'a self,
        c: &'a ContextCandidate,
        _budget_tokens: usize,
    ) -> BoxFuture<'a, Result<RealizedSpan>> {
        let text = c.text.clone();
        Box::pin(async move {
            Ok(RealizedSpan {
                text,
                compacted: false,
            })
        })
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
        Box::pin(async move {
            Ok(default_truncate(&text, target_tokens, counter))
        })
    }
}

/// Default degrade: truncate to ~`target_tokens` on a whitespace boundary.
fn default_truncate(text: &str, target_tokens: usize, counter: &TokenCounter) -> Option<RealizedSpan> {
    if target_tokens == 0 || text.is_empty() {
        return None;
    }
    if counter.count(text) <= target_tokens {
        return Some(RealizedSpan {
            text: text.to_string(),
            compacted: false,
        });
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
    Some(RealizedSpan {
        text: truncated,
        compacted: true,
    })
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
}

impl ContextCompiler {
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            counter: TokenCounter::heuristic(),
            embedder: None,
            session_id: None,
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
            cand_vecs = embedder.embed(&texts).await.unwrap_or_else(|_| {
                vec![Vec::new(); cands.len()]
            });
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
                let band = c.score.clamp(0.0, 1.0)
                    * band_multiplier(weights, &c.source);
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
                    signals: SpanSignals {
                        recency,
                        importance,
                        relevance,
                        redundancy: 0.0,
                    },
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
        let (pinned, rest): (Vec<EntryCarrier>, Vec<EntryCarrier>) = entries
            .into_iter()
            .partition(|e| matches!(e.cand.pin, PinState::NeverEvict | PinState::UserPinned));

        for e in pinned {
            if let Some(left) = self
                .admit(e, &mut free, &mut selected, &mut compaction_events, true)
                .await?
            {
                deferred.push(left);
            }
        }

        // Value-density greedy over the rest.
        // Re-rank by density = value / tokens.
        let mut pool = rest;
        pool.sort_by(|a, b| {
            let da = a.base_value / (a.cand.token_count().max(1) as f32);
            let db = b.base_value / (b.cand.token_count().max(1) as f32);
            db.partial_cmp(&da)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cand.id.cmp(&b.cand.id))
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
            if let Some(left) = self
                .admit(e, &mut free, &mut selected, &mut compaction_events, false)
                .await?
            {
                deferred.push(left);
            }
        }

        // --- 4. Bounded local-improvement: pull the highest-value deferred span
        //         that now fits (whole or degraded) into leftover budget; cap
        //         at improve_iters. A real swap-aware pass over body-bearing
        //         deferred entries.
        self.local_improve(
            &mut selected,
            &mut deferred,
            &mut compaction_events,
            &mut free,
            profile.improve_iters,
        )
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
            ctx_len_effective: total,
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

        Ok(CompiledContext {
            prompt: prompt_parts.join("\n\n"),
            manifest,
        })
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
                let tokens = self.counter.count(&r.text).min(*free).max(usize::from(!is_pin));
                *free = free.saturating_sub(tokens);
                let original_id = e.cand.id.clone();
                let result_id = span_content_id(&e.cand.source, &e.cand.title, &r.text);
                let ratio = tokens as f32 / cost.max(1) as f32;
                let cf = if r.compacted {
                    compaction.push(CompactionEvent {
                        original_id: original_id.clone(),
                        result_id,
                        method: "degrade".to_string(),
                        model: None,
                        ratio,
                    });
                    Some(crate::manifest::CompactedFrom {
                        original_id,
                        method: "degrade".to_string(),
                        ratio,
                    })
                } else {
                    None
                };
                let text = r.text;
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

    fn max_redundancy(
        &self,
        cand_vecs: &[Vec<f32>],
        idx: usize,
        selected: &[SelectedSpan],
    ) -> f32 {
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
            match self
                .admit(e, free, selected, compaction, false)
                .await?
            {
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

fn band_multiplier(
    weights: &crate::profiles::SourceWeights,
    kind: &ContextSourceKind,
) -> f32 {
    weights
        .band_by_kind
        .get(&format!("{kind:?}"))
        .copied()
        .unwrap_or(1.0)
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
        b.value
            .partial_cmp(&a.value)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
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
        fn gather<'a>(
            &'a self,
            _input: &'a CompileInput,
        ) -> BoxFuture<'a, Result<Vec<ContextCandidate>>> {
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
        fn gather<'a>(
            &'a self,
            input: &'a CompileInput,
        ) -> BoxFuture<'a, Result<Vec<ContextCandidate>>> {
            self.candidates(input)
        }
        fn candidates<'a>(
            &'a self,
            _input: &'a CompileInput,
        ) -> BoxFuture<'a, Result<Vec<ContextCandidate>>> {
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
            Box::pin(async move {
                Ok(RealizedSpan {
                    text: body,
                    compacted: false,
                })
            })
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
        let high = compiled
            .manifest
            .retained
            .iter()
            .find(|s| s.title == "high")
            .expect("high retained");
        // The bulky low-value span must either be dropped, or degraded
        // (compacted) to fit — never admitted whole at full size.
        let low_dropped = compiled.manifest.dropped.iter().any(|d| d.id == "low");
        let low_compacted = compiled
            .manifest
            .retained
            .iter()
            .any(|s| s.title == "low" && s.compacted_from.is_some());
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

        let realized_ids: HashSet<String> =
            realized.lock().unwrap().iter().cloned().collect();
        // The selected span was realized.
        assert!(
            realized_ids.contains("keep"),
            "selected candidate must be realized; realized={realized_ids:?}"
        );
        // The dropped span was NEVER realized — its (huge) body was never touched.
        assert!(
            !realized_ids.contains("drop"),
            "dropped candidate must NOT be realized; realized={realized_ids:?}"
        );
        // And it is recorded as a no-fit drop, not silently lost.
        assert!(
            compiled
                .manifest
                .dropped
                .iter()
                .any(|d| d.id == "drop" && d.reason == DropReason::NoFit),
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
            fn gather<'a>(
                &'a self,
                input: &'a CompileInput,
            ) -> BoxFuture<'a, Result<Vec<ContextCandidate>>> {
                self.candidates(input)
            }
            fn candidates<'a>(
                &'a self,
                _input: &'a CompileInput,
            ) -> BoxFuture<'a, Result<Vec<ContextCandidate>>> {
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
                Box::pin(async move {
                    Ok(RealizedSpan {
                        text: "word ".repeat(100_000),
                        compacted: false,
                    })
                })
            }
            fn degrade<'a>(
                &'a self,
                _c: &'a ContextCandidate,
                target_tokens: usize,
                _counter: &'a TokenCounter,
            ) -> BoxFuture<'a, Result<Option<RealizedSpan>>> {
                *self.calls.lock().unwrap() += 1;
                let n = target_tokens.max(1);
                Box::pin(async move {
                    Ok(Some(RealizedSpan {
                        text: "w ".repeat(n),
                        compacted: true,
                    }))
                })
            }
        }

        let mut compiler = ContextCompiler::new();
        compiler.add_source(DegradeSpy {
            calls: degraded_calls.clone(),
        });
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
        assert!(
            !compiled.manifest.compaction_events.is_empty(),
            "degrade should record a compaction event"
        );
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
            compiled
                .manifest
                .dropped
                .iter()
                .any(|d| d.reason == DropReason::Redundant),
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
            ContextCandidate::new(
                "code",
                ContextSourceKind::Code,
                "code",
                body,
                0.5,
                Provenance::trusted("t"),
            ),
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
            compiled
                .manifest
                .retained
                .iter()
                .any(|s| matches!(s.source, ContextSourceKind::Diagnostics)),
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
        let standard_kept_code = standard
            .manifest
            .retained
            .iter()
            .any(|s| s.title == "code");
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
