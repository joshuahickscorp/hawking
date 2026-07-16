//! HIDE model orchestration (bible ch.06).
//!
//! The orchestrator is the shell-side fleet router over one or more
//! `hawking-serve` instances. It is HTTP/interface only — no GPU code here.
//!
//! The pipeline is: a request is **routed** ([`router`]) to a role; the role's
//! endpoint is resolved and called by the [`executor`]; a confidence-gated
//! [`escalation`] cascade retries up a stronger role when the cheap one is
//! uncertain; [`confidence`] supplies the self-consistency signal; [`grammar`]
//! enforces output envelopes via validate-and-retry; [`scheduler`] gates roles
//! against the machine's energy/thermal/RAM budget; and [`adapters`] selects
//! LoRA deltas per role/task. Live model calls cross the [`inference`] trait,
//! implemented over HTTP by [`http_client`].

#[rustfmt::skip]
pub mod adapters {
    //! LoRA / adapter descriptors + selection (ch.06 §4.9).
    //!
    //! No LoRA *serving* exists in the runtime yet ([RUNTIME-SIDE — LATER]); this
    //! module is the **selection policy** the router uses today and the clean seam a
    //! runtime adapter-loader will implement. The policy is real: given a task, a
    //! detected language, and config, it resolves the right adapter set
    //! (`["rust","personal"]` for a Rust edit, `["commit-msg"]` for a commit, etc.)
    //! and validates the choice against a role's `caps.lora`.

    use hide_core::runtime::ModelRole;
    use serde::{Deserialize, Serialize};
    use std::collections::BTreeMap;

    /// A single resident-or-on-disk adapter the runtime can apply as a low-rank
    /// delta over the base forward.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct AdapterDescriptor {
        pub id: String,
        pub kind: AdapterKind,
        /// Path/artifact id the runtime loader resolves (opaque to the shell).
        pub artifact: String,
        /// Default blend weight when selected.
        pub default_scale: f32,
    }

    /// What an adapter specializes.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum AdapterKind {
        /// Per-language idioms (`rust`, `ts`, `python`).
        Language(String),
        /// Per-task (`commit-msg`, `sql`, `test-gen`).
        Task(String),
        /// The user's accepted-edit personal adapter (§4.10).
        Personal,
    }

    /// One selected adapter + its blend weight (the §4.9.1 `AdapterRef`).
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct AdapterRef {
        pub id: String,
        pub scale: f32,
    }

    /// The router's resolved adapter set for a request (the §4.9.1 `AdapterSelection`).
    #[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
    pub struct AdapterSelection {
        /// Usually 0..2 adapters, e.g. `[language, personal]`.
        pub adapters: Vec<AdapterRef>,
    }

    impl AdapterSelection {
        pub fn is_empty(&self) -> bool {
            self.adapters.is_empty()
        }
    }

    /// The catalog of available adapters + the selection policy.
    #[derive(Debug, Clone, Default)]
    pub struct AdapterRegistry {
        by_id: BTreeMap<String, AdapterDescriptor>,
        /// Whether the user's personal adapter is enabled (opt-in, §4.10).
        personal_enabled: bool,
    }

    impl AdapterRegistry {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn register(&mut self, descriptor: AdapterDescriptor) {
            self.by_id.insert(descriptor.id.clone(), descriptor);
        }

        pub fn enable_personal(&mut self, enabled: bool) {
            self.personal_enabled = enabled;
        }

        pub fn get(&self, id: &str) -> Option<&AdapterDescriptor> {
            self.by_id.get(id)
        }

        /// Pick adapters for a request (§4.4.2 `pick_adapter`): a language adapter
        /// (if one exists for the detected language) plus the personal adapter (if
        /// enabled), or a task adapter for task-shaped requests. The choice is
        /// validated against the role's `caps.lora` — a role without LoRA support
        /// gets an empty selection (it degrades to the base model).
        pub fn select(&self, role: &ModelRole, task_kind: &str, language: Option<&str>) -> AdapterSelection {
            if !role.caps.lora {
                return AdapterSelection::default();
            }
            let mut selection = AdapterSelection::default();

            // Task adapters take the slot for task-shaped requests (commit-msg, sql).
            if let Some(task_adapter) = self.task_adapter_for(task_kind) {
                selection.adapters.push(AdapterRef { id: task_adapter.id.clone(), scale: task_adapter.default_scale });
            } else if let Some(lang) = language {
                if let Some(lang_adapter) = self.language_adapter_for(lang) {
                    selection
                        .adapters
                        .push(AdapterRef { id: lang_adapter.id.clone(), scale: lang_adapter.default_scale });
                }
            }

            // The personal adapter composes on top (when enabled and present).
            if self.personal_enabled {
                if let Some(personal) = self.by_id.values().find(|d| d.kind == AdapterKind::Personal) {
                    selection.adapters.push(AdapterRef { id: personal.id.clone(), scale: personal.default_scale });
                }
            }
            selection
        }

        fn language_adapter_for(&self, language: &str) -> Option<&AdapterDescriptor> {
            let lang = language.to_lowercase();
            self.by_id.values().find(|d| matches!(&d.kind, AdapterKind::Language(l) if l.eq_ignore_ascii_case(&lang)))
        }

        fn task_adapter_for(&self, task_kind: &str) -> Option<&AdapterDescriptor> {
            // Map a few task kinds onto their task adapters by id convention.
            let id = match task_kind {
                "commit_msg" | "commit-msg" => "commit-msg",
                "sql" => "sql",
                "test_gen" | "test-gen" => "test-gen",
                _ => return None,
            };
            self.by_id.get(id).filter(|d| matches!(d.kind, AdapterKind::Task(_)))
        }
    }

    /// Map a file extension to the language adapter key the registry uses.
    pub fn language_for_extension(ext: &str) -> Option<&'static str> {
        Some(match ext.trim_start_matches('.') {
            "rs" => "rust",
            "ts" | "tsx" => "ts",
            "js" | "jsx" => "js",
            "py" => "python",
            "go" => "go",
            "java" => "java",
            "c" | "h" => "c",
            "cpp" | "cc" | "hpp" => "cpp",
            _ => return None,
        })
    }

    /// The seam a runtime adapter-loader will implement ([RUNTIME-SIDE — LATER]).
    /// Today only the no-op / not-loaded path exists; the shell-today fallback is a
    /// separately-merged model served as its own role (§4.9.3).
    pub trait AdapterServer: Send + Sync {
        /// Ensure the given adapters are resident; returns the ids actually loaded.
        fn ensure_resident(&self, adapters: &[AdapterRef]) -> hide_core::Result<Vec<String>>;
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use hide_core::ids::{ModelId, RoleId};
        use hide_core::runtime::{ModelArchitecture, ModelDescriptor, ProviderCaps, RolePurpose, SamplerProfile};

        fn role_with_lora(lora: bool) -> ModelRole {
            ModelRole {
                id: RoleId::new(),
                name: "hero".into(),
                purpose: RolePurpose::HeroCoder,
                model: ModelDescriptor {
                    id: ModelId::new(),
                    name: "hero".into(),
                    architecture: ModelArchitecture::Transformer,
                    context_tokens: 4096,
                    tokenizer_signature: "tok".into(),
                    footprint_mb: 4600,
                },
                caps: ProviderCaps { lora, ..ProviderCaps::hawking_local_shell_today() },
                default_sampler: SamplerProfile::deterministic_edit(),
                endpoint: None,
                cost: None,
                escalates_to: None,
                metadata: BTreeMap::new(),
            }
        }

        fn registry() -> AdapterRegistry {
            let mut r = AdapterRegistry::new();
            r.register(AdapterDescriptor {
                id: "rust".into(),
                kind: AdapterKind::Language("rust".into()),
                artifact: "rust.lora".into(),
                default_scale: 1.0,
            });
            r.register(AdapterDescriptor {
                id: "commit-msg".into(),
                kind: AdapterKind::Task("commit-msg".into()),
                artifact: "commit-msg.lora".into(),
                default_scale: 1.0,
            });
            r.register(AdapterDescriptor {
                id: "personal".into(),
                kind: AdapterKind::Personal,
                artifact: "personal.lora".into(),
                default_scale: 0.5,
            });
            r
        }

        #[test]
        fn no_lora_cap_means_empty_selection() {
            let r = registry();
            let sel = r.select(&role_with_lora(false), "edit_code", Some("rust"));
            assert!(sel.is_empty());
        }

        #[test]
        fn rust_edit_selects_language_and_personal() {
            let mut r = registry();
            r.enable_personal(true);
            let sel = r.select(&role_with_lora(true), "edit_code", Some("rust"));
            let ids: Vec<&str> = sel.adapters.iter().map(|a| a.id.as_str()).collect();
            assert_eq!(ids, vec!["rust", "personal"]);
            assert_eq!(sel.adapters[1].scale, 0.5);
        }

        #[test]
        fn task_adapter_takes_the_slot() {
            let r = registry();
            let sel = r.select(&role_with_lora(true), "commit-msg", Some("rust"));
            let ids: Vec<&str> = sel.adapters.iter().map(|a| a.id.as_str()).collect();
            assert_eq!(ids, vec!["commit-msg"]);
        }

        #[test]
        fn extension_mapping() {
            assert_eq!(language_for_extension("rs"), Some("rust"));
            assert_eq!(language_for_extension(".tsx"), Some("ts"));
            assert_eq!(language_for_extension("md"), None);
        }
    }
}
#[rustfmt::skip]
pub mod confidence {
    //! Confidence signals and self-consistency voting (ch.06 §4.7).
    //!
    //! The exact per-token logit signals (token confidence, entropy, self-certainty)
    //! are a gated runtime readback ([RUNTIME-SIDE — LATER]). What works **today**,
    //! with no runtime hook, is **self-consistency voting**: sample `k` completions
    //! (cheap profile), normalize, cluster identical answers, and score agreement.
    //! That is the [SHELL-TODAY] escalation gate the cascade (`escalation.rs`)
    //! consumes.
    //!
    //! This module is pure: it operates on already-collected sample strings, so it
    //! is fully testable without a runtime.

    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;

    /// The result of voting over k samples.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct VoteResult {
        /// The plurality answer (normalized form), or `None` if there were no samples.
        pub winner: Option<String>,
        /// One *original* (un-normalized) sample text that maps to the winner.
        pub winner_original: Option<String>,
        /// Fraction of samples agreeing with the winner, in [0,1].
        pub agreement: f32,
        /// Number of distinct normalized answers.
        pub distinct: usize,
        /// Total samples considered.
        pub total: usize,
        /// Per-answer tallies (normalized answer → count), sorted desc by count.
        pub tallies: Vec<(String, usize)>,
    }

    impl VoteResult {
        /// A scalar confidence in [0,1] derived from agreement and answer spread.
        /// Agreement dominates; a long tail of distinct alternatives discounts it.
        pub fn confidence(&self) -> f32 {
            if self.total == 0 {
                return 0.0;
            }
            // Penalize fragmentation: with d distinct answers over n samples, the
            // "spread factor" shrinks confidence when many singletons appear.
            let spread =
                if self.distinct <= 1 { 1.0 } else { 1.0 - ((self.distinct - 1) as f32 / self.total as f32) * 0.5 };
            (self.agreement * spread).clamp(0.0, 1.0)
        }
    }

    /// How to normalize a sample before clustering. Choosing the right normalizer
    /// is what makes voting work for a given task (a tool-call JSON should be
    /// canonicalized; a classifier label just trimmed/lowercased).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub enum AnswerNormalizer {
        /// Trim only.
        Trimmed,
        /// Trim + lowercase (classifier labels, yes/no).
        CaseFold,
        /// Parse as JSON and re-serialize with sorted keys (tool-calls, structured
        /// single-value answers). Falls back to trimmed text if not JSON.
        CanonicalJson,
    }

    impl AnswerNormalizer {
        pub fn normalize(&self, sample: &str) -> String {
            match self {
                AnswerNormalizer::Trimmed => sample.trim().to_string(),
                AnswerNormalizer::CaseFold => sample.trim().to_lowercase(),
                AnswerNormalizer::CanonicalJson => {
                    canonical_json(sample.trim()).unwrap_or_else(|| sample.trim().to_string())
                }
            }
        }
    }

    /// Re-serialize a JSON document with object keys sorted (recursively), so
    /// `{"a":1,"b":2}` and `{"b":2,"a":1}` cluster together. `None` if not JSON.
    fn canonical_json(s: &str) -> Option<String> {
        let value: serde_json::Value = serde_json::from_str(s).ok()?;
        Some(canonicalize_value(&value).to_string())
    }

    fn canonicalize_value(v: &serde_json::Value) -> serde_json::Value {
        match v {
            serde_json::Value::Object(map) => {
                let mut sorted = serde_json::Map::new();
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                for k in keys {
                    sorted.insert(k.clone(), canonicalize_value(&map[k]));
                }
                serde_json::Value::Object(sorted)
            }
            serde_json::Value::Array(arr) => serde_json::Value::Array(arr.iter().map(canonicalize_value).collect()),
            other => other.clone(),
        }
    }

    /// Vote over a set of samples using the given normalizer.
    pub fn self_consistency_vote(samples: &[String], normalizer: AnswerNormalizer) -> VoteResult {
        let total = samples.len();
        if total == 0 {
            return VoteResult {
                winner: None,
                winner_original: None,
                agreement: 0.0,
                distinct: 0,
                total: 0,
                tallies: Vec::new(),
            };
        }
        let mut counts: HashMap<String, usize> = HashMap::new();
        let mut first_original: HashMap<String, String> = HashMap::new();
        for s in samples {
            let key = normalizer.normalize(s);
            *counts.entry(key.clone()).or_insert(0) += 1;
            first_original.entry(key).or_insert_with(|| s.clone());
        }
        let mut tallies: Vec<(String, usize)> = counts.into_iter().collect();
        // Sort by count desc, then by answer asc for determinism on ties.
        tallies.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let distinct = tallies.len();
        let (winner, winner_count) = tallies.first().map(|(k, c)| (Some(k.clone()), *c)).unwrap_or((None, 0));
        let winner_original = winner.as_ref().and_then(|w| first_original.get(w).cloned());
        let agreement = winner_count as f32 / total as f32;
        VoteResult { winner, winner_original, agreement, distinct, total, tallies }
    }

    /// Shannon entropy (nats) of a probability distribution, ignoring zeros.
    /// Exposed so a future runtime logprob path can reuse the same metric for
    /// first-token entropy (§4.7).
    pub fn entropy(probs: &[f32]) -> f32 {
        probs.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum()
    }

    /// Self-certainty: divergence of a distribution from uniform, normalized to
    /// [0,1] (0 = uniform / maximally uncertain, 1 = a point mass). This is the
    /// §4.7 "separates correct from incorrect better than perplexity" signal,
    /// computed from a distribution once the runtime exposes one.
    pub fn self_certainty(probs: &[f32]) -> f32 {
        let n = probs.len();
        if n <= 1 {
            return 1.0;
        }
        let max_entropy = (n as f32).ln();
        let h = entropy(probs);
        (1.0 - h / max_entropy).clamp(0.0, 1.0)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn unanimous_vote_is_high_confidence() {
            let samples = vec!["yes".into(), "yes".into(), "yes".into()];
            let r = self_consistency_vote(&samples, AnswerNormalizer::CaseFold);
            assert_eq!(r.winner.as_deref(), Some("yes"));
            assert_eq!(r.agreement, 1.0);
            assert_eq!(r.distinct, 1);
            assert!(r.confidence() > 0.95);
        }

        #[test]
        fn split_vote_is_low_confidence() {
            let samples = vec!["a".into(), "b".into(), "c".into()];
            let r = self_consistency_vote(&samples, AnswerNormalizer::Trimmed);
            assert_eq!(r.distinct, 3);
            assert!((r.agreement - 1.0 / 3.0).abs() < 1e-6);
            assert!(r.confidence() < 0.4);
        }

        #[test]
        fn json_clusters_regardless_of_key_order() {
            let samples = vec![
                "{\"name\":\"edit\",\"path\":\"a\"}".into(),
                "{\"path\":\"a\",\"name\":\"edit\"}".into(),
                "{\"name\":\"read\"}".into(),
            ];
            let r = self_consistency_vote(&samples, AnswerNormalizer::CanonicalJson);
            // The two reorderings cluster → winner has 2 of 3.
            assert_eq!(r.distinct, 2);
            assert!((r.agreement - 2.0 / 3.0).abs() < 1e-6);
        }

        #[test]
        fn entropy_and_self_certainty() {
            // Uniform over 2 → entropy ln(2), self-certainty 0.
            let uniform = [0.5, 0.5];
            assert!((entropy(&uniform) - 2.0_f32.ln()).abs() < 1e-5);
            assert!(self_certainty(&uniform) < 1e-5);
            // Point mass → self-certainty 1.
            let point = [1.0, 0.0];
            assert!((self_certainty(&point) - 1.0).abs() < 1e-5);
        }

        #[test]
        fn empty_samples_are_zero_confidence() {
            let r = self_consistency_vote(&[], AnswerNormalizer::Trimmed);
            assert_eq!(r.total, 0);
            assert_eq!(r.confidence(), 0.0);
            assert!(r.winner.is_none());
        }
    }
}
#[rustfmt::skip]
pub mod difficulty {
    use hide_core::runtime::InferenceRequest;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct DifficultyEstimate {
        pub score: f32,
        pub reason: String,
        pub signals: Vec<DifficultySignal>,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct DifficultySignal {
        pub name: String,
        pub value: f32,
    }

    #[derive(Default)]
    pub struct DifficultyEstimator;

    impl DifficultyEstimator {
        pub fn estimate(&self, request: &InferenceRequest) -> DifficultyEstimate {
            let prompt_len = request.prompt.chars().count() as f32;
            let mut score = (prompt_len / 12_000.0).min(0.45);
            let mut signals = vec![DifficultySignal { name: "prompt_length".to_string(), value: score }];
            for marker in ["refactor", "multi-file", "security", "architecture", "failing tests"] {
                if request.prompt.to_lowercase().contains(marker) {
                    score += 0.12;
                    signals.push(DifficultySignal { name: marker.to_string(), value: 0.12 });
                }
            }
            score = score.min(1.0);
            DifficultyEstimate {
                score,
                reason: if score > 0.65 {
                    "high difficulty; route to hero role".to_string()
                } else {
                    "low/medium difficulty; cheap role can try first".to_string()
                },
                signals,
            }
        }
    }
}
#[rustfmt::skip]
pub mod escalation {
    //! The confidence-gated escalation cascade (ch.06 §4.4 step 5, §4.7).
    //!
    //! This is the crate's reason to exist. Given a [`RouteDecision`] (a cheap role)
    //! and a way to measure a generation's confidence, the cascade:
    //!
    //! 1. runs the chosen role and collects its output,
    //! 2. computes a confidence signal,
    //! 3. if confidence is below the task threshold **and** the role declares an
    //!    `escalates_to` target **and** the escalation budget allows, retries on the
    //!    stronger role (sampler forced to `edit`, grammar/adapter carried),
    //! 4. repeats up the cascade graph until confident, out of budget, or at the top.
    //!
    //! Confidence today is [SHELL-TODAY] self-consistency voting (`confidence.rs`)
    //! and grammar validation (`grammar.rs`) — both work without a runtime logprob
    //! hook. The runtime per-token signals (§4.7) slot in behind [`ConfidenceProbe`]
    //! later.

    use crate::confidence::{self_consistency_vote, AnswerNormalizer};
    use crate::grammar::{GrammarMatcher, GrammarValidation};
    use crate::inference::InferenceClient;
    use crate::registry::RoleRegistry;
    use crate::router::RouteDecision;
    use hide_core::error::{HideError, Result};
    use hide_core::ids::RoleId;
    use hide_core::runtime::{GenerationStats, InferenceRequest, ModelRole, SamplerProfile, StreamChunk};
    use serde::{Deserialize, Serialize};
    use std::sync::Arc;

    /// Bounds the cascade so it cannot retry-up forever.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct EscalationBudget {
        /// Maximum number of *escalation hops* (role upgrades) allowed.
        pub max_escalations: u32,
        /// Number of samples to draw for self-consistency voting on a gateable task.
        pub vote_samples: u32,
        /// Confidence at or above which an answer is accepted without escalating.
        pub accept_threshold: f32,
    }

    impl Default for EscalationBudget {
        fn default() -> Self {
            Self { max_escalations: 2, vote_samples: 3, accept_threshold: 0.67 }
        }
    }

    /// Why a confidence pass decided to escalate (or not). Recorded for replay.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub enum EscalationReason {
        /// Confidence below the threshold.
        LowConfidence { confidence: f32, threshold: f32 },
        /// The grammar matcher rejected the output (and could not be repaired here).
        GrammarInvalid { code: String },
        /// Self-consistency vote disagreed across samples.
        VoteDisagreement { agreement: f32 },
    }

    /// One step of the cascade, recorded in the manifest (§4.4 "record both").
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct EscalationStep {
        pub role_id: RoleId,
        pub role_name: String,
        pub confidence: f32,
        pub output: String,
        /// `None` on the accepted (final) step; `Some` when this step escalated.
        pub escalated: Option<EscalationReason>,
    }

    /// The full result of running the cascade.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct EscalationOutcome {
        /// The accepted output text.
        pub output: String,
        /// The role that produced the accepted output.
        pub final_role_id: RoleId,
        /// Confidence of the accepted output.
        pub confidence: f32,
        /// Every step taken, in order (cheap → escalated).
        pub trail: Vec<EscalationStep>,
        /// Stats from the final generation.
        pub stats: GenerationStats,
    }

    impl EscalationOutcome {
        pub fn escalated(&self) -> bool {
            self.trail.len() > 1
        }
    }

    /// How confidence is measured for a generation. The default impl
    /// ([`SelfConsistencyProbe`]) is [SHELL-TODAY]; a runtime logprob probe slots in
    /// behind the same trait later (§4.7).
    pub trait ConfidenceProbe: Send + Sync {
        /// Score a single primary output, optionally given extra samples for voting
        /// and a grammar to validate against. Returns `(confidence, reason_if_low)`.
        fn score(
            &self,
            primary: &str,
            samples: &[String],
            grammar: Option<&GrammarMatcher>,
        ) -> (f32, Option<EscalationReason>);
    }

    /// The [SHELL-TODAY] probe: grammar validity is a hard gate, then
    /// self-consistency voting over the samples gives the confidence scalar.
    #[derive(Debug, Clone, Copy, Default)]
    pub struct SelfConsistencyProbe {
        pub normalizer_is_json: bool,
        pub accept_threshold: f32,
    }

    impl ConfidenceProbe for SelfConsistencyProbe {
        fn score(
            &self,
            primary: &str,
            samples: &[String],
            grammar: Option<&GrammarMatcher>,
        ) -> (f32, Option<EscalationReason>) {
            // Grammar invalidity is decisive: a dead-ended / unparseable envelope is
            // zero-confidence and triggers escalation.
            if let Some(g) = grammar {
                if let GrammarValidation::Retry(hint) = g.validate(primary) {
                    return (0.0, Some(EscalationReason::GrammarInvalid { code: hint.code }));
                }
            }
            let normalizer =
                if self.normalizer_is_json { AnswerNormalizer::CanonicalJson } else { AnswerNormalizer::CaseFold };
            // Include the primary in the vote pool.
            let mut pool: Vec<String> = Vec::with_capacity(samples.len() + 1);
            pool.push(primary.to_string());
            pool.extend_from_slice(samples);
            let vote = self_consistency_vote(&pool, normalizer);
            let confidence = vote.confidence();
            let threshold = if self.accept_threshold > 0.0 { self.accept_threshold } else { 0.67 };
            let reason = if confidence < threshold {
                Some(EscalationReason::VoteDisagreement { agreement: vote.agreement })
            } else {
                None
            };
            (confidence, reason)
        }
    }

    /// The cascade executor. Holds the registry (to resolve `escalates_to`) and an
    /// inference client (to run roles).
    pub struct EscalationCascade {
        registry: Arc<RoleRegistry>,
        client: Arc<dyn InferenceClient>,
        probe: Arc<dyn ConfidenceProbe>,
        budget: EscalationBudget,
    }

    impl EscalationCascade {
        pub fn new(
            registry: Arc<RoleRegistry>,
            client: Arc<dyn InferenceClient>,
            probe: Arc<dyn ConfidenceProbe>,
            budget: EscalationBudget,
        ) -> Self {
            Self { registry, client, probe, budget }
        }

        /// Collect a single completion's full text (concatenating token chunks).
        async fn run_once(&self, request: &InferenceRequest) -> Result<(String, GenerationStats)> {
            let mut text = String::new();
            let mut final_stats: Option<GenerationStats> = None;
            let mut err: Option<String> = None;
            {
                let mut sink = |chunk: StreamChunk| {
                    match chunk {
                        StreamChunk::Token { text: t, .. } => text.push_str(&t),
                        StreamChunk::Done { stats, .. } => final_stats = stats,
                        StreamChunk::Error { message } => err = Some(message),
                    }
                    Ok(())
                };
                self.client.generate(request.clone(), &mut sink).await?;
            }
            if let Some(message) = err {
                return Err(HideError::RuntimeUnavailable(message));
            }
            Ok((
                text,
                final_stats.unwrap_or(GenerationStats {
                    input_tokens: 0,
                    output_tokens: 0,
                    decode_tokens_per_second: None,
                }),
            ))
        }

        /// Build the request for a given role, forcing the edit sampler when
        /// escalating (the §4.4 step-5 contract).
        fn request_for_role(base: &InferenceRequest, role: &ModelRole, force_edit_sampler: bool) -> InferenceRequest {
            let mut req = base.clone();
            req.sampler = Some(if force_edit_sampler {
                SamplerProfile::deterministic_edit()
            } else {
                role.default_sampler.clone()
            });
            req
        }

        /// Run the cascade. `decision` selects the starting (cheap) role; the request
        /// is the task. Returns the accepted output plus the full escalation trail.
        pub async fn execute(&self, decision: &RouteDecision, request: &InferenceRequest) -> Result<EscalationOutcome> {
            let mut current_role = self
                .registry
                .get(&decision.role_id)
                .ok_or_else(|| HideError::NotFound(format!("role {}", decision.role_id)))?;
            let grammar = self.build_grammar(decision)?;
            let mut trail: Vec<EscalationStep> = Vec::new();
            let mut escalations: u32 = 0;
            let mut force_edit = false;

            loop {
                let role_request = Self::request_for_role(request, &current_role, force_edit);
                let (primary, stats) = self.run_once(&role_request).await?;

                // Draw extra samples for voting only if the task is gateable
                // (votable) and the budget asks for more than one.
                let samples = self.collect_vote_samples(&role_request, &primary).await?;

                let (confidence, reason) = self.probe.score(&primary, &samples, grammar.as_ref());

                let at_top = current_role.escalates_to.is_none();
                let budget_left = escalations < self.budget.max_escalations;
                let should_escalate =
                    reason.is_some() && confidence < self.budget.accept_threshold && !at_top && budget_left;

                trail.push(EscalationStep {
                    role_id: current_role.id.clone(),
                    role_name: current_role.name.clone(),
                    confidence,
                    output: primary.clone(),
                    escalated: if should_escalate { reason.clone() } else { None },
                });

                if !should_escalate {
                    return Ok(EscalationOutcome {
                        output: primary,
                        final_role_id: current_role.id.clone(),
                        confidence,
                        trail,
                        stats,
                    });
                }

                // Escalate up the cascade graph.
                let next_id = current_role.escalates_to.clone().expect("checked !at_top above");
                let next_role = self
                    .registry
                    .get(&next_id)
                    .ok_or_else(|| HideError::NotFound(format!("escalation target role {next_id}")))?;
                current_role = next_role;
                escalations += 1;
                force_edit = true; // escalations always use the edit sampler
            }
        }

        fn build_grammar(&self, decision: &RouteDecision) -> Result<Option<GrammarMatcher>> {
            // A grammar string on the decision is treated as a JSON-Schema document
            // when it parses as JSON; otherwise it is ignored for the shell gate.
            let Some(spec_json) = &decision.grammar else {
                return Ok(None);
            };
            match crate::grammar::ShellGrammarCompiler::spec_from_schema(spec_json) {
                Ok(spec) => Ok(Some(GrammarMatcher::new(spec)?)),
                // A non-schema grammar marker (e.g. "tool-call-json") just disables
                // the shell-side gate; the runtime mask handles it later.
                Err(_) => Ok(None),
            }
        }

        async fn collect_vote_samples(&self, role_request: &InferenceRequest, _primary: &str) -> Result<Vec<String>> {
            let extra = self.budget.vote_samples.saturating_sub(1);
            if extra == 0 {
                return Ok(Vec::new());
            }
            let mut samples = Vec::with_capacity(extra as usize);
            for _ in 0..extra {
                let (text, _) = self.run_once(role_request).await?;
                samples.push(text);
            }
            Ok(samples)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::inference::ScriptedInferenceClient;
        use crate::router::RouteDecision;
        use hide_core::ids::{ModelId, RoleId};
        use hide_core::runtime::{ModelArchitecture, ModelDescriptor, ProviderCaps, RolePurpose, SamplerProfile};
        use std::collections::BTreeMap;

        fn make_role(name: &str, id: RoleId, escalates_to: Option<RoleId>) -> ModelRole {
            ModelRole {
                id,
                name: name.to_string(),
                purpose: RolePurpose::FastDraft,
                model: ModelDescriptor {
                    id: ModelId::new(),
                    name: name.to_string(),
                    architecture: ModelArchitecture::Transformer,
                    context_tokens: 4096,
                    tokenizer_signature: "tok".into(),
                    footprint_mb: 500,
                },
                caps: ProviderCaps::hawking_local_shell_today(),
                default_sampler: SamplerProfile::deterministic_edit(),
                endpoint: None,
                cost: None,
                escalates_to,
                metadata: BTreeMap::new(),
            }
        }

        fn base_request() -> InferenceRequest {
            InferenceRequest {
                task_kind: "classify".into(),
                prompt: "is this safe?".into(),
                messages: Vec::new(),
                max_output_tokens: 8,
                sampler: None,
                grammar: None,
                want_logprobs: false,
                metadata: BTreeMap::new(),
            }
        }

        #[tokio::test]
        async fn confident_cheap_role_does_not_escalate() {
            let draft_id = RoleId::new();
            let hero_id = RoleId::new();
            let registry = Arc::new(RoleRegistry::default());
            registry.register(make_role("draft", draft_id.clone(), Some(hero_id.clone())));
            registry.register(make_role("hero", hero_id.clone(), None));

            // Every sample is "yes" → unanimous → high confidence → no escalation.
            let client = Arc::new(ScriptedInferenceClient::new(vec!["yes".into(), "yes".into(), "yes".into()]));
            let probe = Arc::new(SelfConsistencyProbe { normalizer_is_json: false, accept_threshold: 0.67 });
            let cascade = EscalationCascade::new(registry, client, probe, EscalationBudget::default());
            let decision = RouteDecision {
                role_id: draft_id.clone(),
                provider: "hawking-local".into(),
                sampler: SamplerProfile::deterministic_edit(),
                grammar: None,
                reason: "test".into(),
                estimated_difficulty: crate::difficulty::DifficultyEstimate {
                    score: 0.1,
                    reason: "low".into(),
                    signals: vec![],
                },
            };
            let outcome = cascade.execute(&decision, &base_request()).await.unwrap();
            assert!(!outcome.escalated());
            assert_eq!(outcome.final_role_id, draft_id);
            assert_eq!(outcome.output, "yes");
        }

        #[tokio::test]
        async fn disagreement_escalates_to_hero() {
            let draft_id = RoleId::new();
            let hero_id = RoleId::new();
            let registry = Arc::new(RoleRegistry::default());
            registry.register(make_role("draft", draft_id.clone(), Some(hero_id.clone())));
            registry.register(make_role("hero", hero_id.clone(), None));

            // Draft phase: 3 disagreeing samples → escalate.
            // Hero phase: 3 agreeing samples → accept.
            let client = Arc::new(ScriptedInferenceClient::new(vec![
                "yes".into(),
                "no".into(),
                "maybe".into(),
                "yes".into(),
                "yes".into(),
                "yes".into(),
            ]));
            let probe = Arc::new(SelfConsistencyProbe { normalizer_is_json: false, accept_threshold: 0.67 });
            let cascade = EscalationCascade::new(
                registry,
                client,
                probe,
                EscalationBudget { max_escalations: 2, vote_samples: 3, accept_threshold: 0.67 },
            );
            let decision = RouteDecision {
                role_id: draft_id.clone(),
                provider: "hawking-local".into(),
                sampler: SamplerProfile::deterministic_edit(),
                grammar: None,
                reason: "test".into(),
                estimated_difficulty: crate::difficulty::DifficultyEstimate {
                    score: 0.1,
                    reason: "low".into(),
                    signals: vec![],
                },
            };
            let outcome = cascade.execute(&decision, &base_request()).await.unwrap();
            assert!(outcome.escalated());
            assert_eq!(outcome.final_role_id, hero_id);
            assert_eq!(outcome.trail.len(), 2);
            assert!(matches!(outcome.trail[0].escalated, Some(EscalationReason::VoteDisagreement { .. })));
        }

        #[tokio::test]
        async fn top_role_cannot_escalate_even_if_uncertain() {
            let hero_id = RoleId::new();
            let registry = Arc::new(RoleRegistry::default());
            registry.register(make_role("hero", hero_id.clone(), None));
            let client = Arc::new(ScriptedInferenceClient::new(vec!["a".into(), "b".into(), "c".into()]));
            let probe = Arc::new(SelfConsistencyProbe { normalizer_is_json: false, accept_threshold: 0.99 });
            let cascade = EscalationCascade::new(registry, client, probe, EscalationBudget::default());
            let decision = RouteDecision {
                role_id: hero_id.clone(),
                provider: "hawking-local".into(),
                sampler: SamplerProfile::deterministic_edit(),
                grammar: None,
                reason: "test".into(),
                estimated_difficulty: crate::difficulty::DifficultyEstimate {
                    score: 0.9,
                    reason: "high".into(),
                    signals: vec![],
                },
            };
            let outcome = cascade.execute(&decision, &base_request()).await.unwrap();
            // No escalation target → accept the best we have.
            assert!(!outcome.escalated());
            assert_eq!(outcome.final_role_id, hero_id);
            assert_eq!(outcome.trail.len(), 1);
        }
    }
}
#[rustfmt::skip]
pub mod executor {
    //! The executor: `RouteDecision.role_id` → endpoint → `InferenceClient` call.
    //!
    //! The router (`router.rs`) decides *which* role; the cascade (`escalation.rs`)
    //! decides *whether to retry up*. The executor is the glue that turns a role id
    //! into a concrete HTTP client (by its `endpoint`) and runs the call. It holds a
    //! small pool of per-endpoint clients so the fleet's several `hawking-serve`
    //! instances are reached without re-building a client per request.
    //!
    //! For the embedder role it dispatches to [`InferenceClient::embed`]; for every
    //! other role it streams `generate`. Live HTTP is behind the trait, so an
    //! [`Executor`] built with a stub client is fully testable offline.

    use crate::http_client::{GenerateRoute, HawkingHttpClient};
    use crate::inference::InferenceClient;
    use crate::registry::RoleRegistry;
    use crate::router::RouteDecision;
    use hide_core::error::{HideError, Result};
    use hide_core::ids::RoleId;
    use hide_core::runtime::{GenerationStats, InferenceRequest, RolePurpose, StreamChunk, TokenSink};
    use parking_lot::RwLock;
    use std::collections::HashMap;
    use std::sync::Arc;

    /// Resolves clients for endpoints. The default ([`HttpClientFactory`]) builds a
    /// streaming [`HawkingHttpClient`]; tests inject a factory that always returns a
    /// stub.
    pub trait ClientFactory: Send + Sync {
        /// Build (or return a cached) client for the given endpoint and route.
        fn client_for(&self, endpoint: &str, route: GenerateRoute) -> Arc<dyn InferenceClient>;
    }

    /// Builds and caches real HTTP clients per `(endpoint, route)`.
    #[derive(Default)]
    pub struct HttpClientFactory {
        cache: RwLock<HashMap<(String, GenerateRoute), Arc<dyn InferenceClient>>>,
    }

    impl ClientFactory for HttpClientFactory {
        fn client_for(&self, endpoint: &str, route: GenerateRoute) -> Arc<dyn InferenceClient> {
            let key = (endpoint.to_string(), route);
            if let Some(client) = self.cache.read().get(&key) {
                return client.clone();
            }
            let client: Arc<dyn InferenceClient> = Arc::new(HawkingHttpClient::with_route(endpoint, route));
            self.cache.write().insert(key, client.clone());
            client
        }
    }

    /// A factory that always hands back the same client, regardless of endpoint —
    /// for offline tests.
    pub struct FixedClientFactory(pub Arc<dyn InferenceClient>);

    impl ClientFactory for FixedClientFactory {
        fn client_for(&self, _endpoint: &str, _route: GenerateRoute) -> Arc<dyn InferenceClient> {
            self.0.clone()
        }
    }

    /// Maps route decisions to endpoint calls.
    pub struct Executor {
        registry: Arc<RoleRegistry>,
        factory: Arc<dyn ClientFactory>,
        /// Endpoint used when a role declares none (single-instance dev default).
        default_endpoint: String,
    }

    impl Executor {
        pub fn new(registry: Arc<RoleRegistry>, factory: Arc<dyn ClientFactory>) -> Self {
            Self { registry, factory, default_endpoint: "http://127.0.0.1:8080".to_string() }
        }

        pub fn with_default_endpoint(mut self, endpoint: impl Into<String>) -> Self {
            self.default_endpoint = endpoint.into();
            self
        }

        fn resolve_endpoint(&self, role_endpoint: Option<&str>) -> String {
            role_endpoint.map(str::to_string).unwrap_or_else(|| self.default_endpoint.clone())
        }

        /// Resolve the client for a role id, picking the chat route for chat tasks
        /// and the native route otherwise.
        fn client_for_role(&self, role_id: &RoleId, route: GenerateRoute) -> Result<Arc<dyn InferenceClient>> {
            let role = self
                .registry
                .get(role_id)
                .ok_or_else(|| HideError::NotFound(format!("role {role_id} not registered")))?;
            let endpoint = self.resolve_endpoint(role.endpoint.as_deref());
            Ok(self.factory.client_for(&endpoint, route))
        }

        /// Execute a routed generation: resolve the role's endpoint, build the
        /// client, and stream tokens into the sink.
        pub async fn execute<'a>(
            &'a self,
            decision: &RouteDecision,
            request: InferenceRequest,
            sink: TokenSink<'a>,
        ) -> Result<GenerationStats> {
            let route = if request.messages.is_empty() { GenerateRoute::Native } else { GenerateRoute::Chat };
            let client = self.client_for_role(&decision.role_id, route)?;
            client.generate(request, sink).await
        }

        /// Embed text through whichever role is configured as the embedder. If the
        /// decision's role is the embedder, that endpoint is used; otherwise the
        /// first registered embedder role is resolved.
        pub async fn embed(&self, decision: &RouteDecision, text: &str) -> Result<Vec<f32>> {
            let role = self
                .registry
                .get(&decision.role_id)
                .filter(|r| r.purpose == RolePurpose::Embedder)
                .or_else(|| self.registry.by_purpose(RolePurpose::Embedder).into_iter().next())
                .ok_or_else(|| HideError::NotFound("no embedder role registered".to_string()))?;
            let endpoint = self.resolve_endpoint(role.endpoint.as_deref());
            let client = self.factory.client_for(&endpoint, GenerateRoute::Native);
            client.embed(text).await
        }
    }

    /// Convenience: collect a routed generation into a single string (concatenating
    /// token chunks). Useful for callers that don't need streaming.
    pub async fn collect_to_string(
        executor: &Executor,
        decision: &RouteDecision,
        request: InferenceRequest,
    ) -> Result<(String, GenerationStats)> {
        let mut text = String::new();
        let mut stats = GenerationStats { input_tokens: 0, output_tokens: 0, decode_tokens_per_second: None };
        {
            let mut sink = |chunk: StreamChunk| {
                match chunk {
                    StreamChunk::Token { text: t, .. } => text.push_str(&t),
                    StreamChunk::Done { stats: s, .. } => {
                        if let Some(s) = s {
                            stats = s;
                        }
                    }
                    StreamChunk::Error { message } => return Err(HideError::RuntimeUnavailable(message)),
                }
                Ok(())
            };
            executor.execute(decision, request, &mut sink).await?;
        }
        Ok((text, stats))
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::difficulty::DifficultyEstimate;
        use crate::inference::StubInferenceClient;
        use hide_core::ids::{ModelId, RoleId};
        use hide_core::runtime::{ModelArchitecture, ModelDescriptor, ModelRole, ProviderCaps, SamplerProfile};
        use std::collections::BTreeMap;

        fn role(name: &str, id: RoleId, purpose: RolePurpose, endpoint: &str) -> ModelRole {
            ModelRole {
                id,
                name: name.into(),
                purpose,
                model: ModelDescriptor {
                    id: ModelId::new(),
                    name: name.into(),
                    architecture: ModelArchitecture::Transformer,
                    context_tokens: 4096,
                    tokenizer_signature: "tok".into(),
                    footprint_mb: 100,
                },
                caps: ProviderCaps { embeddings: true, ..ProviderCaps::hawking_local_shell_today() },
                default_sampler: SamplerProfile::deterministic_edit(),
                endpoint: Some(endpoint.into()),
                cost: None,
                escalates_to: None,
                metadata: BTreeMap::new(),
            }
        }

        fn decision(role_id: RoleId) -> RouteDecision {
            RouteDecision {
                role_id,
                provider: "hawking-local".into(),
                sampler: SamplerProfile::deterministic_edit(),
                grammar: None,
                reason: "test".into(),
                estimated_difficulty: DifficultyEstimate { score: 0.1, reason: "low".into(), signals: vec![] },
            }
        }

        fn request() -> InferenceRequest {
            InferenceRequest {
                task_kind: "code".into(),
                prompt: "p".into(),
                messages: Vec::new(),
                max_output_tokens: 4,
                sampler: None,
                grammar: None,
                want_logprobs: false,
                metadata: BTreeMap::new(),
            }
        }

        #[tokio::test]
        async fn executor_routes_role_to_client_and_collects() {
            let hero_id = RoleId::new();
            let registry = Arc::new(RoleRegistry::default());
            registry.register(role("hero", hero_id.clone(), RolePurpose::HeroCoder, "http://127.0.0.1:8081"));
            let stub: Arc<dyn InferenceClient> = Arc::new(StubInferenceClient::new("generated"));
            let executor = Executor::new(registry, Arc::new(FixedClientFactory(stub)));
            let (text, _) = collect_to_string(&executor, &decision(hero_id), request()).await.unwrap();
            assert_eq!(text, "generated");
        }

        #[tokio::test]
        async fn executor_embeds_via_embedder_role() {
            let embed_id = RoleId::new();
            let registry = Arc::new(RoleRegistry::default());
            registry.register(role("embedder", embed_id.clone(), RolePurpose::Embedder, "http://127.0.0.1:8083"));
            let stub: Arc<dyn InferenceClient> = Arc::new(StubInferenceClient::new("x"));
            let executor = Executor::new(registry, Arc::new(FixedClientFactory(stub)));
            let v = executor.embed(&decision(embed_id), "hello world").await.unwrap();
            assert_eq!(v.len(), 8);
        }

        #[tokio::test]
        async fn unknown_role_is_an_error() {
            let registry = Arc::new(RoleRegistry::default());
            let stub: Arc<dyn InferenceClient> = Arc::new(StubInferenceClient::new("x"));
            let executor = Executor::new(registry, Arc::new(FixedClientFactory(stub)));
            let result = collect_to_string(&executor, &decision(RoleId::new()), request()).await;
            assert!(result.is_err());
        }
    }
}
#[rustfmt::skip]
pub mod grammar {
    //! Constrained / grammar decode as a shell-side service (ch.06 §4.5).
    //!
    //! The runtime owns the real `mask_logits` primitive; until a `grammar` request
    //! field lands ([RUNTIME-SIDE — LATER]), this crate provides the **[SHELL-TODAY]**
    //! fallback the bible specifies (§4.5.4):
    //!
    //! 1. A [`GrammarSpec`] enum (`JsonObject | Regex | Choices`) describing what the
    //!    output envelope must satisfy.
    //! 2. A [`GrammarMatcher`] that does **validate-and-retry**: parse a completed
    //!    output against the spec, and on failure return a structured [`RetryHint`]
    //!    the caller folds into a re-prompt.
    //! 3. For the `JsonObject` case, a real **JSON-object FSM** ([`JsonObjectFsm`])
    //!    that, given the text emitted so far, reports which *classes* of next
    //!    character are legal — the shell-side analog of the runtime's per-token
    //!    mask. It is exact for a flat `{ "k": v, ... }` object and degrades to a
    //!    permissive state for arbitrary nesting (never a fabricated hash).
    //!
    //! No fabricated hashes: [`compile`] hashes the spec's canonical bytes with a
    //! real FNV-1a digest keyed by tokenizer signature.

    use regex::Regex;
    use serde::{Deserialize, Serialize};
    use serde_json::Value;

    /// What an output's envelope must satisfy. Deliberately small — the bible's
    /// `JsonObject | Regex | Choices` shell subset (§4.5.1 lists more cases that are
    /// runtime-side).
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub enum GrammarSpec {
        /// Any well-formed JSON object (optionally with required top-level keys).
        JsonObject { required_keys: Vec<String> },
        /// The whole output must fully match this regex.
        Regex(String),
        /// The output must be exactly one of these choices (classifier labels, enum).
        Choices(Vec<String>),
    }

    impl GrammarSpec {
        /// Canonical bytes for hashing/caching.
        fn canonical(&self) -> String {
            match self {
                GrammarSpec::JsonObject { required_keys } => {
                    let mut keys = required_keys.clone();
                    keys.sort();
                    format!("json_object:{}", keys.join(","))
                }
                GrammarSpec::Regex(r) => format!("regex:{r}"),
                GrammarSpec::Choices(c) => {
                    let mut c = c.clone();
                    c.sort();
                    format!("choices:{}", c.join("\u{1}"))
                }
            }
        }
    }

    /// A structured retry instruction emitted on a validation failure. The caller
    /// (router/executor or ch.02 kernel) re-prompts with `message` appended.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct RetryHint {
        /// Stable failure code (`NOT_JSON`, `MISSING_KEY`, `REGEX_MISMATCH`, `NOT_A_CHOICE`).
        pub code: String,
        /// Human/agent-readable correction to fold into the re-prompt.
        pub message: String,
    }

    /// Outcome of validating a completed output against a [`GrammarSpec`].
    #[derive(Debug, Clone, PartialEq)]
    pub enum GrammarValidation {
        Valid,
        Retry(RetryHint),
    }

    // ---- Legacy/compat surface (kept stable for any external consumer) ---------

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct GrammarRequest {
        pub name: String,
        pub schema_json: String,
        pub tokenizer_signature: String,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct CompiledGrammar {
        pub name: String,
        pub grammar_hash: String,
        pub tokenizer_signature: String,
        pub mask_cache_key: String,
    }

    pub trait GrammarCompiler: Send + Sync {
        fn compile(&self, request: GrammarRequest) -> hide_core::Result<CompiledGrammar>;
    }

    /// FNV-1a 64-bit, hex. Real digest — no fabricated stub hashes.
    fn fnv1a_hex(bytes: &[u8]) -> String {
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        format!("{h:016x}")
    }

    /// The real grammar compiler: parses the schema JSON into a [`GrammarSpec`],
    /// content-addresses it, and produces a [`CompiledGrammar`] whose hash is a real
    /// digest of `(spec, tokenizer_sig)`.
    #[derive(Default)]
    pub struct ShellGrammarCompiler;

    impl ShellGrammarCompiler {
        /// Derive a [`GrammarSpec`] from a schema JSON document. Supports a JSON
        /// Schema-ish `{"type":"object","required":[...]}`, `{"enum":[...]}`, and
        /// `{"pattern":"..."}`; falls back to a permissive JSON object.
        pub fn spec_from_schema(schema_json: &str) -> hide_core::Result<GrammarSpec> {
            let value: Value = serde_json::from_str(schema_json)?;
            if let Some(choices) = value.get("enum").and_then(|v| v.as_array()) {
                let choices = choices.iter().filter_map(|v| v.as_str().map(str::to_string)).collect();
                return Ok(GrammarSpec::Choices(choices));
            }
            if let Some(pattern) = value.get("pattern").and_then(|v| v.as_str()) {
                return Ok(GrammarSpec::Regex(pattern.to_string()));
            }
            let required = value
                .get("required")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
                .unwrap_or_default();
            Ok(GrammarSpec::JsonObject { required_keys: required })
        }
    }

    impl GrammarCompiler for ShellGrammarCompiler {
        fn compile(&self, request: GrammarRequest) -> hide_core::Result<CompiledGrammar> {
            let spec = Self::spec_from_schema(&request.schema_json)?;
            let canonical = spec.canonical();
            let grammar_hash = fnv1a_hex(canonical.as_bytes());
            let mask_cache_key = fnv1a_hex(format!("{canonical}|{}", request.tokenizer_signature).as_bytes());
            Ok(CompiledGrammar {
                name: request.name,
                grammar_hash,
                tokenizer_signature: request.tokenizer_signature,
                mask_cache_key,
            })
        }
    }

    /// The decode-time / output-time matcher. Holds a spec and (for the JSON case) a
    /// running FSM. Used two ways: feed it the *whole* completed output to validate
    /// (`validate`), or drive the [`JsonObjectFsm`] incrementally for a shell-side
    /// legal-next-char check.
    #[derive(Debug, Clone)]
    pub struct GrammarMatcher {
        spec: GrammarSpec,
        regex: Option<Regex>,
    }

    impl GrammarMatcher {
        pub fn new(spec: GrammarSpec) -> hide_core::Result<Self> {
            let regex = match &spec {
                GrammarSpec::Regex(r) => Some(
                    Regex::new(r)
                        .map_err(|e| hide_core::error::HideError::Config(format!("invalid grammar regex: {e}")))?,
                ),
                _ => None,
            };
            Ok(Self { spec, regex })
        }

        pub fn spec(&self) -> &GrammarSpec {
            &self.spec
        }

        /// Validate a completed output against the spec, returning a structured
        /// retry hint on failure (the validate-and-retry fallback, §4.5.4).
        pub fn validate(&self, output: &str) -> GrammarValidation {
            match &self.spec {
                GrammarSpec::JsonObject { required_keys } => {
                    let parsed: Value = match serde_json::from_str(output.trim()) {
                        Ok(v) => v,
                        Err(e) => {
                            return GrammarValidation::Retry(RetryHint {
                                code: "NOT_JSON".to_string(),
                                message: format!("Output was not valid JSON ({e}). Re-emit ONLY a single JSON object."),
                            })
                        }
                    };
                    let obj = match parsed.as_object() {
                        Some(o) => o,
                        None => {
                            return GrammarValidation::Retry(RetryHint {
                                code: "NOT_JSON".to_string(),
                                message: "Output must be a JSON object, not an array or scalar.".to_string(),
                            })
                        }
                    };
                    for key in required_keys {
                        if !obj.contains_key(key) {
                            return GrammarValidation::Retry(RetryHint {
                                code: "MISSING_KEY".to_string(),
                                message: format!("JSON object is missing required key \"{key}\". Add it and re-emit."),
                            });
                        }
                    }
                    GrammarValidation::Valid
                }
                GrammarSpec::Regex(pattern) => {
                    let re = self.regex.as_ref().expect("regex compiled in new()");
                    if re.is_match(output.trim()) {
                        GrammarValidation::Valid
                    } else {
                        GrammarValidation::Retry(RetryHint {
                            code: "REGEX_MISMATCH".to_string(),
                            message: format!(
                                "Output did not match the required pattern /{pattern}/. Re-emit to match it exactly."
                            ),
                        })
                    }
                }
                GrammarSpec::Choices(choices) => {
                    let trimmed = output.trim();
                    if choices.iter().any(|c| c == trimmed) {
                        GrammarValidation::Valid
                    } else {
                        GrammarValidation::Retry(RetryHint {
                            code: "NOT_A_CHOICE".to_string(),
                            message: format!(
                                "Output must be exactly one of: {}. Re-emit one of these and nothing else.",
                                choices.join(", ")
                            ),
                        })
                    }
                }
            }
        }

        /// A fresh JSON-object FSM for incremental masking (only meaningful for the
        /// `JsonObject` spec).
        pub fn json_fsm(&self) -> Option<JsonObjectFsm> {
            matches!(self.spec, GrammarSpec::JsonObject { .. }).then(JsonObjectFsm::new)
        }
    }

    /// Classes of next character the JSON-object FSM permits in its current state.
    /// This is the shell-side analog of the runtime's per-token logit mask: it tells
    /// a caller which characters keep the prefix on a path to a valid object.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct NextCharMask {
        pub allow_open_brace: bool,
        pub allow_close_brace: bool,
        pub allow_quote: bool,
        pub allow_colon: bool,
        pub allow_comma: bool,
        /// Inside a string literal: any character (until a closing quote).
        pub allow_string_char: bool,
        /// Value position: digits / `t`/`f`/`n` (true/false/null) / `{` / `"`.
        pub allow_value_start: bool,
        pub allow_whitespace: bool,
    }

    /// A minimal pushdown-free FSM over a *flat* JSON object `{ "k": "v", ... }`.
    /// It is exact for flat string-valued objects (the common tool-call envelope)
    /// and, on encountering nesting/other value types, enters a permissive
    /// `Freeform` state rather than rejecting — honoring §4.5.3 ("the grammar
    /// guarantees the envelope, never the thought").
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct JsonObjectFsm {
        state: FsmState,
        /// True once the closing top-level brace has been consumed.
        done: bool,
        /// True if the input has irrecoverably left the flat-object grammar
        /// (a real syntax error: a value where a key was due, etc.).
        dead: bool,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum FsmState {
        /// Before the opening `{`.
        Start,
        /// After `{` or after a `,` — expecting a key string (or `}` to close).
        ExpectKeyOrClose,
        /// Inside a key string literal.
        InKey,
        /// After a key string — expecting `:`.
        ExpectColon,
        /// After `:` — expecting a value.
        ExpectValue,
        /// Inside a string value.
        InStringValue,
        /// After a complete value — expecting `,` or `}`.
        ExpectCommaOrClose,
        /// Permissive: nesting / non-string value detected; we no longer constrain.
        Freeform,
    }

    impl Default for JsonObjectFsm {
        fn default() -> Self {
            Self::new()
        }
    }

    impl JsonObjectFsm {
        pub fn new() -> Self {
            Self { state: FsmState::Start, done: false, dead: false }
        }

        pub fn is_complete(&self) -> bool {
            self.done
        }

        /// No legal next character — a dead end the caller can escalate on (§4.4).
        pub fn is_dead_ended(&self) -> bool {
            self.dead
        }

        /// The legal next-character classes in the current state.
        pub fn mask(&self) -> NextCharMask {
            let none = NextCharMask {
                allow_open_brace: false,
                allow_close_brace: false,
                allow_quote: false,
                allow_colon: false,
                allow_comma: false,
                allow_string_char: false,
                allow_value_start: false,
                allow_whitespace: true,
            };
            if self.dead || self.done {
                return NextCharMask { allow_whitespace: true, ..none };
            }
            match self.state {
                FsmState::Start => NextCharMask { allow_open_brace: true, ..none },
                FsmState::ExpectKeyOrClose => NextCharMask { allow_quote: true, allow_close_brace: true, ..none },
                FsmState::InKey | FsmState::InStringValue => {
                    NextCharMask { allow_quote: true, allow_string_char: true, allow_whitespace: true, ..none }
                }
                FsmState::ExpectColon => NextCharMask { allow_colon: true, ..none },
                FsmState::ExpectValue => {
                    NextCharMask { allow_quote: true, allow_value_start: true, allow_open_brace: true, ..none }
                }
                FsmState::ExpectCommaOrClose => NextCharMask { allow_comma: true, allow_close_brace: true, ..none },
                FsmState::Freeform => NextCharMask {
                    allow_open_brace: true,
                    allow_close_brace: true,
                    allow_quote: true,
                    allow_colon: true,
                    allow_comma: true,
                    allow_string_char: true,
                    allow_value_start: true,
                    allow_whitespace: true,
                },
            }
        }

        /// Advance the FSM by one already-chosen character.
        pub fn accept(&mut self, c: char) {
            if self.dead || self.done {
                return;
            }
            if c.is_whitespace() && !matches!(self.state, FsmState::InKey | FsmState::InStringValue) {
                return; // whitespace is structurally inert between tokens
            }
            match self.state {
                FsmState::Start => {
                    if c == '{' {
                        self.state = FsmState::ExpectKeyOrClose;
                    } else {
                        self.dead = true;
                    }
                }
                FsmState::ExpectKeyOrClose => match c {
                    '"' => self.state = FsmState::InKey,
                    '}' => self.done = true,
                    _ => self.dead = true,
                },
                FsmState::InKey => {
                    if c == '"' {
                        self.state = FsmState::ExpectColon;
                    }
                    // else: still in the key string
                }
                FsmState::ExpectColon => {
                    if c == ':' {
                        self.state = FsmState::ExpectValue;
                    } else {
                        self.dead = true;
                    }
                }
                FsmState::ExpectValue => match c {
                    '"' => self.state = FsmState::InStringValue,
                    '{' | '[' => self.state = FsmState::Freeform, // nesting → permissive
                    _ => self.state = FsmState::Freeform,         // numbers/bools/null → permissive
                },
                FsmState::InStringValue => {
                    if c == '"' {
                        self.state = FsmState::ExpectCommaOrClose;
                    }
                }
                FsmState::ExpectCommaOrClose => match c {
                    ',' => self.state = FsmState::ExpectKeyOrClose,
                    '}' => self.done = true,
                    _ => self.dead = true,
                },
                FsmState::Freeform => {
                    // Best-effort: a top-level close ends the object.
                    if c == '}' {
                        self.done = true;
                    }
                }
            }
        }

        /// Drive the FSM over a whole string (convenience for tests/validation).
        pub fn accept_str(&mut self, s: &str) {
            for c in s.chars() {
                self.accept(c);
                if self.done || self.dead {
                    break;
                }
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn compiler_produces_real_hashes() {
            let c = ShellGrammarCompiler;
            let out = c
                .compile(GrammarRequest {
                    name: "tool".into(),
                    schema_json: "{\"type\":\"object\",\"required\":[\"name\"]}".into(),
                    tokenizer_signature: "tok-a".into(),
                })
                .unwrap();
            assert!(!out.grammar_hash.starts_with("stub:"));
            assert_eq!(out.grammar_hash.len(), 16);
            // Tokenizer sig changes the mask cache key but not the grammar hash.
            let out2 = c
                .compile(GrammarRequest {
                    name: "tool".into(),
                    schema_json: "{\"type\":\"object\",\"required\":[\"name\"]}".into(),
                    tokenizer_signature: "tok-b".into(),
                })
                .unwrap();
            assert_eq!(out.grammar_hash, out2.grammar_hash);
            assert_ne!(out.mask_cache_key, out2.mask_cache_key);
        }

        #[test]
        fn schema_to_choices_and_regex() {
            assert_eq!(
                ShellGrammarCompiler::spec_from_schema("{\"enum\":[\"a\",\"b\"]}").unwrap(),
                GrammarSpec::Choices(vec!["a".into(), "b".into()])
            );
            assert_eq!(
                ShellGrammarCompiler::spec_from_schema("{\"pattern\":\"^\\\\d+$\"}").unwrap(),
                GrammarSpec::Regex("^\\d+$".into())
            );
        }

        #[test]
        fn json_object_validate_missing_key() {
            let m = GrammarMatcher::new(GrammarSpec::JsonObject { required_keys: vec!["name".into()] }).unwrap();
            assert_eq!(m.validate("{\"name\":\"x\"}"), GrammarValidation::Valid);
            match m.validate("{\"other\":1}") {
                GrammarValidation::Retry(h) => assert_eq!(h.code, "MISSING_KEY"),
                _ => panic!("expected retry"),
            }
            match m.validate("not json") {
                GrammarValidation::Retry(h) => assert_eq!(h.code, "NOT_JSON"),
                _ => panic!("expected retry"),
            }
        }

        #[test]
        fn choices_and_regex_validate() {
            let m = GrammarMatcher::new(GrammarSpec::Choices(vec!["yes".into(), "no".into()])).unwrap();
            assert_eq!(m.validate(" yes "), GrammarValidation::Valid);
            match m.validate("maybe") {
                GrammarValidation::Retry(h) => assert_eq!(h.code, "NOT_A_CHOICE"),
                _ => panic!("expected retry"),
            }
            let r = GrammarMatcher::new(GrammarSpec::Regex("^v\\d+$".into())).unwrap();
            assert_eq!(r.validate("v12"), GrammarValidation::Valid);
            assert!(matches!(r.validate("x"), GrammarValidation::Retry(_)));
        }

        #[test]
        fn fsm_accepts_flat_object_and_completes() {
            let mut fsm = JsonObjectFsm::new();
            // At Start only `{` is legal.
            assert!(fsm.mask().allow_open_brace);
            assert!(!fsm.mask().allow_quote);
            fsm.accept_str("{\"name\":\"edit_file\"}");
            assert!(fsm.is_complete());
            assert!(!fsm.is_dead_ended());
        }

        #[test]
        fn fsm_dead_ends_on_value_where_key_expected() {
            let mut fsm = JsonObjectFsm::new();
            fsm.accept_str("{1");
            assert!(fsm.is_dead_ended());
        }

        #[test]
        fn fsm_goes_permissive_on_nesting() {
            let mut fsm = JsonObjectFsm::new();
            fsm.accept_str("{\"a\":{\"b\":1}}");
            // Did not dead-end on the nested object.
            assert!(!fsm.is_dead_ended());
        }
    }
}
#[rustfmt::skip]
pub mod http_client {
    //! Live HTTP client for a `hawking-serve` instance.
    //!
    //! Replaces the old hand-rolled blocking `TcpStream` with a `reqwest` client
    //! that streams **incrementally** — bytes arrive, are parsed into SSE events,
    //! and forwarded to the caller's [`TokenSink`] as they land (no buffering the
    //! whole response first). Three endpoints are spoken:
    //!
    //! * `/v1/hawking/generate` — native SSE, `{ "text", "tok_index" }` token frames
    //!   plus a trailing `{ "stats": { … } }` frame and a `[DONE]` terminator.
    //! * `/v1/chat/completions` — OpenAI-compatible SSE, `choices[].delta.content`.
    //! * `/v1/embeddings` — JSON, `data[0].embedding`.
    //!
    //! The runtime is **not** running during tests, so the streaming and parsing
    //! logic is factored into pure functions ([`parse_native_sse_event`],
    //! [`parse_openai_sse_event`], [`extract_embedding`]) that are unit-tested
    //! directly; the network path is exercised only behind the live trait.

    use crate::inference::InferenceClient;
    use eventsource_stream::Eventsource;
    use futures::future::BoxFuture;
    use futures::StreamExt;
    use hide_core::error::{HideError, Result};
    use hide_core::runtime::{GenerationStats, InferenceRequest, SamplerProfile, StreamChunk, TokenSink};
    use serde_json::{json, Value};
    use std::time::Duration;

    /// Which serve route a generation should target. The native route is preferred
    /// when the role advertises `native_tokens_endpoint`; chat is the portable
    /// fallback (and the only route for message-shaped requests against a model
    /// that wants a chat template).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum GenerateRoute {
        Native,
        Chat,
    }

    #[derive(Debug, Clone)]
    pub struct HawkingHttpClient {
        pub base_url: String,
        pub route: GenerateRoute,
        pub timeout: Duration,
        client: reqwest::Client,
    }

    impl HawkingHttpClient {
        pub fn new(base_url: impl Into<String>) -> Self {
            Self::with_route(base_url, GenerateRoute::Native)
        }

        pub fn with_route(base_url: impl Into<String>, route: GenerateRoute) -> Self {
            let timeout = Duration::from_secs(300);
            let client = reqwest::Client::builder().timeout(timeout).build().unwrap_or_default();
            Self { base_url: base_url.into().trim_end_matches('/').to_string(), route, timeout, client }
        }

        fn url(&self, path: &str) -> String {
            format!("{}{}", self.base_url, path)
        }

        fn sampler_or_default(request: &InferenceRequest) -> SamplerProfile {
            request.sampler.clone().unwrap_or_else(SamplerProfile::deterministic_edit)
        }

        fn prompt_text(request: &InferenceRequest) -> String {
            if request.prompt.is_empty() {
                request.messages.iter().map(|m| format!("{}: {}", m.role, m.content)).collect::<Vec<_>>().join("\n")
            } else {
                request.prompt.clone()
            }
        }

        /// Body for `/v1/hawking/generate`.
        pub fn build_native_generate_body(request: &InferenceRequest) -> Value {
            let sampler = Self::sampler_or_default(request);
            let mut body = json!({
                "prompt": Self::prompt_text(request),
                "max_tokens": request.max_output_tokens,
                "temperature": sampler.temperature,
                "stream": true,
                "stop": [],
            });
            if let Some(top_p) = sampler.top_p {
                body["top_p"] = json!(top_p);
            }
            if let Some(top_k) = sampler.top_k {
                body["top_k"] = json!(top_k);
            }
            if let Some(seed) = sampler.seed {
                body["seed"] = json!(seed);
            }
            if let Some(rp) = sampler.repetition_penalty {
                body["repetition_penalty"] = json!(rp);
            }
            if request.grammar.is_some() {
                // Today's surface only honors generic json_mode; richer grammar is a
                // runtime ask (ch.06 §4.5.4). Flag it so the server can opt in.
                body["json_mode"] = json!(true);
            }
            body
        }

        /// Body for `/v1/chat/completions`.
        pub fn build_chat_body(request: &InferenceRequest) -> Value {
            let sampler = Self::sampler_or_default(request);
            let messages: Vec<Value> = if request.messages.is_empty() {
                vec![json!({ "role": "user", "content": request.prompt })]
            } else {
                request.messages.iter().map(|m| json!({ "role": m.role, "content": m.content })).collect()
            };
            let mut body = json!({
                "messages": messages,
                "max_tokens": request.max_output_tokens,
                "temperature": sampler.temperature,
                "stream": true,
            });
            if let Some(top_p) = sampler.top_p {
                body["top_p"] = json!(top_p);
            }
            if let Some(seed) = sampler.seed {
                body["seed"] = json!(seed);
            }
            if request.grammar.is_some() {
                body["response_format"] = json!({ "type": "json_object" });
            }
            body
        }

        async fn stream_sse(
            &self,
            path: &str,
            body: Value,
            route: GenerateRoute,
            sink: TokenSink<'_>,
        ) -> Result<GenerationStats> {
            let resp = self
                .client
                .post(self.url(path))
                .header("Accept", "text/event-stream")
                .json(&body)
                .send()
                .await
                .map_err(|e| HideError::RuntimeUnavailable(format!("request to {path} failed: {e}")))?;
            if !resp.status().is_success() {
                return Err(HideError::RuntimeUnavailable(format!("{path} returned HTTP {}", resp.status())));
            }

            let mut stats = GenerationStats { input_tokens: 0, output_tokens: 0, decode_tokens_per_second: None };
            let mut stream = resp.bytes_stream().eventsource();
            while let Some(event) = stream.next().await {
                let event =
                    event.map_err(|e| HideError::RuntimeUnavailable(format!("SSE stream error on {path}: {e}")))?;
                let parsed = match route {
                    GenerateRoute::Native => parse_native_sse_event(&event.data, &mut stats),
                    GenerateRoute::Chat => parse_openai_sse_event(&event.data, &mut stats),
                };
                match parsed {
                    SseStep::Token(chunk) => sink(chunk)?,
                    SseStep::Done(reason) => {
                        sink(StreamChunk::Done { reason, stats: Some(stats.clone()) })?;
                        return Ok(stats);
                    }
                    SseStep::Ignore => {}
                }
            }
            // Stream ended without an explicit terminator — still close it out.
            sink(StreamChunk::Done { reason: "eof".to_string(), stats: Some(stats.clone()) })?;
            Ok(stats)
        }
    }

    impl InferenceClient for HawkingHttpClient {
        fn generate<'a>(
            &'a self,
            request: InferenceRequest,
            sink: TokenSink<'a>,
        ) -> BoxFuture<'a, Result<GenerationStats>> {
            Box::pin(async move {
                match self.route {
                    GenerateRoute::Native => {
                        let body = Self::build_native_generate_body(&request);
                        self.stream_sse("/v1/hawking/generate", body, GenerateRoute::Native, sink).await
                    }
                    GenerateRoute::Chat => {
                        let body = Self::build_chat_body(&request);
                        self.stream_sse("/v1/chat/completions", body, GenerateRoute::Chat, sink).await
                    }
                }
            })
        }

        fn embed<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>>> {
            Box::pin(async move {
                let body = json!({ "input": text, "model": "hawking-local" });
                let resp = self
                    .client
                    .post(self.url("/v1/embeddings"))
                    .json(&body)
                    .send()
                    .await
                    .map_err(|e| HideError::RuntimeUnavailable(format!("embeddings request failed: {e}")))?;
                if !resp.status().is_success() {
                    return Err(HideError::RuntimeUnavailable(format!(
                        "/v1/embeddings returned HTTP {}",
                        resp.status()
                    )));
                }
                let value: Value = resp
                    .json()
                    .await
                    .map_err(|e| HideError::RuntimeUnavailable(format!("embeddings decode failed: {e}")))?;
                extract_embedding(&value)
                    .ok_or_else(|| HideError::RuntimeUnavailable("no embedding in response".into()))
            })
        }
    }

    /// Outcome of parsing one SSE `data:` payload.
    enum SseStep {
        Token(StreamChunk),
        Done(String),
        Ignore,
    }

    /// Parse a native `/v1/hawking/generate` SSE frame, mutating running stats.
    fn parse_native_sse_event(data: &str, stats: &mut GenerationStats) -> SseStep {
        let data = data.trim();
        if data == "[DONE]" {
            return SseStep::Done("stop".to_string());
        }
        let value: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => return SseStep::Ignore,
        };
        if let Some(raw_stats) = value.get("stats") {
            if let Some(pt) = raw_stats.get("prompt_tokens").and_then(|v| v.as_u64()) {
                stats.input_tokens = pt as usize;
            }
            if let Some(ct) = raw_stats.get("completion_tokens").and_then(|v| v.as_u64()) {
                stats.output_tokens = ct as usize;
            }
            if let Some(tps) = raw_stats.get("dec_tps").and_then(|v| v.as_f64()) {
                stats.decode_tokens_per_second = Some(tps as f32);
            }
        }
        if let Some(text) = value.get("text").and_then(|v| v.as_str()) {
            stats.output_tokens += 1;
            return SseStep::Token(StreamChunk::Token {
                token_id: value.get("tok_index").and_then(|v| v.as_u64()).map(|v| v as u32),
                text: text.to_string(),
            });
        }
        SseStep::Ignore
    }

    /// Parse an OpenAI `/v1/chat/completions` SSE delta frame.
    fn parse_openai_sse_event(data: &str, stats: &mut GenerationStats) -> SseStep {
        let data = data.trim();
        if data == "[DONE]" {
            return SseStep::Done("stop".to_string());
        }
        let value: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => return SseStep::Ignore,
        };
        if let Some(usage) = value.get("usage") {
            if let Some(pt) = usage.get("prompt_tokens").and_then(|v| v.as_u64()) {
                stats.input_tokens = pt as usize;
            }
            if let Some(ct) = usage.get("completion_tokens").and_then(|v| v.as_u64()) {
                stats.output_tokens = ct as usize;
            }
        }
        let choice = value.get("choices").and_then(|c| c.get(0));
        if let Some(choice) = choice {
            if let Some(reason) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                if !reason.is_empty() {
                    // A delta with content AND a finish_reason can co-occur; emit the
                    // content first if present, otherwise close.
                    if let Some(content) = choice
                        .get("delta")
                        .and_then(|d| d.get("content"))
                        .and_then(|v| v.as_str())
                        .filter(|s| !s.is_empty())
                    {
                        stats.output_tokens += 1;
                        return SseStep::Token(StreamChunk::Token { token_id: None, text: content.to_string() });
                    }
                    return SseStep::Done(reason.to_string());
                }
            }
            if let Some(content) = choice.get("delta").and_then(|d| d.get("content")).and_then(|v| v.as_str()) {
                if !content.is_empty() {
                    stats.output_tokens += 1;
                    return SseStep::Token(StreamChunk::Token { token_id: None, text: content.to_string() });
                }
            }
        }
        SseStep::Ignore
    }

    /// Pull the first embedding vector out of an `/v1/embeddings` response.
    fn extract_embedding(value: &Value) -> Option<Vec<f32>> {
        let arr = value
            .get("data")
            .and_then(|d| d.get(0))
            .and_then(|e| e.get("embedding"))
            // Allow a bare {"embedding": [...]} too.
            .or_else(|| value.get("embedding"))?;
        let vec = arr.as_array()?.iter().map(|v| v.as_f64().map(|f| f as f32)).collect::<Option<Vec<f32>>>()?;
        Some(vec)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::collections::BTreeMap;

        fn empty_stats() -> GenerationStats {
            GenerationStats { input_tokens: 0, output_tokens: 0, decode_tokens_per_second: None }
        }

        #[test]
        fn native_event_emits_token() {
            let mut stats = empty_stats();
            match parse_native_sse_event("{\"tok_index\":3,\"text\":\"hi\"}", &mut stats) {
                SseStep::Token(StreamChunk::Token { token_id, text }) => {
                    assert_eq!(token_id, Some(3));
                    assert_eq!(text, "hi");
                }
                _ => panic!("expected token"),
            }
        }

        #[test]
        fn native_event_folds_stats_and_done() {
            let mut stats = empty_stats();
            let _ = parse_native_sse_event(
                "{\"stats\":{\"prompt_tokens\":5,\"completion_tokens\":9,\"dec_tps\":42.5}}",
                &mut stats,
            );
            assert_eq!(stats.input_tokens, 5);
            assert_eq!(stats.output_tokens, 9);
            assert_eq!(stats.decode_tokens_per_second, Some(42.5));
            assert!(matches!(parse_native_sse_event("[DONE]", &mut stats), SseStep::Done(_)));
        }

        #[test]
        fn openai_delta_emits_content() {
            let mut stats = empty_stats();
            let frame = "{\"choices\":[{\"delta\":{\"content\":\"foo\"},\"finish_reason\":null}]}";
            match parse_openai_sse_event(frame, &mut stats) {
                SseStep::Token(StreamChunk::Token { text, .. }) => assert_eq!(text, "foo"),
                _ => panic!("expected token"),
            }
            assert_eq!(stats.output_tokens, 1);
        }

        #[test]
        fn openai_finish_reason_closes() {
            let mut stats = empty_stats();
            let frame = "{\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}";
            assert!(matches!(parse_openai_sse_event(frame, &mut stats), SseStep::Done(_)));
        }

        #[test]
        fn embedding_extracted_from_openai_shape() {
            let v: Value = serde_json::from_str("{\"data\":[{\"embedding\":[0.1,0.2,0.3],\"index\":0}]}").unwrap();
            assert_eq!(extract_embedding(&v), Some(vec![0.1, 0.2, 0.3]));
        }

        #[test]
        fn chat_body_uses_messages_or_prompt() {
            let req = InferenceRequest {
                task_kind: "chat".into(),
                prompt: "hello".into(),
                messages: Vec::new(),
                max_output_tokens: 16,
                sampler: None,
                grammar: None,
                want_logprobs: false,
                metadata: BTreeMap::new(),
            };
            let body = HawkingHttpClient::build_chat_body(&req);
            assert_eq!(body["messages"][0]["content"], "hello");
            assert_eq!(body["stream"], true);
        }

        #[test]
        fn native_body_carries_sampler() {
            let mut req = InferenceRequest {
                task_kind: "t".into(),
                prompt: "p".into(),
                messages: Vec::new(),
                max_output_tokens: 7,
                sampler: Some(SamplerProfile {
                    temperature: 0.3,
                    top_k: Some(40),
                    top_p: Some(0.9),
                    repetition_penalty: Some(1.1),
                    seed: Some(11),
                    deterministic: false,
                }),
                grammar: Some("g".into()),
                want_logprobs: false,
                metadata: BTreeMap::new(),
            };
            let body = HawkingHttpClient::build_native_generate_body(&req);
            assert_eq!(body["max_tokens"], 7);
            assert_eq!(body["top_k"], 40);
            assert_eq!(body["seed"], 11);
            assert_eq!(body["json_mode"], true);
            req.grammar = None;
            let body = HawkingHttpClient::build_native_generate_body(&req);
            assert!(body.get("json_mode").is_none());
        }
    }
}
#[rustfmt::skip]
pub mod inference {
    //! The uniform inference seam.
    //!
    //! [`InferenceClient`] is the single boundary the orchestrator (and `hide-kernel`
    //! via `KernelRuntimeClient`) crosses to reach a live model. There are three
    //! capabilities behind it, mirroring `hawking-serve`'s HTTP surface:
    //!
    //! * [`InferenceClient::generate`] — streaming completion / chat
    //!   (`/v1/hawking/generate` native SSE, or `/v1/chat/completions` OpenAI SSE).
    //! * [`InferenceClient::embed`] — a vector embedding (`/v1/embeddings`), the
    //!   embedder role's only capability.
    //!
    //! Live HTTP is gated behind this trait; tests use [`StubInferenceClient`].

    use futures::future::BoxFuture;
    use hide_core::error::Result;
    use hide_core::runtime::{GenerationStats, InferenceRequest, StreamChunk, TokenSink};

    /// The boundary every model call crosses. Implemented by [`crate::http_client`]
    /// for live HTTP and by [`StubInferenceClient`] for tests / offline routing.
    pub trait InferenceClient: Send + Sync {
        /// Stream a completion. The sink receives `Token` chunks then a terminal
        /// `Done` (or `Error`). Returns aggregate [`GenerationStats`].
        fn generate<'a>(
            &'a self,
            request: InferenceRequest,
            sink: TokenSink<'a>,
        ) -> BoxFuture<'a, Result<GenerationStats>>;

        /// Embed a single text into a vector (`/v1/embeddings`). The embedder role
        /// is driven entirely through this method.
        fn embed<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>>>;
    }

    /// A deterministic test double. `generate` emits `response` as one token then a
    /// `Done`; `embed` returns a stable hashed pseudo-vector so retrieval/voting
    /// tests are reproducible without a runtime.
    #[derive(Debug, Clone)]
    pub struct StubInferenceClient {
        pub response: String,
        /// Dimension of the deterministic embedding vector.
        pub embed_dim: usize,
    }

    impl StubInferenceClient {
        pub fn new(response: impl Into<String>) -> Self {
            Self { response: response.into(), embed_dim: 8 }
        }
    }

    impl Default for StubInferenceClient {
        fn default() -> Self {
            Self::new(String::new())
        }
    }

    /// A stable, content-derived pseudo-embedding: hashes byte-windows into buckets
    /// so identical inputs map to identical vectors and similar inputs overlap.
    /// Real enough for cosine-based voting/dedup tests; not a semantic embedding.
    pub fn deterministic_embedding(text: &str, dim: usize) -> Vec<f32> {
        let dim = dim.max(1);
        let mut v = vec![0.0f32; dim];
        for token in text.split(|c: char| !c.is_alphanumeric()).filter(|t| !t.is_empty()) {
            // FNV-1a over the lowercased token.
            let mut h: u64 = 0xcbf29ce484222325;
            for b in token.to_ascii_lowercase().bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            let bucket = (h % dim as u64) as usize;
            v[bucket] += 1.0;
        }
        // L2-normalize so cosine similarity is a dot product.
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }

    impl InferenceClient for StubInferenceClient {
        fn generate<'a>(
            &'a self,
            _request: InferenceRequest,
            sink: TokenSink<'a>,
        ) -> BoxFuture<'a, Result<GenerationStats>> {
            Box::pin(async move {
                sink(StreamChunk::Token { token_id: None, text: self.response.clone() })?;
                sink(StreamChunk::Done { reason: "stop".to_string(), stats: None })?;
                Ok(GenerationStats { input_tokens: 0, output_tokens: 1, decode_tokens_per_second: None })
            })
        }

        fn embed<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>>> {
            let dim = self.embed_dim;
            let owned = text.to_string();
            Box::pin(async move { Ok(deterministic_embedding(&owned, dim)) })
        }
    }

    /// A scripted stub that returns a different completion on each successive
    /// `generate` call — used to test the escalation cascade (cheap role stumbles,
    /// stronger role succeeds).
    #[derive(Debug)]
    pub struct ScriptedInferenceClient {
        responses: parking_lot::Mutex<std::collections::VecDeque<String>>,
        fallback: String,
    }

    impl ScriptedInferenceClient {
        pub fn new(responses: impl IntoIterator<Item = String>) -> Self {
            let responses: std::collections::VecDeque<String> = responses.into_iter().collect();
            let fallback = responses.back().cloned().unwrap_or_default();
            Self { responses: parking_lot::Mutex::new(responses), fallback }
        }
    }

    impl InferenceClient for ScriptedInferenceClient {
        fn generate<'a>(
            &'a self,
            _request: InferenceRequest,
            sink: TokenSink<'a>,
        ) -> BoxFuture<'a, Result<GenerationStats>> {
            let next = self.responses.lock().pop_front().unwrap_or_else(|| self.fallback.clone());
            Box::pin(async move {
                sink(StreamChunk::Token { token_id: None, text: next })?;
                sink(StreamChunk::Done { reason: "stop".to_string(), stats: None })?;
                Ok(GenerationStats { input_tokens: 0, output_tokens: 1, decode_tokens_per_second: None })
            })
        }

        fn embed<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>>> {
            let owned = text.to_string();
            Box::pin(async move { Ok(deterministic_embedding(&owned, 8)) })
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[tokio::test]
        async fn stub_embeddings_are_deterministic_and_normalized() {
            let client = StubInferenceClient::new("hello");
            let a = client.embed("fn main() {}").await.unwrap();
            let b = client.embed("fn main() {}").await.unwrap();
            assert_eq!(a, b);
            let norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5 || norm == 0.0);
        }

        #[tokio::test]
        async fn scripted_client_advances_per_call() {
            let client = ScriptedInferenceClient::new(vec!["first".to_string(), "second".to_string()]);
            let mut got = Vec::new();
            let mut sink = |chunk: StreamChunk| {
                if let StreamChunk::Token { text, .. } = chunk {
                    got.push(text);
                }
                Ok(())
            };
            let req = InferenceRequest {
                task_kind: "t".into(),
                prompt: "p".into(),
                messages: Vec::new(),
                max_output_tokens: 1,
                sampler: None,
                grammar: None,
                want_logprobs: false,
                metadata: Default::default(),
            };
            client.generate(req.clone(), &mut sink).await.unwrap();
            client.generate(req, &mut sink).await.unwrap();
            assert_eq!(got, vec!["first".to_string(), "second".to_string()]);
        }
    }
}
#[rustfmt::skip]
pub mod registry {
    use hide_core::error::{HideError, Result};
    use hide_core::ids::ModelId;
    use hide_core::ids::RoleId;
    use hide_core::runtime::{
        ModelArchitecture, ModelDescriptor, ModelRole, ProviderCaps, RolePurpose, SamplerProfile,
    };
    use parking_lot::RwLock;
    use serde::Deserialize;
    use std::collections::BTreeMap;
    use std::path::Path;

    #[derive(Default)]
    pub struct RoleRegistry {
        roles: RwLock<BTreeMap<RoleId, ModelRole>>,
    }

    impl RoleRegistry {
        pub fn with_default_local_roles() -> Self {
            let registry = Self::default();
            registry.register_default_local_roles();
            registry
        }

        pub fn register_default_local_roles(&self) {
            for role in default_hawking_local_roles() {
                self.register(role);
            }
        }

        pub fn register(&self, role: ModelRole) {
            self.roles.write().insert(role.id.clone(), role);
        }

        pub fn get(&self, id: &RoleId) -> Option<ModelRole> {
            self.roles.read().get(id).cloned()
        }

        pub fn by_purpose(&self, purpose: RolePurpose) -> Vec<ModelRole> {
            self.roles.read().values().filter(|role| role.purpose == purpose).cloned().collect()
        }

        pub fn all(&self) -> Vec<ModelRole> {
            self.roles.read().values().cloned().collect()
        }

        pub fn is_empty(&self) -> bool {
            self.roles.read().is_empty()
        }

        /// Resolve a role by its human name (the key used in `roles.toml`), since
        /// `escalates_to`/`draft_for` reference roles by name in the file.
        pub fn by_name(&self, name: &str) -> Option<ModelRole> {
            self.roles.read().values().find(|r| r.name == name).cloned()
        }

        /// Load roles from a `.hide/roles.toml` (or any TOML) file if it exists,
        /// else fall back to the built-in defaults. `escalates_to` names are
        /// resolved to the registered roles' ids in a second pass.
        pub fn load_from_file_or_default(path: impl AsRef<Path>) -> Result<Self> {
            let path = path.as_ref();
            if !path.exists() {
                return Ok(Self::with_default_local_roles());
            }
            let text = std::fs::read_to_string(path)?;
            Self::from_roles_toml(&text)
        }

        /// Parse a `roles.toml` document into a registry (the §4.3 schema).
        pub fn from_roles_toml(text: &str) -> Result<Self> {
            let file: RolesFile =
                toml::from_str(text).map_err(|e| HideError::Config(format!("invalid roles.toml: {e}")))?;
            let registry = Self::default();

            // First pass: register every role with a stable id == its name, so the
            // escalates_to wiring can resolve by name.
            for (name, spec) in &file.roles {
                registry.register(spec.to_model_role(name));
            }
            // Second pass: resolve escalates_to (a role *name*) to that role's id.
            let resolved: Vec<ModelRole> = registry
                .all()
                .into_iter()
                .map(|mut role| {
                    if let Some(target_name) = file.roles.get(&role.name).and_then(|s| s.escalates_to.clone()) {
                        role.escalates_to = registry.by_name(&target_name).map(|r| r.id);
                    }
                    role
                })
                .collect();
            for role in resolved {
                registry.register(role);
            }
            Ok(registry)
        }
    }

    // ---- roles.toml schema (ch.06 §4.3) ----------------------------------------

    #[derive(Debug, Deserialize)]
    struct RolesFile {
        #[serde(default)]
        roles: BTreeMap<String, RoleSpec>,
    }

    #[derive(Debug, Deserialize)]
    struct RoleSpec {
        role_kind: String,
        model: ModelSpec,
        #[serde(default)]
        endpoint: Option<String>,
        #[serde(default)]
        default_sampler: Option<String>,
        #[serde(default)]
        caps: CapsSpec,
        #[serde(default)]
        cost: Option<f32>,
        #[serde(default)]
        escalates_to: Option<String>,
    }

    #[derive(Debug, Deserialize)]
    struct ModelSpec {
        id: String,
        #[serde(default = "default_arch")]
        arch: String,
        #[serde(default = "default_ctx")]
        ctx_len_native: usize,
        #[serde(default)]
        tokenizer_sig: Option<String>,
        #[serde(default)]
        footprint_mb: u64,
    }

    fn default_arch() -> String {
        "transformer".to_string()
    }
    fn default_ctx() -> usize {
        8_192
    }

    #[derive(Debug, Default, Deserialize)]
    struct CapsSpec {
        #[serde(default)]
        grammar: bool,
        #[serde(default)]
        logprobs: bool,
        #[serde(default)]
        embeddings: bool,
        #[serde(default)]
        lora: bool,
        #[serde(default = "default_true")]
        streaming: bool,
    }

    fn default_true() -> bool {
        true
    }

    impl RoleSpec {
        fn to_model_role(&self, name: &str) -> ModelRole {
            let purpose = parse_role_kind(&self.role_kind);
            let arch = match self.model.arch.as_str() {
                "ssm" => ModelArchitecture::Ssm,
                "hybrid" => ModelArchitecture::Hybrid,
                "transformer" => ModelArchitecture::Transformer,
                _ => ModelArchitecture::Unknown,
            };
            let sampler = match self.default_sampler.as_deref() {
                Some("greedy") => SamplerProfile {
                    temperature: 0.0,
                    top_k: None,
                    top_p: None,
                    repetition_penalty: None,
                    seed: Some(0),
                    deterministic: true,
                },
                Some("balanced") => SamplerProfile {
                    temperature: 0.7,
                    top_k: Some(40),
                    top_p: Some(0.9),
                    repetition_penalty: Some(1.05),
                    seed: None,
                    deterministic: false,
                },
                // "edit" and anything else → deterministic edit profile.
                _ => SamplerProfile::deterministic_edit(),
            };
            ModelRole {
                // Stable id == name so escalates_to can resolve before random ids.
                id: RoleId::from(name),
                name: name.to_string(),
                purpose,
                model: ModelDescriptor {
                    id: ModelId::from(self.model.id.clone()),
                    name: self.model.id.clone(),
                    architecture: arch,
                    context_tokens: self.model.ctx_len_native,
                    tokenizer_signature: self.model.tokenizer_sig.clone().unwrap_or_else(|| "unknown".to_string()),
                    footprint_mb: self.model.footprint_mb,
                },
                caps: ProviderCaps {
                    streaming: self.caps.streaming,
                    embeddings: self.caps.embeddings,
                    grammar: self.caps.grammar,
                    raw_logits: false,
                    logprobs: self.caps.logprobs,
                    lora: self.caps.lora,
                    kv_handles: false,
                    native_tokens_endpoint: true,
                },
                default_sampler: sampler,
                endpoint: self.endpoint.clone(),
                cost: self.cost,
                escalates_to: None, // resolved in the second pass
                metadata: BTreeMap::new(),
            }
        }
    }

    fn parse_role_kind(kind: &str) -> RolePurpose {
        match kind {
            "hero" | "hero_coder" => RolePurpose::HeroCoder,
            "fast_draft" => RolePurpose::FastDraft,
            "embedder" => RolePurpose::Embedder,
            "reranker" => RolePurpose::Reranker,
            "compactor" | "summarizer" => RolePurpose::Summarizer,
            "classifier" => RolePurpose::Classifier,
            "ssm_long" => RolePurpose::SsmLong,
            _ => RolePurpose::ToolPlanner,
        }
    }

    pub fn default_hawking_local_roles() -> Vec<ModelRole> {
        vec![
            local_role(
                "hawking-fast-draft",
                RolePurpose::FastDraft,
                16_384,
                4_096,
                ProviderCaps::hawking_local_shell_today(),
                SamplerProfile {
                    temperature: 0.2,
                    top_k: Some(40),
                    top_p: Some(0.95),
                    repetition_penalty: None,
                    seed: None,
                    deterministic: false,
                },
            ),
            local_role(
                "hawking-hero-coder",
                RolePurpose::HeroCoder,
                32_768,
                8_192,
                ProviderCaps::hawking_local_shell_today(),
                SamplerProfile::deterministic_edit(),
            ),
            local_role(
                "hawking-embedder",
                RolePurpose::Embedder,
                8_192,
                1_024,
                ProviderCaps {
                    streaming: false,
                    embeddings: true,
                    grammar: false,
                    raw_logits: false,
                    logprobs: false,
                    lora: false,
                    kv_handles: false,
                    native_tokens_endpoint: false,
                },
                SamplerProfile::deterministic_edit(),
            ),
            local_role(
                "hawking-tool-planner",
                RolePurpose::ToolPlanner,
                16_384,
                4_096,
                ProviderCaps { grammar: true, ..ProviderCaps::hawking_local_shell_today() },
                SamplerProfile::deterministic_edit(),
            ),
        ]
    }

    fn local_role(
        name: &str,
        purpose: RolePurpose,
        context_tokens: usize,
        footprint_mb: u64,
        caps: ProviderCaps,
        default_sampler: SamplerProfile,
    ) -> ModelRole {
        ModelRole {
            id: RoleId::new(),
            name: name.to_string(),
            purpose,
            model: ModelDescriptor {
                id: ModelId::new(),
                name: name.to_string(),
                architecture: ModelArchitecture::Transformer,
                context_tokens,
                tokenizer_signature: "hawking-local".to_string(),
                footprint_mb,
            },
            caps,
            default_sampler,
            endpoint: None,
            cost: None,
            escalates_to: None,
            metadata: BTreeMap::new(),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn default_registry_contains_core_runtime_roles() {
            let registry = RoleRegistry::with_default_local_roles();
            assert!(!registry.by_purpose(RolePurpose::FastDraft).is_empty());
            assert!(!registry.by_purpose(RolePurpose::HeroCoder).is_empty());
            assert!(!registry.by_purpose(RolePurpose::Embedder).is_empty());
            assert!(registry.by_purpose(RolePurpose::ToolPlanner).iter().any(|role| role.caps.grammar));
        }

        #[test]
        fn roles_toml_loads_and_resolves_escalation() {
            let toml = r#"
    [roles.fast_draft]
    role_kind = "fast_draft"
    endpoint = "http://127.0.0.1:8082"
    default_sampler = "greedy"
    escalates_to = "hero"
    [roles.fast_draft.model]
    id = "qwen2.5-0.5b"
    arch = "transformer"
    ctx_len_native = 32768
    footprint_mb = 500
    [roles.fast_draft.caps]
    grammar = true

    [roles.hero]
    role_kind = "hero"
    endpoint = "http://127.0.0.1:8081"
    default_sampler = "edit"
    [roles.hero.model]
    id = "qwen2.5-7b-instruct"
    arch = "transformer"
    ctx_len_native = 32768
    footprint_mb = 4600
    [roles.hero.caps]
    grammar = true
    "#;
            let registry = RoleRegistry::from_roles_toml(toml).unwrap();
            let draft = registry.by_name("fast_draft").unwrap();
            let hero = registry.by_name("hero").unwrap();
            assert_eq!(draft.purpose, RolePurpose::FastDraft);
            assert_eq!(draft.endpoint.as_deref(), Some("http://127.0.0.1:8082"));
            assert_eq!(draft.escalates_to, Some(hero.id.clone()));
            assert!(hero.escalates_to.is_none());
            assert_eq!(hero.model.footprint_mb, 4600);
        }

        #[test]
        fn missing_file_falls_back_to_defaults() {
            let registry = RoleRegistry::load_from_file_or_default("/nonexistent/path/roles.toml").unwrap();
            assert!(!registry.by_purpose(RolePurpose::HeroCoder).is_empty());
        }
    }
}
#[rustfmt::skip]
pub mod router {
    use crate::difficulty::{DifficultyEstimate, DifficultyEstimator};
    use crate::registry::RoleRegistry;
    use hide_core::error::{HideError, Result};
    use hide_core::ids::RoleId;
    use hide_core::runtime::{InferenceRequest, ModelRole, RolePurpose, SamplerProfile};
    use serde::{Deserialize, Serialize};
    use std::sync::Arc;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct RouteDecision {
        pub role_id: RoleId,
        pub provider: String,
        pub sampler: SamplerProfile,
        pub grammar: Option<String>,
        pub reason: String,
        pub estimated_difficulty: DifficultyEstimate,
    }

    pub trait Router: Send + Sync {
        fn route(&self, request: &InferenceRequest) -> Result<RouteDecision>;
    }

    pub struct SimpleRouter {
        registry: Arc<RoleRegistry>,
        estimator: DifficultyEstimator,
    }

    impl SimpleRouter {
        pub fn new(registry: Arc<RoleRegistry>) -> Self {
            Self { registry, estimator: DifficultyEstimator }
        }

        fn choose_role(&self, request: &InferenceRequest, difficulty: &DifficultyEstimate) -> Option<ModelRole> {
            if request.task_kind == "embedding" {
                return self.registry.by_purpose(RolePurpose::Embedder).into_iter().next();
            }
            if request.grammar.is_some() {
                if let Some(role) = self
                    .registry
                    .all()
                    .into_iter()
                    .find(|role| role.caps.grammar && role.purpose == RolePurpose::ToolPlanner)
                {
                    return Some(role);
                }
            }
            if difficulty.score > 0.65 {
                self.registry.by_purpose(RolePurpose::HeroCoder).into_iter().next()
            } else {
                self.registry
                    .by_purpose(RolePurpose::FastDraft)
                    .into_iter()
                    .next()
                    .or_else(|| self.registry.by_purpose(RolePurpose::HeroCoder).into_iter().next())
            }
        }
    }

    impl Router for SimpleRouter {
        fn route(&self, request: &InferenceRequest) -> Result<RouteDecision> {
            let difficulty = self.estimator.estimate(request);
            let role = self
                .choose_role(request, &difficulty)
                .ok_or_else(|| HideError::Config("no model role registered for request".to_string()))?;
            Ok(RouteDecision {
                role_id: role.id,
                provider: "hawking-local".to_string(),
                sampler: request.sampler.clone().unwrap_or_else(|| role.default_sampler.clone()),
                grammar: request.grammar.clone().filter(|_| role.caps.grammar),
                reason: difficulty.reason.clone(),
                estimated_difficulty: difficulty,
            })
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use hide_core::ids::{ModelId, RoleId};
        use hide_core::runtime::{
            ModelArchitecture, ModelDescriptor, ModelRole, ProviderCaps, RolePurpose, SamplerProfile,
        };
        use std::collections::BTreeMap;

        #[test]
        fn router_prefers_grammar_capable_tool_planner() {
            let registry = Arc::new(RoleRegistry::default());
            registry.register(ModelRole {
                id: RoleId::new(),
                name: "tool".to_string(),
                purpose: RolePurpose::ToolPlanner,
                model: ModelDescriptor {
                    id: ModelId::new(),
                    name: "tool".to_string(),
                    architecture: ModelArchitecture::Transformer,
                    context_tokens: 4096,
                    tokenizer_signature: "tok".to_string(),
                    footprint_mb: 128,
                },
                caps: ProviderCaps { grammar: true, ..ProviderCaps::hawking_local_shell_today() },
                default_sampler: SamplerProfile::deterministic_edit(),
                endpoint: None,
                cost: None,
                escalates_to: None,
                metadata: BTreeMap::new(),
            });
            let router = SimpleRouter::new(registry);
            let decision = router
                .route(&InferenceRequest {
                    task_kind: "tool_call".to_string(),
                    prompt: "{}".to_string(),
                    messages: Vec::new(),
                    max_output_tokens: 128,
                    sampler: None,
                    grammar: Some("tool-call-json".to_string()),
                    want_logprobs: false,
                    metadata: BTreeMap::new(),
                })
                .unwrap();
            assert_eq!(decision.grammar.as_deref(), Some("tool-call-json"));
        }

        #[test]
        fn router_uses_default_roles_for_hard_requests() {
            let registry = Arc::new(RoleRegistry::with_default_local_roles());
            let router = SimpleRouter::new(registry.clone());
            let decision = router
                .route(&InferenceRequest {
                    task_kind: "code".to_string(),
                    prompt: format!(
                        "{} architecture security refactor multi-file failing tests",
                        "large context ".repeat(5000)
                    ),
                    messages: Vec::new(),
                    max_output_tokens: 512,
                    sampler: None,
                    grammar: None,
                    want_logprobs: false,
                    metadata: BTreeMap::new(),
                })
                .unwrap();
            let role = registry.get(&decision.role_id).unwrap();
            assert_eq!(role.purpose, RolePurpose::HeroCoder);
        }
    }
}
#[rustfmt::skip]
pub mod sampler {
    use hide_core::runtime::SamplerProfile;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct SamplerCatalog {
        pub edit: SamplerProfile,
        pub planning: SamplerProfile,
        pub brainstorm: SamplerProfile,
    }

    impl Default for SamplerCatalog {
        fn default() -> Self {
            Self {
                edit: SamplerProfile::deterministic_edit(),
                planning: SamplerProfile {
                    temperature: 0.2,
                    top_k: Some(40),
                    top_p: Some(0.9),
                    repetition_penalty: Some(1.05),
                    seed: Some(0),
                    deterministic: false,
                },
                brainstorm: SamplerProfile {
                    temperature: 0.8,
                    top_k: Some(80),
                    top_p: Some(0.95),
                    repetition_penalty: Some(1.05),
                    seed: None,
                    deterministic: false,
                },
            }
        }
    }
}
#[rustfmt::skip]
pub mod scheduler {
    //! Energy / thermal / RAM-aware admission control (ch.06 §4.11).
    //!
    //! On a laptop the model fleet is a *power budget*. Before a role is admitted to
    //! run, the scheduler checks it against a [`ResourceSnapshot`] (free RAM, a
    //! thermal-headroom proxy, in-flight count, battery/mode) and returns
    //! [`Admission::Admit`] or [`Admission::Defer`] with a **structured reason**
    //! (`Ram` / `Thermal` / `Concurrency` / `Energy`). The router (§4.4 step 4)
    //! consults it to pick a smaller role or back off spec when the budget is tight.
    //!
    //! Pure policy — no OS probing here (that is the host's `ResourceProbe`); the
    //! snapshot is fed in, so the predicates are fully testable.

    use hide_core::runtime::ModelRole;
    use serde::{Deserialize, Serialize};

    /// User-facing power mode (the §4.11 dial).
    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum PowerMode {
        /// Full fleet, full spec, biggest roles.
        PluggedPerf,
        /// Default.
        #[default]
        Balanced,
        /// Smaller roles, throttled concurrency, quieter fans.
        Quiet,
    }

    /// A point-in-time view of the machine's budget. Supplied by the host.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ResourceSnapshot {
        /// Free unified memory in MB (RAM == VRAM on Apple Silicon).
        pub ram_free_mb: u64,
        /// Thermal headroom proxy in [0,1]: 1.0 = cool, 0.0 = throttling.
        pub thermal_headroom: f32,
        /// Number of generations currently in flight across the fleet.
        pub in_flight: u32,
        /// Whether the machine is on battery.
        pub on_battery: bool,
        /// Active power mode.
        pub mode: PowerMode,
    }

    impl Default for ResourceSnapshot {
        fn default() -> Self {
            Self {
                ram_free_mb: u64::MAX,
                thermal_headroom: 1.0,
                in_flight: 0,
                on_battery: false,
                mode: PowerMode::Balanced,
            }
        }
    }

    /// Why a role was deferred.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum DeferReason {
        /// The role's footprint won't fit in free RAM (plus headroom).
        Ram,
        /// Thermal headroom too low for a role this heavy.
        Thermal,
        /// Too many generations already in flight.
        Concurrency,
        /// On battery in quiet mode: this role is too expensive.
        Energy,
    }

    /// The admission verdict.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub enum Admission {
        Admit,
        Defer { reason: DeferReason, detail: String },
    }

    impl Admission {
        pub fn is_admit(&self) -> bool {
            matches!(self, Admission::Admit)
        }
    }

    /// Admission policy thresholds.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct AdmissionPolicy {
        /// RAM kept free as headroom beyond the role's footprint (MB).
        pub ram_headroom_mb: u64,
        /// Below this thermal headroom, only light roles are admitted.
        pub thermal_low: f32,
        /// A role footprint (MB) considered "heavy" for thermal/energy gating.
        pub heavy_footprint_mb: u64,
        /// Max concurrent generations (balanced/plugged).
        pub max_concurrency: u32,
        /// Max concurrent generations on battery / quiet.
        pub max_concurrency_battery: u32,
    }

    impl Default for AdmissionPolicy {
        fn default() -> Self {
            Self {
                ram_headroom_mb: 1_024,
                thermal_low: 0.2,
                heavy_footprint_mb: 4_000,
                max_concurrency: 6,
                max_concurrency_battery: 2,
            }
        }
    }

    /// The admission controller.
    #[derive(Debug, Clone, Default)]
    pub struct Scheduler {
        pub policy: AdmissionPolicy,
    }

    impl Scheduler {
        pub fn new(policy: AdmissionPolicy) -> Self {
            Self { policy }
        }

        /// Decide whether `role` may run under `snapshot`. Order matters: RAM is a
        /// hard wall, then concurrency, then thermal/energy (which prefer a smaller
        /// role rather than block outright — the router uses the reason to downgrade).
        pub fn admit(&self, role: &ModelRole, snapshot: &ResourceSnapshot) -> Admission {
            let footprint = role.model.footprint_mb;

            // 1. RAM: a shared-process role (footprint 0) always fits.
            if footprint > 0 {
                let needed = footprint.saturating_add(self.policy.ram_headroom_mb);
                if needed > snapshot.ram_free_mb {
                    return Admission::Defer {
                        reason: DeferReason::Ram,
                        detail: format!(
                            "role '{}' needs {}MB (+{}MB headroom) but only {}MB free",
                            role.name, footprint, self.policy.ram_headroom_mb, snapshot.ram_free_mb
                        ),
                    };
                }
            }

            // 2. Concurrency cap (battery/quiet lowers it).
            let cap = if snapshot.on_battery || snapshot.mode == PowerMode::Quiet {
                self.policy.max_concurrency_battery
            } else {
                self.policy.max_concurrency
            };
            if snapshot.in_flight >= cap {
                return Admission::Defer {
                    reason: DeferReason::Concurrency,
                    detail: format!("{} in flight ≥ cap {cap}", snapshot.in_flight),
                };
            }

            let heavy = footprint >= self.policy.heavy_footprint_mb;

            // 3. Thermal: throttling blocks heavy roles.
            if heavy && snapshot.thermal_headroom < self.policy.thermal_low {
                return Admission::Defer {
                    reason: DeferReason::Thermal,
                    detail: format!(
                        "thermal headroom {:.2} < {:.2}; defer heavy role '{}'",
                        snapshot.thermal_headroom, self.policy.thermal_low, role.name
                    ),
                };
            }

            // 4. Energy: on battery in quiet mode, defer heavy roles to prefer a
            //    smaller one (the router downgrades on this reason).
            if heavy && snapshot.on_battery && snapshot.mode == PowerMode::Quiet {
                return Admission::Defer {
                    reason: DeferReason::Energy,
                    detail: format!("on battery + quiet: defer heavy role '{}' to a lighter one", role.name),
                };
            }

            Admission::Admit
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use hide_core::ids::{ModelId, RoleId};
        use hide_core::runtime::{ModelArchitecture, ModelDescriptor, ProviderCaps, RolePurpose, SamplerProfile};
        use std::collections::BTreeMap;

        fn role(footprint_mb: u64) -> ModelRole {
            ModelRole {
                id: RoleId::new(),
                name: "test".into(),
                purpose: RolePurpose::HeroCoder,
                model: ModelDescriptor {
                    id: ModelId::new(),
                    name: "test".into(),
                    architecture: ModelArchitecture::Transformer,
                    context_tokens: 4096,
                    tokenizer_signature: "tok".into(),
                    footprint_mb,
                },
                caps: ProviderCaps::hawking_local_shell_today(),
                default_sampler: SamplerProfile::deterministic_edit(),
                endpoint: None,
                cost: None,
                escalates_to: None,
                metadata: BTreeMap::new(),
            }
        }

        #[test]
        fn admits_when_budget_is_ample() {
            let s = Scheduler::default();
            assert!(s.admit(&role(4_600), &ResourceSnapshot::default()).is_admit());
        }

        #[test]
        fn defers_on_insufficient_ram() {
            let s = Scheduler::default();
            let snap = ResourceSnapshot { ram_free_mb: 2_000, ..ResourceSnapshot::default() };
            match s.admit(&role(4_600), &snap) {
                Admission::Defer { reason, .. } => assert_eq!(reason, DeferReason::Ram),
                _ => panic!("expected RAM defer"),
            }
        }

        #[test]
        fn shared_process_role_ignores_ram() {
            let s = Scheduler::default();
            let snap = ResourceSnapshot { ram_free_mb: 0, ..ResourceSnapshot::default() };
            assert!(s.admit(&role(0), &snap).is_admit());
        }

        #[test]
        fn defers_heavy_role_when_throttling() {
            let s = Scheduler::default();
            let snap = ResourceSnapshot { thermal_headroom: 0.1, ..ResourceSnapshot::default() };
            match s.admit(&role(5_000), &snap) {
                Admission::Defer { reason, .. } => assert_eq!(reason, DeferReason::Thermal),
                _ => panic!("expected thermal defer"),
            }
            // A light role still admits when hot.
            assert!(s.admit(&role(500), &snap).is_admit());
        }

        #[test]
        fn defers_on_concurrency_cap() {
            let s = Scheduler::default();
            let snap = ResourceSnapshot { in_flight: 6, ..ResourceSnapshot::default() };
            match s.admit(&role(500), &snap) {
                Admission::Defer { reason, .. } => assert_eq!(reason, DeferReason::Concurrency),
                _ => panic!("expected concurrency defer"),
            }
        }

        #[test]
        fn battery_quiet_defers_heavy_for_energy() {
            let s = Scheduler::default();
            let snap = ResourceSnapshot {
                on_battery: true,
                mode: PowerMode::Quiet,
                in_flight: 0,
                ..ResourceSnapshot::default()
            };
            match s.admit(&role(5_000), &snap) {
                Admission::Defer { reason, .. } => assert_eq!(reason, DeferReason::Energy),
                _ => panic!("expected energy defer"),
            }
        }
    }
}
#[rustfmt::skip]
pub mod supervisor {
    use hide_core::runtime::RuntimeSupervisorState;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct RuntimeLock {
        pub pid: Option<u32>,
        pub port: u16,
        pub model_id: String,
        pub started_at_ms: u64,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct RuntimeSupervisorStatus {
        pub state: RuntimeSupervisorState,
        pub lock: Option<RuntimeLock>,
        pub consecutive_failures: u32,
        pub last_error: Option<String>,
    }

    impl RuntimeSupervisorStatus {
        pub fn down() -> Self {
            Self { state: RuntimeSupervisorState::Down, lock: None, consecutive_failures: 0, last_error: None }
        }
    }
}
#[rustfmt::skip]
pub mod tool_spec_decode {
    //! Tool-call spec-decode: the "small spec-decode layer for tools" (see
    //! `docs/RESEARCH.md`, tool-call decoding).
    //!
    //! A tool call is 60-80 percent deterministic given the schema, so most of its
    //! tokens can be emitted with no forward pass at all. This module provides the
    //! two training-free, lossless primitives that exploit that, at the string level
    //! (the runtime maps them onto its per-token logit mask and its verifier):
    //!
    //! * [`ToolCallGrammar`] - jump-forward. It knows the canonical tool-call
    //!   envelope `{"name": "<tool>", "arguments": {...}}` and the registered tool
    //!   names, and reports the continuation that is the ONLY legal one from a given
    //!   state: the opening scaffolding, the shared prefix of the still-consistent
    //!   tool names, and the full skeleton once the tool is fixed. Emitting those is
    //!   free and cannot change the sampled distribution (they were forced anyway).
    //! * [`PromptLookup`] - draft argument values that the model is copying out of
    //!   context (a path it just read, a symbol from the diff) by matching the tail
    //!   of what it has generated against the context and proposing the continuation.
    //!   The target still verifies each drafted token, so acceptance is lossless.
    //!
    //! Both are pure and fully unit-tested with no model. The runtime consumes them:
    //! the grammar feeds `mask_logits` / a fast-forward emit, and the lookup feeds
    //! the existing `speculate` verifier.

    use serde::{Deserialize, Serialize};
    use serde_json::Value;

    // ---------------------------------------------------------------------------
    // schema-aware tool-call grammar (jump-forward)
    // ---------------------------------------------------------------------------

    /// The minimal schema shape the grammar needs: a tool name and its required
    /// argument keys (in declared order). Derived from a `ToolSpec.input_schema`.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct ToolSchema {
        pub name: String,
        #[serde(default)]
        pub required_keys: Vec<String>,
        /// Total number of DECLARED argument properties (the schema's `properties`
        /// map), or 0 when unknown. Needed to tell whether a single required key is
        /// the ONLY possible first key. Without it we cannot safely jump the key.
        #[serde(default)]
        pub declarable_keys: usize,
        /// Whether the schema forbids undeclared keys (`additionalProperties:false`).
        /// Only a closed schema can guarantee no other key appears first.
        #[serde(default)]
        pub closed: bool,
    }

    impl ToolSchema {
        /// Convenience constructor. `declarable_keys`/`closed` default to
        /// unknown/false, so key jump-forward stays OFF unless the shape is known via
        /// [`ToolSchema::from_input_schema`] or [`ToolSchema::with_shape`]. This keeps
        /// the losslessness invariant safe by default.
        pub fn new(name: impl Into<String>, required_keys: Vec<String>) -> Self {
            Self { name: name.into(), required_keys, declarable_keys: 0, closed: false }
        }

        /// Declare the full argument shape (total declared property count + whether the
        /// schema is closed) so the grammar can decide when key jump-forward is safe.
        pub fn with_shape(mut self, declarable_keys: usize, closed: bool) -> Self {
            self.declarable_keys = declarable_keys;
            self.closed = closed;
            self
        }

        /// Build from a JSON-Schema-ish object, reading `required`, the full
        /// `properties` set, and `additionalProperties`, as carried on every
        /// `ToolSpec.input_schema`.
        pub fn from_input_schema(name: impl Into<String>, schema: &Value) -> Self {
            let required_keys = schema
                .get("required")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
                .unwrap_or_default();
            let declarable_keys = schema.get("properties").and_then(|v| v.as_object()).map(|p| p.len()).unwrap_or(0);
            let closed = schema.get("additionalProperties").and_then(|v| v.as_bool()) == Some(false);
            Self { name: name.into(), required_keys, declarable_keys, closed }
        }
    }

    /// What the grammar reports about the tool-name field given the chars typed so
    /// far (the characters after the opening quote of the name).
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct NameJump {
        /// The continuation that is the only legal one: the shared prefix of every
        /// registered name still consistent with `typed`, beyond what is typed.
        pub forced: String,
        /// `Some(name)` once `typed + forced` uniquely identifies a full tool name.
        pub resolved: Option<String>,
        /// True when `typed` is a prefix of no registered tool (a dead branch the
        /// constraint would have prevented; surfaced so the caller can reject).
        pub dead: bool,
    }

    /// The canonical tool-call grammar over a fixed set of registered tools.
    #[derive(Debug, Clone)]
    pub struct ToolCallGrammar {
        schemas: Vec<ToolSchema>,
    }

    impl ToolCallGrammar {
        /// Build from the registered tool schemas. Names are de-duplicated and sorted
        /// so common-prefix computation is stable.
        pub fn new(mut schemas: Vec<ToolSchema>) -> Self {
            schemas.sort_by(|a, b| a.name.cmp(&b.name));
            schemas.dedup_by(|a, b| a.name == b.name);
            Self { schemas }
        }

        /// The registered tool names, sorted.
        pub fn names(&self) -> Vec<&str> {
            self.schemas.iter().map(|s| s.name.as_str()).collect()
        }

        /// The forced opening scaffolding: from an empty output, this exact string is
        /// the only legal start of a tool call, so it can be emitted with no forward
        /// pass. (The runtime uses the whitespace-canonical form.)
        pub fn envelope_prefix(&self) -> &'static str {
            "{\"name\": \""
        }

        /// The closing brace of the outer envelope, forced once the arguments object
        /// is complete.
        pub fn envelope_suffix(&self) -> &'static str {
            "}"
        }

        /// Jump-forward within the tool-name field. Given the name characters typed so
        /// far, return the forced shared continuation and, once unambiguous, the
        /// resolved name.
        pub fn name_jump(&self, typed: &str) -> NameJump {
            let consistent: Vec<&str> =
                self.schemas.iter().map(|s| s.name.as_str()).filter(|n| n.starts_with(typed)).collect();
            if consistent.is_empty() {
                return NameJump { forced: String::new(), resolved: None, dead: true };
            }
            let lcp = longest_common_prefix(&consistent);
            let forced = lcp[typed.len().min(lcp.len())..].to_string();
            let resolved = if consistent.len() == 1 { Some(consistent[0].to_string()) } else { None };
            NameJump { forced, resolved, dead: false }
        }

        /// The maximal deterministic skeleton once the tool is known: the whole
        /// envelope up to the point where free content (the first argument value)
        /// begins. This is the single biggest jump-forward win: given the resolved
        /// tool, the runtime emits this entire string for free.
        ///
        /// Returns `None` if `name` is not registered. The first argument key is
        /// jumped ONLY when it is provably the sole legal opening key: the schema is
        /// closed (`additionalProperties:false`) AND its single required key is its
        /// only declared property. If the schema has optional properties (so the
        /// object could legally begin with a different key) the scaffold stops at the
        /// object opener `{`, preserving losslessness (never force a non-forced token).
        pub fn scaffold_for(&self, name: &str, sole_key: bool) -> Option<String> {
            let schema = self.schemas.iter().find(|s| s.name == name)?;
            let mut out = format!("{{\"name\": \"{name}\", \"arguments\": {{");
            let key_is_forced =
                sole_key && schema.closed && schema.required_keys.len() == 1 && schema.declarable_keys == 1;
            if key_is_forced {
                out.push_str(&format!("\"{}\": ", schema.required_keys[0]));
            }
            Some(out)
        }

        /// Validity gate (Phase 2): is `(name, args)` a legal call under the grammar.
        /// Used both as the constrained-decode invariant and as a cheap post-hoc check.
        pub fn is_valid_call(&self, name: &str, args: &Value) -> Result<(), String> {
            let Some(schema) = self.schemas.iter().find(|s| s.name == name) else {
                return Err(format!("unknown tool \"{name}\""));
            };
            let Some(obj) = args.as_object() else {
                return Err("arguments must be a JSON object".to_string());
            };
            for key in &schema.required_keys {
                if !obj.contains_key(key) {
                    return Err(format!("missing required argument \"{key}\""));
                }
            }
            Ok(())
        }

        /// The fraction of a fully-rendered canonical call for `name` that is grammar
        /// -forced scaffolding (envelope + name + punctuation), given the rendered
        /// arguments JSON. This is the "how much did jump-forward save" measure the
        /// runtime reports; it is a lower bound (prompt-lookup saves more on top).
        pub fn forced_fraction(&self, name: &str, arguments_json: &str) -> Option<f64> {
            let scaffold = self.scaffold_for(name, false)?;
            let total = scaffold.len() + arguments_json.len() + self.envelope_suffix().len();
            if total == 0 {
                return Some(0.0);
            }
            let forced = scaffold.len() + self.envelope_suffix().len();
            Some(forced as f64 / total as f64)
        }
    }

    /// Longest common prefix of a non-empty slice of strings (byte-safe: only cuts on
    /// a char boundary because all inputs are `&str` and we compare whole chars).
    fn longest_common_prefix(items: &[&str]) -> String {
        let Some(first) = items.first() else {
            return String::new();
        };
        let mut prefix = *first;
        for s in &items[1..] {
            let mut end = 0;
            for ((i, a), b) in prefix.char_indices().zip(s.chars()) {
                if a == b {
                    end = i + a.len_utf8();
                } else {
                    break;
                }
            }
            prefix = &prefix[..end];
            if prefix.is_empty() {
                break;
            }
        }
        prefix.to_string()
    }

    // ---------------------------------------------------------------------------
    // prompt-lookup drafter (copied argument values)
    // ---------------------------------------------------------------------------

    /// A training-free n-gram / prompt-lookup drafter. It proposes a continuation for
    /// the generated text by finding the tail of what has been generated inside a
    /// haystack (the prompt / a file just read) and copying what follows there.
    #[derive(Debug, Clone)]
    pub struct PromptLookup {
        /// Longest suffix (in chars) to try to match first.
        pub max_ngram: usize,
        /// Shortest suffix to accept a match on.
        pub min_ngram: usize,
    }

    impl Default for PromptLookup {
        fn default() -> Self {
            Self { max_ngram: 32, min_ngram: 3 }
        }
    }

    impl PromptLookup {
        pub fn new(min_ngram: usize, max_ngram: usize) -> Self {
            Self { max_ngram: max_ngram.max(min_ngram), min_ngram: min_ngram.max(1) }
        }

        /// Draft up to `max_draft` characters continuing `generated`, by matching the
        /// longest suffix of `generated` (between `min_ngram` and `max_ngram`) that
        /// occurs in `haystack` and returning the text that follows it there. Returns
        /// `None` when no suffix of the allowed lengths matches with any follow-on.
        ///
        /// The runtime feeds the drafted chars (tokenized) to the target verifier, so
        /// a wrong guess costs nothing but the verify it would have done anyway.
        pub fn draft(&self, generated: &str, haystack: &str, max_draft: usize) -> Option<String> {
            if max_draft == 0 || generated.is_empty() || haystack.is_empty() {
                return None;
            }
            let gen_chars: Vec<char> = generated.chars().collect();
            let hi = self.max_ngram.min(gen_chars.len());
            for k in (self.min_ngram..=hi).rev() {
                let suffix: String = gen_chars[gen_chars.len() - k..].iter().collect();
                // First occurrence that has following characters.
                let mut search_from = 0;
                while let Some(rel) = haystack[search_from..].find(&suffix) {
                    let end = search_from + rel + suffix.len();
                    if end < haystack.len() {
                        let draft: String = haystack[end..].chars().take(max_draft).collect();
                        if !draft.is_empty() {
                            return Some(draft);
                        }
                    }
                    // advance past this occurrence to look for a later one with follow-on
                    search_from = search_from + rel + 1;
                    if search_from >= haystack.len() {
                        break;
                    }
                }
            }
            None
        }
    }

    /// How many leading characters of `draft` the target actually accepts, given the
    /// ground-truth continuation `truth` the target would have produced. This is the
    /// lossless-verify accounting: the accepted count is the length of the common
    /// prefix, and the first mismatch is where real decoding resumes. Pure, so the
    /// governor and tests can reason about acceptance without a model.
    pub fn accepted_prefix_len(draft: &str, truth: &str) -> usize {
        draft.chars().zip(truth.chars()).take_while(|(a, b)| a == b).count()
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use serde_json::json;

        fn grammar() -> ToolCallGrammar {
            ToolCallGrammar::new(vec![
                ToolSchema::new("fs.read", vec!["path".into()]),
                ToolSchema::new("fs.list", vec!["path".into()]),
                ToolSchema::new("git.status", vec![]),
                ToolSchema::new("shell.run", vec!["argv".into()]),
            ])
        }

        #[test]
        fn envelope_prefix_is_forced_opening() {
            assert_eq!(grammar().envelope_prefix(), "{\"name\": \"");
        }

        #[test]
        fn name_jump_shares_prefix_then_resolves() {
            let g = grammar();
            // Empty typed: fs.* and git.* and shell.* share nothing -> no forced chars.
            assert_eq!(g.name_jump("").forced, "");
            // "fs." is shared by fs.read and fs.list: typing "f" forces "s." (the LCP).
            let j = g.name_jump("f");
            assert_eq!(j.forced, "s.");
            assert_eq!(j.resolved, None);
            // "fs.r" uniquely identifies fs.read: forced completes it, resolved set.
            let j = g.name_jump("fs.r");
            assert_eq!(j.forced, "ead");
            assert_eq!(j.resolved.as_deref(), Some("fs.read"));
            // "git." uniquely identifies git.status.
            assert_eq!(g.name_jump("git.").resolved.as_deref(), Some("git.status"));
        }

        #[test]
        fn name_jump_flags_dead_branch() {
            let j = grammar().name_jump("nope");
            assert!(j.dead);
            assert_eq!(j.resolved, None);
        }

        #[test]
        fn scaffold_forces_sole_key_only_when_closed_single_prop() {
            // A closed schema whose ONLY declared property IS the single required key:
            // "path" is provably the only legal first key, so jumping it is lossless.
            let g = ToolCallGrammar::new(vec![ToolSchema::from_input_schema(
                "x.single",
                &json!({
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"],
                    "additionalProperties": false
                }),
            )]);
            assert_eq!(
                g.scaffold_for("x.single", true).unwrap(),
                "{\"name\": \"x.single\", \"arguments\": {\"path\": "
            );
            // sole_key=false always stops at the object opener.
            assert_eq!(g.scaffold_for("x.single", false).unwrap(), "{\"name\": \"x.single\", \"arguments\": {");
            assert_eq!(g.scaffold_for("made.up", true), None);
        }

        #[test]
        fn scaffold_does_not_force_key_when_optional_props_exist() {
            // fs.read's REAL schema: required [path] but optional range/encoding, so the
            // arguments object may legally begin with a non-required key. Forcing "path"
            // would emit a token that was not the only legal continuation (a losslessness
            // violation), so the scaffold must stop at the object opener.
            let g = ToolCallGrammar::new(vec![ToolSchema::from_input_schema(
                "fs.read",
                &json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "range": { "type": "array" },
                        "encoding": { "type": "string" }
                    },
                    "required": ["path"],
                    "additionalProperties": false
                }),
            )]);
            assert_eq!(g.scaffold_for("fs.read", true).unwrap(), "{\"name\": \"fs.read\", \"arguments\": {");
        }

        #[test]
        fn scaffold_does_not_force_key_when_multiple_required() {
            let g = ToolCallGrammar::new(vec![ToolSchema::new("x.multi", vec!["a".into(), "b".into()])]);
            // Two required keys: order is not forced, so only the object opener is emitted.
            assert_eq!(g.scaffold_for("x.multi", true).unwrap(), "{\"name\": \"x.multi\", \"arguments\": {");
        }

        #[test]
        fn validity_gate_checks_name_and_required_keys() {
            let g = grammar();
            assert!(g.is_valid_call("fs.read", &json!({ "path": "a" })).is_ok());
            assert!(g.is_valid_call("fs.read", &json!({})).is_err());
            assert!(g.is_valid_call("made.up", &json!({})).is_err());
            assert!(g.is_valid_call("fs.read", &json!("not object")).is_err());
            assert!(g.is_valid_call("git.status", &json!({})).is_ok());
        }

        #[test]
        fn forced_fraction_is_a_real_lower_bound() {
            let g = grammar();
            // git.status with empty args: almost all of it is forced scaffolding.
            let frac = g.forced_fraction("git.status", "{}").unwrap();
            assert!(frac > 0.9, "expected mostly-forced, got {frac}");
            // A call with a long argument value has a lower forced fraction.
            let low = g.forced_fraction("shell.run", "{\"argv\": [\"a very long command here\"]}").unwrap();
            assert!(low < frac);
        }

        #[test]
        fn schema_from_input_schema_reads_required() {
            let s = ToolSchema::from_input_schema("fs.read", &json!({ "type": "object", "required": ["path"] }));
            assert_eq!(s.required_keys, vec!["path".to_string()]);
        }

        #[test]
        fn prompt_lookup_copies_run_from_context() {
            let lookup = PromptLookup::default();
            // The model has emitted a path prefix that appears in the file it read.
            let haystack = "files: src/main.rs, src/lib.rs, README.md";
            let generated = "open src/li";
            let draft = lookup.draft(generated, haystack, 6).unwrap();
            assert_eq!(draft, "b.rs, ");
        }

        #[test]
        fn prompt_lookup_returns_none_without_match() {
            let lookup = PromptLookup::default();
            assert!(lookup.draft("zzz qqq", "nothing alike here", 8).is_none());
        }

        #[test]
        fn accepted_prefix_len_is_common_prefix() {
            assert_eq!(accepted_prefix_len("src/lib.rs", "src/lib.rs"), 10);
            assert_eq!(accepted_prefix_len("src/lib.rs", "src/main"), 4);
            assert_eq!(accepted_prefix_len("abc", "xyz"), 0);
        }

        #[test]
        fn longest_common_prefix_basic() {
            assert_eq!(longest_common_prefix(&["fs.read", "fs.list"]), "fs.");
            assert_eq!(longest_common_prefix(&["a", "b"]), "");
            assert_eq!(longest_common_prefix(&["only"]), "only");
        }
    }
}

pub use adapters::{AdapterRegistry, AdapterSelection};
pub use confidence::{self_consistency_vote, AnswerNormalizer, VoteResult};
pub use escalation::{
    EscalationBudget, EscalationCascade, EscalationOutcome, SelfConsistencyProbe,
};
pub use executor::{Executor, HttpClientFactory};
pub use grammar::{GrammarMatcher, GrammarSpec, GrammarValidation, ShellGrammarCompiler};
pub use http_client::{GenerateRoute, HawkingHttpClient};
pub use inference::{InferenceClient, StubInferenceClient};
pub use registry::{default_hawking_local_roles, RoleRegistry};
pub use router::{RouteDecision, Router, SimpleRouter};
pub use scheduler::{Admission, ResourceSnapshot, Scheduler};
