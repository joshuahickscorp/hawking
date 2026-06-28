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
use hide_core::runtime::{
    GenerationStats, InferenceRequest, ModelRole, SamplerProfile, StreamChunk,
};
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
        Self {
            max_escalations: 2,
            vote_samples: 3,
            accept_threshold: 0.67,
        }
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
                return (
                    0.0,
                    Some(EscalationReason::GrammarInvalid { code: hint.code }),
                );
            }
        }
        let normalizer = if self.normalizer_is_json {
            AnswerNormalizer::CanonicalJson
        } else {
            AnswerNormalizer::CaseFold
        };
        // Include the primary in the vote pool.
        let mut pool: Vec<String> = Vec::with_capacity(samples.len() + 1);
        pool.push(primary.to_string());
        pool.extend_from_slice(samples);
        let vote = self_consistency_vote(&pool, normalizer);
        let confidence = vote.confidence();
        let threshold = if self.accept_threshold > 0.0 {
            self.accept_threshold
        } else {
            0.67
        };
        let reason = if confidence < threshold {
            Some(EscalationReason::VoteDisagreement {
                agreement: vote.agreement,
            })
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
        Self {
            registry,
            client,
            probe,
            budget,
        }
    }

    /// Collect a single completion's full text (concatenating token chunks).
    async fn run_once(
        &self,
        request: &InferenceRequest,
    ) -> Result<(String, GenerationStats)> {
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
    fn request_for_role(
        base: &InferenceRequest,
        role: &ModelRole,
        force_edit_sampler: bool,
    ) -> InferenceRequest {
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
    pub async fn execute(
        &self,
        decision: &RouteDecision,
        request: &InferenceRequest,
    ) -> Result<EscalationOutcome> {
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
            let samples = self
                .collect_vote_samples(&role_request, &primary)
                .await?;

            let (confidence, reason) =
                self.probe.score(&primary, &samples, grammar.as_ref());

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
            let next_id = current_role
                .escalates_to
                .clone()
                .expect("checked !at_top above");
            let next_role = self.registry.get(&next_id).ok_or_else(|| {
                HideError::NotFound(format!("escalation target role {next_id}"))
            })?;
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

    async fn collect_vote_samples(
        &self,
        role_request: &InferenceRequest,
        _primary: &str,
    ) -> Result<Vec<String>> {
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
    use hide_core::runtime::{
        ModelArchitecture, ModelDescriptor, ProviderCaps, RolePurpose, SamplerProfile,
    };
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
        let client = Arc::new(ScriptedInferenceClient::new(vec![
            "yes".into(),
            "yes".into(),
            "yes".into(),
        ]));
        let probe = Arc::new(SelfConsistencyProbe {
            normalizer_is_json: false,
            accept_threshold: 0.67,
        });
        let cascade = EscalationCascade::new(
            registry,
            client,
            probe,
            EscalationBudget::default(),
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
        let probe = Arc::new(SelfConsistencyProbe {
            normalizer_is_json: false,
            accept_threshold: 0.67,
        });
        let cascade = EscalationCascade::new(
            registry,
            client,
            probe,
            EscalationBudget {
                max_escalations: 2,
                vote_samples: 3,
                accept_threshold: 0.67,
            },
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
        assert!(matches!(
            outcome.trail[0].escalated,
            Some(EscalationReason::VoteDisagreement { .. })
        ));
    }

    #[tokio::test]
    async fn top_role_cannot_escalate_even_if_uncertain() {
        let hero_id = RoleId::new();
        let registry = Arc::new(RoleRegistry::default());
        registry.register(make_role("hero", hero_id.clone(), None));
        let client = Arc::new(ScriptedInferenceClient::new(vec![
            "a".into(),
            "b".into(),
            "c".into(),
        ]));
        let probe = Arc::new(SelfConsistencyProbe {
            normalizer_is_json: false,
            accept_threshold: 0.99,
        });
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
