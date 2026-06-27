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
        Self {
            registry,
            estimator: DifficultyEstimator::default(),
        }
    }

    fn choose_role(
        &self,
        request: &InferenceRequest,
        difficulty: &DifficultyEstimate,
    ) -> Option<ModelRole> {
        if request.task_kind == "embedding" {
            return self
                .registry
                .by_purpose(RolePurpose::Embedder)
                .into_iter()
                .next();
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
            self.registry
                .by_purpose(RolePurpose::HeroCoder)
                .into_iter()
                .next()
        } else {
            self.registry
                .by_purpose(RolePurpose::FastDraft)
                .into_iter()
                .next()
                .or_else(|| {
                    self.registry
                        .by_purpose(RolePurpose::HeroCoder)
                        .into_iter()
                        .next()
                })
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
            sampler: request
                .sampler
                .clone()
                .unwrap_or_else(|| role.default_sampler.clone()),
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
            caps: ProviderCaps {
                grammar: true,
                ..ProviderCaps::hawking_local_shell_today()
            },
            default_sampler: SamplerProfile::deterministic_edit(),
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
