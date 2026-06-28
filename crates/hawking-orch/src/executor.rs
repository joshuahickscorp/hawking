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
        let client: Arc<dyn InferenceClient> =
            Arc::new(HawkingHttpClient::with_route(endpoint, route));
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
        Self {
            registry,
            factory,
            default_endpoint: "http://127.0.0.1:8080".to_string(),
        }
    }

    pub fn with_default_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.default_endpoint = endpoint.into();
        self
    }

    fn resolve_endpoint(&self, role_endpoint: Option<&str>) -> String {
        role_endpoint
            .map(str::to_string)
            .unwrap_or_else(|| self.default_endpoint.clone())
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
        let route = if request.messages.is_empty() {
            GenerateRoute::Native
        } else {
            GenerateRoute::Chat
        };
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
            .or_else(|| {
                self.registry
                    .by_purpose(RolePurpose::Embedder)
                    .into_iter()
                    .next()
            })
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
    let mut stats = GenerationStats {
        input_tokens: 0,
        output_tokens: 0,
        decode_tokens_per_second: None,
    };
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
    use hide_core::runtime::{
        ModelArchitecture, ModelDescriptor, ModelRole, ProviderCaps, SamplerProfile,
    };
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
            caps: ProviderCaps {
                embeddings: true,
                ..ProviderCaps::hawking_local_shell_today()
            },
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
            estimated_difficulty: DifficultyEstimate {
                score: 0.1,
                reason: "low".into(),
                signals: vec![],
            },
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
        registry.register(role(
            "hero",
            hero_id.clone(),
            RolePurpose::HeroCoder,
            "http://127.0.0.1:8081",
        ));
        let stub: Arc<dyn InferenceClient> = Arc::new(StubInferenceClient::new("generated"));
        let executor = Executor::new(registry, Arc::new(FixedClientFactory(stub)));
        let (text, _) = collect_to_string(&executor, &decision(hero_id), request())
            .await
            .unwrap();
        assert_eq!(text, "generated");
    }

    #[tokio::test]
    async fn executor_embeds_via_embedder_role() {
        let embed_id = RoleId::new();
        let registry = Arc::new(RoleRegistry::default());
        registry.register(role(
            "embedder",
            embed_id.clone(),
            RolePurpose::Embedder,
            "http://127.0.0.1:8083",
        ));
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
