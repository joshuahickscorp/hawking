use futures::future::BoxFuture;
use hawking_orch::inference::InferenceClient;
use hawking_orch::router::{RouteDecision, Router};
use hide_core::error::Result;
use hide_core::runtime::{GenerationStats, InferenceRequest, StreamChunk};
use std::sync::Arc;

pub struct KernelRuntimeClient {
    router: Arc<dyn Router>,
    inference: Arc<dyn InferenceClient>,
}

impl KernelRuntimeClient {
    pub fn new(router: Arc<dyn Router>, inference: Arc<dyn InferenceClient>) -> Self {
        Self { router, inference }
    }

    pub fn route(&self, request: &InferenceRequest) -> Result<RouteDecision> {
        self.router.route(request)
    }

    pub fn generate<'a>(
        &'a self,
        request: InferenceRequest,
        sink: &'a mut (dyn FnMut(StreamChunk) -> Result<()> + Send),
    ) -> BoxFuture<'a, Result<GenerationStats>> {
        self.inference.generate(request, sink)
    }
}
