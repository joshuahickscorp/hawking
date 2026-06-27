use crate::error::Result;
use crate::ids::{ModelId, RoleId};
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderCaps {
    pub streaming: bool,
    pub embeddings: bool,
    pub grammar: bool,
    pub raw_logits: bool,
    pub logprobs: bool,
    pub lora: bool,
    pub kv_handles: bool,
    pub native_tokens_endpoint: bool,
}

impl ProviderCaps {
    pub fn hawking_local_shell_today() -> Self {
        Self {
            streaming: true,
            embeddings: true,
            grammar: false,
            raw_logits: false,
            logprobs: false,
            lora: false,
            kv_handles: false,
            native_tokens_endpoint: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelDescriptor {
    pub id: ModelId,
    pub name: String,
    pub architecture: ModelArchitecture,
    pub context_tokens: usize,
    pub tokenizer_signature: String,
    pub footprint_mb: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelArchitecture {
    Transformer,
    Ssm,
    Hybrid,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelRole {
    pub id: RoleId,
    pub name: String,
    pub purpose: RolePurpose,
    pub model: ModelDescriptor,
    pub caps: ProviderCaps,
    pub default_sampler: SamplerProfile,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RolePurpose {
    HeroCoder,
    FastDraft,
    Embedder,
    Reranker,
    Summarizer,
    Classifier,
    ToolPlanner,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplerProfile {
    pub temperature: f32,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub deterministic: bool,
}

impl SamplerProfile {
    pub fn deterministic_edit() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            top_p: None,
            repetition_penalty: None,
            seed: Some(0),
            deterministic: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub task_kind: String,
    pub prompt: String,
    pub messages: Vec<InferenceMessage>,
    pub max_output_tokens: usize,
    pub sampler: Option<SamplerProfile>,
    pub grammar: Option<String>,
    pub want_logprobs: bool,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamChunk {
    Token {
        token_id: Option<u32>,
        text: String,
    },
    Done {
        reason: String,
        stats: Option<GenerationStats>,
    },
    Error {
        message: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationStats {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub decode_tokens_per_second: Option<f32>,
}

pub type TokenSink<'a> = &'a mut (dyn FnMut(StreamChunk) -> Result<()> + Send);

pub trait ModelProvider: Send + Sync {
    fn id(&self) -> &str;
    fn capabilities(&self) -> ProviderCaps;
    fn generate<'a>(
        &'a self,
        request: InferenceRequest,
        sink: TokenSink<'a>,
    ) -> BoxFuture<'a, Result<GenerationStats>>;
    fn embed<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>>>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeSupervisorState {
    Down,
    Booting,
    Ready,
    Degraded,
    Failed,
}
