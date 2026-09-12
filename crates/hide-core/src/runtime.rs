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
    /// Localhost endpoint this role is served from (ch.06 §4.4). Optional so
    /// roles can be declared before an endpoint is resolved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    /// Relative cost hint for the scheduler/admission (ch.06 §4.11).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<f32>,
    /// The role to escalate to when confidence is low — expresses the
    /// confidence-aware cascade graph (ch.06 §4.4, the chapter's thesis).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub escalates_to: Option<RoleId>,
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
    /// Long-context SSM (RWKV-7 / Mamba-2) routing (ch.06).
    SsmLong,
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
    /// Decode wall time reported by the runtime, excluding prompt prefill,
    /// in monotonic integer nanoseconds. `None` means the provider did not
    /// expose this metric.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_ns: Option<u64>,
    /// Compatibility projection for older providers and readers. New runtime
    /// paths should populate `decode_ns` and derive this view from it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_ms: Option<f64>,
    /// Completed decode forwards represented by `decode_ns`. This is distinct
    /// from emitted tokens because token zero can be sampled from prompt prefill.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_decode_forwards: Option<usize>,
    /// Runtime-reported decode throughput. Consumers must pair it with
    /// `completed_decode_forwards` when claiming complete-forward TPS.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_tokens_per_second: Option<f32>,
}

impl GenerationStats {
    /// Return the exact runtime duration when available, accepting the old
    /// floating-point milliseconds field only as a read-only compatibility
    /// fallback.
    pub fn effective_decode_ns(&self) -> Option<u64> {
        self.decode_ns
            .filter(|value| *value > 0)
            .or_else(|| self.decode_ms.and_then(decode_ms_to_ns))
    }

    /// Compatibility projection for existing TPS/report code.
    pub fn effective_decode_ms(&self) -> Option<f64> {
        self.effective_decode_ns()
            .map(|nanoseconds| nanoseconds as f64 / 1_000_000.0)
    }
}

fn decode_ms_to_ns(milliseconds: f64) -> Option<u64> {
    if !milliseconds.is_finite() || milliseconds <= 0.0 {
        return None;
    }
    let nanoseconds = milliseconds * 1_000_000.0;
    (nanoseconds <= u64::MAX as f64).then(|| nanoseconds.round() as u64)
}

#[cfg(test)]
mod generation_stats_tests {
    use super::GenerationStats;

    #[test]
    fn prefers_exact_nanoseconds_and_derives_legacy_milliseconds() {
        let stats = GenerationStats {
            input_tokens: 1,
            output_tokens: 2,
            decode_ns: Some(1_234_567),
            decode_ms: Some(99.0),
            completed_decode_forwards: Some(2),
            decode_tokens_per_second: None,
        };
        assert_eq!(stats.effective_decode_ns(), Some(1_234_567));
        assert_eq!(stats.effective_decode_ms(), Some(1.234567));
    }

    #[test]
    fn converts_legacy_milliseconds_only_as_a_compatibility_fallback() {
        let stats = GenerationStats {
            input_tokens: 0,
            output_tokens: 0,
            decode_ns: None,
            decode_ms: Some(2.5),
            completed_decode_forwards: None,
            decode_tokens_per_second: None,
        };
        assert_eq!(stats.effective_decode_ns(), Some(2_500_000));
    }
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
