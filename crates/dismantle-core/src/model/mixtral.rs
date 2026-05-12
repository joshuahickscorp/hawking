//! Mixtral 8x7B preview skeleton.
//!
//! v1.0.0's memory-diff launch needs architecture detection and the public
//! shape of a Mixtral engine, but the full standard-MHA + 8-expert forward path
//! remains v1.0.1 work. This module keeps the fallback honest: Mixtral GGUFs
//! are recognized distinctly from dense LLaMA models and fail with a clear
//! preview error instead of being reported as an unknown architecture.

use crate::engine::{Engine, EngineConfig, GenStats, GenerateRequest, StreamEvent};
use crate::gguf::GgufFile;
use crate::tokenizer::Tokenizer;
use crate::{Error, Result};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MixtralConfig {
    pub n_layers: usize,
    pub hidden: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
}

impl MixtralConfig {
    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let get_u32 = |k: &str| g.metadata.get(k).and_then(|v| v.as_u32());
        let n_layers = get_u32("llama.block_count")
            .ok_or_else(|| Error::Model("missing llama.block_count".into()))?
            as usize;
        let hidden = get_u32("llama.embedding_length")
            .ok_or_else(|| Error::Model("missing llama.embedding_length".into()))?
            as usize;
        let n_heads = get_u32("llama.attention.head_count")
            .ok_or_else(|| Error::Model("missing llama.attention.head_count".into()))?
            as usize;
        let n_kv_heads =
            get_u32("llama.attention.head_count_kv").unwrap_or(n_heads as u32) as usize;
        let intermediate = get_u32("llama.feed_forward_length")
            .ok_or_else(|| Error::Model("missing llama.feed_forward_length".into()))?
            as usize;
        let n_experts = get_u32("llama.expert_count")
            .ok_or_else(|| Error::Model("missing llama.expert_count".into()))?
            as usize;
        let top_k = get_u32("llama.expert_used_count").unwrap_or(2) as usize;
        let vocab_size = get_u32("llama.vocab_size")
            .map(|v| v as usize)
            .or_else(|| {
                g.tensor("token_embd.weight")
                    .and_then(|t| t.dims.iter().copied().max())
                    .map(|v| v as usize)
            })
            .ok_or_else(|| Error::Model("missing llama vocab size".into()))?;

        Ok(Self {
            n_layers,
            hidden,
            n_heads,
            n_kv_heads,
            head_dim: hidden / n_heads,
            intermediate,
            n_experts,
            top_k,
            vocab_size,
            max_seq_len: get_u32("llama.context_length").unwrap_or(32768) as usize,
        })
    }

    pub fn synthetic_for_test() -> Self {
        Self {
            n_layers: 2,
            hidden: 256,
            n_heads: 8,
            n_kv_heads: 2,
            head_dim: 32,
            intermediate: 512,
            n_experts: 8,
            top_k: 2,
            vocab_size: 128,
            max_seq_len: 256,
        }
    }
}

pub struct MixtralEngine {
    pub config: MixtralConfig,
    pub model_id: String,
    pub _weights_path: PathBuf,
}

impl MixtralEngine {
    pub fn load_tokenizer_preview(weights: &Path, gguf: &GgufFile) -> Result<Tokenizer> {
        let sidecar = weights
            .parent()
            .map(|d| d.join("tokenizer.json"))
            .filter(|p| p.exists());
        if let Some(path) = sidecar {
            Tokenizer::from_file(path)
        } else {
            Tokenizer::from_gguf(gguf)
        }
    }

    pub fn synthetic_forward_shape_for_test(config: &MixtralConfig, token: u32) -> Vec<f32> {
        (0..config.vocab_size)
            .map(|i| ((i as u32 ^ token) as f32) / config.vocab_size as f32)
            .collect()
    }
}

impl Engine for MixtralEngine {
    fn load(weights: &Path, _config: EngineConfig) -> Result<Self> {
        let gguf = GgufFile::open(weights)?;
        if !is_mixtral_gguf(&gguf) {
            return Err(Error::Model("not a Mixtral MoE GGUF".into()));
        }
        let _cfg = MixtralConfig::from_gguf(&gguf)?;
        let _tokenizer = Self::load_tokenizer_preview(weights, &gguf)?;
        Err(Error::Unimplemented(
            "MixtralEngine preview: detection/tokenizer scaffolding landed; full forward path is v1.0.1",
        ))
    }

    fn generate(
        &mut self,
        _req: GenerateRequest,
        _sink: &mut dyn FnMut(StreamEvent),
    ) -> Result<GenStats> {
        Err(Error::Unimplemented("MixtralEngine::generate"))
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn model_arch(&self) -> &str {
        "mixtral"
    }

    fn forward_tokens_for_test(
        &mut self,
        _tokens: &[u32],
        _positions: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        Err(Error::Unimplemented(
            "MixtralEngine::forward_tokens_for_test",
        ))
    }
}

pub fn is_mixtral_gguf(gguf: &GgufFile) -> bool {
    gguf.architecture() == Some("llama") && gguf.tensor("blk.0.ffn_gate_exps.weight").is_some()
}
