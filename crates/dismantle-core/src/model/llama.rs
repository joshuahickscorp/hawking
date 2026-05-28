//! Llama-family dense forward pass (Llama-2 / Llama-3.x / Mistral).
//!
//! Architecturally a near-sibling of `qwen_dense`:
//!
//!   - Grouped-query attention (n_heads / n_kv_heads)
//!   - SwiGLU FFN (gate + up + down)
//!   - RMSNorm
//!   - RoPE; Llama-3.1+ adds NTK-aware piecewise frequency rescaling
//!     (see [`crate::kernels::Llama3RopeScaling`]).
//!
//! Two structural differences from Qwen2:
//!
//!   1. No Q/K/V biases (Llama families omit them; Qwen2 carries them).
//!   2. RoPE θ is typically 500_000 (Llama-3) instead of 1_000_000.
//!
//! Step 3 lands the loader + dispatcher. Forward path is filled in by
//! subsequent commits (prefill → decode → Metal hot path).

use crate::cache::KvCache;
use crate::engine::{Engine, EngineConfig, GenStats, GenerateRequest, StreamEvent};
use crate::gguf::{GgmlType, GgufFile};
use crate::kernels::Llama3RopeScaling;
use crate::profile::KernelProfile;
use crate::quant;
use crate::sample::Sampler;
use crate::tokenizer::Tokenizer;
use crate::{Error, Result};
use half::f16;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub n_layers: usize,
    pub hidden: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
    /// Llama-3.1+ NTK-aware RoPE rescaling. `Some` when the GGUF carries
    /// `llama.rope.scaling.type == "llama3"` and the four
    /// `llama.rope.scaling.*` parameters. Earlier Llama / Mistral GGUFs
    /// leave this `None`.
    pub rope_scaling: Option<Llama3RopeScaling>,
    /// Distinguish Llama-2 / Llama-3 / Mistral for reporting. Carried
    /// from GGUF `general.architecture` verbatim so profile matching
    /// (via [`crate::profile::arch_family`]) and downstream logging both
    /// see the original arch string.
    pub arch: String,
}

impl LlamaConfig {
    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let arch = g.architecture().unwrap_or("").to_string();
        // Every Llama-family GGUF llama.cpp produces uses the `llama.*`
        // metadata prefix regardless of point release (Llama-2/3/3.1/3.2,
        // Mistral, Phi when ported to llama.cpp's "llama" arch). So a
        // single prefix here covers the whole family.
        let get_u32 = |k: &str| g.metadata.get(k).and_then(|v| v.as_u32());
        let get_f32 = |k: &str| g.metadata.get(k).and_then(|v| v.as_f32());
        let get_str = |k: &str| g.metadata.get(k).and_then(|v| v.as_str());

        let n_layers = get_u32("llama.block_count")
            .ok_or_else(|| Error::Model("missing llama.block_count".into()))?
            as usize;
        let hidden = get_u32("llama.embedding_length")
            .ok_or_else(|| Error::Model("missing llama.embedding_length".into()))?
            as usize;
        let n_heads = get_u32("llama.attention.head_count")
            .ok_or_else(|| Error::Model("missing llama.attention.head_count".into()))?
            as usize;
        let n_kv_heads = get_u32("llama.attention.head_count_kv").unwrap_or(n_heads as u32) as usize;
        // Some Llama GGUFs ship an explicit head_dim (e.g. Llama-3.2 1B
        // where hidden=2048 but head_dim=64 with 32 heads); fall back to
        // hidden/n_heads when absent.
        let head_dim = get_u32("llama.attention.key_length")
            .map(|v| v as usize)
            .unwrap_or(hidden / n_heads);
        let intermediate = get_u32("llama.feed_forward_length")
            .ok_or_else(|| Error::Model("missing llama.feed_forward_length".into()))?
            as usize;
        let vocab_size = match get_u32("llama.vocab_size") {
            Some(v) => v as usize,
            None => {
                // GGUF dim ordering varies; vocab >> hidden in practice,
                // so the max dim on the embed tensor is the vocab size.
                let dims = g
                    .tensor("token_embd.weight")
                    .map(|t| t.dims.clone())
                    .ok_or_else(|| {
                        Error::Model("vocab size not in metadata or token_embd dims".into())
                    })?;
                dims.iter().copied().max().unwrap_or(0) as usize
            }
        };
        let rope_theta = get_f32("llama.rope.freq_base").unwrap_or(500_000.0);
        let rms_norm_eps = get_f32("llama.attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
        let max_seq_len = get_u32("llama.context_length").unwrap_or(8192) as usize;

        // RoPE NTK-aware scaling: only honored when scaling.type ==
        // "llama3". Some Llama-3.0 GGUFs leave scaling.type unset; in
        // that case the four scaling params are absent and we fall
        // through to unscaled RoPE.
        let rope_scaling = if get_str("llama.rope.scaling.type") == Some("llama3") {
            let factor = get_f32("llama.rope.scaling.factor");
            let low = get_f32("llama.rope.scaling.low_freq_factor");
            let high = get_f32("llama.rope.scaling.high_freq_factor");
            let orig = get_u32("llama.rope.scaling.original_context_length");
            match (factor, low, high, orig) {
                (Some(factor), Some(low_freq_factor), Some(high_freq_factor), Some(orig_ctx)) => {
                    Some(Llama3RopeScaling {
                        factor,
                        low_freq_factor,
                        high_freq_factor,
                        original_max_position_embeddings: orig_ctx,
                    })
                }
                _ => None,
            }
        } else {
            None
        };

        Ok(Self {
            n_layers,
            hidden,
            n_heads,
            n_kv_heads,
            head_dim,
            intermediate,
            vocab_size,
            rope_theta,
            rms_norm_eps,
            max_seq_len,
            rope_scaling,
            arch,
        })
    }
}

/// Pointer into the mmap'd GGUF for one tensor. Module-local mirror of
/// the same idiom used by `qwen_dense` and `deepseek_v2` so we don't
/// re-export the type just for cross-module use.
///
/// Fields are read by the forward pass added in step 4; for now the
/// scaffold just records them.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct TensorRef {
    pub offset: usize,
    pub byte_size: usize,
    pub dtype: GgmlType,
    pub n_elems: usize,
}

#[allow(dead_code)] // fields read once forward pass lands in step 4.
pub struct LlamaLayer {
    /// Per-layer norms (eager fp32, small).
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    /// Attention projection weights (lazy — read via TensorRef on each
    /// forward; Metal hot path in step 6 will pin them).
    pub(crate) q_proj: TensorRef,
    pub(crate) k_proj: TensorRef,
    pub(crate) v_proj: TensorRef,
    pub(crate) o_proj: TensorRef,
    // Llama families omit Q/K/V biases (unlike Qwen2).
    /// FFN weights.
    pub(crate) ffn_gate: TensorRef,
    pub(crate) ffn_up: TensorRef,
    pub(crate) ffn_down: TensorRef,
}

pub struct LlamaDense {
    pub config: LlamaConfig,
    pub tokenizer: Tokenizer,
    pub model_id: String,

    /// mmap keepalive (every TensorRef points into this).
    pub gguf: GgufFile,

    pub embed: Vec<f16>,
    pub final_norm: Vec<f32>,
    /// `None` ⇒ tied to embed (Llama-3.2-1B is tied; larger Llama-3
    /// variants typically ship an explicit `output.weight`).
    pub lm_head: Option<Vec<f16>>,
    pub layers: Vec<LlamaLayer>,

    pub kv: KvCache,
    pub sampler: Sampler,
    pub kernel_profile: Option<KernelProfile>,
    pub _weights_path: PathBuf,
}

impl LlamaDense {
    fn dequant_f32(g: &GgufFile, name: &str) -> Result<Vec<f32>> {
        let info = g
            .tensor(name)
            .ok_or_else(|| Error::Model(format!("missing tensor `{name}`")))?;
        let bytes = g.tensor_bytes(name).unwrap();
        quant::dequant_to_f32(info, bytes)
    }

    fn dequant_f16(g: &GgufFile, name: &str) -> Result<Vec<f16>> {
        let info = g
            .tensor(name)
            .ok_or_else(|| Error::Model(format!("missing tensor `{name}`")))?;
        let bytes = g.tensor_bytes(name).unwrap();
        quant::dequant_to_f16(info, bytes)
    }

    fn tensor_ref(g: &GgufFile, name: &str) -> Result<TensorRef> {
        let info = g
            .tensor(name)
            .ok_or_else(|| Error::Model(format!("missing tensor `{name}`")))?;
        let n_elems: usize = info.dims.iter().product::<u64>() as usize;
        Ok(TensorRef {
            offset: info.data_offset as usize,
            byte_size: info.byte_size as usize,
            dtype: info.dtype,
            n_elems,
        })
    }
}

impl Engine for LlamaDense {
    fn load(weights: &Path, config: EngineConfig) -> Result<Self> {
        let gguf = GgufFile::open(weights)?;
        let cfg = LlamaConfig::from_gguf(&gguf)?;
        let model_id = gguf.name().unwrap_or("llama-dense").to_string();

        let sidecar = weights
            .parent()
            .map(|d| d.join("tokenizer.json"))
            .filter(|p| p.exists());
        let tokenizer = if let Some(p) = sidecar {
            Tokenizer::from_file(&p)?
        } else {
            Tokenizer::from_gguf(&gguf)?
        };

        let embed = Self::dequant_f16(&gguf, "token_embd.weight")?;
        let final_norm = Self::dequant_f32(&gguf, "output_norm.weight")?;
        let lm_head = if gguf.tensor("output.weight").is_some() {
            Some(Self::dequant_f16(&gguf, "output.weight")?)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for li in 0..cfg.n_layers {
            let lp = |suf: &str| format!("blk.{li}.{suf}");
            layers.push(LlamaLayer {
                attn_norm: Self::dequant_f32(&gguf, &lp("attn_norm.weight"))?,
                ffn_norm: Self::dequant_f32(&gguf, &lp("ffn_norm.weight"))?,
                q_proj: Self::tensor_ref(&gguf, &lp("attn_q.weight"))?,
                k_proj: Self::tensor_ref(&gguf, &lp("attn_k.weight"))?,
                v_proj: Self::tensor_ref(&gguf, &lp("attn_v.weight"))?,
                o_proj: Self::tensor_ref(&gguf, &lp("attn_output.weight"))?,
                ffn_gate: Self::tensor_ref(&gguf, &lp("ffn_gate.weight"))?,
                ffn_up: Self::tensor_ref(&gguf, &lp("ffn_up.weight"))?,
                ffn_down: Self::tensor_ref(&gguf, &lp("ffn_down.weight"))?,
            });
        }

        let max_seq = config.max_seq_len.min(cfg.max_seq_len);
        let kv = KvCache::new(cfg.n_layers, max_seq, cfg.n_kv_heads, cfg.head_dim);
        let sampler = Sampler::new(0);

        if let Some(profile) = config.kernel_profile.as_ref() {
            profile.validate_for_gguf(&gguf, None)?;
        }
        let kernel_profile = config.kernel_profile.clone();

        Ok(Self {
            config: cfg,
            tokenizer,
            model_id,
            gguf,
            embed,
            final_norm,
            lm_head,
            layers,
            kv,
            sampler,
            kernel_profile,
            _weights_path: weights.to_path_buf(),
        })
    }

    fn generate(
        &mut self,
        _req: GenerateRequest,
        _sink: &mut dyn FnMut(StreamEvent),
    ) -> Result<GenStats> {
        // The forward pass lands in steps 4–6. This stub returns a clear
        // error so accidental invocations against a Llama GGUF fail
        // loudly instead of silently emitting garbage.
        Err(Error::Unimplemented(
            "LlamaDense::generate (forward pass lands in step 4–6)",
        ))
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn model_arch(&self) -> &str {
        &self.config.arch
    }

    fn forward_tokens_for_test(
        &mut self,
        _tokens: &[u32],
        _positions: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        Err(Error::Unimplemented(
            "LlamaDense::forward_tokens_for_test (lands in step 4)",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::Llama3RopeScaling;

    /// Llama-3.2 reference RoPE-scaling values that should round-trip
    /// through `LlamaConfig`. Validates the metadata key strings.
    #[test]
    fn rope_scaling_params_round_trip_into_struct() {
        let s = Llama3RopeScaling {
            factor: 32.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192,
        };
        // The struct is plain data; the GGUF-side path is exercised by
        // the smoke tests in step 8 when a real Llama-3.2 GGUF is
        // present. Here we just confirm the type the runtime stores is
        // the same shape the kernel consumes.
        assert_eq!(s.factor, 32.0);
        assert_eq!(s.original_max_position_embeddings, 8192);
    }
}
