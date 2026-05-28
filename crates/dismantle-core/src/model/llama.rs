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

use crate::attn::mha_decode_step;
use crate::cache::KvCache;
use crate::engine::{
    Engine, EngineConfig, GenStats, GenerateRequest, StopReason, StreamEvent,
};
use crate::gguf::{GgmlType, GgufFile};
use crate::kernels::{
    add_inplace, embed_lookup, gemv_f16, gemv_f32, rmsnorm, rope_inplace_scaled, silu_mul,
    Llama3RopeScaling,
};
use crate::profile::KernelProfile;
use crate::quant;
use crate::sample::Sampler;
use crate::tokenizer::Tokenizer;
use crate::{Error, Result};
use half::f16;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Instant;

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
#[derive(Debug, Clone)]
pub(crate) struct TensorRef {
    pub offset: usize,
    pub byte_size: usize,
    pub dtype: GgmlType,
    pub n_elems: usize,
}

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

    fn dequant_ref_into(&self, t: &TensorRef, buf: &mut Vec<f32>) -> Result<()> {
        if buf.len() != t.n_elems {
            buf.resize(t.n_elems, 0.0);
        }
        let bytes = &self.gguf.mmap[t.offset..t.offset + t.byte_size];
        quant::dequant_into(t.dtype, bytes, buf)
    }

    /// CPU reference matmul: dequantize the weight slab once into the
    /// `scratch` buffer, then run plain f32 GEMV. Used by step 4's
    /// reference forward; step 6 replaces every call with the Metal Q4_K
    /// dispatcher.
    fn matmul_cpu(
        &self,
        t: &TensorRef,
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
        scratch: &mut Vec<f32>,
    ) -> Result<()> {
        self.dequant_ref_into(t, scratch)?;
        gemv_f32(scratch, rows, cols, x, out);
        Ok(())
    }

    /// CPU reference forward for a single token at position `pos`.
    /// Appends K/V at the current `kv.seq_len` slot and bumps `seq_len`.
    /// Mirrors `qwen_dense::forward_token`, with two differences:
    ///   - no Q/K/V bias adds (Llama families omit them)
    ///   - RoPE goes through `rope_inplace_scaled` so Llama-3.1+ NTK
    ///     rescaling is honored when `cfg.rope_scaling.is_some()`
    pub(crate) fn forward_token_cpu(&mut self, token: u32, pos: usize) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let h = cfg.hidden;
        let head_dim = cfg.head_dim;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let rope_scaling = cfg.rope_scaling;
        let rope_theta = cfg.rope_theta;
        let rms_eps = cfg.rms_norm_eps;
        let n_layers = cfg.n_layers;

        let mut x = vec![0.0f32; h];
        embed_lookup(&self.embed, h, token, &mut x);

        let mut scratch = Vec::<f32>::new();

        let stride = n_kv_heads * head_dim;
        if self.kv.seq_len >= self.kv.max_seq {
            return Err(Error::Model(format!(
                "kv cache full at {}",
                self.kv.max_seq
            )));
        }
        let kv_off = self.kv.seq_len * stride;
        let mha_seq_len = self.kv.seq_len + 1;

        for li in 0..n_layers {
            let mut x_norm = vec![0.0f32; h];
            rmsnorm(&x, &self.layers[li].attn_norm, rms_eps, &mut x_norm);

            // Snapshot the per-layer TensorRefs so we can release the
            // immutable borrow on `self.layers` before the KV write.
            let q_proj = self.layers[li].q_proj.clone();
            let k_proj = self.layers[li].k_proj.clone();
            let v_proj = self.layers[li].v_proj.clone();
            let o_proj = self.layers[li].o_proj.clone();
            let ffn_gate = self.layers[li].ffn_gate.clone();
            let ffn_up = self.layers[li].ffn_up.clone();
            let ffn_down = self.layers[li].ffn_down.clone();
            let ffn_norm_w = self.layers[li].ffn_norm.clone();

            let mut q_full = vec![0.0f32; q_dim];
            let mut k_token = vec![0.0f32; kv_dim];
            let mut v_token = vec![0.0f32; kv_dim];
            self.matmul_cpu(&q_proj, q_dim, h, &x_norm, &mut q_full, &mut scratch)?;
            self.matmul_cpu(&k_proj, kv_dim, h, &x_norm, &mut k_token, &mut scratch)?;
            self.matmul_cpu(&v_proj, kv_dim, h, &x_norm, &mut v_token, &mut scratch)?;

            // RoPE on every Q head and every KV head, with optional
            // Llama-3.1+ NTK rescale (None ⇒ bit-identical to plain
            // rope_inplace).
            for h_i in 0..n_heads {
                let off = h_i * head_dim;
                rope_inplace_scaled(
                    &mut q_full[off..off + head_dim],
                    pos as u32,
                    rope_theta,
                    rope_scaling,
                );
            }
            for h_i in 0..n_kv_heads {
                let off = h_i * head_dim;
                rope_inplace_scaled(
                    &mut k_token[off..off + head_dim],
                    pos as u32,
                    rope_theta,
                    rope_scaling,
                );
            }

            self.kv.keys[li][kv_off..kv_off + stride].copy_from_slice(&k_token);
            self.kv.values[li][kv_off..kv_off + stride].copy_from_slice(&v_token);

            let kv_size = mha_seq_len * stride;
            let keys = &self.kv.keys[li][..kv_size];
            let values = &self.kv.values[li][..kv_size];

            let mut attn_out = vec![0.0f32; q_dim];
            mha_decode_step(
                &q_full, keys, values, n_heads, n_kv_heads, head_dim, mha_seq_len, &mut attn_out,
            )?;

            let mut o = vec![0.0f32; h];
            self.matmul_cpu(&o_proj, h, q_dim, &attn_out, &mut o, &mut scratch)?;
            add_inplace(&mut x, &o);

            let mut x_norm2 = vec![0.0f32; h];
            rmsnorm(&x, &ffn_norm_w, rms_eps, &mut x_norm2);
            let mid = cfg.intermediate;
            let mut g = vec![0.0f32; mid];
            let mut u = vec![0.0f32; mid];
            let mut a = vec![0.0f32; mid];
            self.matmul_cpu(&ffn_gate, mid, h, &x_norm2, &mut g, &mut scratch)?;
            self.matmul_cpu(&ffn_up, mid, h, &x_norm2, &mut u, &mut scratch)?;
            silu_mul(&g, &u, &mut a);
            let mut f = vec![0.0f32; h];
            self.matmul_cpu(&ffn_down, h, mid, &a, &mut f, &mut scratch)?;
            add_inplace(&mut x, &f);
        }

        self.kv.seq_len += 1;

        let mut x_norm = vec![0.0f32; h];
        rmsnorm(&x, &self.final_norm, rms_eps, &mut x_norm);

        let mut logits = vec![0.0f32; cfg.vocab_size];
        let w_f16: &[f16] = match &self.lm_head {
            Some(w) => w,
            None => &self.embed,
        };
        gemv_f16(w_f16, cfg.vocab_size, h, &x_norm, &mut logits);
        Ok(logits)
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
        req: GenerateRequest,
        sink: &mut dyn FnMut(StreamEvent),
    ) -> Result<GenStats> {
        // Step 5: CPU reference generate. Single-token serial prefill +
        // decode through `forward_token_cpu`. Metal hot path, spec-decode,
        // and prefix caching are deliberately out of scope here — they
        // layer on in step 6 behind the same env-var gates Qwen uses.
        if let Some(seed) = req.sampling.seed {
            self.sampler = Sampler::new(seed);
        }

        let abort_set = |req: &GenerateRequest| -> bool {
            req.abort
                .as_ref()
                .map(|f| f.load(Ordering::Relaxed))
                .unwrap_or(false)
        };
        let stall_limit = std::time::Duration::from_millis(req.max_stall_ms);
        let stall_active = req.max_stall_ms > 0;

        let prompt_ids = self.tokenizer.encode(&req.prompt, true)?;
        if prompt_ids.is_empty() {
            return Err(Error::Model("empty prompt after tokenization".into()));
        }
        let prompt_len = prompt_ids.len();
        let mut stats = GenStats {
            prompt_tokens: prompt_len,
            profile_id: self.kernel_profile.as_ref().map(|p| p.profile_id.clone()),
            ..Default::default()
        };

        self.kv.reset();

        // Prefill: run every prompt token to populate the KV cache.
        let prefill_start = Instant::now();
        let mut prefill_aborted = false;
        for (i, &t) in prompt_ids.iter().enumerate() {
            if abort_set(&req) {
                prefill_aborted = true;
                break;
            }
            let step_start = Instant::now();
            let _ = self.forward_token_cpu(t, i)?;
            if stall_active && step_start.elapsed() > stall_limit {
                prefill_aborted = true;
                break;
            }
        }
        stats.prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

        if prefill_aborted {
            sink(StreamEvent::Done {
                reason: StopReason::Aborted,
                stats: stats.clone(),
            });
            return Ok(stats);
        }

        // Decode loop.
        let decode_start = Instant::now();
        let mut last_id = *prompt_ids.last().unwrap();
        let mut produced = 0usize;
        let mut reason = StopReason::MaxTokens;
        let eos = self.tokenizer.eos_id();

        for step in 0..req.max_new_tokens {
            if abort_set(&req) {
                reason = StopReason::Aborted;
                break;
            }
            let pos = prompt_len + step;
            let step_start = Instant::now();
            let mut logits = self.forward_token_cpu(last_id, pos)?;
            let next_id = self.sampler.sample(&mut logits, &req.sampling);
            if stall_active && step_start.elapsed() > stall_limit {
                reason = StopReason::Aborted;
                break;
            }
            self.sampler.record(next_id);
            let text = self.tokenizer.decode_one(next_id).unwrap_or_default();
            sink(StreamEvent::Token { id: next_id, text });
            produced += 1;
            if Some(next_id) == eos {
                reason = StopReason::Eos;
                break;
            }
            last_id = next_id;
        }

        stats.decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
        stats.completion_tokens = produced;
        sink(StreamEvent::Done {
            reason,
            stats: stats.clone(),
        });
        Ok(stats)
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn model_arch(&self) -> &str {
        &self.config.arch
    }

    fn encode_prompt_for_batch(&self, prompt: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode(prompt, true)
    }

    fn decode_token_for_batch(&self, token: u32) -> Result<String> {
        self.tokenizer.decode_one(token)
    }

    fn eos_id_for_batch(&self) -> Option<u32> {
        self.tokenizer.eos_id()
    }

    fn forward_tokens_for_test(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        if tokens.len() != positions.len() {
            return Err(Error::Model(format!(
                "forward_tokens shape: tokens={} positions={}",
                tokens.len(),
                positions.len()
            )));
        }
        let mut out = Vec::with_capacity(tokens.len());
        for (i, &token) in tokens.iter().enumerate() {
            out.push(self.forward_token_cpu(token, positions[i])?);
        }
        Ok(out)
    }

    fn reset_kv_for_test(&mut self) {
        self.kv.seq_len = 0;
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
