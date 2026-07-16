//! OLMoE loader — 64-expert top-8 MoE with standard GQA + QK-norms.
//!
//! Target: OLMoE-1B-7B-*-Q4_K_M.gguf
//! GGUF arch string: "olmoe"
//! Metadata keys:    "llama.*" (same as LLaMA/Mixtral exports)
//!
//! Differences from Mixtral:
//!  - Expert weights are fused: blk.N.ffn_{gate,up,down}_exps.weight (3D tensor,
//!    outer dim = n_experts).  Sliced into per-expert TensorRefs without copying.
//!  - Per-head QK norms: blk.N.attn_q_norm.weight / blk.N.attn_k_norm.weight
//!    (shape [head_dim]) applied after QKV projection, before RoPE.
//!  - 64 experts, top-8.
//!
//! CPU reference path only for the initial implementation.

use super::arch_config::token_embd_vocab_size_opt;
use super::weights::{dequant_f16, dequant_f32, tensor_ref, TensorRef};
use crate::attn::mha_decode_step;
use crate::cache::KvCache;
use crate::engine::{Engine, EngineConfig, GenStats, GenerateRequest, StopReason, StreamEvent};
use crate::gguf::GgufFile;
use crate::kernels::{add_inplace, embed_lookup, gemv_f32, rmsnorm, rope_inplace, silu_mul};
use crate::moe::topk_gate;
use crate::quant;
use crate::sample::Sampler;
use crate::tokenizer::Tokenizer;
use crate::{Error, Result};
use half::f16;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Instant;

// ─── Config ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct OlmoeConfig {
    pub n_layers: usize,
    pub hidden: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize, // per-expert FFN intermediate dim
    pub n_experts: usize,    // 64
    pub top_k: usize,        // 8
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
}

impl OlmoeConfig {
    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let get_u32 = |k: &str| g.metadata.get(k).and_then(|v| v.as_u32());
        let get_f32 = |k: &str| g.metadata.get(k).and_then(|v| v.as_f32());

        let n_layers = get_u32("llama.block_count").ok_or_else(|| Error::Model("olmoe: missing llama.block_count".into()))? as usize;
        let hidden = get_u32("llama.embedding_length").ok_or_else(|| Error::Model("olmoe: missing llama.embedding_length".into()))? as usize;
        let n_heads = get_u32("llama.attention.head_count").ok_or_else(|| Error::Model("olmoe: missing llama.attention.head_count".into()))? as usize;
        let n_kv_heads = get_u32("llama.attention.head_count_kv").unwrap_or(n_heads as u32) as usize;
        let intermediate = get_u32("llama.feed_forward_length").ok_or_else(|| Error::Model("olmoe: missing llama.feed_forward_length".into()))? as usize;
        let n_experts = get_u32("llama.expert_count").ok_or_else(|| Error::Model("olmoe: missing llama.expert_count".into()))? as usize;
        let top_k = get_u32("llama.expert_used_count").unwrap_or(8) as usize;
        let vocab_size = get_u32("llama.vocab_size").map(|v| v as usize).or_else(|| token_embd_vocab_size_opt(g)).ok_or_else(|| Error::Model("olmoe: cannot determine vocab_size".into()))?;

        if n_heads == 0 {
            return Err(Error::Model("olmoe: n_heads == 0".into()));
        }
        if hidden % n_heads != 0 {
            return Err(Error::Model(format!("olmoe: hidden={hidden} not divisible by n_heads={n_heads}")));
        }

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
            rope_theta: get_f32("llama.rope.freq_base").unwrap_or(10_000.0),
            rms_norm_eps: get_f32("llama.attention.layer_norm_rms_epsilon").unwrap_or(1e-5),
            max_seq_len: get_u32("llama.context_length").unwrap_or(4096) as usize,
        })
    }
}

// ─── Layer weights ────────────────────────────────────────────────────────────

pub struct OlmoeLayer {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    // Attention projections — TensorRefs into mmap, dequanted on demand.
    pub attn_q: TensorRef,
    pub attn_k: TensorRef,
    pub attn_v: TensorRef,
    pub attn_output: TensorRef,
    // QK-norms: shape [head_dim].  Applied per-head before RoPE.
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    // Router: [n_experts × hidden], dequanted eagerly (small).
    pub router: Vec<f32>,
    // Per-expert FFN slices from fused tensors.
    pub ffn_gate: Vec<TensorRef>, // n_experts × [intermediate, hidden]
    pub ffn_up: Vec<TensorRef>,
    pub ffn_down: Vec<TensorRef>, // n_experts × [hidden, intermediate]
}

// ─── Engine ───────────────────────────────────────────────────────────────────

pub struct OlmoeEngine {
    pub config: OlmoeConfig,
    pub tokenizer: Tokenizer,
    pub model_id: String,
    pub gguf: GgufFile,
    pub embed: Vec<f16>,
    pub final_norm: Vec<f32>,
    pub lm_head: Option<Vec<f16>>,
    pub layers: Vec<OlmoeLayer>,
    pub kv: KvCache,
    pub sampler: Sampler,
    pub _weights_path: PathBuf,
}

impl OlmoeEngine {
    /// Slice a fused 3D expert tensor into per-expert TensorRefs without
    /// copying. Mirrors deepseek_v2's fused_expert_refs.
    fn fused_expert_refs(g: &GgufFile, name: &str, n_experts: usize) -> Result<Vec<TensorRef>> {
        let info = g.tensor(name).ok_or_else(|| Error::Model(format!("olmoe: missing tensor `{name}`")))?;
        let total_elems: usize = info.dims.iter().product::<u64>() as usize;
        let total_bytes = info.byte_size as usize;
        if total_elems % n_experts != 0 {
            return Err(Error::Model(format!("olmoe tensor {name}: {total_elems} elems not divisible by {n_experts}")));
        }
        if total_bytes % n_experts != 0 {
            return Err(Error::Model(format!("olmoe tensor {name}: {total_bytes} bytes not divisible by {n_experts}")));
        }
        let per_elems = total_elems / n_experts;
        let per_bytes = total_bytes / n_experts;
        let base = info.data_offset as usize;
        Ok((0..n_experts).map(|e| TensorRef { offset: base + e * per_bytes, byte_size: per_bytes, dtype: info.dtype, n_elems: per_elems }).collect())
    }

    /// Dequantize a TensorRef into buf (resizes in place).
    fn dequant_ref(&self, t: &TensorRef, buf: &mut Vec<f32>) -> Result<()> {
        if buf.len() != t.n_elems {
            buf.resize(t.n_elems, 0.0);
        }
        let bytes = &self.gguf.mmap[t.offset..t.offset + t.byte_size];
        quant::dequant_into(t.dtype, bytes, buf)
    }
}

impl Engine for OlmoeEngine {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn load(weights: &Path, config: EngineConfig) -> Result<Self> {
        let gguf = GgufFile::open(weights)?;
        let arch = gguf.metadata.get("general.architecture").and_then(|v| v.as_str()).unwrap_or("");
        if arch != "olmoe" {
            return Err(Error::Model(format!("olmoe loader: expected arch 'olmoe', got '{arch}'")));
        }

        let cfg = OlmoeConfig::from_gguf(&gguf)?;
        let sidecar = weights.parent().map(|d| d.join("tokenizer.json")).filter(|p| p.exists());
        let tokenizer = if let Some(path) = sidecar { Tokenizer::from_file(path)? } else { Tokenizer::from_gguf(&gguf)? };

        let embed = dequant_f16(&gguf, "token_embd.weight")?;
        let final_norm = dequant_f32(&gguf, "output_norm.weight")?;
        let lm_head = if gguf.tensor("output.weight").is_some() { Some(dequant_f16(&gguf, "output.weight")?) } else { None };

        let max_seq = config.max_seq_len.min(cfg.max_seq_len);
        let kv = KvCache::new(cfg.n_layers, max_seq, cfg.n_kv_heads, cfg.head_dim);
        let sampler = Sampler::new(42);
        let model_id = gguf.name().unwrap_or("olmoe").to_string();

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for li in 0..cfg.n_layers {
            let lp = |suf: &str| format!("blk.{li}.{suf}");

            let attn_norm = dequant_f32(&gguf, &lp("attn_norm.weight"))?;
            let ffn_norm = dequant_f32(&gguf, &lp("ffn_norm.weight"))?;
            let q_norm = dequant_f32(&gguf, &lp("attn_q_norm.weight"))?;
            let k_norm = dequant_f32(&gguf, &lp("attn_k_norm.weight"))?;

            let attn_q = tensor_ref(&gguf, &lp("attn_q.weight"))?;
            let attn_k = tensor_ref(&gguf, &lp("attn_k.weight"))?;
            let attn_v = tensor_ref(&gguf, &lp("attn_v.weight"))?;
            let attn_output = tensor_ref(&gguf, &lp("attn_output.weight"))?;

            // Router: dequant eagerly (n_experts × hidden is small for OLMoE).
            let router = dequant_f32(&gguf, &lp("ffn_gate_inp.weight"))?;

            let ffn_gate = Self::fused_expert_refs(&gguf, &lp("ffn_gate_exps.weight"), cfg.n_experts)?;
            let ffn_up = Self::fused_expert_refs(&gguf, &lp("ffn_up_exps.weight"), cfg.n_experts)?;
            let ffn_down = Self::fused_expert_refs(&gguf, &lp("ffn_down_exps.weight"), cfg.n_experts)?;

            layers.push(OlmoeLayer { attn_norm, ffn_norm, attn_q, attn_k, attn_v, attn_output, q_norm, k_norm, router, ffn_gate, ffn_up, ffn_down });
        }

        eprintln!("[olmoe] loaded {} layers, {} experts top-{} (CPU path)", cfg.n_layers, cfg.n_experts, cfg.top_k);

        Ok(Self { config: cfg, tokenizer, model_id, gguf, embed, final_norm, lm_head, layers, kv, sampler, _weights_path: weights.to_owned() })
    }

    fn generate(&mut self, req: GenerateRequest, sink: &mut dyn FnMut(StreamEvent)) -> Result<GenStats> {
        if let Some(seed) = req.sampling.seed {
            self.sampler = Sampler::new(seed);
        }

        let prompt_ids = self.tokenizer.encode(&req.prompt, true)?;
        let n_prompt = prompt_ids.len();
        self.kv.reset();

        let t0 = Instant::now();
        // Prefill.
        for &tok in &prompt_ids {
            let _ = self.forward_one(tok)?;
        }
        let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let max_new = req.max_new_tokens.min(self.config.max_seq_len.saturating_sub(n_prompt));
        let t1 = Instant::now();
        let mut n_gen = 0usize;
        let mut stop_reason = StopReason::MaxTokens;
        let mut last_tok = *prompt_ids.last().unwrap_or(&0);

        let vocab_index = if req.json_mode {
            let tok = &self.tokenizer;
            let vs = self.config.vocab_size;
            Some(crate::json_constrain::JsonVocabIndex::build(vs, |id| tok.decode_one(id).unwrap_or_default()))
        } else {
            None
        };
        let mut constraint = if req.json_mode { Some(crate::json_constrain::JsonConstraint::new()) } else { None };

        loop {
            if let Some(sig) = &req.abort {
                if sig.load(Ordering::Relaxed) {
                    stop_reason = StopReason::Aborted;
                    break;
                }
            }
            if n_gen >= max_new {
                break;
            }

            let mut logits = self.forward_one(last_tok)?;
            if let (Some(vi), Some(c)) = (&vocab_index, &constraint) {
                c.mask_logits(vi, &mut logits);
            }
            let tok = self.sampler.sample(&mut logits, &req.sampling);
            n_gen += 1;

            if self.tokenizer.is_eog(tok) {
                stop_reason = StopReason::Eos;
                break;
            }

            let text = self.tokenizer.decode_one(tok)?;
            let json_done = if let Some(c) = &mut constraint {
                c.advance(&text);
                c.is_done()
            } else {
                false
            };
            sink(StreamEvent::Token { id: tok, text });
            if json_done {
                stop_reason = StopReason::Eos;
                break;
            }
            last_tok = tok;
        }

        let decode_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let stats = GenStats { prompt_tokens: n_prompt, completion_tokens: n_gen, prefill_ms, decode_ms, ..Default::default() };
        sink(StreamEvent::Done { reason: stop_reason, stats: stats.clone() });
        Ok(stats)
    }

    fn forward_tokens_for_test(&mut self, tokens: &[u32], _positions: &[usize]) -> Result<Vec<Vec<f32>>> {
        self.kv.reset();
        let mut out = Vec::with_capacity(tokens.len());
        for &tok in tokens {
            out.push(self.forward_one(tok)?);
        }
        Ok(out)
    }

    fn reset_kv_for_test(&mut self) {
        self.kv.reset();
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
}

// ─── Core forward step ────────────────────────────────────────────────────────

impl OlmoeEngine {
    fn forward_one(&mut self, token_id: u32) -> Result<Vec<f32>> {
        let hidden = self.config.hidden;
        let n_heads = self.config.n_heads;
        let n_kv_heads = self.config.n_kv_heads;
        let head_dim = self.config.head_dim;
        let kv_hidden = n_kv_heads * head_dim;
        let eps = self.config.rms_norm_eps;
        let rope_theta = self.config.rope_theta;

        if self.kv.seq_len >= self.kv.max_seq {
            return Err(Error::Model(format!("olmoe: kv cache full at {}", self.kv.max_seq)));
        }

        let kv_off = self.kv.seq_len * kv_hidden;
        let mha_seq_len = self.kv.seq_len + 1;
        let pos = self.kv.seq_len as u32;

        let mut x = vec![0.0f32; hidden];
        embed_lookup(&self.embed, hidden, token_id, &mut x);

        let mut x_norm = vec![0.0f32; hidden];
        let mut q_buf = vec![0.0f32; hidden];
        let mut k_buf = vec![0.0f32; kv_hidden];
        let mut v_buf = vec![0.0f32; kv_hidden];
        let mut attn_out = vec![0.0f32; hidden];
        let mut proj_out = vec![0.0f32; hidden];
        let mut gate_logits = vec![0.0f32; self.config.n_experts];
        let mut expert_gate = vec![0.0f32; self.config.intermediate];
        let mut expert_up = vec![0.0f32; self.config.intermediate];
        let mut expert_act = vec![0.0f32; self.config.intermediate];
        let mut expert_out = vec![0.0f32; hidden];
        let mut ffn_accum = vec![0.0f32; hidden];
        let mut w_buf = Vec::<f32>::new();

        for li in 0..self.config.n_layers {
            // ── Attention ───────────────────────────────────────────────────
            rmsnorm(&x, &self.layers[li].attn_norm, eps, &mut x_norm);

            let attn_q_ref = self.layers[li].attn_q.clone();
            self.dequant_ref(&attn_q_ref, &mut w_buf)?;
            gemv_f32(&w_buf, hidden, hidden, &x_norm, &mut q_buf);

            let attn_k_ref = self.layers[li].attn_k.clone();
            self.dequant_ref(&attn_k_ref, &mut w_buf)?;
            gemv_f32(&w_buf, kv_hidden, hidden, &x_norm, &mut k_buf);

            let attn_v_ref = self.layers[li].attn_v.clone();
            self.dequant_ref(&attn_v_ref, &mut w_buf)?;
            gemv_f32(&w_buf, kv_hidden, hidden, &x_norm, &mut v_buf);

            // Per-head QK norms then RoPE.
            {
                let q_norm_w = self.layers[li].q_norm.clone();
                let k_norm_w = self.layers[li].k_norm.clone();
                apply_qk_norm_rope(&mut q_buf, &q_norm_w, n_heads, head_dim, eps, pos, rope_theta);
                apply_qk_norm_rope(&mut k_buf, &k_norm_w, n_kv_heads, head_dim, eps, pos, rope_theta);
            }

            // Write K and V directly into cache for this layer.
            self.kv.keys[li][kv_off..kv_off + kv_hidden].copy_from_slice(&k_buf);
            self.kv.values[li][kv_off..kv_off + kv_hidden].copy_from_slice(&v_buf);

            let kv_size = mha_seq_len * kv_hidden;
            let keys = &self.kv.keys[li][..kv_size];
            let values = &self.kv.values[li][..kv_size];
            mha_decode_step(&q_buf, keys, values, n_heads, n_kv_heads, head_dim, mha_seq_len, &mut attn_out)?;

            let attn_out_ref = self.layers[li].attn_output.clone();
            self.dequant_ref(&attn_out_ref, &mut w_buf)?;
            gemv_f32(&w_buf, hidden, hidden, &attn_out, &mut proj_out);
            add_inplace(&mut x, &proj_out);

            // ── FFN / MoE ───────────────────────────────────────────────────
            rmsnorm(&x, &self.layers[li].ffn_norm, eps, &mut x_norm);

            let router = self.layers[li].router.clone();
            gemv_f32(&router, self.config.n_experts, hidden, &x_norm, &mut gate_logits);
            let top_experts = topk_gate(&mut gate_logits, self.config.top_k, true);

            ffn_accum.iter_mut().for_each(|v| *v = 0.0);
            for (eid, weight) in &top_experts {
                let gate_ref = self.layers[li].ffn_gate[*eid].clone();
                self.dequant_ref(&gate_ref, &mut w_buf)?;
                gemv_f32(&w_buf, self.config.intermediate, hidden, &x_norm, &mut expert_gate);

                let up_ref = self.layers[li].ffn_up[*eid].clone();
                self.dequant_ref(&up_ref, &mut w_buf)?;
                gemv_f32(&w_buf, self.config.intermediate, hidden, &x_norm, &mut expert_up);

                silu_mul(&expert_gate, &expert_up, &mut expert_act);

                let down_ref = self.layers[li].ffn_down[*eid].clone();
                self.dequant_ref(&down_ref, &mut w_buf)?;
                gemv_f32(&w_buf, hidden, self.config.intermediate, &expert_act, &mut expert_out);

                for (acc, v) in ffn_accum.iter_mut().zip(expert_out.iter()) {
                    *acc += weight * v;
                }
            }

            add_inplace(&mut x, &ffn_accum);
        }

        self.kv.seq_len += 1;

        // Final norm + LM head.
        rmsnorm(&x, &self.final_norm, eps, &mut x_norm);
        let lm_f32: Vec<f32> = self.lm_head.as_deref().unwrap_or(&self.embed).iter().map(|v| v.to_f32()).collect();
        let vocab_size = self.config.vocab_size;
        let mut logits = vec![0.0f32; vocab_size];
        gemv_f32(&lm_f32, vocab_size, hidden, &x_norm, &mut logits);
        Ok(logits)
    }
}

// ─── Per-head QK-norm + RoPE ─────────────────────────────────────────────────

/// Apply RMSNorm per-head then RoPE in-place.
/// x: [n_heads × head_dim], norm_w: [head_dim].
fn apply_qk_norm_rope(x: &mut [f32], norm_w: &[f32], n_heads: usize, head_dim: usize, eps: f32, pos: u32, rope_theta: f32) {
    let mut tmp = vec![0.0f32; head_dim];
    for h in 0..n_heads {
        let head = &mut x[h * head_dim..(h + 1) * head_dim];
        rmsnorm(head, norm_w, eps, &mut tmp);
        head.copy_from_slice(&tmp);
        rope_inplace(head, pos, rope_theta);
    }
}
