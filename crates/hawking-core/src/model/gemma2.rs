//! Gemma-2 dense forward pass (Gemma-2 2B / 9B / 27B).
//!
//! Gemma-2 diverges from the Llama/Qwen dense template in several ways,
//! all handled here:
//!
//!   - **Sandwich norms.** Each block has four RMSNorms: pre-attn
//!     (`attn_norm`), post-attn (`post_attention_norm`, applied to the
//!     attention output *before* the residual add), pre-ffn (`ffn_norm`),
//!     and post-ffn (`post_ffw_norm`, before its residual add).
//!   - **(1 + weight) RMSNorm.** Gemma stores norm weights centered at 0;
//!     we add 1.0 at load so the shared rmsnorm kernel is reused.
//!   - **Embedding scale.** Input embeddings are scaled by √hidden.
//!   - **GeGLU FFN** (tanh-approx GELU on the gate), not SwiGLU.
//!   - **Logit soft-capping** on both attention scores (≈50) and final
//!     logits (≈30).
//!   - **Explicit head_dim** (256 for 2B), independent of hidden/n_heads,
//!     and a `query_pre_attn_scalar` that sets the attention score scale.
//!   - **Tied LM head** (no `output.weight`).
//!
//! Sliding-window attention (even layers, window 4096) is *not* applied —
//! this engine runs full causal attention, which is exact for prompts up
//! to the window and drifts only beyond it. A guard in `from_gguf` warns.
//!
//! On macOS the Q4_K projections, f16 LM head, and rmsnorm run on Metal;
//! attention and non-Q4_K weights use the CPU reference path.

use super::arch_config::ArchReader;
use super::weights::{dequant_f16, dequant_f32, tensor_ref, TensorRef};
use crate::attn::mha_decode_step_gemma;
use crate::cache::KvCache;
use crate::engine::{Engine, EngineConfig, GenStats, GenerateRequest, StopReason, StreamEvent};
use crate::gguf::{GgmlType, GgufFile};
use crate::kernels::{
    add_inplace, embed_lookup, gelu_mul, gemv_f16, gemv_f32, logit_softcap_inplace, rmsnorm,
    rope_inplace,
};
use crate::metal::MetalContext;
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
pub struct Gemma2Config {
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
    /// Attention score scale = 1/sqrt(query_pre_attn_scalar). For 2B this
    /// equals 1/sqrt(head_dim); for 9B/27B it does not.
    pub attn_scale: f32,
    /// Attention-logit soft-cap (0 ⇒ disabled).
    pub attn_logit_softcap: f32,
    /// Final-logit soft-cap (0 ⇒ disabled).
    pub final_logit_softcap: f32,
    /// √hidden, applied to the input embeddings.
    pub embed_scale: f32,
}

impl Gemma2Config {
    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let get_u32 = |k: &str| g.metadata.get(k).and_then(|v| v.as_u32());
        let get_f32 = |k: &str| g.metadata.get(k).and_then(|v| v.as_f32());
        // P1-D1: shared core reads via ArchReader; the vocab fallback and the
        // soft-cap / query-scalar / sliding-window extras keep the closures.
        let r = ArchReader::new(g, "gemma2");

        let n_layers = r.req_usize("block_count")?;
        let hidden = r.req_usize("embedding_length")?;
        let n_heads = r.req_usize("attention.head_count")?;
        let n_kv_heads = r.opt_usize("attention.head_count_kv", n_heads);
        // Gemma carries an explicit head_dim (key_length) that is NOT
        // hidden/n_heads — 2B is hidden=2304, n_heads=8, head_dim=256.
        let head_dim = r.opt_usize("attention.key_length", hidden / n_heads);
        let intermediate = r.req_usize("feed_forward_length")?;
        let vocab_size = match get_u32("gemma2.vocab_size") {
            Some(v) => v as usize,
            None => {
                let dims = g
                    .tensor("token_embd.weight")
                    .map(|t| t.dims.clone())
                    .ok_or_else(|| {
                        Error::Model("vocab size not in metadata or token_embd dims".into())
                    })?;
                dims.iter().copied().max().unwrap_or(0) as usize
            }
        };
        let rope_theta = r.opt_f32("rope.freq_base", 10_000.0);
        let rms_norm_eps = r.opt_f32("attention.layer_norm_rms_epsilon", 1e-6);
        let max_seq_len = r.opt_usize("context_length", 8192);

        // query_pre_attn_scalar defaults to head_dim when absent.
        let qpas = get_f32("gemma2.attention.query_pre_attn_scalar").unwrap_or(head_dim as f32);
        let attn_scale = 1.0 / qpas.sqrt();
        let attn_logit_softcap = get_f32("gemma2.attn_logit_softcapping").unwrap_or(0.0);
        let final_logit_softcap = get_f32("gemma2.final_logit_softcapping").unwrap_or(0.0);
        let embed_scale = (hidden as f32).sqrt();

        if let Some(win) = get_u32("gemma2.attention.sliding_window") {
            if (win as usize) < max_seq_len {
                eprintln!(
                    "hawking: note — Gemma-2 uses sliding-window attention on alternating \
                     layers (window={win}); this engine runs full causal attention, exact up \
                     to {win} tokens of context and drifting beyond it"
                );
            }
        }

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
            attn_scale,
            attn_logit_softcap,
            final_logit_softcap,
            embed_scale,
        })
    }
}

pub struct Gemma2Layer {
    /// Sandwich norms (all already +1.0 at load for the (1+w) convention).
    pub attn_norm: Vec<f32>,
    pub post_attention_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub post_ffw_norm: Vec<f32>,
    pub(crate) q_proj: TensorRef,
    pub(crate) k_proj: TensorRef,
    pub(crate) v_proj: TensorRef,
    pub(crate) o_proj: TensorRef,
    pub(crate) ffn_gate: TensorRef,
    pub(crate) ffn_up: TensorRef,
    pub(crate) ffn_down: TensorRef,
}

pub struct Gemma2 {
    pub config: Gemma2Config,
    pub tokenizer: Tokenizer,
    pub model_id: String,
    pub gguf: GgufFile,
    pub embed: Vec<f16>,
    /// final_norm already carries the +1.0 (1+w) offset.
    pub final_norm: Vec<f32>,
    pub layers: Vec<Gemma2Layer>,
    pub kv: KvCache,
    pub sampler: Sampler,
    pub kernel_profile: Option<KernelProfile>,
    pub _weights_path: PathBuf,
    pub metal_ctx: Option<MetalContext>,
}

impl Gemma2 {
    /// Gemma RMSNorm weights are stored centered at 0; the effective
    /// scale is `(1 + w)`. Fold the +1.0 in at load so the standard
    /// rmsnorm kernel applies the right gain.
    fn dequant_norm_plus_one(g: &GgufFile, name: &str) -> Result<Vec<f32>> {
        let mut w = dequant_f32(g, name)?;
        for v in w.iter_mut() {
            *v += 1.0;
        }
        Ok(w)
    }

    fn dequant_ref_into(&self, t: &TensorRef, buf: &mut Vec<f32>) -> Result<()> {
        if buf.len() != t.n_elems {
            buf.resize(t.n_elems, 0.0);
        }
        let bytes = &self.gguf.mmap[t.offset..t.offset + t.byte_size];
        quant::dequant_into(t.dtype, bytes, buf)
    }

    fn rmsnorm_dispatch(&self, x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) -> Result<()> {
        #[cfg(target_os = "macos")]
        if let Some(ctx) = &self.metal_ctx {
            return crate::kernels::rmsnorm_metal(ctx, x, weight, eps, out);
        }
        rmsnorm(x, weight, eps, out);
        Ok(())
    }

    fn matmul_q4_dispatch(
        &self,
        t: &TensorRef,
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
        scratch: &mut Vec<f32>,
    ) -> Result<()> {
        #[cfg(target_os = "macos")]
        if let Some(ctx) = &self.metal_ctx {
            if t.dtype == GgmlType::Q4_K {
                let bytes = &self.gguf.mmap[t.offset..t.offset + t.byte_size];
                return crate::kernels::gemv_q4_k_m(ctx, bytes, rows, cols, x, out);
            }
        }
        self.dequant_ref_into(t, scratch)?;
        gemv_f32(scratch, rows, cols, x, out);
        Ok(())
    }

    fn gemv_f16_dispatch(
        &self,
        w_f16: &[f16],
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        #[cfg(target_os = "macos")]
        if let Some(ctx) = &self.metal_ctx {
            let w_bytes = bytemuck::cast_slice::<f16, u8>(w_f16);
            return crate::kernels::gemv_f16_metal(ctx, w_bytes, rows, cols, x, out);
        }
        gemv_f16(w_f16, rows, cols, x, out);
        Ok(())
    }

    pub(crate) fn forward_token(&mut self, token: u32, pos: usize) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let h = cfg.hidden;
        let head_dim = cfg.head_dim;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let rope_theta = cfg.rope_theta;
        let rms_eps = cfg.rms_norm_eps;
        let n_layers = cfg.n_layers;
        let mid = cfg.intermediate;
        let vocab_size = cfg.vocab_size;
        let attn_scale = cfg.attn_scale;
        let attn_softcap = cfg.attn_logit_softcap;
        let final_softcap = cfg.final_logit_softcap;
        let embed_scale = cfg.embed_scale;

        let mut x = vec![0.0f32; h];
        embed_lookup(&self.embed, h, token, &mut x);
        // Gemma scales the input embeddings by √hidden.
        for v in x.iter_mut() {
            *v *= embed_scale;
        }

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
            // Per-layer weights read in place (shared borrows of self); no
            // per-token clones. The four sandwich norms were the worst
            // offenders — 4 hidden-sized allocs/layer/token before this.
            // Pre-attn norm.
            let mut x_norm = vec![0.0f32; h];
            self.rmsnorm_dispatch(&x, &self.layers[li].attn_norm, rms_eps, &mut x_norm)?;

            let mut q_full = vec![0.0f32; q_dim];
            let mut k_token = vec![0.0f32; kv_dim];
            let mut v_token = vec![0.0f32; kv_dim];
            self.matmul_q4_dispatch(
                &self.layers[li].q_proj,
                q_dim,
                h,
                &x_norm,
                &mut q_full,
                &mut scratch,
            )?;
            self.matmul_q4_dispatch(
                &self.layers[li].k_proj,
                kv_dim,
                h,
                &x_norm,
                &mut k_token,
                &mut scratch,
            )?;
            self.matmul_q4_dispatch(
                &self.layers[li].v_proj,
                kv_dim,
                h,
                &x_norm,
                &mut v_token,
                &mut scratch,
            )?;

            for h_i in 0..n_heads {
                let off = h_i * head_dim;
                rope_inplace(&mut q_full[off..off + head_dim], pos as u32, rope_theta);
            }
            for h_i in 0..n_kv_heads {
                let off = h_i * head_dim;
                rope_inplace(&mut k_token[off..off + head_dim], pos as u32, rope_theta);
            }

            self.kv.keys[li][kv_off..kv_off + stride].copy_from_slice(&k_token);
            self.kv.values[li][kv_off..kv_off + stride].copy_from_slice(&v_token);

            let kv_size = mha_seq_len * stride;
            let keys = &self.kv.keys[li][..kv_size];
            let values = &self.kv.values[li][..kv_size];

            let mut attn_out = vec![0.0f32; q_dim];
            mha_decode_step_gemma(
                &q_full,
                keys,
                values,
                n_heads,
                n_kv_heads,
                head_dim,
                mha_seq_len,
                attn_scale,
                attn_softcap,
                &mut attn_out,
            )?;

            let mut o = vec![0.0f32; h];
            self.matmul_q4_dispatch(
                &self.layers[li].o_proj,
                h,
                q_dim,
                &attn_out,
                &mut o,
                &mut scratch,
            )?;
            // Post-attention norm BEFORE the residual add (sandwich).
            let mut o_norm = vec![0.0f32; h];
            self.rmsnorm_dispatch(
                &o,
                &self.layers[li].post_attention_norm,
                rms_eps,
                &mut o_norm,
            )?;
            add_inplace(&mut x, &o_norm);

            // Pre-ffn norm.
            let mut x_norm2 = vec![0.0f32; h];
            self.rmsnorm_dispatch(&x, &self.layers[li].ffn_norm, rms_eps, &mut x_norm2)?;
            let mut g = vec![0.0f32; mid];
            let mut u = vec![0.0f32; mid];
            let mut a = vec![0.0f32; mid];
            self.matmul_q4_dispatch(
                &self.layers[li].ffn_gate,
                mid,
                h,
                &x_norm2,
                &mut g,
                &mut scratch,
            )?;
            self.matmul_q4_dispatch(
                &self.layers[li].ffn_up,
                mid,
                h,
                &x_norm2,
                &mut u,
                &mut scratch,
            )?;
            // GeGLU (not SwiGLU).
            gelu_mul(&g, &u, &mut a);
            let mut f = vec![0.0f32; h];
            self.matmul_q4_dispatch(&self.layers[li].ffn_down, h, mid, &a, &mut f, &mut scratch)?;
            // Post-ffn norm BEFORE the residual add (sandwich).
            let mut f_norm = vec![0.0f32; h];
            self.rmsnorm_dispatch(&f, &self.layers[li].post_ffw_norm, rms_eps, &mut f_norm)?;
            add_inplace(&mut x, &f_norm);
        }

        self.kv.seq_len += 1;

        let mut x_norm = vec![0.0f32; h];
        self.rmsnorm_dispatch(&x, &self.final_norm, rms_eps, &mut x_norm)?;

        // Tied LM head: project against the (unscaled) embedding table.
        // Both the receiver and `&self.embed` are shared borrows, so no
        // clone is needed.
        let mut logits = vec![0.0f32; vocab_size];
        self.gemv_f16_dispatch(&self.embed, vocab_size, h, &x_norm, &mut logits)?;

        // Final logit soft-cap.
        logit_softcap_inplace(&mut logits, final_softcap);
        Ok(logits)
    }
}

impl Engine for Gemma2 {
    fn load(weights: &Path, config: EngineConfig) -> Result<Self> {
        let gguf = GgufFile::open(weights)?;
        let cfg = Gemma2Config::from_gguf(&gguf)?;
        let model_id = gguf.name().unwrap_or("gemma2").to_string();

        let sidecar = weights
            .parent()
            .map(|d| d.join("tokenizer.json"))
            .filter(|p| p.exists());
        let tokenizer = if let Some(p) = sidecar {
            Tokenizer::from_file(&p)?
        } else {
            Tokenizer::from_gguf(&gguf)?
        };

        let embed = dequant_f16(&gguf, "token_embd.weight")?;
        let final_norm = Self::dequant_norm_plus_one(&gguf, "output_norm.weight")?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for li in 0..cfg.n_layers {
            let lp = |suf: &str| format!("blk.{li}.{suf}");
            layers.push(Gemma2Layer {
                attn_norm: Self::dequant_norm_plus_one(&gguf, &lp("attn_norm.weight"))?,
                post_attention_norm: Self::dequant_norm_plus_one(
                    &gguf,
                    &lp("post_attention_norm.weight"),
                )?,
                ffn_norm: Self::dequant_norm_plus_one(&gguf, &lp("ffn_norm.weight"))?,
                post_ffw_norm: Self::dequant_norm_plus_one(&gguf, &lp("post_ffw_norm.weight"))?,
                q_proj: tensor_ref(&gguf, &lp("attn_q.weight"))?,
                k_proj: tensor_ref(&gguf, &lp("attn_k.weight"))?,
                v_proj: tensor_ref(&gguf, &lp("attn_v.weight"))?,
                o_proj: tensor_ref(&gguf, &lp("attn_output.weight"))?,
                ffn_gate: tensor_ref(&gguf, &lp("ffn_gate.weight"))?,
                ffn_up: tensor_ref(&gguf, &lp("ffn_up.weight"))?,
                ffn_down: tensor_ref(&gguf, &lp("ffn_down.weight"))?,
            });
        }

        let max_seq = config.max_seq_len.min(cfg.max_seq_len);
        let kv = KvCache::new(cfg.n_layers, max_seq, cfg.n_kv_heads, cfg.head_dim);
        let sampler = Sampler::new(0);

        let metal_ctx = MetalContext::new_with_trace(config.trace_dispatch).ok();
        let device_name = metal_ctx.as_ref().map(|ctx| ctx.device_name());
        if let Some(profile) = config.kernel_profile.as_ref() {
            profile.validate_for_gguf(&gguf, device_name.as_deref())?;
        }
        let kernel_profile = config.kernel_profile.clone();

        Ok(Self {
            config: cfg,
            tokenizer,
            model_id,
            gguf,
            embed,
            final_norm,
            layers,
            kv,
            sampler,
            kernel_profile,
            _weights_path: weights.to_path_buf(),
            metal_ctx,
        })
    }

    fn generate(
        &mut self,
        req: GenerateRequest,
        sink: &mut dyn FnMut(StreamEvent),
    ) -> Result<GenStats> {
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
        let prefill_start = Instant::now();
        let mut prefill_aborted = false;
        for (i, &t) in prompt_ids.iter().enumerate() {
            if abort_set(&req) {
                prefill_aborted = true;
                break;
            }
            let step_start = Instant::now();
            let _ = self.forward_token(t, i)?;
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
            let mut logits = self.forward_token(last_id, pos)?;
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
        "gemma2"
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
            out.push(self.forward_token(token, positions[i])?);
        }
        Ok(out)
    }

    fn reset_kv_for_test(&mut self) {
        self.kv.seq_len = 0;
    }
}
