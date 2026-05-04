//! Qwen2 / Qwen2.5 dense forward pass (Stage 2 of the dual-path
//! Phase-2 plan).
//!
//! Targets `general.architecture == "qwen2"` GGUFs (Qwen2.5-3B-Instruct
//! is the reference). Architecture: standard MHA with GQA (n_heads=16,
//! n_kv_heads=2 for the 3B variant), full-head RoPE (no nope/rope
//! split), standard SwiGLU FFN (no MoE), tied LM head (no separate
//! output.weight in the 3B Q4_K_M).
//!
//! Reuses the same Metal kernels as DeepSeek-V2-Lite:
//! - `rmsnorm_metal` for the per-layer norms + final norm
//! - `gemv_q4_k_m` for the Q4_K_M matmuls (q/k/v/o projections and
//!   gate/up/down FFN)
//! - `gemv_f16_metal` for the (potentially tied) LM head
//!
//! Weights are stored as `TensorRef` into the mmap'd GGUF and dispatched
//! to Metal via the `gemv_q4_k_m` fused-dequant kernel — no eager fp32
//! materialization of the bulk weights. Per-layer norms + bias vectors
//! are dequanted eagerly (small).

use crate::attn::mha_decode_step;
use crate::cache::KvCache;
use crate::engine::{Engine, EngineConfig, GenStats, GenerateRequest, StopReason, StreamEvent};
use crate::gguf::{GgmlType, GgufFile};
use crate::kernels::{
    add_inplace, embed_lookup, gemv_f16, gemv_f32, rmsnorm, rope_inplace, silu_mul,
};
use crate::metal::MetalContext;
use crate::quant;
use crate::sample::Sampler;
use crate::tokenizer::Tokenizer;
use crate::{Error, Result};
use half::f16;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct QwenConfig {
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
}

impl QwenConfig {
    fn from_gguf(g: &GgufFile) -> Result<Self> {
        let get_u32 = |k: &str| g.metadata.get(k).and_then(|v| v.as_u32());
        let get_f32 = |k: &str| g.metadata.get(k).and_then(|v| v.as_f32());

        let n_layers = get_u32("qwen2.block_count")
            .ok_or_else(|| Error::Model("missing qwen2.block_count".into()))?
            as usize;
        let hidden = get_u32("qwen2.embedding_length")
            .ok_or_else(|| Error::Model("missing qwen2.embedding_length".into()))?
            as usize;
        let n_heads = get_u32("qwen2.attention.head_count")
            .ok_or_else(|| Error::Model("missing qwen2.attention.head_count".into()))?
            as usize;
        let n_kv_heads =
            get_u32("qwen2.attention.head_count_kv").unwrap_or(n_heads as u32) as usize;
        // Qwen2 GGUFs don't carry head_dim explicitly; derive from
        // hidden/n_heads (matches all published Qwen2/2.5 variants).
        let head_dim = hidden / n_heads;
        let intermediate = get_u32("qwen2.feed_forward_length")
            .ok_or_else(|| Error::Model("missing qwen2.feed_forward_length".into()))?
            as usize;
        // Qwen2 GGUFs frequently omit `qwen2.vocab_size`; derive it from
        // the embedding-table tensor dims as a fallback.
        let vocab_size = match get_u32("qwen2.vocab_size").or_else(|| get_u32("llama.vocab_size")) {
            Some(v) => v as usize,
            None => {
                // GGUF dim ordering varies; vocab >> hidden in
                // practice, so the max of the embed tensor's dims is
                // the vocab size.
                let dims = g
                    .tensor("token_embd.weight")
                    .map(|t| t.dims.clone())
                    .ok_or_else(|| {
                        Error::Model("vocab size not in metadata or token_embd dims".into())
                    })?;
                dims.iter().copied().max().unwrap_or(0) as usize
            }
        };

        Ok(Self {
            n_layers,
            hidden,
            n_heads,
            n_kv_heads,
            head_dim,
            intermediate,
            vocab_size,
            rope_theta: get_f32("qwen2.rope.freq_base").unwrap_or(1_000_000.0),
            rms_norm_eps: get_f32("qwen2.attention.layer_norm_rms_epsilon").unwrap_or(1e-6),
            max_seq_len: get_u32("qwen2.context_length").unwrap_or(32768) as usize,
        })
    }
}

/// Pointer into the mmap'd GGUF for one tensor — same shape as the
/// DeepSeek path's `TensorRef` but kept module-local so qwen_dense
/// doesn't import internals from `model::deepseek_v2`.
#[derive(Debug, Clone)]
struct TensorRef {
    offset: usize,
    byte_size: usize,
    dtype: GgmlType,
    n_elems: usize,
}

pub struct QwenLayer {
    // Per-layer norms (eager fp32, small).
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    // Attention projection weights (lazy — dispatch via gemv_q4_k_m).
    q_proj: TensorRef,
    k_proj: TensorRef,
    v_proj: TensorRef,
    o_proj: TensorRef,
    // Qwen2 carries biases on Q, K, V (not O). Eager fp32 (small).
    q_bias: Vec<f32>,
    k_bias: Vec<f32>,
    v_bias: Vec<f32>,
    // FFN weights (lazy).
    ffn_gate: TensorRef,
    ffn_up: TensorRef,
    ffn_down: TensorRef,
}

pub struct QwenDense {
    pub config: QwenConfig,
    pub tokenizer: Tokenizer,
    pub model_id: String,

    /// mmap keepalive (every TensorRef points into this).
    pub gguf: GgufFile,

    pub embed: Vec<f16>,
    pub final_norm: Vec<f32>,
    /// `None` ⇒ tied to embed (Qwen2.5-3B-Q4_K_M is tied).
    pub lm_head: Option<Vec<f16>>,
    pub layers: Vec<QwenLayer>,

    pub kv: KvCache,
    pub sampler: Sampler,
    pub _weights_path: PathBuf,
    pub metal_ctx: Option<MetalContext>,
}

impl QwenDense {
    fn dequant_f32(g: &GgufFile, name: &str) -> Result<Vec<f32>> {
        let info = g
            .tensor(name)
            .ok_or_else(|| Error::Model(format!("missing tensor `{name}`")))?;
        let bytes = g.tensor_bytes(name).unwrap();
        quant::dequant_to_f32(info, bytes)
    }

    fn dequant_f32_opt(g: &GgufFile, name: &str) -> Result<Option<Vec<f32>>> {
        if g.tensor(name).is_some() {
            Ok(Some(Self::dequant_f32(g, name)?))
        } else {
            Ok(None)
        }
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
}

impl Engine for QwenDense {
    fn load(weights: &Path, config: EngineConfig) -> Result<Self> {
        let gguf = GgufFile::open(weights)?;
        let cfg = QwenConfig::from_gguf(&gguf)?;
        let model_id = gguf.name().unwrap_or("qwen2-dense").to_string();

        // Tokenizer: prefer sidecar tokenizer.json, fall back to GGUF.
        let sidecar = weights
            .parent()
            .map(|d| d.join("tokenizer.json"))
            .filter(|p| p.exists());
        let tokenizer = if let Some(p) = sidecar {
            Tokenizer::from_file(&p)?
        } else {
            Tokenizer::from_gguf(&gguf)?
        };

        // Embed table — typically fp16 in Q4_K_M GGUFs but read whatever
        // dtype the GGUF carries.
        let embed = Self::dequant_f16(&gguf, "token_embd.weight")?;
        let final_norm = Self::dequant_f32(&gguf, "output_norm.weight")?;
        // Qwen2.5-3B-Q4_K_M ties LM head to embed (no separate
        // output.weight); larger Qwen variants may carry it explicitly.
        let lm_head = if gguf.tensor("output.weight").is_some() {
            Some(Self::dequant_f16(&gguf, "output.weight")?)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for li in 0..cfg.n_layers {
            let lp = |suf: &str| format!("blk.{li}.{suf}");

            let attn_norm = Self::dequant_f32(&gguf, &lp("attn_norm.weight"))?;
            let ffn_norm = Self::dequant_f32(&gguf, &lp("ffn_norm.weight"))?;

            let q_proj = Self::tensor_ref(&gguf, &lp("attn_q.weight"))?;
            let k_proj = Self::tensor_ref(&gguf, &lp("attn_k.weight"))?;
            let v_proj = Self::tensor_ref(&gguf, &lp("attn_v.weight"))?;
            let o_proj = Self::tensor_ref(&gguf, &lp("attn_output.weight"))?;

            // Biases are present on q/k/v in Qwen2; absent on o.
            let q_bias = Self::dequant_f32_opt(&gguf, &lp("attn_q.bias"))?.unwrap_or_default();
            let k_bias = Self::dequant_f32_opt(&gguf, &lp("attn_k.bias"))?.unwrap_or_default();
            let v_bias = Self::dequant_f32_opt(&gguf, &lp("attn_v.bias"))?.unwrap_or_default();

            let ffn_gate = Self::tensor_ref(&gguf, &lp("ffn_gate.weight"))?;
            let ffn_up = Self::tensor_ref(&gguf, &lp("ffn_up.weight"))?;
            let ffn_down = Self::tensor_ref(&gguf, &lp("ffn_down.weight"))?;

            layers.push(QwenLayer {
                attn_norm,
                ffn_norm,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_bias,
                k_bias,
                v_bias,
                ffn_gate,
                ffn_up,
                ffn_down,
            });
        }

        let max_seq = config.max_seq_len.min(cfg.max_seq_len);
        let kv = KvCache::new(cfg.n_layers, max_seq, cfg.n_kv_heads, cfg.head_dim);
        let sampler = Sampler::new(0);
        let metal_ctx = MetalContext::new_with_trace(config.trace_dispatch).ok();

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
            _weights_path: weights.to_owned(),
            metal_ctx,
        })
    }

    fn generate(
        &mut self,
        req: GenerateRequest,
        sink: &mut dyn FnMut(StreamEvent),
    ) -> Result<GenStats> {
        use std::sync::atomic::Ordering;

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
        let prompt_len = prompt_ids.len();
        let mut stats = GenStats {
            prompt_tokens: prompt_len,
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
            let mut logits = self.forward_token(last_id, pos)?;
            if stall_active && step_start.elapsed() > stall_limit {
                reason = StopReason::Aborted;
                break;
            }
            let next_id = self.sampler.sample(&mut logits, &req.sampling);
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
        let (buffers_created, bytes_allocated, commits) = self
            .metal_ctx
            .as_ref()
            .map(|ctx| ctx.drain_stats())
            .unwrap_or_default();
        stats.metal_buffers_created = buffers_created;
        stats.metal_bytes_allocated = bytes_allocated;
        stats.metal_commits = commits;
        sink(StreamEvent::Done {
            reason,
            stats: stats.clone(),
        });
        Ok(stats)
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn forward_tokens_for_test(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        if tokens.len() != positions.len() {
            return Err(crate::Error::Model(format!(
                "forward_tokens shape: tokens={} positions={}",
                tokens.len(), positions.len()
            )));
        }
        let mut out = Vec::with_capacity(tokens.len());
        for (i, &token) in tokens.iter().enumerate() {
            out.push(self.forward_token(token, positions[i])?);
        }
        Ok(out)
    }
}

impl QwenDense {
    fn rmsnorm_dispatch(&self, x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) -> Result<()> {
        #[cfg(target_os = "macos")]
        if let Some(ctx) = &self.metal_ctx {
            return crate::kernels::rmsnorm_metal(ctx, x, weight, eps, out);
        }
        rmsnorm(x, weight, eps, out);
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

    /// Q4_K_M matmul dispatcher used for every per-layer matmul (q/k/v/o
    /// projections + gate/up/down FFN). On macOS with Metal alive, reads
    /// raw 4-bit bytes from the GGUF mmap and dispatches `gemv_q4_k_m`
    /// (dequant fused inside FMA). Off-macOS or non-Q4_K, falls back to
    /// dequant-into-scratch + CPU gemv_f32 — slow but correct.
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

    fn forward_token(&mut self, token: u32, pos: usize) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let h = cfg.hidden;
        let head_dim = cfg.head_dim;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let mut x = vec![0.0f32; h];
        embed_lookup(&self.embed, h, token, &mut x);

        // Reused scratch for lazy dequant fallback.
        let mut scratch = Vec::<f32>::new();

        // KV cache append offset for this token: shared across layers,
        // so compute once before the layer loop. seq_len bumps once
        // after all layers finish (kv.seq_len reflects "tokens already
        // in cache *including* this token" only after the layer loop).
        let stride = n_kv_heads * head_dim;
        if self.kv.seq_len >= self.kv.max_seq {
            return Err(Error::Model(format!(
                "kv cache full at {}",
                self.kv.max_seq
            )));
        }
        let kv_off = self.kv.seq_len * stride;
        let mha_seq_len = self.kv.seq_len + 1;

        for li in 0..cfg.n_layers {
            // ---- Attention block ----
            let mut x_norm = vec![0.0f32; h];
            self.rmsnorm_dispatch(
                &x,
                &self.layers[li].attn_norm,
                cfg.rms_norm_eps,
                &mut x_norm,
            )?;

            // Q / K / V projections (Q4_K_M weights, fp32 biases).
            let layer = &self.layers[li];
            let mut q_full = vec![0.0f32; q_dim];
            let mut k_token = vec![0.0f32; kv_dim];
            let mut v_token = vec![0.0f32; kv_dim];
            self.matmul_q4_dispatch(&layer.q_proj, q_dim, h, &x_norm, &mut q_full, &mut scratch)?;
            self.matmul_q4_dispatch(
                &layer.k_proj,
                kv_dim,
                h,
                &x_norm,
                &mut k_token,
                &mut scratch,
            )?;
            self.matmul_q4_dispatch(
                &layer.v_proj,
                kv_dim,
                h,
                &x_norm,
                &mut v_token,
                &mut scratch,
            )?;
            // Add biases (Qwen2 carries them on q/k/v).
            if !layer.q_bias.is_empty() {
                add_inplace(&mut q_full, &layer.q_bias);
            }
            if !layer.k_bias.is_empty() {
                add_inplace(&mut k_token, &layer.k_bias);
            }
            if !layer.v_bias.is_empty() {
                add_inplace(&mut v_token, &layer.v_bias);
            }

            // RoPE on the full head_dim of every Q head and every KV head.
            for h_i in 0..n_heads {
                let off = h_i * head_dim;
                rope_inplace(&mut q_full[off..off + head_dim], pos as u32, cfg.rope_theta);
            }
            for h_i in 0..n_kv_heads {
                let off = h_i * head_dim;
                rope_inplace(
                    &mut k_token[off..off + head_dim],
                    pos as u32,
                    cfg.rope_theta,
                );
            }

            // Append this token's K, V into the KV cache for layer `li`.
            // We write at the pre-computed offset (shared across layers
            // since seq_len doesn't bump until after the loop).
            self.kv.keys[li][kv_off..kv_off + stride].copy_from_slice(&k_token);
            self.kv.values[li][kv_off..kv_off + stride].copy_from_slice(&v_token);

            let kv_size = mha_seq_len * stride;
            let keys = &self.kv.keys[li][..kv_size];
            let values = &self.kv.values[li][..kv_size];

            let mut attn_out = vec![0.0f32; q_dim];
            mha_decode_step(
                &q_full,
                keys,
                values,
                n_heads,
                n_kv_heads,
                head_dim,
                mha_seq_len,
                &mut attn_out,
            )?;

            // O projection.
            let mut o = vec![0.0f32; h];
            self.matmul_q4_dispatch(&layer.o_proj, h, q_dim, &attn_out, &mut o, &mut scratch)?;
            add_inplace(&mut x, &o);

            // ---- FFN block (standard SwiGLU) ----
            let mut x_norm = vec![0.0f32; h];
            self.rmsnorm_dispatch(&x, &layer.ffn_norm, cfg.rms_norm_eps, &mut x_norm)?;
            let mid = cfg.intermediate;
            let mut g = vec![0.0f32; mid];
            let mut u = vec![0.0f32; mid];
            let mut a = vec![0.0f32; mid];
            self.matmul_q4_dispatch(&layer.ffn_gate, mid, h, &x_norm, &mut g, &mut scratch)?;
            self.matmul_q4_dispatch(&layer.ffn_up, mid, h, &x_norm, &mut u, &mut scratch)?;
            silu_mul(&g, &u, &mut a);
            let mut f = vec![0.0f32; h];
            self.matmul_q4_dispatch(&layer.ffn_down, h, mid, &a, &mut f, &mut scratch)?;
            add_inplace(&mut x, &f);
        }

        // Bump KV cache seq_len now that every layer has written its
        // slice for this token.
        self.kv.seq_len += 1;

        // Final norm + LM head.
        let mut x_norm = vec![0.0f32; h];
        self.rmsnorm_dispatch(&x, &self.final_norm, cfg.rms_norm_eps, &mut x_norm)?;

        let mut logits = vec![0.0f32; cfg.vocab_size];
        let w_f16: &[f16] = match &self.lm_head {
            Some(w) => w,
            None => &self.embed,
        };
        self.gemv_f16_dispatch(w_f16, cfg.vocab_size, h, &x_norm, &mut logits)?;
        Ok(logits)
    }
}
