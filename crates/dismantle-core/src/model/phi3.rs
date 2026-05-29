//! Phi-3 / Phi-3.5-mini dense forward pass (llama.cpp arch "phi3").
//!
//! Phi-3 diverges from the Llama dense template in three ways, all
//! handled here:
//!
//!   - **Fused projections.** Q/K/V live in one `attn_qkv.weight`
//!     tensor and the FFN gate+up live in one `ffn_up.weight` tensor.
//!     We split them into sub-`TensorRef`s at load by row-offset
//!     arithmetic, which preserves the underlying quant (Q4_K stays
//!     Q4_K) because GGML quant blocks pack row-contiguously and the
//!     splits land on row boundaries.
//!   - **NEOX RoPE + longrope scaling.** Phi-3 uses NEOX pairing; Phi-3.5
//!     adds su/longrope per-dimension factor arrays
//!     (`rope_factors_short/long.weight`) selected by context length,
//!     plus an attention mscale. See [`crate::kernels::rope_inplace_longrope`].
//!   - **Standard pre-norm + SwiGLU**, no biases, LM head either separate
//!     (`output.weight`) or tied to the embedding table.
//!
//! On macOS the Q4_K projections, f16 LM head, and rmsnorm run on Metal;
//! attention and non-Q4_K weights use the CPU reference path. Full causal
//! attention (Phi-3's large sliding window is not applied).

use crate::attn::mha_decode_step;
use crate::cache::KvCache;
use crate::engine::{
    Engine, EngineConfig, GenStats, GenerateRequest, StopReason, StreamEvent,
};
use crate::gguf::{GgmlType, GgufFile};
use crate::kernels::{
    add_inplace, embed_lookup, gemv_f16, gemv_f32, rmsnorm, rope_inplace_longrope, silu_mul,
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
pub struct Phi3Config {
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

impl Phi3Config {
    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let get_u32 = |k: &str| g.metadata.get(k).and_then(|v| v.as_u32());
        let get_f32 = |k: &str| g.metadata.get(k).and_then(|v| v.as_f32());

        let n_layers = get_u32("phi3.block_count")
            .ok_or_else(|| Error::Model("missing phi3.block_count".into()))?
            as usize;
        let hidden = get_u32("phi3.embedding_length")
            .ok_or_else(|| Error::Model("missing phi3.embedding_length".into()))?
            as usize;
        let n_heads = get_u32("phi3.attention.head_count")
            .ok_or_else(|| Error::Model("missing phi3.attention.head_count".into()))?
            as usize;
        let n_kv_heads =
            get_u32("phi3.attention.head_count_kv").unwrap_or(n_heads as u32) as usize;
        let head_dim = get_u32("phi3.attention.key_length")
            .map(|v| v as usize)
            .unwrap_or(hidden / n_heads);
        let intermediate = get_u32("phi3.feed_forward_length")
            .ok_or_else(|| Error::Model("missing phi3.feed_forward_length".into()))?
            as usize;
        let vocab_size = match get_u32("phi3.vocab_size") {
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
        let rope_theta = get_f32("phi3.rope.freq_base").unwrap_or(10_000.0);
        let rms_norm_eps = get_f32("phi3.attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
        let max_seq_len = get_u32("phi3.context_length").unwrap_or(4096) as usize;

        if let Some(win) = get_u32("phi3.attention.sliding_window") {
            if (win as usize) < max_seq_len {
                eprintln!(
                    "dismantle: warning — GGUF declares phi3 sliding_window={win} but the \
                     engine runs full causal attention; output may drift beyond {win} tokens"
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
        })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TensorRef {
    pub offset: usize,
    pub byte_size: usize,
    pub dtype: GgmlType,
    pub n_elems: usize,
}

pub struct Phi3Layer {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    // Sub-views of the fused attn_qkv tensor (row-offset splits).
    pub(crate) q_proj: TensorRef,
    pub(crate) k_proj: TensorRef,
    pub(crate) v_proj: TensorRef,
    pub(crate) o_proj: TensorRef,
    // Sub-views of the fused ffn_up tensor (gate = first half, up = second).
    pub(crate) ffn_gate: TensorRef,
    pub(crate) ffn_up: TensorRef,
    pub(crate) ffn_down: TensorRef,
}

pub struct Phi3 {
    pub config: Phi3Config,
    pub tokenizer: Tokenizer,
    pub model_id: String,
    pub gguf: GgufFile,
    pub embed: Vec<f16>,
    pub final_norm: Vec<f32>,
    pub lm_head: Option<Vec<f16>>,
    pub layers: Vec<Phi3Layer>,
    /// Per-dimension RoPE factors (head_dim/2) selected at load: long
    /// when max_seq_len exceeds the original context, else short. All
    /// 1.0 when the GGUF carries no longrope tensors (plain NEOX RoPE).
    pub rope_ext_factors: Vec<f32>,
    /// Long-context attention mscale (1.0 unless extended context).
    pub rope_mscale: f32,
    pub kv: KvCache,
    pub sampler: Sampler,
    pub kernel_profile: Option<KernelProfile>,
    pub _weights_path: PathBuf,
    pub metal_ctx: Option<MetalContext>,
}

impl Phi3 {
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

    /// Carve a row-range out of a (rows, cols) weight `t` into a new
    /// `TensorRef` pointing into the same mmap. Valid because every GGML
    /// quant block packs row-contiguously and the splits land on row
    /// boundaries, so the quant type (Q4_K etc.) is preserved.
    ///
    /// `cols` is the input dimension; `row_start`/`n_rows` are in rows of
    /// the output dimension. Errors if the byte size isn't an exact
    /// multiple of the row count (i.e. the split would land mid-block).
    fn sub_rows(t: &TensorRef, cols: usize, row_start: usize, n_rows: usize) -> Result<TensorRef> {
        let total_rows = t.n_elems / cols;
        if total_rows * cols != t.n_elems {
            return Err(Error::Model(format!(
                "phi3 fused split: n_elems={} not divisible by cols={cols}",
                t.n_elems
            )));
        }
        if t.byte_size % total_rows != 0 {
            return Err(Error::Model(format!(
                "phi3 fused split: byte_size={} not divisible by rows={total_rows}",
                t.byte_size
            )));
        }
        let bytes_per_row = t.byte_size / total_rows;
        if row_start + n_rows > total_rows {
            return Err(Error::Model(format!(
                "phi3 fused split out of range: {row_start}+{n_rows} > {total_rows}"
            )));
        }
        Ok(TensorRef {
            offset: t.offset + row_start * bytes_per_row,
            byte_size: n_rows * bytes_per_row,
            dtype: t.dtype,
            n_elems: n_rows * cols,
        })
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
        let mscale = self.rope_mscale;

        let mut x = vec![0.0f32; h];
        embed_lookup(&self.embed, h, token, &mut x);

        let mut scratch = Vec::<f32>::new();

        let stride = n_kv_heads * head_dim;
        if self.kv.seq_len >= self.kv.max_seq {
            return Err(Error::Model(format!("kv cache full at {}", self.kv.max_seq)));
        }
        let kv_off = self.kv.seq_len * stride;
        let mha_seq_len = self.kv.seq_len + 1;

        for li in 0..n_layers {
            // Per-layer weights (incl. the split fused Q/K/V and gate/up
            // sub-views) read in place; no per-token clones. ext_factors
            // is read straight off `self.rope_ext_factors` at the rope
            // call sites — both shared borrows of self.
            let mut x_norm = vec![0.0f32; h];
            self.rmsnorm_dispatch(&x, &self.layers[li].attn_norm, rms_eps, &mut x_norm)?;

            let mut q_full = vec![0.0f32; q_dim];
            let mut k_token = vec![0.0f32; kv_dim];
            let mut v_token = vec![0.0f32; kv_dim];
            self.matmul_q4_dispatch(&self.layers[li].q_proj, q_dim, h, &x_norm, &mut q_full, &mut scratch)?;
            self.matmul_q4_dispatch(&self.layers[li].k_proj, kv_dim, h, &x_norm, &mut k_token, &mut scratch)?;
            self.matmul_q4_dispatch(&self.layers[li].v_proj, kv_dim, h, &x_norm, &mut v_token, &mut scratch)?;

            // NEOX longrope on every Q and KV head.
            for h_i in 0..n_heads {
                let off = h_i * head_dim;
                rope_inplace_longrope(
                    &mut q_full[off..off + head_dim],
                    pos as u32,
                    rope_theta,
                    &self.rope_ext_factors,
                    mscale,
                );
            }
            for h_i in 0..n_kv_heads {
                let off = h_i * head_dim;
                rope_inplace_longrope(
                    &mut k_token[off..off + head_dim],
                    pos as u32,
                    rope_theta,
                    &self.rope_ext_factors,
                    mscale,
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
            self.matmul_q4_dispatch(&self.layers[li].o_proj, h, q_dim, &attn_out, &mut o, &mut scratch)?;
            add_inplace(&mut x, &o);

            let mut x_norm2 = vec![0.0f32; h];
            self.rmsnorm_dispatch(&x, &self.layers[li].ffn_norm, rms_eps, &mut x_norm2)?;
            let mut g = vec![0.0f32; mid];
            let mut u = vec![0.0f32; mid];
            let mut a = vec![0.0f32; mid];
            self.matmul_q4_dispatch(&self.layers[li].ffn_gate, mid, h, &x_norm2, &mut g, &mut scratch)?;
            self.matmul_q4_dispatch(&self.layers[li].ffn_up, mid, h, &x_norm2, &mut u, &mut scratch)?;
            silu_mul(&g, &u, &mut a);
            let mut f = vec![0.0f32; h];
            self.matmul_q4_dispatch(&self.layers[li].ffn_down, h, mid, &a, &mut f, &mut scratch)?;
            add_inplace(&mut x, &f);
        }

        self.kv.seq_len += 1;

        let mut x_norm = vec![0.0f32; h];
        self.rmsnorm_dispatch(&x, &self.final_norm, rms_eps, &mut x_norm)?;

        let mut logits = vec![0.0f32; vocab_size];
        let w_f16: &[f16] = match &self.lm_head {
            Some(w) => w,
            None => &self.embed,
        };
        self.gemv_f16_dispatch(w_f16, vocab_size, h, &x_norm, &mut logits)?;
        Ok(logits)
    }
}

impl Engine for Phi3 {
    fn load(weights: &Path, config: EngineConfig) -> Result<Self> {
        let gguf = GgufFile::open(weights)?;
        let cfg = Phi3Config::from_gguf(&gguf)?;
        let model_id = gguf.name().unwrap_or("phi3").to_string();

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

        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for li in 0..cfg.n_layers {
            let lp = |suf: &str| format!("blk.{li}.{suf}");

            // Fused QKV → Q[0,q_dim) K[q_dim,q_dim+kv_dim) V[..]. cols=hidden.
            let qkv = Self::tensor_ref(&gguf, &lp("attn_qkv.weight"))?;
            let q_proj = Self::sub_rows(&qkv, cfg.hidden, 0, q_dim)?;
            let k_proj = Self::sub_rows(&qkv, cfg.hidden, q_dim, kv_dim)?;
            let v_proj = Self::sub_rows(&qkv, cfg.hidden, q_dim + kv_dim, kv_dim)?;

            // Fused gate+up → gate[0,mid) up[mid,2*mid). cols=hidden.
            let gate_up = Self::tensor_ref(&gguf, &lp("ffn_up.weight"))?;
            let ffn_gate = Self::sub_rows(&gate_up, cfg.hidden, 0, cfg.intermediate)?;
            let ffn_up = Self::sub_rows(&gate_up, cfg.hidden, cfg.intermediate, cfg.intermediate)?;

            layers.push(Phi3Layer {
                attn_norm: Self::dequant_f32(&gguf, &lp("attn_norm.weight"))?,
                ffn_norm: Self::dequant_f32(&gguf, &lp("ffn_norm.weight"))?,
                q_proj,
                k_proj,
                v_proj,
                o_proj: Self::tensor_ref(&gguf, &lp("attn_output.weight"))?,
                ffn_gate,
                ffn_up,
                ffn_down: Self::tensor_ref(&gguf, &lp("ffn_down.weight"))?,
            });
        }

        // Runtime context (matches llama.cpp, which keys longrope factor
        // selection on n_ctx, not the model's trained max). A 128k model
        // run at n_ctx<=orig therefore uses the SHORT factors.
        let max_seq = config.max_seq_len.min(cfg.max_seq_len);

        // longrope factor selection. Phi-3.5 ships short/long factor
        // tensors (length head_dim/2) and an original context length.
        let half = cfg.head_dim / 2;
        let orig_ctx = gguf
            .metadata
            .get("phi3.rope.scaling.original_context_length")
            .and_then(|v| v.as_u32())
            .map(|v| v as usize)
            .unwrap_or(max_seq);
        let use_long = max_seq > orig_ctx;
        let short_f = Self::dequant_f32_opt(&gguf, "rope_factors_short.weight")?;
        let long_f = Self::dequant_f32_opt(&gguf, "rope_factors_long.weight")?;
        let has_factors = short_f.is_some() || long_f.is_some();
        let rope_ext_factors = if use_long {
            long_f.or(short_f)
        } else {
            short_f.or(long_f)
        }
        .filter(|f| f.len() == half)
        .unwrap_or_else(|| vec![1.0f32; half]);
        // Attention mscale: prefer the converter-baked value
        // (phi3.rope.scaling.attn_factor); else the su/longrope closed
        // form from the model's trained ratio, applied only when the
        // model actually declares longrope factors.
        let rope_mscale = gguf
            .metadata
            .get("phi3.rope.scaling.attn_factor")
            .and_then(|v| v.as_f32())
            .unwrap_or_else(|| {
                if has_factors && cfg.max_seq_len > orig_ctx && orig_ctx > 0 {
                    let scale = cfg.max_seq_len as f32 / orig_ctx as f32;
                    (1.0 + scale.ln() / (orig_ctx as f32).ln()).sqrt()
                } else {
                    1.0
                }
            });

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
            lm_head,
            layers,
            rope_ext_factors,
            rope_mscale,
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
        "phi3"
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Row-offset splitting must preserve dtype and land on clean
    /// boundaries: a fused (rows=12, cols=256) Q4_K tensor split 4/8
    /// yields sub-views whose byte offsets/sizes are exact row multiples.
    #[test]
    fn sub_rows_splits_on_block_boundaries() {
        // Q4_K: 256 elems/block, 144 bytes/block. cols=256 ⇒ 1 block/row.
        let cols = 256usize;
        let rows = 12usize;
        let bytes_per_row = 144usize;
        let fused = TensorRef {
            offset: 1000,
            byte_size: rows * bytes_per_row,
            dtype: GgmlType::Q4_K,
            n_elems: rows * cols,
        };
        let a = Phi3::sub_rows(&fused, cols, 0, 4).unwrap();
        let b = Phi3::sub_rows(&fused, cols, 4, 8).unwrap();
        assert_eq!(a.offset, 1000);
        assert_eq!(a.byte_size, 4 * bytes_per_row);
        assert_eq!(a.n_elems, 4 * cols);
        assert_eq!(a.dtype, GgmlType::Q4_K);
        assert_eq!(b.offset, 1000 + 4 * bytes_per_row);
        assert_eq!(b.byte_size, 8 * bytes_per_row);
        assert_eq!(b.n_elems, 8 * cols);
        // The two sub-views exactly tile the fused tensor.
        assert_eq!(a.byte_size + b.byte_size, fused.byte_size);
        assert_eq!(a.n_elems + b.n_elems, fused.n_elems);
    }

    #[test]
    fn sub_rows_rejects_out_of_range() {
        let fused = TensorRef {
            offset: 0,
            byte_size: 12 * 144,
            dtype: GgmlType::Q4_K,
            n_elems: 12 * 256,
        };
        assert!(Phi3::sub_rows(&fused, 256, 8, 8).is_err());
    }
}
