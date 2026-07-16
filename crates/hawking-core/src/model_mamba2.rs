//! Mamba-2 SSD (Selective State Space Duality) inference engine.
//!
//! Supports mamba2-370M-Q4_K_M.gguf and similar GGUF models with
//! `general.architecture = "mamba2"`.
//!
//! CPU reference path for correctness. Q4_K projections are accelerated via
//! the shared `matmul_q4_dispatch` helper (same Metal kernel as Phi3/Llama).
//! No KV cache — per-layer SSM state replaces it.

use super::arch_config::{token_embd_vocab_size, ArchReader};
use super::weights::{dequant_f32, tensor_ref, TensorRef};
use crate::engine::{Engine, EngineConfig, GenStats, GenerateRequest, StopReason, StreamEvent};
use crate::gguf::GgufFile;
use crate::kernels::{add_inplace, embed_lookup, gemv_f32, rmsnorm};
use crate::metal::MetalContext;
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
pub struct Mamba2Config {
    pub n_layers: usize,
    pub hidden: usize,
    pub inner: usize,       // expand * hidden_size  (2048 for 370M)
    pub n_heads: usize,     // 32
    pub head_dim: usize,    // inner / n_heads = 64
    pub state_size: usize,  // 128
    pub conv_kernel: usize, // 4
    pub n_groups: usize,    // 1 (number of B/C groups)
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
}

impl Mamba2Config {
    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let r = ArchReader::new(g, "mamba2");
        let n_layers = r.req_usize("block_count")?;
        let hidden = r.req_usize("embedding_length")?;
        let n_heads = r.req_usize("attention.head_count")?;
        let inner = r.req_usize("ssm.inner_size")?;
        let head_dim = if n_heads > 0 { inner / n_heads } else { 64 };
        let state_size = r.opt_usize("ssm.state_size", 128);
        let conv_kernel = r.opt_usize("ssm.conv_kernel", 4);
        let n_groups = r.opt_usize("ssm.group_count", 1);
        let rms_norm_eps = r.opt_f32("attention.layer_norm_rms_epsilon", 1e-5);
        let vocab_size = token_embd_vocab_size(g, "mamba2: cannot determine vocab_size")?;
        if vocab_size == 0 {
            return Err(Error::Model("mamba2: cannot determine vocab_size".into()));
        }
        Ok(Self { n_layers, hidden, inner, n_heads, head_dim, state_size, conv_kernel, n_groups, vocab_size, rms_norm_eps })
    }

    // Dimension of xBC slice (x_ssm + B + C).
    fn xbc_dim(&self) -> usize {
        self.inner + 2 * self.n_groups * self.state_size
    }

    // Dimension of dt slice (one value per head).
    fn dt_dim(&self) -> usize {
        self.n_heads
    }

    // Total output dimension of ssm_in projection.
    fn proj_dim(&self) -> usize {
        self.inner + self.xbc_dim() + self.dt_dim()
    }

    // Width of conv ring buffer (== xbc_dim).
    fn conv_width(&self) -> usize {
        self.xbc_dim()
    }
}

// ─── Per-layer weights ────────────────────────────────────────────────────────

pub struct Mamba2Layer {
    // Pre-SSM RMSNorm weights [hidden].
    pub attn_norm: Vec<f32>,
    // Post-SSD RMSNorm weights [inner] (before z gating).
    pub ssm_norm: Vec<f32>,
    // Input projection: [proj_dim, hidden] stored Q4_K in GGUF (rows=proj_dim, cols=hidden).
    pub ssm_in: TensorRef,
    // Output projection: [hidden, inner] stored Q4_K in GGUF (rows=hidden, cols=inner).
    pub ssm_out: TensorRef,
    // SSM decay logs [n_heads].
    pub ssm_a: Vec<f32>,
    // SSM skip connection [n_heads].
    pub ssm_d: Vec<f32>,
    // dt bias [n_heads].
    pub ssm_dt_bias: Vec<f32>,
    // Depthwise conv weights [conv_width * conv_kernel] (channel-major: W[d*conv_kernel+k]).
    pub conv_weight: Vec<f32>,
    // Depthwise conv bias [conv_width].
    pub conv_bias: Vec<f32>,
}

// ─── Per-layer recurrent state ────────────────────────────────────────────────

pub struct Mamba2LayerState {
    // Ring buffer of last (conv_kernel) xBC inputs.
    // Layout: ring[step * conv_width + d], step ∈ [0, conv_kernel), oldest-first.
    conv_ring: Vec<f32>,
    // Position in the ring (0..conv_kernel).
    conv_pos: usize,
    // SSM hidden state [n_heads * head_dim * state_size].
    ssm_state: Vec<f32>,
}

impl Mamba2LayerState {
    fn new(conv_kernel: usize, conv_width: usize, n_heads: usize, head_dim: usize, state_size: usize) -> Self {
        Self { conv_ring: vec![0.0f32; conv_kernel * conv_width], conv_pos: 0, ssm_state: vec![0.0f32; n_heads * head_dim * state_size] }
    }

    fn reset(&mut self) {
        self.conv_ring.iter_mut().for_each(|v| *v = 0.0);
        self.conv_pos = 0;
        self.ssm_state.iter_mut().for_each(|v| *v = 0.0);
    }
}

// ─── Engine struct ────────────────────────────────────────────────────────────

pub struct Mamba2 {
    pub config: Mamba2Config,
    pub tokenizer: Tokenizer,
    pub model_id: String,
    pub gguf: GgufFile,
    // Token embedding dequanted to f16 [vocab_size * hidden].
    pub embed: Vec<f16>,
    // Final RMSNorm weights [hidden].
    pub output_norm: Vec<f32>,
    // LM head as f16 [vocab_size * hidden]; None = tied to embed.
    pub lm_head: Option<Vec<f16>>,
    pub layers: Vec<Mamba2Layer>,
    pub states: Vec<Mamba2LayerState>,
    pub sampler: Sampler,
    pub metal_ctx: Option<MetalContext>,
    pub _weights_path: PathBuf,
}

impl Mamba2 {
    fn dequant_ref_into(&self, t: &TensorRef, buf: &mut Vec<f32>) -> Result<()> {
        if buf.len() != t.n_elems {
            buf.resize(t.n_elems, 0.0);
        }
        let bytes = &self.gguf.mmap[t.offset..t.offset + t.byte_size];
        quant::dequant_into(t.dtype, bytes, buf)
    }

    fn matmul_q4_dispatch(&self, t: &TensorRef, rows: usize, cols: usize, x: &[f32], out: &mut [f32], scratch: &mut Vec<f32>) -> Result<()> {
        #[cfg(target_os = "macos")]
        if let Some(ctx) = &self.metal_ctx {
            use crate::gguf::GgmlType;
            if t.dtype == GgmlType::Q4_K {
                let bytes = &self.gguf.mmap[t.offset..t.offset + t.byte_size];
                return crate::kernels::gemv_q4_k_m(ctx, bytes, rows, cols, x, out);
            }
        }
        self.dequant_ref_into(t, scratch)?;
        gemv_f32(scratch, rows, cols, x, out);
        Ok(())
    }

    fn rmsnorm_dispatch(&self, x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) -> Result<()> {
        #[cfg(target_os = "macos")]
        if let Some(ctx) = &self.metal_ctx {
            return crate::kernels::rmsnorm_metal(ctx, x, weight, eps, out);
        }
        rmsnorm(x, weight, eps, out);
        Ok(())
    }

    fn gemv_f16_dispatch(&self, w: &[f16], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
        #[cfg(target_os = "macos")]
        if let Some(ctx) = &self.metal_ctx {
            let bytes = bytemuck::cast_slice::<f16, u8>(w);
            return crate::kernels::gemv_f16_metal(ctx, bytes, rows, cols, x, out);
        }
        use crate::kernels::gemv_f16;
        gemv_f16(w, rows, cols, x, out);
        Ok(())
    }

    fn reset_states(&mut self) {
        for s in &mut self.states {
            s.reset();
        }
    }

    /// SSD decode step for one token. Returns logits over vocab.
    fn forward_token(&mut self, token: u32) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let h = cfg.hidden;
        let inner = cfg.inner;
        let n_heads = cfg.n_heads;
        let head_dim = cfg.head_dim;
        let state_size = cfg.state_size;
        let conv_kernel = cfg.conv_kernel;
        let conv_width = cfg.conv_width();
        let xbc_dim = cfg.xbc_dim();
        let proj_dim = cfg.proj_dim();
        let eps = cfg.rms_norm_eps;

        // 1. Token embedding lookup.
        let mut x = vec![0.0f32; h];
        embed_lookup(&self.embed, h, token, &mut x);

        let mut scratch = Vec::<f32>::new();
        let mut proj = vec![0.0f32; proj_dim];
        let mut y_flat = vec![0.0f32; inner];
        let mut y_norm = vec![0.0f32; inner];
        let mut y_out = vec![0.0f32; inner];
        let mut delta = vec![0.0f32; h];

        let n_groups = cfg.n_groups;

        for li in 0..cfg.n_layers {
            // 2. Pre-SSM RMSNorm.
            let mut xn = vec![0.0f32; h];
            self.rmsnorm_dispatch(&x, &self.layers[li].attn_norm, eps, &mut xn)?;

            // 3. Input projection: xn (hidden) → proj (proj_dim).
            //    ssm_in stored as (proj_dim, hidden) → rows=proj_dim, cols=hidden.
            self.matmul_q4_dispatch(&self.layers[li].ssm_in, proj_dim, h, &xn, &mut proj, &mut scratch)?;

            // 4. Split projection: z | xBC | dt.
            let (z, rest) = proj.split_at(inner);
            let (xbc, dt) = rest.split_at(xbc_dim);

            // 5. Conv ring buffer update + depthwise conv.
            {
                let state = &mut self.states[li];
                let pos = state.conv_pos;
                state.conv_ring[pos * conv_width..pos * conv_width + conv_width].copy_from_slice(xbc);
                let new_pos = (pos + 1) % conv_kernel;
                state.conv_pos = new_pos;
            }
            let new_pos = self.states[li].conv_pos;
            let conv_w = &self.layers[li].conv_weight;
            let conv_b = &self.layers[li].conv_bias;
            let mut y_conv = vec![0.0f32; conv_width];
            {
                let ring = &self.states[li].conv_ring;
                for d in 0..conv_width {
                    let mut acc = conv_b[d];
                    for k in 0..conv_kernel {
                        let ring_slot = (new_pos + k) % conv_kernel;
                        acc += ring[ring_slot * conv_width + d] * conv_w[d * conv_kernel + k];
                    }
                    y_conv[d] = acc;
                }
            }
            for v in &mut y_conv {
                *v *= 1.0 / (1.0 + (-*v).exp());
            }

            // 6. Split y_conv into x_ssm | B | C.
            let (x_ssm, bc) = y_conv.split_at(inner);
            let b_vec = &bc[..n_groups * state_size];
            let c_vec = &bc[n_groups * state_size..];

            // 7. Per-head SSM update (all layer weights read first, then state mutated).
            y_flat.iter_mut().for_each(|v| *v = 0.0);
            {
                let ssm_a_ref: Vec<f32> = self.layers[li].ssm_a.clone();
                let ssm_d_ref: Vec<f32> = self.layers[li].ssm_d.clone();
                let ssm_dt_bias_ref: Vec<f32> = self.layers[li].ssm_dt_bias.clone();
                let x_ssm_owned: Vec<f32> = x_ssm.to_vec();
                let b_owned: Vec<f32> = b_vec.to_vec();
                let c_owned: Vec<f32> = c_vec.to_vec();
                let dt_owned: Vec<f32> = dt.to_vec();
                let st = &mut self.states[li].ssm_state;
                for hi in 0..n_heads {
                    let raw_dt = dt_owned[hi] + ssm_dt_bias_ref[hi];
                    let dt_h = if raw_dt > 20.0 { raw_dt } else { (raw_dt.exp() + 1.0).ln() };
                    let a_h = (-dt_h * ssm_a_ref[hi].exp()).exp();
                    let x_h = &x_ssm_owned[hi * head_dim..(hi + 1) * head_dim];
                    let state_off = hi * head_dim * state_size;
                    for d in 0..head_dim {
                        let so = state_off + d * state_size;
                        let xd = dt_h * x_h[d];
                        let mut y_hd = ssm_d_ref[hi] * x_h[d];
                        for s in 0..state_size {
                            st[so + s] = a_h * st[so + s] + xd * b_owned[s];
                            y_hd += st[so + s] * c_owned[s];
                        }
                        y_flat[hi * head_dim + d] = y_hd;
                    }
                }
            }

            // 8. Post-SSD RMSNorm.
            self.rmsnorm_dispatch(&y_flat, &self.layers[li].ssm_norm, eps, &mut y_norm)?;

            // 9. Gate: y_out = y_norm * silu(z).
            for i in 0..inner {
                let zi = z[i];
                let gate = zi / (1.0 + (-zi).exp());
                y_out[i] = y_norm[i] * gate;
            }

            // 10. Output projection: y_out (inner) → delta (hidden).
            //     ssm_out stored as (hidden, inner) → rows=hidden, cols=inner.
            self.matmul_q4_dispatch(&self.layers[li].ssm_out, h, inner, &y_out, &mut delta, &mut scratch)?;

            // 11. Residual add.
            add_inplace(&mut x, &delta);
        }

        // Final norm.
        let mut x_norm = vec![0.0f32; h];
        self.rmsnorm_dispatch(&x, &self.output_norm, eps, &mut x_norm)?;

        // LM head (tied to embed when output.weight is absent).
        let vocab = cfg.vocab_size;
        let mut logits = vec![0.0f32; vocab];
        let w_f16: &[f16] = match &self.lm_head {
            Some(w) => w,
            None => &self.embed,
        };
        self.gemv_f16_dispatch(w_f16, vocab, h, &x_norm, &mut logits)?;
        Ok(logits)
    }
}

// ─── Engine trait ─────────────────────────────────────────────────────────────

impl Engine for Mamba2 {
    fn load(weights: &Path, config: EngineConfig) -> Result<Self> {
        let gguf = GgufFile::open(weights)?;
        let cfg = Mamba2Config::from_gguf(&gguf)?;
        let model_id = gguf.name().unwrap_or("mamba2").to_string();

        let sidecar = weights.parent().map(|d| d.join("tokenizer.json")).filter(|p| p.exists());
        let tokenizer = if let Some(p) = sidecar { Tokenizer::from_file(&p)? } else { Tokenizer::from_gguf(&gguf)? };

        let embed = super::weights::dequant_f16(&gguf, "token_embd.weight")?;
        let output_norm = dequant_f32(&gguf, "output_norm.weight")?;
        let lm_head = if gguf.tensor("output.weight").is_some() { Some(super::weights::dequant_f16(&gguf, "output.weight")?) } else { None };

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for li in 0..cfg.n_layers {
            let lp = |suf: &str| format!("blk.{li}.{suf}");
            layers.push(Mamba2Layer {
                attn_norm: dequant_f32(&gguf, &lp("attn_norm.weight"))?,
                ssm_norm: dequant_f32(&gguf, &lp("ssm_norm.weight"))?,
                ssm_in: tensor_ref(&gguf, &lp("ssm_in.weight"))?,
                ssm_out: tensor_ref(&gguf, &lp("ssm_out.weight"))?,
                ssm_a: dequant_f32(&gguf, &lp("ssm_a"))?,
                ssm_d: dequant_f32(&gguf, &lp("ssm_d"))?,
                ssm_dt_bias: dequant_f32(&gguf, &lp("ssm_dt.bias"))?,
                conv_weight: dequant_f32(&gguf, &lp("ssm_conv1d.weight"))?,
                conv_bias: dequant_f32(&gguf, &lp("ssm_conv1d.bias"))?,
            });
        }

        let states: Vec<Mamba2LayerState> = (0..cfg.n_layers).map(|_| Mamba2LayerState::new(cfg.conv_kernel, cfg.conv_width(), cfg.n_heads, cfg.head_dim, cfg.state_size)).collect();

        let sampler = Sampler::new(42);

        #[allow(unused_mut)]
        let mut metal_ctx: Option<MetalContext> = None;
        #[cfg(target_os = "macos")]
        if !config.force_cpu {
            let ctx_result = if config.trace_dispatch { MetalContext::new_with_trace(true) } else { MetalContext::new() };
            match ctx_result {
                Ok(ctx) => metal_ctx = Some(ctx),
                Err(e) => eprintln!("hawking: mamba2 Metal init failed ({e}), using CPU"),
            }
        }

        Ok(Self { config: cfg, tokenizer, model_id, gguf, embed, output_norm, lm_head, layers, states, sampler, metal_ctx, _weights_path: weights.to_path_buf() })
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn model_arch(&self) -> &str {
        "mamba2"
    }

    fn generate(&mut self, req: GenerateRequest, sink: &mut dyn FnMut(StreamEvent)) -> Result<GenStats> {
        let prompt_ids = self.tokenizer.encode(&req.prompt, true)?;
        let n_prompt = prompt_ids.len();

        self.reset_states();

        let t0 = Instant::now();

        // Prefill all prompt tokens; only last-position logits matter.
        for &tok in &prompt_ids {
            let _ = self.forward_token(tok)?;
        }
        let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let mut n_gen = 0usize;
        let max_new = req.max_new_tokens.min(4096usize.saturating_sub(n_prompt));
        let t1 = Instant::now();
        let mut last_tok = *prompt_ids.last().unwrap_or(&0);

        let mut stop_reason = StopReason::MaxTokens;

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

            let mut logits = self.forward_token(last_tok)?;
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

    fn forward_tokens_for_test(&mut self, tokens: &[u32], positions: &[usize]) -> Result<Vec<Vec<f32>>> {
        let _ = positions;
        let mut out = Vec::with_capacity(tokens.len());
        self.reset_states();
        for &tok in tokens {
            out.push(self.forward_token(tok)?);
        }
        Ok(out)
    }

    fn reset_kv_for_test(&mut self) {
        self.reset_states();
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
