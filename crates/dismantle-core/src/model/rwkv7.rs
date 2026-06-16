//! RWKV-7 ("Goose") inference path — slices 1-2: GGUF loader + CPU-reference
//! forward + numerical parity gate against llama.cpp.
//!
//! RWKV-7 is a state-space model: it carries a **fixed-size recurrent state per
//! layer** instead of a growing KV cache, so decode is O(1) in context and the
//! state footprint is constant (~6 MiB for the 0.4B at any depth). This module
//! is the correctness layer (CPU f32) that de-risks the recurrence math before
//! the Metal WKV-7 kernel (a later slice).
//!
//! The math mirrors llama.cpp's `rwkv7-base.cpp` (`build_rwkv7_time_mix` /
//! `build_rwkv7_channel_mix`) and the scalar recurrence in
//! `ggml_compute_forward_rwkv_wkv7_f32`. See `docs/strand/`-adjacent
//! `reports/ssm_derisk_m3.md` (branch `ssm/derisk`) for the spec.
//!
//! ## Per-token forward (decode, one token)
//! ```text
//!   x = embed[token]
//!   x = layernorm(x, tok_norm)                      // LN0 (embedding norm)
//!   for each layer:
//!       att_in  = layernorm(x, attn_norm)
//!       x_prev  = att_token_shift_state             // previous token's att_in
//!       cur     = time_mix(att_in, x_prev)          // WKV-7 recurrence (per-head SxS state)
//!       att_token_shift_state = att_in              // store for next token
//!       ffn_inp = cur + x                           // residual
//!       ffn_in  = layernorm(ffn_inp, attn_norm_2)
//!       x_prev2 = ffn_token_shift_state
//!       cmix    = channel_mix(ffn_in, x_prev2)      // ReLU^2 MLP
//!       ffn_token_shift_state = ffn_in              // store for next token
//!       x       = cmix + ffn_inp                    // residual
//!   x = layernorm(x, output_norm)
//!   logits = output_weight @ x
//! ```
//!
//! ## WKV-7 recurrence (per head, state `S` is `head_size x head_size`)
//! With `a = -kk`, `b = kk * iclr` (per head, after l2-norm of `k*k_k`):
//! ```text
//!   sa[i]    = sum_j a[j] * S_prev[i][j]
//!   S[i][j]  = S_prev[i][j]*w[j] + v[i]*k[j] + sa[i]*b[j]
//!   out[i]   = sum_j S[i][j] * r[j]
//! ```
//! `S[i][j]` is stored row-major at `i*head_size + j` (row `i`, col `j`),
//! matching ggml's `state_prev[i*h_stride + j]` indexing.

use crate::engine::{Engine, EngineConfig, GenStats, GenerateRequest, StopReason, StreamEvent};
use crate::gguf::GgufFile;
use crate::kernels::gemv_f32;
use crate::model::weights::{dequant_f32, tensor_ref};
use crate::sample::Sampler;
use crate::tokenizer::Tokenizer;
use crate::{quant, Error, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Instant;

#[cfg(target_os = "macos")]
use crate::metal::MetalContext;

/// Static architecture parameters read from the `rwkv7.*` GGUF metadata.
#[derive(Debug, Clone)]
pub struct RwkvConfig {
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_ff: usize,
    pub head_size: usize,
    pub head_count: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub ln_eps: f32,
    /// LoRA ranks for the time-mix sub-projections.
    pub decay_lora: usize,
    pub iclr_lora: usize,
    pub value_res_lora: usize,
    pub gate_lora: usize,
    pub token_shift_count: usize,
}

impl RwkvConfig {
    fn from_gguf(g: &GgufFile) -> Result<Self> {
        let get_u32 = |k: &str| g.metadata.get(k).and_then(|v| v.as_u32());
        let req = |k: &str| -> Result<usize> {
            get_u32(k)
                .map(|v| v as usize)
                .ok_or_else(|| Error::Model(format!("missing {k}")))
        };
        let n_embd = req("rwkv7.embedding_length")?;
        let n_layer = req("rwkv7.block_count")?;
        let n_ff = req("rwkv7.feed_forward_length")?;
        let head_size = req("rwkv7.wkv.head_size")?;
        // `attention.head_count` is 0 in the World GGUFs — derive from embd/head.
        let head_count_meta = get_u32("rwkv7.attention.head_count").unwrap_or(0) as usize;
        let head_count = if head_count_meta == 0 {
            n_embd / head_size
        } else {
            head_count_meta
        };
        if head_count * head_size != n_embd {
            return Err(Error::Model(format!(
                "rwkv7: head_count({head_count}) * head_size({head_size}) != n_embd({n_embd})"
            )));
        }
        // vocab from the embedding tensor's outer dim (ne[1]).
        let vocab_size = g
            .tensor("token_embd.weight")
            .map(|t| t.dims.get(1).copied().unwrap_or(0) as usize)
            .filter(|&v| v > 0)
            .ok_or_else(|| Error::Model("rwkv7: cannot determine vocab from token_embd".into()))?;
        let max_seq_len = req("rwkv7.context_length").unwrap_or(n_embd.max(4096));
        let ln_eps = g
            .metadata
            .get("rwkv7.attention.layer_norm_epsilon")
            .and_then(|v| v.as_f32())
            .unwrap_or(1e-5);
        let decay_lora = req("rwkv7.attention.decay_lora_rank")?;
        let iclr_lora = req("rwkv7.attention.iclr_lora_rank")?;
        let value_res_lora = req("rwkv7.attention.value_residual_mix_lora_rank")?;
        let gate_lora = get_u32("rwkv7.attention.gate_lora_rank").unwrap_or(0) as usize;
        let token_shift_count = get_u32("rwkv7.token_shift_count").unwrap_or(2) as usize;
        Ok(Self {
            n_embd,
            n_layer,
            n_ff,
            head_size,
            head_count,
            vocab_size,
            max_seq_len,
            ln_eps,
            decay_lora,
            iclr_lora,
            value_res_lora,
            gate_lora,
            token_shift_count,
        })
    }
}

/// All per-layer weights for one RWKV-7 block. Norms/vectors are eager f32
/// (small); the large projection matrices are dequantized to f32 once at load
/// (this is the CPU reference path — the Metal slice keeps them as `TensorRef`
/// and reads the quantized bytes on the GPU).
pub struct RwkvLayer {
    // LayerNorms (with bias — RWKV uses LN, not RMSNorm).
    pub attn_norm_w: Vec<f32>,
    pub attn_norm_b: Vec<f32>,
    pub attn_norm2_w: Vec<f32>,
    pub attn_norm2_b: Vec<f32>,

    // Token-shift lerp coefficients.
    /// `[6 * n_embd]` packed as slot-major: slot order r,w,k,v,a,g.
    pub time_mix_lerp_fused: Vec<f32>,
    /// `[n_embd]`.
    pub channel_mix_lerp_k: Vec<f32>,

    // Time-mix main projections (`[out, in]` row-major after dequant).
    pub time_mix_receptance: Vec<f32>, // [n_embd, n_embd]
    pub time_mix_key: Vec<f32>,        // [n_embd, n_embd]
    pub time_mix_value: Vec<f32>,      // [n_embd, n_embd]
    pub time_mix_output: Vec<f32>,     // [n_embd, n_embd]

    // Decay LoRA (w): w1 [decay, n_embd], w2 [n_embd, decay], w0 [n_embd].
    pub time_mix_w0: Vec<f32>,
    pub time_mix_w1: Vec<f32>,
    pub time_mix_w2: Vec<f32>,
    // In-context learning rate LoRA (a): a1 [iclr, n_embd], a2 [n_embd, iclr], a0 [n_embd].
    pub time_mix_a0: Vec<f32>,
    pub time_mix_a1: Vec<f32>,
    pub time_mix_a2: Vec<f32>,
    // Value-residual mix LoRA (v): v1 [vres, n_embd], v2 [n_embd, vres], v0 [n_embd].
    // Present on every layer in the GGUF (layer 0's is unused; a dummy copy of `a`).
    pub time_mix_v0: Vec<f32>,
    pub time_mix_v1: Vec<f32>,
    pub time_mix_v2: Vec<f32>,
    // Gate LoRA (g): g1 [gate, n_embd], g2 [n_embd, gate]. Optional.
    pub time_mix_g1: Option<Vec<f32>>,
    pub time_mix_g2: Option<Vec<f32>>,

    // Per-channel time-mix vectors `[n_embd]`.
    pub time_mix_k_k: Vec<f32>,
    pub time_mix_k_a: Vec<f32>,
    pub time_mix_r_k: Vec<f32>,

    // WKV head group-norm (`[n_embd]`), applied over head_count groups.
    pub time_mix_ln_w: Vec<f32>,
    pub time_mix_ln_b: Vec<f32>,

    // Channel-mix (FFN) — ReLU^2.
    pub channel_mix_key: Vec<f32>,   // [n_ff, n_embd]
    pub channel_mix_value: Vec<f32>, // [n_embd, n_ff]
}

/// Per-layer recurrent state — the fixed-size replacement for the KV cache.
/// `wkv[layer]` is `head_count * head_size * head_size` floats (the per-head
/// `S` matrices, row-major `i*head_size + j`). `att_shift`/`ffn_shift` hold the
/// previous token's post-LN hidden for the two token-shift lerps (`n_embd`
/// each). Nothing here grows with sequence length.
pub struct RwkvState {
    pub wkv: Vec<Vec<f32>>,
    pub att_shift: Vec<Vec<f32>>,
    pub ffn_shift: Vec<Vec<f32>>,
    /// `true` until the first token has been consumed; used to mark the
    /// token-shift state as "no previous token" (x_prev = 0).
    pub fresh: bool,
}

impl RwkvState {
    fn new(cfg: &RwkvConfig) -> Self {
        let s_per_layer = cfg.head_count * cfg.head_size * cfg.head_size;
        Self {
            wkv: (0..cfg.n_layer)
                .map(|_| vec![0.0f32; s_per_layer])
                .collect(),
            att_shift: (0..cfg.n_layer).map(|_| vec![0.0f32; cfg.n_embd]).collect(),
            ffn_shift: (0..cfg.n_layer).map(|_| vec![0.0f32; cfg.n_embd]).collect(),
            fresh: true,
        }
    }

    /// Reset to the zero state (start of a fresh sequence).
    pub fn reset(&mut self) {
        for s in &mut self.wkv {
            s.iter_mut().for_each(|v| *v = 0.0);
        }
        for s in &mut self.att_shift {
            s.iter_mut().for_each(|v| *v = 0.0);
        }
        for s in &mut self.ffn_shift {
            s.iter_mut().for_each(|v| *v = 0.0);
        }
        self.fresh = true;
    }

    /// Total state size in bytes (the "constant 6 MiB" headline number).
    pub fn size_bytes(&self) -> usize {
        let f = |v: &Vec<Vec<f32>>| v.iter().map(|x| x.len()).sum::<usize>();
        (f(&self.wkv) + f(&self.att_shift) + f(&self.ffn_shift)) * std::mem::size_of::<f32>()
    }
}

pub struct RwkvSeven {
    pub config: RwkvConfig,
    pub model_id: String,
    /// mmap keepalive + metadata.
    pub gguf: GgufFile,
    /// RWKV "World" tokenizer is a custom trie not covered by the GGUF-fallback
    /// tokenizer; `None` when it could not be loaded. The parity gate feeds
    /// token ids directly via `forward_tokens_for_test`, so this only gates the
    /// text `generate` entrypoint.
    pub tokenizer: Option<Tokenizer>,

    pub embed: Vec<f32>, // [vocab, n_embd] dequantized
    pub tok_norm_w: Vec<f32>,
    pub tok_norm_b: Vec<f32>,
    pub output_norm_w: Vec<f32>,
    pub output_norm_b: Vec<f32>,
    pub output: Vec<f32>, // [vocab, n_embd] dequantized (LM head)

    pub layers: Vec<RwkvLayer>,
    pub state: RwkvState,
    pub sampler: Sampler,
    pub _weights_path: PathBuf,

    /// Metal context (None on non-macOS or when Metal init fails / is forced
    /// off). Present ⇒ the GPU decode path (`forward_token_gpu`) is available.
    #[cfg(target_os = "macos")]
    pub metal_ctx: Option<MetalContext>,
    /// GPU-resident weights + recurrent-state arena for the decode path. Built
    /// once in `load` when `metal_ctx` is `Some`. The CPU `state`/weights stay
    /// as the correctness oracle and the fallback.
    #[cfg(target_os = "macos")]
    pub gpu: Option<gpu::RwkvGpu>,
}

/// LayerNorm with weight + bias over the whole vector (population variance,
/// matching ggml's `ggml_norm`: subtract mean, divide by sqrt(var+eps), then
/// `*w + b`).
fn layernorm(x: &[f32], w: &[f32], b: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let mean = x.iter().copied().sum::<f32>() / n as f32;
    let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv = 1.0 / (var + eps).sqrt();
    for i in 0..n {
        out[i] = (x[i] - mean) * inv * w[i] + b[i];
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl RwkvSeven {
    fn dequant_layer_matrix(g: &GgufFile, name: &str) -> Result<Vec<f32>> {
        dequant_f32(g, name)
    }

    fn dequant_opt(g: &GgufFile, name: &str) -> Result<Option<Vec<f32>>> {
        if g.tensor(name).is_some() {
            Ok(Some(dequant_f32(g, name)?))
        } else {
            Ok(None)
        }
    }

    /// The full per-token RWKV-7 forward. Returns the `vocab`-sized logit row.
    /// Mutates `self.state` (the recurrent state replaces the KV cache).
    pub fn forward_token(&mut self, token: u32) -> Result<Vec<f32>> {
        // Copy the scalar config out so no borrow of `self.config` persists
        // across the `&mut self` time_mix/channel_mix calls below.
        let n = self.config.n_embd;
        let n_layer = self.config.n_layer;
        let vocab_size = self.config.vocab_size;
        let eps = self.config.ln_eps;

        // Embedding lookup + LN0.
        let row = token as usize * n;
        if row + n > self.embed.len() {
            return Err(Error::Model(format!("rwkv7: token {token} out of vocab")));
        }
        let mut x = self.embed[row..row + n].to_vec();
        {
            let mut tmp = vec![0.0f32; n];
            layernorm(&x, &self.tok_norm_w, &self.tok_norm_b, eps, &mut tmp);
            x = tmp;
        }

        // `v_first`: the value projection of layer 0, reused as the residual
        // target for the value-residual mix in deeper layers.
        let mut v_first: Vec<f32> = Vec::new();

        for li in 0..n_layer {
            // ---- token-shift for the time-mix branch ----
            let mut att_in = vec![0.0f32; n];
            {
                let layer = &self.layers[li];
                layernorm(&x, &layer.attn_norm_w, &layer.attn_norm_b, eps, &mut att_in);
            }
            // x_prev = previous token's att_in (zero on the first token).
            let cur = self.time_mix(li, &att_in, &mut v_first)?;
            // store att_in as the next token-shift for this layer.
            self.state.att_shift[li].copy_from_slice(&att_in);

            // residual: ffn_inp = cur + x
            let mut ffn_inp = x.clone();
            for i in 0..n {
                ffn_inp[i] += cur[i];
            }

            // ---- channel-mix branch ----
            let mut ffn_in = vec![0.0f32; n];
            {
                let layer = &self.layers[li];
                layernorm(
                    &ffn_inp,
                    &layer.attn_norm2_w,
                    &layer.attn_norm2_b,
                    eps,
                    &mut ffn_in,
                );
            }
            let cmix = self.channel_mix(li, &ffn_in)?;
            self.state.ffn_shift[li].copy_from_slice(&ffn_in);

            // residual: x = cmix + ffn_inp
            x = ffn_inp;
            for i in 0..n {
                x[i] += cmix[i];
            }
        }

        // final norm + LM head.
        let mut x_norm = vec![0.0f32; n];
        layernorm(
            &x,
            &self.output_norm_w,
            &self.output_norm_b,
            eps,
            &mut x_norm,
        );
        let mut logits = vec![0.0f32; vocab_size];
        gemv_f32(&self.output, vocab_size, n, &x_norm, &mut logits);

        self.state.fresh = false;
        Ok(logits)
    }

    /// Time-mix: token-shift lerp → r/w/k/v/a/g projections (+ LoRA paths) →
    /// WKV-7 recurrence → group-norm → r·k·r_k bonus → gate → output proj.
    fn time_mix(&mut self, li: usize, att_in: &[f32], v_first: &mut Vec<f32>) -> Result<Vec<f32>> {
        // Disjoint field borrows so the immutable `layer`/config reads coexist
        // with the in-place mutable update of `state.wkv[li]` in the recurrence.
        let RwkvSeven {
            config,
            layers,
            state,
            ..
        } = self;
        let cfg = &*config;
        let n = cfg.n_embd;
        let hs = cfg.head_size;
        let hc = cfg.head_count;
        let layer = &layers[li];

        // sx = x_prev - x   (x_prev = stored att token-shift; zero when fresh).
        let x_prev = &state.att_shift[li];
        let mut sx = vec![0.0f32; n];
        if state.fresh {
            for i in 0..n {
                sx[i] = -att_in[i];
            }
        } else {
            for i in 0..n {
                sx[i] = x_prev[i] - att_in[i];
            }
        }

        // Per-slot lerp: x_s = att_in + sx * lerp_fused[slot] (slots r,w,k,v,a,g).
        let lerp = |slot: usize, out: &mut [f32]| {
            let base = slot * n;
            for i in 0..n {
                out[i] = att_in[i] + sx[i] * layer.time_mix_lerp_fused[base + i];
            }
        };
        let mut xr = vec![0.0f32; n];
        let mut xw = vec![0.0f32; n];
        let mut xk = vec![0.0f32; n];
        let mut xv = vec![0.0f32; n];
        let mut xa = vec![0.0f32; n];
        let mut xg = vec![0.0f32; n];
        lerp(0, &mut xr);
        lerp(1, &mut xw);
        lerp(2, &mut xk);
        lerp(3, &mut xv);
        lerp(4, &mut xa);
        let has_gate = layer.time_mix_g1.is_some() && layer.time_mix_g2.is_some();
        if has_gate {
            lerp(5, &mut xg);
        }

        // r = Wr @ xr
        let mut r = vec![0.0f32; n];
        gemv_f32(&layer.time_mix_receptance, n, n, &xr, &mut r);

        // w = exp(-0.606531 * sigmoid(w0 + W2 @ tanh(W1 @ xw)))
        let mut w_lo = vec![0.0f32; cfg.decay_lora];
        gemv_f32(&layer.time_mix_w1, cfg.decay_lora, n, &xw, &mut w_lo);
        for v in &mut w_lo {
            *v = v.tanh();
        }
        let mut w = vec![0.0f32; n];
        gemv_f32(&layer.time_mix_w2, n, cfg.decay_lora, &w_lo, &mut w);
        for i in 0..n {
            w[i] = (-0.606531_f32 * sigmoid(w[i] + layer.time_mix_w0[i])).exp();
        }

        // k = Wk @ xk ; v = Wv @ xv
        let mut k = vec![0.0f32; n];
        gemv_f32(&layer.time_mix_key, n, n, &xk, &mut k);
        let mut v = vec![0.0f32; n];
        gemv_f32(&layer.time_mix_value, n, n, &xv, &mut v);

        // value-residual mix (skipped on layer 0, where v_first is established).
        if v_first.is_empty() {
            *v_first = v.clone();
        } else {
            // v += (v_first - v) * sigmoid(v0 + V2 @ (V1 @ xv))
            let mut v_lo = vec![0.0f32; cfg.value_res_lora];
            gemv_f32(&layer.time_mix_v1, cfg.value_res_lora, n, &xv, &mut v_lo);
            let mut v_mix = vec![0.0f32; n];
            gemv_f32(&layer.time_mix_v2, n, cfg.value_res_lora, &v_lo, &mut v_mix);
            for i in 0..n {
                let g = sigmoid(v_mix[i] + layer.time_mix_v0[i]);
                v[i] += (v_first[i] - v[i]) * g;
            }
        }

        // gate g = G2 @ sigmoid(G1 @ xg)
        let g_vec = if has_gate {
            let g1 = layer.time_mix_g1.as_ref().unwrap();
            let g2 = layer.time_mix_g2.as_ref().unwrap();
            let mut g_lo = vec![0.0f32; cfg.gate_lora];
            gemv_f32(g1, cfg.gate_lora, n, &xg, &mut g_lo);
            for vv in &mut g_lo {
                *vv = sigmoid(*vv);
            }
            let mut g = vec![0.0f32; n];
            gemv_f32(g2, n, cfg.gate_lora, &g_lo, &mut g);
            Some(g)
        } else {
            None
        };

        // a = sigmoid(a0 + A2 @ (A1 @ xa))   (in-context learning rate)
        let mut a_lo = vec![0.0f32; cfg.iclr_lora];
        gemv_f32(&layer.time_mix_a1, cfg.iclr_lora, n, &xa, &mut a_lo);
        let mut a = vec![0.0f32; n];
        gemv_f32(&layer.time_mix_a2, n, cfg.iclr_lora, &a_lo, &mut a);
        for i in 0..n {
            a[i] = sigmoid(a[i] + layer.time_mix_a0[i]);
        }

        // kk = l2norm_per_head(k * k_k)
        let mut kk = vec![0.0f32; n];
        for i in 0..n {
            kk[i] = k[i] * layer.time_mix_k_k[i];
        }
        for h in 0..hc {
            let off = h * hs;
            let mut sum = 0.0f32;
            for j in 0..hs {
                sum += kk[off + j] * kk[off + j];
            }
            // ggml_l2_norm: divide by max(sqrt(sum), eps), eps = 1e-12.
            let scale = 1.0 / sum.sqrt().max(1e-12);
            for j in 0..hs {
                kk[off + j] *= scale;
            }
        }

        // k = k + (a - 1) * (k * k_a)   [== k + a*ka - ka]
        for i in 0..n {
            let ka = k[i] * layer.time_mix_k_a[i];
            k[i] += a[i] * ka - ka;
        }

        // WKV-7 op inputs: a_op = -kk, b_op = kk * a.
        let mut a_op = vec![0.0f32; n];
        let mut b_op = vec![0.0f32; n];
        for i in 0..n {
            a_op[i] = -kk[i];
            b_op[i] = kk[i] * a[i];
        }

        // ---- WKV-7 recurrence, per head, in place on state.wkv[li] ----
        let wkv = &mut state.wkv[li];
        let mut out = vec![0.0f32; n];
        for h in 0..hc {
            let ho = h * hs; // offset into the n-dim vectors for this head
            let so = h * hs * hs; // offset into the state for this head
            for i in 0..hs {
                let v_i = v[ho + i];
                // sa[i] = sum_j a_op[j] * S_prev[i][j]
                let row = so + i * hs;
                let mut sa = 0.0f32;
                for j in 0..hs {
                    sa += a_op[ho + j] * wkv[row + j];
                }
                // S[i][j] = S_prev[i][j]*w[j] + v[i]*k[j] + sa*b_op[j]
                // out[i]  = sum_j S[i][j] * r[j]
                let mut result = 0.0f32;
                for j in 0..hs {
                    let s_new = wkv[row + j] * w[ho + j] + v_i * k[ho + j] + sa * b_op[ho + j];
                    wkv[row + j] = s_new;
                    result += s_new * r[ho + j];
                }
                out[ho + i] = result;
            }
        }

        // group-norm over head_count groups (eps = 64e-5), then *ln_w + ln_b.
        let gn_eps = 64e-5_f32;
        for h in 0..hc {
            let off = h * hs;
            let seg = &out[off..off + hs];
            let mean = seg.iter().copied().sum::<f32>() / hs as f32;
            let var = seg.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / hs as f32;
            let inv = 1.0 / (var + gn_eps).sqrt();
            for j in 0..hs {
                out[off + j] = (out[off + j] - mean) * inv;
            }
        }
        for i in 0..n {
            out[i] = out[i] * layer.time_mix_ln_w[i] + layer.time_mix_ln_b[i];
        }

        // bonus: out += v * (rowsum_per_head(k * r * r_k))
        for h in 0..hc {
            let off = h * hs;
            let mut rk = 0.0f32;
            for j in 0..hs {
                rk += k[off + j] * r[off + j] * layer.time_mix_r_k[off + j];
            }
            for j in 0..hs {
                out[off + j] += v[off + j] * rk;
            }
        }

        // gate
        if let Some(g) = g_vec {
            for i in 0..n {
                out[i] *= g[i];
            }
        }

        // output projection
        let mut y = vec![0.0f32; n];
        gemv_f32(&layer.time_mix_output, n, n, &out, &mut y);
        Ok(y)
    }

    /// Channel-mix (FFN): token-shift lerp → ReLU(Wk@xk)^2 → Wv@k.
    fn channel_mix(&self, li: usize, ffn_in: &[f32]) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let n = cfg.n_embd;
        let layer = &self.layers[li];

        let x_prev = &self.state.ffn_shift[li];
        // xk = ffn_in + (x_prev - ffn_in) * lerp_k
        let mut xk = vec![0.0f32; n];
        if self.state.fresh {
            for i in 0..n {
                xk[i] = ffn_in[i] + (-ffn_in[i]) * layer.channel_mix_lerp_k[i];
            }
        } else {
            for i in 0..n {
                xk[i] = ffn_in[i] + (x_prev[i] - ffn_in[i]) * layer.channel_mix_lerp_k[i];
            }
        }

        // k = relu(Wk @ xk)^2
        let mut k = vec![0.0f32; cfg.n_ff];
        gemv_f32(&layer.channel_mix_key, cfg.n_ff, n, &xk, &mut k);
        for v in &mut k {
            let r = v.max(0.0);
            *v = r * r;
        }
        // out = Wv @ k
        let mut out = vec![0.0f32; n];
        gemv_f32(&layer.channel_mix_value, n, cfg.n_ff, &k, &mut out);
        Ok(out)
    }

    /// Route one decode step to the GPU path when `use_gpu`, else the CPU
    /// reference. The two states are advanced independently, so callers MUST run
    /// a whole sequence on ONE path (they reset both up front). On non-macOS
    /// `use_gpu` is always false and this is exactly `forward_token`.
    #[inline]
    fn forward_token_routed(&mut self, token: u32, use_gpu: bool) -> Result<Vec<f32>> {
        #[cfg(target_os = "macos")]
        if use_gpu {
            return self.forward_token_gpu(token);
        }
        let _ = use_gpu;
        self.forward_token(token)
    }
}

impl Engine for RwkvSeven {
    fn load(weights: &Path, _config: EngineConfig) -> Result<Self> {
        let gguf = GgufFile::open(weights)?;
        let cfg = RwkvConfig::from_gguf(&gguf)?;
        let model_id = gguf.name().unwrap_or("rwkv7").to_string();

        // RWKV World tokenizer is a custom trie; best-effort load (the parity
        // gate does not need it).
        let tokenizer = Tokenizer::from_gguf(&gguf).ok();

        // Embedding + LM head dequantized to f32 (CPU reference path).
        let embed = {
            let t = tensor_ref(&gguf, "token_embd.weight")?;
            let mut buf = vec![0.0f32; t.n_elems];
            let bytes = &gguf.mmap[t.offset..t.offset + t.byte_size];
            quant::dequant_into(t.dtype, bytes, &mut buf)?;
            buf
        };
        let output = {
            let t = tensor_ref(&gguf, "output.weight")?;
            let mut buf = vec![0.0f32; t.n_elems];
            let bytes = &gguf.mmap[t.offset..t.offset + t.byte_size];
            quant::dequant_into(t.dtype, bytes, &mut buf)?;
            buf
        };
        let tok_norm_w = dequant_f32(&gguf, "token_embd_norm.weight")?;
        let tok_norm_b = dequant_f32(&gguf, "token_embd_norm.bias")?;
        let output_norm_w = dequant_f32(&gguf, "output_norm.weight")?;
        let output_norm_b = dequant_f32(&gguf, "output_norm.bias")?;

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for li in 0..cfg.n_layer {
            let p = |suf: &str| format!("blk.{li}.{suf}");
            let g1 = Self::dequant_opt(&gguf, &p("time_mix_g1.weight"))?;
            let g2 = Self::dequant_opt(&gguf, &p("time_mix_g2.weight"))?;
            layers.push(RwkvLayer {
                attn_norm_w: dequant_f32(&gguf, &p("attn_norm.weight"))?,
                attn_norm_b: dequant_f32(&gguf, &p("attn_norm.bias"))?,
                attn_norm2_w: dequant_f32(&gguf, &p("attn_norm_2.weight"))?,
                attn_norm2_b: dequant_f32(&gguf, &p("attn_norm_2.bias"))?,
                time_mix_lerp_fused: dequant_f32(&gguf, &p("time_mix_lerp_fused.weight"))?,
                channel_mix_lerp_k: dequant_f32(&gguf, &p("channel_mix_lerp_k.weight"))?,
                time_mix_receptance: Self::dequant_layer_matrix(
                    &gguf,
                    &p("time_mix_receptance.weight"),
                )?,
                time_mix_key: Self::dequant_layer_matrix(&gguf, &p("time_mix_key.weight"))?,
                time_mix_value: Self::dequant_layer_matrix(&gguf, &p("time_mix_value.weight"))?,
                time_mix_output: Self::dequant_layer_matrix(&gguf, &p("time_mix_output.weight"))?,
                time_mix_w0: dequant_f32(&gguf, &p("time_mix_w0.weight"))?,
                time_mix_w1: dequant_f32(&gguf, &p("time_mix_w1.weight"))?,
                time_mix_w2: dequant_f32(&gguf, &p("time_mix_w2.weight"))?,
                time_mix_a0: dequant_f32(&gguf, &p("time_mix_a0.weight"))?,
                time_mix_a1: dequant_f32(&gguf, &p("time_mix_a1.weight"))?,
                time_mix_a2: dequant_f32(&gguf, &p("time_mix_a2.weight"))?,
                time_mix_v0: dequant_f32(&gguf, &p("time_mix_v0.weight"))?,
                time_mix_v1: dequant_f32(&gguf, &p("time_mix_v1.weight"))?,
                time_mix_v2: dequant_f32(&gguf, &p("time_mix_v2.weight"))?,
                time_mix_g1: g1,
                time_mix_g2: g2,
                time_mix_k_k: dequant_f32(&gguf, &p("time_mix_k_k.weight"))?,
                time_mix_k_a: dequant_f32(&gguf, &p("time_mix_k_a.weight"))?,
                time_mix_r_k: dequant_f32(&gguf, &p("time_mix_r_k.weight"))?,
                time_mix_ln_w: dequant_f32(&gguf, &p("time_mix_ln.weight"))?,
                time_mix_ln_b: dequant_f32(&gguf, &p("time_mix_ln.bias"))?,
                channel_mix_key: Self::dequant_layer_matrix(&gguf, &p("channel_mix_key.weight"))?,
                channel_mix_value: Self::dequant_layer_matrix(
                    &gguf,
                    &p("channel_mix_value.weight"),
                )?,
            });
        }

        let state = RwkvState::new(&cfg);
        let sampler = Sampler::new(0);

        // ── GPU decode path (macOS/Metal) ─────────────────────────────────────
        // Build the Metal context + upload the (already-dequantized) f32 weights
        // into a RwkvDecodeArena once. The arena holds the FIXED recurrent state
        // (no growing KV) and is advanced in place each decode step. Forcing CPU
        // (`force_cpu` / DISMANTLE_FORCE_CPU=1) or any init failure leaves the
        // engine on the validated CPU reference path.
        #[cfg(target_os = "macos")]
        let (metal_ctx, gpu) = {
            let force_cpu = _config.force_cpu || crate::env_on("DISMANTLE_FORCE_CPU");
            if force_cpu {
                (None, None)
            } else {
                match MetalContext::new_with_trace(_config.trace_dispatch) {
                    Ok(ctx) => {
                        let built = gpu::RwkvGpu::build(
                            &ctx,
                            &cfg,
                            &embed,
                            &tok_norm_w,
                            &tok_norm_b,
                            &output_norm_w,
                            &output_norm_b,
                            &output,
                            &layers,
                        );
                        match built {
                            Ok(g) => (Some(ctx), Some(g)),
                            Err(e) => {
                                eprintln!("[rwkv7] GPU arena build failed ({e}); CPU path only");
                                (None, None)
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("[rwkv7] Metal unavailable ({e}); CPU path only");
                        (None, None)
                    }
                }
            }
        };

        Ok(Self {
            config: cfg,
            model_id,
            gguf,
            tokenizer,
            embed,
            tok_norm_w,
            tok_norm_b,
            output_norm_w,
            output_norm_b,
            output,
            layers,
            state,
            sampler,
            _weights_path: weights.to_path_buf(),
            #[cfg(target_os = "macos")]
            metal_ctx,
            #[cfg(target_os = "macos")]
            gpu,
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
        // Encode + read eos up front, then release the tokenizer borrow so the
        // `&mut self` forward calls below are unobstructed; re-borrow only for
        // `decode_one` inside the decode loop.
        let (prompt_ids, eos) = {
            let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
                Error::Model(
                    "rwkv7: World tokenizer not available from GGUF; drive the model via \
                     forward_tokens_for_test (token-id API). Text generate() needs a tokenizer."
                        .into(),
                )
            })?;
            (tokenizer.encode(&req.prompt, true)?, tokenizer.eos_id())
        };
        if prompt_ids.is_empty() {
            return Err(Error::Model("empty prompt after tokenization".into()));
        }
        let prompt_len = prompt_ids.len();
        let mut stats = GenStats {
            prompt_tokens: prompt_len,
            ..Default::default()
        };

        let abort_set = |req: &GenerateRequest| -> bool {
            req.abort
                .as_ref()
                .map(|f| f.load(Ordering::Relaxed))
                .unwrap_or(false)
        };

        // Reset BOTH the CPU reference state and (if present) the GPU arena
        // state, then run the whole sequence on a single path so the recurrent
        // state stays coherent. The GPU path is used end-to-end when Metal is
        // available; otherwise the CPU reference path. (RWKV-7 prefill is the
        // same per-step kernel as decode — no separate batched prefill here.)
        self.reset_kv_for_test();
        #[cfg(target_os = "macos")]
        let use_gpu = self.has_gpu();
        #[cfg(not(target_os = "macos"))]
        let use_gpu = false;

        // Prefill: run the whole prompt through the recurrence (state carries).
        let prefill_start = Instant::now();
        for &t in &prompt_ids {
            if abort_set(&req) {
                sink(StreamEvent::Done {
                    reason: StopReason::Aborted,
                    stats: stats.clone(),
                });
                return Ok(stats);
            }
            let _ = self.forward_token_routed(t, use_gpu)?;
        }
        stats.prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

        // Decode.
        let decode_start = Instant::now();
        let mut last_id = *prompt_ids.last().unwrap();
        let mut produced = 0usize;
        let mut reason = StopReason::MaxTokens;
        for _ in 0..req.max_new_tokens {
            if abort_set(&req) {
                reason = StopReason::Aborted;
                break;
            }
            let mut logits = self.forward_token_routed(last_id, use_gpu)?;
            let next_id = self.sampler.sample(&mut logits, &req.sampling);
            self.sampler.record(next_id);
            let text = self
                .tokenizer
                .as_ref()
                .and_then(|t| t.decode_one(next_id).ok())
                .unwrap_or_default();
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
        "rwkv7"
    }

    /// Token-id forward seam used by the parity gate. RWKV-7 carries recurrent
    /// state, so `positions` is informational only — the engine consumes the
    /// tokens in order against the current state. Callers that want a fresh
    /// sequence must `reset_kv_for_test()` first.
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
        for &t in tokens {
            out.push(self.forward_token(t)?);
        }
        Ok(out)
    }

    fn reset_kv_for_test(&mut self) {
        self.state.reset();
        #[cfg(target_os = "macos")]
        if let Some(g) = self.gpu.as_mut() {
            g.reset();
        }
    }
}

#[cfg(target_os = "macos")]
impl RwkvSeven {
    /// Whether the GPU decode path is wired and available.
    pub fn has_gpu(&self) -> bool {
        self.metal_ctx.is_some() && self.gpu.is_some()
    }

    /// GPU counterpart of [`forward_token`]: one RWKV-7 decode step on Metal,
    /// advancing the GPU-resident recurrent state in place. Bit-for-bit (within
    /// f32 tolerance) against [`forward_token`] — it reuses the same f32 weights
    /// and the same op order. Returns the `vocab`-sized logit row. Falls back to
    /// the CPU path when the GPU is unavailable.
    pub fn forward_token_gpu(&mut self, token: u32) -> Result<Vec<f32>> {
        if self.metal_ctx.is_none() || self.gpu.is_none() {
            return self.forward_token(token);
        }
        // Split borrows: ctx (immutable) + gpu (mutable) + cfg/embed (immutable).
        let RwkvSeven {
            config,
            metal_ctx,
            gpu,
            embed,
            ..
        } = self;
        let ctx = metal_ctx.as_ref().unwrap();
        let g = gpu.as_mut().unwrap();
        let logits = gpu::forward_token_gpu(ctx, g, config, embed, token)?;
        Ok(logits)
    }
}

/// RWKV-7 GPU decode path: weight upload, the recurrent-state arena, and the
/// single-step forward that drives the WKV-7 kernel + glue kernels. macOS-only.
#[cfg(target_os = "macos")]
pub mod gpu {
    use super::{RwkvConfig, RwkvLayer};
    use crate::metal::rwkv_decode_arena::{
        rwkv7_add_into_tcb, rwkv7_channel_mix_shift_tcb, rwkv7_copy_tcb, rwkv7_decay_act_tcb,
        rwkv7_gemv_f32_off_tcb, rwkv7_kk_kmix_tcb, rwkv7_layernorm_tcb, rwkv7_relu_sq_inplace_tcb,
        rwkv7_sigmoid_bias_tcb, rwkv7_sigmoid_inplace_tcb, rwkv7_tanh_inplace_tcb,
        rwkv7_token_shift_lerp_tcb, rwkv7_value_residual_mix_tcb, rwkv7_wkv_decode_tcb,
    };
    use crate::metal::{MetalContext, PinnedBuffer, RwkvDecodeArena, TokenCommandBuffer};
    use crate::{Error, Result};

    /// Group-norm epsilon used inside the WKV-7 per-head norm (matches the CPU
    /// reference `time_mix`: `gn_eps = 64e-5`).
    const GN_EPS: f32 = 64e-5;

    /// GPU-resident f32 weights for one RWKV-7 layer (uploaded once from the
    /// already-dequantized CPU `RwkvLayer`). All matrices are row-major `[out,in]`
    /// exactly as the CPU `gemv_f32` expects, so the GPU `gemv_f32_attn` kernel
    /// reproduces the CPU MAC.
    pub struct RwkvGpuLayer {
        pub attn_norm_w: PinnedBuffer,
        pub attn_norm_b: PinnedBuffer,
        pub attn_norm2_w: PinnedBuffer,
        pub attn_norm2_b: PinnedBuffer,
        pub lerp_fused: PinnedBuffer,
        pub channel_mix_lerp_k: PinnedBuffer,
        pub receptance: PinnedBuffer,
        pub key: PinnedBuffer,
        pub value: PinnedBuffer,
        pub output: PinnedBuffer,
        pub w0: PinnedBuffer,
        pub w1: PinnedBuffer,
        pub w2: PinnedBuffer,
        pub a0: PinnedBuffer,
        pub a1: PinnedBuffer,
        pub a2: PinnedBuffer,
        pub v0: PinnedBuffer,
        pub v1: PinnedBuffer,
        pub v2: PinnedBuffer,
        pub g1: Option<PinnedBuffer>,
        pub g2: Option<PinnedBuffer>,
        pub k_k: PinnedBuffer,
        pub k_a: PinnedBuffer,
        pub r_k: PinnedBuffer,
        pub ln_w: PinnedBuffer,
        pub ln_b: PinnedBuffer,
        pub channel_mix_key: PinnedBuffer,
        pub channel_mix_value: PinnedBuffer,
        pub has_gate: bool,
    }

    /// GPU decode bundle: per-layer weights, global weights, the recurrent-state
    /// arena, and the `fresh` flag (mirrors `RwkvState::fresh`).
    pub struct RwkvGpu {
        pub layers: Vec<RwkvGpuLayer>,
        pub tok_norm_w: PinnedBuffer,
        pub tok_norm_b: PinnedBuffer,
        pub output_norm_w: PinnedBuffer,
        pub output_norm_b: PinnedBuffer,
        pub lm_head: PinnedBuffer,
        pub arena: RwkvDecodeArena,
        pub fresh: bool,
    }

    impl RwkvGpu {
        /// Reset the recurrent state to zero and mark the sequence fresh.
        pub fn reset(&mut self) {
            self.arena.reset_state();
            self.fresh = true;
        }

        #[allow(clippy::too_many_arguments)]
        pub fn build(
            ctx: &MetalContext,
            cfg: &RwkvConfig,
            embed: &[f32],
            tok_norm_w: &[f32],
            tok_norm_b: &[f32],
            output_norm_w: &[f32],
            output_norm_b: &[f32],
            lm_head: &[f32],
            layers: &[RwkvLayer],
        ) -> Result<Self> {
            let _ = embed; // embed gather is a host memcpy of one row (see forward).
            let up = |v: &[f32]| ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(v));
            let up_opt = |v: &Option<Vec<f32>>| v.as_ref().map(|x| up(x));

            let gpu_layers = layers
                .iter()
                .map(|l| RwkvGpuLayer {
                    attn_norm_w: up(&l.attn_norm_w),
                    attn_norm_b: up(&l.attn_norm_b),
                    attn_norm2_w: up(&l.attn_norm2_w),
                    attn_norm2_b: up(&l.attn_norm2_b),
                    lerp_fused: up(&l.time_mix_lerp_fused),
                    channel_mix_lerp_k: up(&l.channel_mix_lerp_k),
                    receptance: up(&l.time_mix_receptance),
                    key: up(&l.time_mix_key),
                    value: up(&l.time_mix_value),
                    output: up(&l.time_mix_output),
                    w0: up(&l.time_mix_w0),
                    w1: up(&l.time_mix_w1),
                    w2: up(&l.time_mix_w2),
                    a0: up(&l.time_mix_a0),
                    a1: up(&l.time_mix_a1),
                    a2: up(&l.time_mix_a2),
                    v0: up(&l.time_mix_v0),
                    v1: up(&l.time_mix_v1),
                    v2: up(&l.time_mix_v2),
                    g1: up_opt(&l.time_mix_g1),
                    g2: up_opt(&l.time_mix_g2),
                    k_k: up(&l.time_mix_k_k),
                    k_a: up(&l.time_mix_k_a),
                    r_k: up(&l.time_mix_r_k),
                    ln_w: up(&l.time_mix_ln_w),
                    ln_b: up(&l.time_mix_ln_b),
                    channel_mix_key: up(&l.channel_mix_key),
                    channel_mix_value: up(&l.channel_mix_value),
                    has_gate: l.time_mix_g1.is_some() && l.time_mix_g2.is_some(),
                })
                .collect();

            let arena = RwkvDecodeArena::new(
                ctx,
                cfg.n_layer,
                cfg.n_embd,
                cfg.n_ff,
                cfg.head_size,
                cfg.head_count,
                cfg.vocab_size,
                cfg.decay_lora,
                cfg.iclr_lora,
                cfg.value_res_lora,
                cfg.gate_lora,
            );
            arena.reset_state();

            Ok(Self {
                layers: gpu_layers,
                tok_norm_w: up(tok_norm_w),
                tok_norm_b: up(tok_norm_b),
                output_norm_w: up(output_norm_w),
                output_norm_b: up(output_norm_b),
                lm_head: up(lm_head),
                arena,
                fresh: true,
            })
        }
    }

    /// Copy `n` f32 from a host slice into the head of a shared GPU buffer.
    fn write_row(buf: &PinnedBuffer, src: &[f32]) {
        let ptr = buf.contents() as *mut f32;
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), ptr, src.len()) };
    }

    /// Read `n` f32 from the head of a shared GPU buffer into a fresh Vec.
    fn read_vec(buf: &PinnedBuffer, n: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
    }

    /// One RWKV-7 decode step on the GPU. Mirrors `RwkvSeven::forward_token` +
    /// `time_mix` + `channel_mix` op-for-op; advances `g.arena` state in place.
    pub fn forward_token_gpu(
        ctx: &MetalContext,
        g: &mut RwkvGpu,
        cfg: &RwkvConfig,
        embed: &[f32],
        token: u32,
    ) -> Result<Vec<f32>> {
        let n = cfg.n_embd;
        let eps = cfg.ln_eps;
        let a = &g.arena;

        // Embedding lookup (host gather of one f32 row into the shared x buffer).
        let row = token as usize * n;
        if row + n > embed.len() {
            return Err(Error::Model(format!("rwkv7: token {token} out of vocab")));
        }
        write_row(&a.x, &embed[row..row + n]);

        let mut tcb = TokenCommandBuffer::new(ctx);

        // LN0 (embedding norm): x_norm = layernorm(x, tok_norm); x <- x_norm.
        rwkv7_layernorm_tcb(
            &mut tcb,
            &a.x,
            0,
            &g.tok_norm_w,
            &g.tok_norm_b,
            &a.x_norm,
            0,
            n,
            eps,
        )?;
        rwkv7_copy_tcb(&mut tcb, &a.x_norm, &a.x, 0, n)?;

        // v_first established on layer 0, reused by deeper layers.
        let mut have_v_first = false;
        // byte offset of slot `s` in the slot-major xs buffer.
        let f32b = std::mem::size_of::<f32>();
        let slot_off = |s: usize| s * n * f32b;

        for li in 0..cfg.n_layer {
            let layer = &g.layers[li];
            let shift_off = a.shift_layer_byte_offset(li);
            let wkv_off = a.wkv_layer_byte_offset(li);

            // ── time-mix ──
            // att_in = layernorm(x, attn_norm)
            rwkv7_layernorm_tcb(
                &mut tcb,
                &a.x,
                0,
                &layer.attn_norm_w,
                &layer.attn_norm_b,
                &a.att_in,
                0,
                n,
                eps,
            )?;
            // token-shift lerp using stored att_shift[li] (x_prev) → xs slot-major.
            let n_slots = if layer.has_gate { 6 } else { 5 };
            rwkv7_token_shift_lerp_tcb(
                &mut tcb,
                &a.att_in,
                &a.att_shift,
                shift_off,
                &layer.lerp_fused,
                &a.xs,
                n,
                n_slots,
                g.fresh,
            )?;

            // r = Wr @ xr   (slot 0)
            rwkv7_gemv_f32_off_tcb(&mut tcb, &layer.receptance, n, n, &a.xs, slot_off(0), &a.r)?;

            // w = exp(-0.606531 * sigmoid(w0 + W2 @ tanh(W1 @ xw)))  (slot 1)
            rwkv7_gemv_f32_off_tcb(
                &mut tcb,
                &layer.w1,
                cfg.decay_lora,
                n,
                &a.xs,
                slot_off(1),
                &a.w_lo,
            )?;
            rwkv7_tanh_inplace_tcb(&mut tcb, &a.w_lo, cfg.decay_lora)?;
            rwkv7_gemv_f32_off_tcb(&mut tcb, &layer.w2, n, cfg.decay_lora, &a.w_lo, 0, &a.w_raw)?;
            rwkv7_decay_act_tcb(&mut tcb, &a.w_raw, &layer.w0, &a.w, n)?;

            // k = Wk @ xk (slot 2) ; v = Wv @ xv (slot 3)
            rwkv7_gemv_f32_off_tcb(&mut tcb, &layer.key, n, n, &a.xs, slot_off(2), &a.k)?;
            rwkv7_gemv_f32_off_tcb(&mut tcb, &layer.value, n, n, &a.xs, slot_off(3), &a.v)?;

            // value-residual mix (skip on layer 0 where v_first is established).
            if !have_v_first {
                rwkv7_copy_tcb(&mut tcb, &a.v, &a.v_first, 0, n)?;
                have_v_first = true;
            } else {
                rwkv7_gemv_f32_off_tcb(
                    &mut tcb,
                    &layer.v1,
                    cfg.value_res_lora,
                    n,
                    &a.xs,
                    slot_off(3),
                    &a.v_lo,
                )?;
                rwkv7_gemv_f32_off_tcb(
                    &mut tcb,
                    &layer.v2,
                    n,
                    cfg.value_res_lora,
                    &a.v_lo,
                    0,
                    &a.v_mix,
                )?;
                rwkv7_value_residual_mix_tcb(&mut tcb, &a.v, &a.v_first, &a.v_mix, &layer.v0, n)?;
            }

            // gate g = G2 @ sigmoid(G1 @ xg)   (slot 5; only if gated)
            if layer.has_gate {
                let g1 = layer.g1.as_ref().unwrap();
                let g2 = layer.g2.as_ref().unwrap();
                rwkv7_gemv_f32_off_tcb(
                    &mut tcb,
                    g1,
                    cfg.gate_lora,
                    n,
                    &a.xs,
                    slot_off(5),
                    &a.g_lo,
                )?;
                rwkv7_sigmoid_inplace_tcb(&mut tcb, &a.g_lo, cfg.gate_lora)?;
                rwkv7_gemv_f32_off_tcb(&mut tcb, g2, n, cfg.gate_lora, &a.g_lo, 0, &a.gate)?;
            }

            // a = sigmoid(a0 + A2 @ (A1 @ xa))   (slot 4)
            rwkv7_gemv_f32_off_tcb(
                &mut tcb,
                &layer.a1,
                cfg.iclr_lora,
                n,
                &a.xs,
                slot_off(4),
                &a.a_lo,
            )?;
            rwkv7_gemv_f32_off_tcb(&mut tcb, &layer.a2, n, cfg.iclr_lora, &a.a_lo, 0, &a.a)?;
            rwkv7_sigmoid_bias_tcb(&mut tcb, &a.a, &layer.a0, n)?;

            // kk = l2norm_per_head(k*k_k); k += (a-1)*(k*k_a); a_op=-kk; b_op=kk*a.
            rwkv7_kk_kmix_tcb(
                &mut tcb,
                &a.k,
                &layer.k_k,
                &layer.k_a,
                &a.a,
                &a.a_op,
                &a.b_op,
                cfg.head_size,
                cfg.head_count,
            )?;

            // WKV-7 recurrence + per-head group-norm + bonus + gate (folded).
            rwkv7_wkv_decode_tcb(
                &mut tcb,
                &a.wkv_state,
                wkv_off,
                &a.r,
                &a.w,
                &a.k,
                &a.v,
                &a.a_op,
                &a.b_op,
                &layer.r_k,
                &layer.ln_w,
                &layer.ln_b,
                &a.gate,
                &a.out_wkv,
                cfg.head_size,
                cfg.head_count,
                GN_EPS,
                layer.has_gate,
            )?;

            // y = Wo @ out  → cur
            rwkv7_gemv_f32_off_tcb(&mut tcb, &layer.output, n, n, &a.out_wkv, 0, &a.cur)?;

            // store att_in as next token-shift for this layer.
            rwkv7_copy_tcb(&mut tcb, &a.att_in, &a.att_shift, shift_off, n)?;

            // ffn_inp = cur + x
            rwkv7_add_into_tcb(&mut tcb, &a.cur, &a.x, &a.ffn_inp, n)?;

            // ── channel-mix ──
            // ffn_in = layernorm(ffn_inp, attn_norm_2)
            rwkv7_layernorm_tcb(
                &mut tcb,
                &a.ffn_inp,
                0,
                &layer.attn_norm2_w,
                &layer.attn_norm2_b,
                &a.ffn_in,
                0,
                n,
                eps,
            )?;
            // xk = ffn_in + (x_prev - ffn_in) * lerp_k  (per-layer ffn_shift[li]).
            rwkv7_channel_mix_shift_tcb(
                &mut tcb,
                &a.ffn_in,
                &a.ffn_shift,
                shift_off,
                &layer.channel_mix_lerp_k,
                &a.xk_ffn,
                n,
                g.fresh,
            )?;
            // k = relu(Wk @ xk)^2
            rwkv7_gemv_f32_off_tcb(
                &mut tcb,
                &layer.channel_mix_key,
                cfg.n_ff,
                n,
                &a.xk_ffn,
                0,
                &a.ffn_k,
            )?;
            rwkv7_relu_sq_inplace_tcb(&mut tcb, &a.ffn_k, cfg.n_ff)?;
            // cmix = Wv @ k
            rwkv7_gemv_f32_off_tcb(
                &mut tcb,
                &layer.channel_mix_value,
                n,
                cfg.n_ff,
                &a.ffn_k,
                0,
                &a.cmix,
            )?;
            // store ffn_in as next token-shift.
            rwkv7_copy_tcb(&mut tcb, &a.ffn_in, &a.ffn_shift, shift_off, n)?;

            // x = cmix + ffn_inp
            rwkv7_add_into_tcb(&mut tcb, &a.cmix, &a.ffn_inp, &a.x, n)?;
        }

        // final norm + LM head.
        rwkv7_layernorm_tcb(
            &mut tcb,
            &a.x,
            0,
            &g.output_norm_w,
            &g.output_norm_b,
            &a.x_norm,
            0,
            n,
            eps,
        )?;
        rwkv7_gemv_f32_off_tcb(
            &mut tcb,
            &g.lm_head,
            cfg.vocab_size,
            n,
            &a.x_norm,
            0,
            &a.logits,
        )?;

        tcb.commit_and_wait()?;
        g.fresh = false;
        Ok(read_vec(&a.logits, cfg.vocab_size))
    }
}
