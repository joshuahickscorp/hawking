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

/// B INDEPENDENT recurrent states for the continuous-batch (multi-seq) decode —
/// the RWKV-7 analogue of B per-slot KV caches in the Qwen multiseq path. Each
/// slot carries its own `RwkvState` (per-head S matrices + token-shift planes +
/// `fresh`); decode advances all B in one pass while every projection/LM-head
/// weight is read ONCE across the B activation columns (the bandwidth win).
///
/// State is per-stream and never shared; only the weights are. A slot can be
/// reset independently (`reset_slot`) so a finished stream's slot can be reused
/// by a new sequence without disturbing the others (continuous batching).
pub struct RwkvMultiState {
    pub slots: Vec<RwkvState>,
}

impl RwkvMultiState {
    /// `b` fresh (zeroed) states sized for `cfg`.
    pub fn new(cfg: &RwkvConfig, b: usize) -> Self {
        Self {
            slots: (0..b).map(|_| RwkvState::new(cfg)).collect(),
        }
    }

    /// Number of streams.
    pub fn batch(&self) -> usize {
        self.slots.len()
    }

    /// Reset all slots to the zero state (start of B fresh sequences).
    pub fn reset(&mut self) {
        for s in &mut self.slots {
            s.reset();
        }
    }

    /// Reset a single slot (its stream finished; reuse it for a new sequence).
    pub fn reset_slot(&mut self, slot: usize) {
        self.slots[slot].reset();
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
        // Single-stream decode is exactly the per-stream core advancing the
        // engine's own state. The multi-stream path (`forward_tokens_multiseq_cpu`)
        // calls the SAME core per stream with each stream's state, so a B-stream
        // batch is bit-for-bit B independent `forward_token` runs.
        let RwkvSeven {
            config,
            layers,
            embed,
            tok_norm_w,
            tok_norm_b,
            output_norm_w,
            output_norm_b,
            output,
            state,
            ..
        } = self;
        Self::forward_token_core(
            config,
            layers,
            embed,
            tok_norm_w,
            tok_norm_b,
            output_norm_w,
            output_norm_b,
            output,
            state,
            token,
        )
    }

    /// Per-stream RWKV-7 forward core: advances `state` (any `RwkvState`) by one
    /// token and returns the `vocab` logit row. All weight tensors are borrowed
    /// read-only so MANY streams can share one weight set while each owns its
    /// recurrent state — the CPU oracle for the multiseq continuous-batch path.
    #[allow(clippy::too_many_arguments)]
    fn forward_token_core(
        config: &RwkvConfig,
        layers: &[RwkvLayer],
        embed: &[f32],
        tok_norm_w: &[f32],
        tok_norm_b: &[f32],
        output_norm_w: &[f32],
        output_norm_b: &[f32],
        output: &[f32],
        state: &mut RwkvState,
        token: u32,
    ) -> Result<Vec<f32>> {
        let n = config.n_embd;
        let n_layer = config.n_layer;
        let vocab_size = config.vocab_size;
        let eps = config.ln_eps;

        // Embedding lookup + LN0.
        let row = token as usize * n;
        if row + n > embed.len() {
            return Err(Error::Model(format!("rwkv7: token {token} out of vocab")));
        }
        let mut x = embed[row..row + n].to_vec();
        {
            let mut tmp = vec![0.0f32; n];
            layernorm(&x, tok_norm_w, tok_norm_b, eps, &mut tmp);
            x = tmp;
        }

        // `v_first`: the value projection of layer 0, reused as the residual
        // target for the value-residual mix in deeper layers.
        let mut v_first: Vec<f32> = Vec::new();

        for li in 0..n_layer {
            // ---- token-shift for the time-mix branch ----
            let mut att_in = vec![0.0f32; n];
            {
                let layer = &layers[li];
                layernorm(&x, &layer.attn_norm_w, &layer.attn_norm_b, eps, &mut att_in);
            }
            // x_prev = previous token's att_in (zero on the first token).
            let cur = Self::time_mix_with_state(config, layers, state, li, &att_in, &mut v_first)?;
            // store att_in as the next token-shift for this layer.
            state.att_shift[li].copy_from_slice(&att_in);

            // residual: ffn_inp = cur + x
            let mut ffn_inp = x.clone();
            for i in 0..n {
                ffn_inp[i] += cur[i];
            }

            // ---- channel-mix branch ----
            let mut ffn_in = vec![0.0f32; n];
            {
                let layer = &layers[li];
                layernorm(
                    &ffn_inp,
                    &layer.attn_norm2_w,
                    &layer.attn_norm2_b,
                    eps,
                    &mut ffn_in,
                );
            }
            let cmix = Self::channel_mix_with_state(config, layers, state, li, &ffn_in)?;
            state.ffn_shift[li].copy_from_slice(&ffn_in);

            // residual: x = cmix + ffn_inp
            x = ffn_inp;
            for i in 0..n {
                x[i] += cmix[i];
            }
        }

        // final norm + LM head.
        let mut x_norm = vec![0.0f32; n];
        layernorm(&x, output_norm_w, output_norm_b, eps, &mut x_norm);
        let mut logits = vec![0.0f32; vocab_size];
        gemv_f32(output, vocab_size, n, &x_norm, &mut logits);

        state.fresh = false;
        Ok(logits)
    }

    /// CPU multi-stream decode: advance B streams by one token each against the
    /// shared weights, returning B `vocab`-sized logit rows (slot order matches
    /// `tokens`). `tokens[b]` feeds slot `b`'s `multi.slots[b]` state.
    ///
    /// This is the CORRECTNESS ORACLE for the continuous-batch path: it is, by
    /// construction, B independent `forward_token` runs that happen to share the
    /// weight set — so it is bit-for-bit identical to decoding each stream alone.
    /// The GPU path (`forward_token_gpu_multiseq`) reproduces these exact logits
    /// while reading each weight ONCE across the B columns; this method is what
    /// the multiseq parity gate diffs against.
    pub fn forward_tokens_multiseq_cpu(
        &mut self,
        tokens: &[u32],
        multi: &mut RwkvMultiState,
    ) -> Result<Vec<Vec<f32>>> {
        if tokens.len() != multi.batch() {
            return Err(Error::Model(format!(
                "rwkv7 multiseq: tokens={} != states={}",
                tokens.len(),
                multi.batch()
            )));
        }
        let RwkvSeven {
            config,
            layers,
            embed,
            tok_norm_w,
            tok_norm_b,
            output_norm_w,
            output_norm_b,
            output,
            ..
        } = self;
        let mut out = Vec::with_capacity(tokens.len());
        for (b, &tok) in tokens.iter().enumerate() {
            out.push(Self::forward_token_core(
                config,
                layers,
                embed,
                tok_norm_w,
                tok_norm_b,
                output_norm_w,
                output_norm_b,
                output,
                &mut multi.slots[b],
                tok,
            )?);
        }
        Ok(out)
    }

    /// CPU multi-stream greedy step: [`forward_tokens_multiseq_cpu`] followed by
    /// a per-slot argmax — the slot-major decode token for each stream. Mirrors
    /// the Qwen `forward_tokens_multiseq_greedy` shape so the bench/serve seam is
    /// uniform across architectures.
    pub fn forward_tokens_multiseq_greedy_cpu(
        &mut self,
        tokens: &[u32],
        multi: &mut RwkvMultiState,
    ) -> Result<Vec<u32>> {
        let logits = self.forward_tokens_multiseq_cpu(tokens, multi)?;
        Ok(logits
            .iter()
            .map(|row| {
                let mut bi = 0u32;
                let mut bv = f32::NEG_INFINITY;
                for (i, &x) in row.iter().enumerate() {
                    if x > bv {
                        bv = x;
                        bi = i as u32;
                    }
                }
                bi
            })
            .collect())
    }

    /// Time-mix: token-shift lerp → r/w/k/v/a/g projections (+ LoRA paths) →
    /// WKV-7 recurrence → group-norm → r·k·r_k bonus → gate → output proj.
    ///
    /// Operates on an EXPLICIT recurrent `state` (not necessarily `self.state`),
    /// so the multi-stream path can advance B independent states with the SAME
    /// op order — single-stream `forward_token` passes `&mut self.state`, the
    /// reference oracle for the multiseq parity gate passes each stream's state.
    /// `config`/`layers` are borrowed read-only, disjoint from `state`.
    fn time_mix_with_state(
        config: &RwkvConfig,
        layers: &[RwkvLayer],
        state: &mut RwkvState,
        li: usize,
        att_in: &[f32],
        v_first: &mut Vec<f32>,
    ) -> Result<Vec<f32>> {
        let cfg = config;
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
    ///
    /// Reads the channel-mix token-shift from an EXPLICIT `state` (mirrors
    /// `time_mix_with_state`) so B streams share the weights but carry their own
    /// `ffn_shift`. The `fresh` flag is read off the same `state`.
    fn channel_mix_with_state(
        config: &RwkvConfig,
        layers: &[RwkvLayer],
        state: &RwkvState,
        li: usize,
        ffn_in: &[f32],
    ) -> Result<Vec<f32>> {
        let cfg = config;
        let n = cfg.n_embd;
        let layer = &layers[li];

        let x_prev = &state.ffn_shift[li];
        // xk = ffn_in + (x_prev - ffn_in) * lerp_k
        let mut xk = vec![0.0f32; n];
        if state.fresh {
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
                            &gguf,
                            &embed,
                            &tok_norm_w,
                            &tok_norm_b,
                            &output_norm_w,
                            &output_norm_b,
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

    /// Metal compute dispatches encoded by the most recent `forward_token_gpu`
    /// step (0 before the first GPU step). Used by the bench/probe to track the
    /// per-step dispatch count — the headline figure for the LoRA-fusion lever.
    pub fn last_gpu_dispatch_count(&self) -> usize {
        self.gpu.as_ref().map(|g| g.last_dispatch_count).unwrap_or(0)
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

    /// Ensure the GPU decode bundle is sized for `batch` independent streams,
    /// rebuilding (re-uploading weights into a B-stream arena) only when the
    /// current bundle's batch differs. A rebuild starts all B streams fresh
    /// (zeroed state). No-op when the GPU is unavailable. Returns whether the GPU
    /// path is active.
    pub fn ensure_gpu_batch(&mut self, batch: usize) -> Result<bool> {
        if self.metal_ctx.is_none() || self.gpu.is_none() {
            return Ok(false);
        }
        let need = batch.max(1);
        if self.gpu.as_ref().unwrap().arena.batch == need {
            return Ok(true);
        }
        let ctx = self.metal_ctx.as_ref().unwrap();
        let rebuilt = gpu::RwkvGpu::build_with_batch(
            ctx,
            &self.config,
            &self.gguf,
            &self.embed,
            &self.tok_norm_w,
            &self.tok_norm_b,
            &self.output_norm_w,
            &self.output_norm_b,
            &self.layers,
            need,
        )?;
        self.gpu = Some(rebuilt);
        Ok(true)
    }

    /// Reset all B streams of the GPU multiseq bundle to the zero state (start of
    /// B fresh sequences). No-op when the GPU is unavailable.
    pub fn reset_gpu_multiseq(&mut self) {
        if let Some(g) = self.gpu.as_mut() {
            g.reset();
        }
    }

    /// Reset ONE stream's GPU recurrent state (its sequence finished; reuse the
    /// slot for a new sequence — the continuous-batch reuse path).
    pub fn reset_gpu_slot(&mut self, slot: usize) {
        if let Some(g) = self.gpu.as_mut() {
            g.arena.reset_slot(slot);
        }
    }

    /// B-STREAM (continuous-batch) GPU decode: advance B independent streams by
    /// one token each in ONE pass, returning B `vocab`-sized logit rows (slot
    /// order matches `tokens`). Sizes the GPU bundle for `tokens.len()` streams on
    /// first use (or batch change). Falls back to B sequential CPU `forward_token`
    /// runs (over independent states) when the GPU is unavailable — still correct,
    /// just no bandwidth win.
    ///
    /// This is the GPU realization of the CPU oracle
    /// [`forward_tokens_multiseq_cpu`]: by construction the two agree
    /// stream-for-stream (within f32 reduction tolerance), which the multiseq
    /// parity gate checks.
    pub fn forward_token_gpu_multiseq(&mut self, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        if !self.ensure_gpu_batch(tokens.len())? {
            // CPU fallback: B independent fresh-less single-stream forwards. The
            // caller owns sequencing/state via the GPU bundle on the fast path, so
            // here we advance the engine's own state per stream is NOT valid;
            // instead require the GPU for the batched path and surface clearly.
            return Err(Error::Model(
                "rwkv7 forward_token_gpu_multiseq: GPU unavailable (use the CPU \
                 multiseq oracle forward_tokens_multiseq_cpu with an explicit \
                 RwkvMultiState)"
                    .into(),
            ));
        }
        let RwkvSeven {
            config,
            metal_ctx,
            gpu,
            embed,
            ..
        } = self;
        let ctx = metal_ctx.as_ref().unwrap();
        let g = gpu.as_mut().unwrap();
        gpu::forward_token_gpu_multiseq(ctx, g, config, embed, tokens)
    }
}

/// RWKV-7 GPU decode path: weight upload, the recurrent-state arena, and the
/// single-step forward that drives the WKV-7 kernel + glue kernels. macOS-only.
#[cfg(target_os = "macos")]
pub mod gpu {
    use super::{RwkvConfig, RwkvLayer};
    use crate::gguf::{GgmlType, GgufFile};
    use crate::kernels::{
        gemm_q4_k_m_batched_v3w_predec_xoff_pinned_tcb, gemv_q4_k_v4_predec_xoff_pinned_tcb,
        gemv_q6_k_pinned_off_tcb, gemv_q6_k_pinned_tcb, predecode_q4_k_scale_table,
    };
    use crate::metal::rwkv_decode_arena::{
        rwkv7_add_into_flat_tcb, rwkv7_add_into_tcb, rwkv7_channel_mix_shift_multiseq_tcb,
        rwkv7_channel_mix_shift_tcb, rwkv7_copy_tcb, rwkv7_decay_act_multiseq_tcb,
        rwkv7_decay_act_tcb, rwkv7_gemv_f32_off_tcb, rwkv7_gemv_f32_xoff_yoff_tcb,
        rwkv7_kk_kmix_multiseq_tcb, rwkv7_kk_kmix_tcb, rwkv7_layernorm_multiseq_tcb,
        rwkv7_layernorm_tcb, rwkv7_lora_grouped_gemv_tcb, rwkv7_lora_mid_act_tcb,
        rwkv7_relu_sq_inplace_tcb, rwkv7_shift_writeback_multiseq_tcb,
        rwkv7_sigmoid_bias_multiseq_tcb, rwkv7_sigmoid_bias_tcb, rwkv7_sigmoid_inplace_tcb,
        rwkv7_tanh_inplace_tcb, rwkv7_token_shift_lerp_multiseq_tcb, rwkv7_token_shift_lerp_tcb,
        rwkv7_value_residual_mix_multiseq_tcb, rwkv7_value_residual_mix_tcb,
        rwkv7_wkv_decode_multiseq_tcb, rwkv7_wkv_decode_tcb,
    };
    use crate::metal::{MetalContext, PinnedBuffer, RwkvDecodeArena, TokenCommandBuffer};
    use crate::{Error, Result};

    /// Group-norm epsilon used inside the WKV-7 per-head norm (matches the CPU
    /// reference `time_mix`: `gn_eps = 64e-5`).
    const GN_EPS: f32 = 64e-5;

    /// One large projection weight, GPU-resident in whatever precision the GGUF
    /// stored it. The RWKV-7 World GGUFs ship the six big per-layer projections
    /// (r/k/v/o time-mix + channel-mix key/value) as Q4_K and the LM head as
    /// Q6_K; streaming those QUANTIZED (4× / 2.46× fewer bytes than f32) is the
    /// headline decode-bandwidth lever. An all-F32 GGUF (used by the exact-parity
    /// gate) keeps the bit-identical f32 GEMV path. `rows = out`, `cols = in`,
    /// matching the row-major `[out,in]` ggml layout the GEMV kernels expect.
    pub enum ProjWeight {
        /// Q4_K bytes + the pre-decoded (ds, dm) f32 scale sidecar.
        Q4k {
            q4: PinnedBuffer,
            scales: PinnedBuffer,
            rows: usize,
            cols: usize,
        },
        /// Q6_K bytes (decoded inline by `gemm_q6_k_fused_v2`).
        Q6k {
            q6: PinnedBuffer,
            rows: usize,
            cols: usize,
        },
        /// f32-dequantized fallback (F32/F16/BF16 tensors, or any non-Q4_K/Q6_K).
        F32 {
            w: PinnedBuffer,
            rows: usize,
            cols: usize,
        },
    }

    impl ProjWeight {
        /// Build a `ProjWeight` for tensor `name`, serving it quantized when the
        /// GGUF stores Q4_K/Q6_K (the fast path) and falling back to an
        /// f32-dequant upload otherwise. `rows`/`cols` are the logical GEMV dims
        /// (`rows = out`, `cols = in`); both quant kernels require `cols % 256 == 0`.
        fn build(ctx: &MetalContext, g: &GgufFile, name: &str, rows: usize, cols: usize) -> Self {
            let info = g
                .tensor(name)
                .unwrap_or_else(|| panic!("rwkv7 gpu: missing tensor {name}"));
            let bytes = g.tensor_bytes(name).unwrap();
            // Quant kernels need cols (the reduction dim) block-aligned; every
            // RWKV-7 projection has cols ∈ {768,1024,4096} (all %256==0), but
            // guard so a non-aligned tensor cleanly takes the f32 path.
            let aligned = cols % 256 == 0;
            match info.dtype {
                GgmlType::Q4_K if aligned => {
                    let scales = predecode_q4_k_scale_table(bytes);
                    ProjWeight::Q4k {
                        q4: ctx.new_buffer_with_bytes(bytes),
                        scales: ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&scales)),
                        rows,
                        cols,
                    }
                }
                GgmlType::Q6_K if aligned => ProjWeight::Q6k {
                    q6: ctx.new_buffer_with_bytes(bytes),
                    rows,
                    cols,
                },
                _ => {
                    // F32/F16/BF16 (or unaligned): dequantize to f32 and serve via
                    // the bit-identical f32 GEMV (preserves the exact-parity gate
                    // on all-F32 GGUFs).
                    let w = crate::model::weights::dequant_f32(g, name)
                        .unwrap_or_else(|e| panic!("rwkv7 gpu: dequant {name}: {e}"));
                    ProjWeight::F32 {
                        w: ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&w)),
                        rows,
                        cols,
                    }
                }
            }
        }

        /// GEMV reading the activation at byte offset `x_off` into `x`, writing
        /// `out` at offset 0. Dispatches the quantized kernel for Q4_K/Q6_K, else
        /// the f32 kernel — all numerically equal to the CPU reference's
        /// `gemv_f32` over the same dequantized weights (the Q4_K/Q6_K inline
        /// decode reproduces the CPU dequant value-for-value, so only GEMV
        /// reduction order differs).
        fn gemv(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            x: &PinnedBuffer,
            x_off: usize,
            out: &PinnedBuffer,
        ) -> Result<()> {
            match self {
                ProjWeight::Q4k {
                    q4,
                    scales,
                    rows,
                    cols,
                } => gemv_q4_k_v4_predec_xoff_pinned_tcb(
                    tcb,
                    q4,
                    0,
                    q4.length() as usize,
                    scales,
                    0,
                    *rows,
                    *cols,
                    x,
                    x_off,
                    out,
                ),
                ProjWeight::Q6k { q6, rows, cols } => {
                    // The Q6_K kernel reads x from offset 0; the only Q6_K user
                    // (the LM head) reads x_norm at offset 0, so assert rather
                    // than silently read the wrong slice.
                    debug_assert_eq!(x_off, 0, "Q6_K GEMV has no x-offset variant");
                    gemv_q6_k_pinned_tcb(tcb, q6, 0, q6.length() as usize, *rows, *cols, x, out)
                }
                ProjWeight::F32 { w, rows, cols } => {
                    rwkv7_gemv_f32_off_tcb(tcb, w, *rows, *cols, x, x_off, out)
                }
            }
        }

        /// B-stream projection: y_batch (B, rows) = W · x_batch (B, cols).
        ///
        /// `x` must hold a CONTIGUOUS (B, cols) f32 block whose first element is at
        /// byte offset `x_off` (so `x[x_off ..]` is exactly the (B, cols) the
        /// batched GEMM reads); `out` is written as a contiguous (B, rows) block at
        /// offset 0. For Q4_K this is ONE `gemm_q4_k_m_batched_v3w_predec` dispatch
        /// — the weight is read once across the B columns (the bandwidth win). The
        /// Q6_K LM head and the f32 LoRA/fallback projections have no batched
        /// kernel, so they loop B single-vector GEMVs over the SAME resident weight
        /// buffer (still one upload; per-stream x/out offsets `bi*cols` / `bi*rows`).
        /// Every path is numerically equal to `gemv` per stream, so the B-stream
        /// result is bit-for-bit B independent single-stream projections.
        fn gemv_batched(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            x: &PinnedBuffer,
            x_off: usize,
            out: &PinnedBuffer,
            batch: usize,
        ) -> Result<()> {
            let f = std::mem::size_of::<f32>();
            match self {
                ProjWeight::Q4k {
                    q4,
                    scales,
                    rows,
                    cols,
                } => {
                    // ONE dispatch reads the weight across the B columns (the BW
                    // win). The kernel reads its (B, cols) input starting at the
                    // byte offset `x_off` (the slot block in the (slot,B,n) xs
                    // buffer) and writes (B, rows) at offset 0. The dispatcher caps
                    // batch at 8 and validates the contiguous-stride contract.
                    gemm_q4_k_m_batched_v3w_predec_xoff_pinned_tcb(
                        tcb,
                        q4,
                        0,
                        q4.length() as usize,
                        scales,
                        0,
                        *rows,
                        *cols,
                        batch,
                        x,
                        x_off,
                        out,
                    )
                }
                ProjWeight::Q6k { q6, rows, cols } => {
                    for bi in 0..batch {
                        gemv_q6_k_pinned_off_tcb(
                            tcb,
                            q6,
                            0,
                            q6.length() as usize,
                            *rows,
                            *cols,
                            x,
                            x_off + bi * *cols * f,
                            out,
                            bi * *rows * f,
                        )?;
                    }
                    Ok(())
                }
                ProjWeight::F32 { w, rows, cols } => {
                    for bi in 0..batch {
                        rwkv7_gemv_f32_xoff_yoff_tcb(
                            tcb,
                            w,
                            *rows,
                            *cols,
                            x,
                            x_off + bi * *cols * f,
                            out,
                            bi * *rows * f,
                        )?;
                    }
                    Ok(())
                }
            }
        }
    }

    /// GPU-resident weights for one RWKV-7 layer. The six big projections
    /// (r/k/v/o time-mix + channel-mix key/value) are served from their NATIVE
    /// GGUF precision via [`ProjWeight`] — Q4_K on the shipped World models — so
    /// decode streams ~4× fewer projection bytes than the old f32-dequant upload.
    /// The small norm/lerp/vector tensors and the LoRA matrices (F32/F16 in the
    /// GGUF) stay as f32 `PinnedBuffer`s; they are tiny and feed the
    /// `gemv_f32_attn` / glue kernels unchanged. All projection matrices are
    /// row-major `[out,in]`, matching what the GEMV kernels expect.
    pub struct RwkvGpuLayer {
        pub attn_norm_w: PinnedBuffer,
        pub attn_norm_b: PinnedBuffer,
        pub attn_norm2_w: PinnedBuffer,
        pub attn_norm2_b: PinnedBuffer,
        pub lerp_fused: PinnedBuffer,
        pub channel_mix_lerp_k: PinnedBuffer,
        pub receptance: ProjWeight,
        pub key: ProjWeight,
        pub value: ProjWeight,
        pub output: ProjWeight,
        pub w0: PinnedBuffer,
        pub a0: PinnedBuffer,
        pub v0: PinnedBuffer,
        /// LoRA-fusion: the down-projections (W1) stacked row-major into one
        /// buffer `[w1 | a1 | v1 (| g1)]` (each `rank_i × n`), and the
        /// up-projections (W2) stacked `[w2 | a2 | v2 (| g2)]` (each
        /// `n × rank_i`), in the group order w,a,v,g. The gate group is included
        /// iff `has_gate`. Fed to `rwkv7_lora_grouped_gemv` so the (up to) eight
        /// tiny LoRA GEMVs collapse to two batched dispatches. The per-LoRA W1/W2
        /// are no longer uploaded separately — these stacks are their sole GPU
        /// home.
        pub lora_w1_stacked: PinnedBuffer,
        pub lora_w2_stacked: PinnedBuffer,
        pub k_k: PinnedBuffer,
        pub k_a: PinnedBuffer,
        pub r_k: PinnedBuffer,
        pub ln_w: PinnedBuffer,
        pub ln_b: PinnedBuffer,
        pub channel_mix_key: ProjWeight,
        pub channel_mix_value: ProjWeight,
        pub has_gate: bool,
    }

    /// GPU decode bundle: per-layer weights, global weights, the recurrent-state
    /// arena, and the `fresh` flag (mirrors `RwkvState::fresh`). The LM head is a
    /// [`ProjWeight`] too — Q6_K on the shipped models (served inline, ~2.46× the
    /// f32 bandwidth saving on the single largest GEMV of the step).
    pub struct RwkvGpu {
        pub layers: Vec<RwkvGpuLayer>,
        pub tok_norm_w: PinnedBuffer,
        pub tok_norm_b: PinnedBuffer,
        pub output_norm_w: PinnedBuffer,
        pub output_norm_b: PinnedBuffer,
        pub lm_head: ProjWeight,
        pub arena: RwkvDecodeArena,
        pub fresh: bool,
        /// Number of Metal compute dispatches encoded by the most recent
        /// `forward_token_gpu` step. Set just before `commit_and_wait`; read by
        /// the bench/probe to track the per-step dispatch count (the LoRA-fusion
        /// headline). Not used by the forward math.
        pub last_dispatch_count: usize,
        /// LoRA-fusion group tables (`RwkvLoraGroup[ngroups]`), built ONCE — their
        /// offsets depend only on the (per-model constant) ranks/n/has_gate, so
        /// the same two buffers serve every layer. `lora_down_table` drives the
        /// stacked W1 GEMV (each group reads a different xs slot, all cols=n);
        /// `lora_up_table` drives the stacked W2 GEMV (rows=n, cols=rank_i).
        pub lora_down_table: PinnedBuffer,
        pub lora_up_table: PinnedBuffer,
        /// Number of LoRA groups (3 without gate, 4 with) and the total stacked
        /// down-output rows (== sum of ranks) and up-output rows (== ngroups * n).
        pub lora_groups: usize,
        pub lora_down_rows: usize,
        pub lora_up_rows: usize,
    }

    /// Host mirror of the shader `RwkvLoraGroup` (5 × u32, tightly packed).
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct LoraGroup {
        row_start: u32,
        w_off: u32,
        x_off: u32,
        out_off: u32,
        cols: u32,
    }
    unsafe impl bytemuck::Zeroable for LoraGroup {}
    unsafe impl bytemuck::Pod for LoraGroup {}

    impl RwkvGpu {
        /// Reset the recurrent state to zero and mark the sequence fresh.
        pub fn reset(&mut self) {
            self.arena.reset_state();
            self.fresh = true;
        }

        /// Build the GPU decode bundle. The six big projections + LM head are
        /// read in their NATIVE precision straight from the GGUF (`g`) via
        /// [`ProjWeight::build`] (Q4_K/Q6_K fast path; f32 fallback on all-F32
        /// GGUFs). The small norm/lerp/vector + LoRA tensors come from the
        /// already-dequantized CPU `layers` (they are F32/F16 in the GGUF and
        /// tiny, so f32 on the GPU is both correct and cheap). `g` MUST be the
        /// same GGUF the CPU `layers` were loaded from.
        #[allow(clippy::too_many_arguments)]
        pub fn build(
            ctx: &MetalContext,
            cfg: &RwkvConfig,
            g: &GgufFile,
            embed: &[f32],
            tok_norm_w: &[f32],
            tok_norm_b: &[f32],
            output_norm_w: &[f32],
            output_norm_b: &[f32],
            layers: &[RwkvLayer],
        ) -> Result<Self> {
            Self::build_with_batch(
                ctx,
                cfg,
                g,
                embed,
                tok_norm_w,
                tok_norm_b,
                output_norm_w,
                output_norm_b,
                layers,
                1,
            )
        }

        /// Build the GPU decode bundle sized for `batch` INDEPENDENT streams (the
        /// continuous-batch path). Identical weight upload to [`build`]; only the
        /// arena scales — every per-token scratch + the three state planes are
        /// sized for B streams (`RwkvDecodeArena::new_with_batch`). `batch == 1`
        /// is exactly the single-stream bundle.
        #[allow(clippy::too_many_arguments)]
        pub fn build_with_batch(
            ctx: &MetalContext,
            cfg: &RwkvConfig,
            g: &GgufFile,
            embed: &[f32],
            tok_norm_w: &[f32],
            tok_norm_b: &[f32],
            output_norm_w: &[f32],
            output_norm_b: &[f32],
            layers: &[RwkvLayer],
            batch: usize,
        ) -> Result<Self> {
            let _ = embed; // embed gather is a host memcpy of one row (see forward).
            let up = |v: &[f32]| ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(v));
            let n = cfg.n_embd;
            let n_ff = cfg.n_ff;

            // Stack the LoRA down-(W1) and up-(W2) projections of one layer into
            // two contiguous buffers in group order w,a,v,(g). W1 rows are
            // `rank_i × n`; W2 rows are `n × rank_i`. Concatenating the row-major
            // f32 slices reproduces the exact per-LoRA layout the grouped-GEMV
            // kernel indexes by `(w_off + local_row*cols)`. Gate is appended iff
            // present (the shipped World models always have it).
            //
            // NB: layer 0's value-residual `v1`/`v2` are a dummy copy of the iclr
            // `a` tensors (iclr_lora rows, not value_res_lora) and are unused; the
            // CPU reference truncates them to `value_res_lora` via the GEMV dims.
            // We MUST take exactly `value_res_lora` rows here too (leading
            // `rows*n` floats, since W1 is `[rank, n]` row-major), otherwise the
            // gate block's offset in the stack shifts and the grouped-GEMV reads
            // the wrong gate weights.
            let stack_w1 = |l: &RwkvLayer| -> PinnedBuffer {
                let mut v =
                    Vec::with_capacity((cfg.decay_lora + cfg.iclr_lora + cfg.value_res_lora) * n);
                v.extend_from_slice(&l.time_mix_w1[..cfg.decay_lora * n]);
                v.extend_from_slice(&l.time_mix_a1[..cfg.iclr_lora * n]);
                v.extend_from_slice(&l.time_mix_v1[..cfg.value_res_lora * n]);
                if let Some(g1) = &l.time_mix_g1 {
                    v.extend_from_slice(&g1[..cfg.gate_lora * n]);
                }
                up(&v)
            };
            // W2 is `[n, rank]` row-major (n*rank floats). The CPU reference reads
            // v2 with `gemv_f32(v2, n, value_res_lora, ..)`, i.e. it reinterprets
            // the leading `n*value_res_lora` flat floats as `[n, value_res_lora]`
            // — true for both the canonical layers and the dummy layer-0 v2
            // (`[n, iclr]`). Mirroring that flat truncation keeps the stack
            // bit-identical to the CPU and keeps the gate block at the right
            // offset.
            let stack_w2 = |l: &RwkvLayer| -> PinnedBuffer {
                let mut v =
                    Vec::with_capacity((cfg.decay_lora + cfg.iclr_lora + cfg.value_res_lora) * n);
                v.extend_from_slice(&l.time_mix_w2[..n * cfg.decay_lora]);
                v.extend_from_slice(&l.time_mix_a2[..n * cfg.iclr_lora]);
                v.extend_from_slice(&l.time_mix_v2[..n * cfg.value_res_lora]);
                if let Some(g2) = &l.time_mix_g2 {
                    v.extend_from_slice(&g2[..n * cfg.gate_lora]);
                }
                up(&v)
            };

            let gpu_layers = layers
                .iter()
                .enumerate()
                .map(|(li, l)| {
                    let p = |suf: &str| format!("blk.{li}.{suf}");
                    // Projection dims are (rows = out, cols = in).
                    let proj = |suf: &str, rows: usize, cols: usize| {
                        ProjWeight::build(ctx, g, &p(suf), rows, cols)
                    };
                    RwkvGpuLayer {
                        attn_norm_w: up(&l.attn_norm_w),
                        attn_norm_b: up(&l.attn_norm_b),
                        attn_norm2_w: up(&l.attn_norm2_w),
                        attn_norm2_b: up(&l.attn_norm2_b),
                        lerp_fused: up(&l.time_mix_lerp_fused),
                        channel_mix_lerp_k: up(&l.channel_mix_lerp_k),
                        receptance: proj("time_mix_receptance.weight", n, n),
                        key: proj("time_mix_key.weight", n, n),
                        value: proj("time_mix_value.weight", n, n),
                        output: proj("time_mix_output.weight", n, n),
                        w0: up(&l.time_mix_w0),
                        a0: up(&l.time_mix_a0),
                        v0: up(&l.time_mix_v0),
                        lora_w1_stacked: stack_w1(l),
                        lora_w2_stacked: stack_w2(l),
                        k_k: up(&l.time_mix_k_k),
                        k_a: up(&l.time_mix_k_a),
                        r_k: up(&l.time_mix_r_k),
                        ln_w: up(&l.time_mix_ln_w),
                        ln_b: up(&l.time_mix_ln_b),
                        channel_mix_key: proj("channel_mix_key.weight", n_ff, n),
                        channel_mix_value: proj("channel_mix_value.weight", n, n_ff),
                        has_gate: l.time_mix_g1.is_some() && l.time_mix_g2.is_some(),
                    }
                })
                .collect();

            let arena = RwkvDecodeArena::new_with_batch(
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
                batch,
            );
            arena.reset_state();

            // LM head from its native GGUF precision (Q6_K on the World models).
            let lm_head = ProjWeight::build(ctx, g, "output.weight", cfg.vocab_size, cfg.n_embd);

            // ── LoRA-fusion group tables (built once; constant offsets) ──
            // Group order w,a,v,(g). Ranks and the xs slot each group's input
            // lives in (time-mix slots r=0,w=1,k=2,v=3,a=4,g=5).
            let has_gate = layers
                .first()
                .map(|l| l.time_mix_g1.is_some() && l.time_mix_g2.is_some())
                .unwrap_or(false);
            // (rank, xs_slot) per group, in stack order.
            let mut groups: Vec<(usize, usize)> = vec![
                (cfg.decay_lora, 1),     // w  ← xw (slot 1)
                (cfg.iclr_lora, 4),      // a  ← xa (slot 4)
                (cfg.value_res_lora, 3), // v  ← xv (slot 3)
            ];
            if has_gate {
                groups.push((cfg.gate_lora, 5)); // g ← xg (slot 5)
            }
            // Down table: cols=n; output stacked by rank into `lora_lo`.
            let mut down = Vec::with_capacity(groups.len());
            let mut w_off = 0usize; // W1 element offset (rows are rank×n)
            let mut lo_off = 0usize; // stacked low-buffer offset
            for &(rank, slot) in &groups {
                down.push(LoraGroup {
                    row_start: lo_off as u32,
                    w_off: w_off as u32,
                    x_off: (slot * n) as u32,
                    out_off: lo_off as u32,
                    cols: n as u32,
                });
                w_off += rank * n;
                lo_off += rank;
            }
            let lora_down_rows = lo_off; // == sum of ranks
            // Up table: rows=n each; cols=rank_i; reads `lora_lo` segments; writes
            // stacked `lora_up` = [w_raw | a | v_mix | gate], each n.
            let mut up_tbl = Vec::with_capacity(groups.len());
            let mut w2_off = 0usize; // W2 element offset (rows are n×rank)
            let mut lo_in = 0usize; // low-buffer input segment offset
            for (gi, &(rank, _)) in groups.iter().enumerate() {
                up_tbl.push(LoraGroup {
                    row_start: (gi * n) as u32,
                    w_off: w2_off as u32,
                    x_off: lo_in as u32,
                    out_off: (gi * n) as u32,
                    cols: rank as u32,
                });
                w2_off += n * rank;
                lo_in += rank;
            }
            let lora_up_rows = groups.len() * n;
            let tbl_bytes = |t: &[LoraGroup]| ctx.new_buffer_with_bytes(bytemuck::cast_slice(t));

            Ok(Self {
                layers: gpu_layers,
                tok_norm_w: up(tok_norm_w),
                tok_norm_b: up(tok_norm_b),
                output_norm_w: up(output_norm_w),
                output_norm_b: up(output_norm_b),
                lm_head,
                arena,
                fresh: true,
                last_dispatch_count: 0,
                lora_down_table: tbl_bytes(&down),
                lora_up_table: tbl_bytes(&up_tbl),
                lora_groups: groups.len(),
                lora_down_rows,
                lora_up_rows,
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

            // r = Wr @ xr   (slot 0) — Q4_K projection
            layer.receptance.gemv(&mut tcb, &a.xs, slot_off(0), &a.r)?;

            // ── fused LoRA: 4 down-GEMVs + 2 inter-acts + 4 up-GEMVs → 3 dispatches.
            // down: W1_stacked @ {xw,xa,xv,xg slots} → lora_lo = [w_lo|a_lo|v_lo|g_lo]
            // mid : tanh(w_lo) + sigmoid(g_lo) in place (a/v identity)
            // up  : W2_stacked @ lora_lo segments → lora_up = [w_raw|a|v_mix|gate] (each n)
            // The per-row arithmetic is bit-identical to the standalone GEMVs.
            rwkv7_lora_grouped_gemv_tcb(
                &mut tcb,
                &layer.lora_w1_stacked,
                &a.xs,
                &a.lora_lo,
                &g.lora_down_table,
                g.lora_groups,
                g.lora_down_rows,
            )?;
            // w segment [0,decay); g segment [decay+iclr+vres, +gate).
            let w_end = cfg.decay_lora;
            let g_begin = cfg.decay_lora + cfg.iclr_lora + cfg.value_res_lora;
            rwkv7_lora_mid_act_tcb(&mut tcb, &a.lora_lo, w_end, g_begin, g.lora_down_rows)?;
            rwkv7_lora_grouped_gemv_tcb(
                &mut tcb,
                &layer.lora_w2_stacked,
                &a.lora_lo,
                &a.lora_up,
                &g.lora_up_table,
                g.lora_groups,
                g.lora_up_rows,
            )?;
            // lora_up segment byte offsets: w_raw@0, a@n, v_mix@2n, gate@3n.
            // (`f32b` is the f32 byte size declared above for `slot_off`.)
            let up_a_off = n * f32b;
            let up_vmix_off = 2 * n * f32b;
            let up_gate_off = 3 * n * f32b;

            // w = exp(-0.606531 * sigmoid(w0 + w_raw))   (w_raw = lora_up[0..n])
            rwkv7_decay_act_tcb(&mut tcb, &a.lora_up, &layer.w0, &a.w, n)?;

            // k = Wk @ xk (slot 2) ; v = Wv @ xv (slot 3) — Q4_K projections
            layer.key.gemv(&mut tcb, &a.xs, slot_off(2), &a.k)?;
            layer.value.gemv(&mut tcb, &a.xs, slot_off(3), &a.v)?;

            // value-residual mix (skip on layer 0 where v_first is established).
            if !have_v_first {
                rwkv7_copy_tcb(&mut tcb, &a.v, &a.v_first, 0, n)?;
                have_v_first = true;
            } else {
                // v_mix = lora_up[2n..3n]; v += (v_first - v) * sigmoid(v_mix + v0).
                rwkv7_value_residual_mix_tcb(
                    &mut tcb,
                    &a.v,
                    &a.v_first,
                    &a.lora_up,
                    up_vmix_off,
                    &layer.v0,
                    n,
                )?;
            }

            // a = sigmoid(a0 + a_raw)   (a_raw = lora_up[n..2n], in place)
            rwkv7_sigmoid_bias_tcb(&mut tcb, &a.lora_up, up_a_off, &layer.a0, n)?;

            // kk = l2norm_per_head(k*k_k); k += (a-1)*(k*k_a); a_op=-kk; b_op=kk*a.
            // `a` is read from lora_up[n..2n].
            rwkv7_kk_kmix_tcb(
                &mut tcb,
                &a.k,
                &layer.k_k,
                &layer.k_a,
                &a.lora_up,
                up_a_off,
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
                &a.lora_up,
                up_gate_off,
                &a.out_wkv,
                cfg.head_size,
                cfg.head_count,
                GN_EPS,
                layer.has_gate,
            )?;

            // y = Wo @ out  → cur — Q4_K projection (x from out_wkv, offset 0)
            layer.output.gemv(&mut tcb, &a.out_wkv, 0, &a.cur)?;

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
            // k = relu(Wk @ xk)^2 — Q4_K projection
            layer
                .channel_mix_key
                .gemv(&mut tcb, &a.xk_ffn, 0, &a.ffn_k)?;
            rwkv7_relu_sq_inplace_tcb(&mut tcb, &a.ffn_k, cfg.n_ff)?;
            // cmix = Wv @ k — Q4_K projection
            layer
                .channel_mix_value
                .gemv(&mut tcb, &a.ffn_k, 0, &a.cmix)?;
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
        // LM head — Q6_K on the World models (reads x_norm at offset 0).
        g.lm_head.gemv(&mut tcb, &a.x_norm, 0, &a.logits)?;

        g.last_dispatch_count = tcb.dispatch_count();
        tcb.commit_and_wait()?;
        g.fresh = false;
        Ok(read_vec(&a.logits, cfg.vocab_size))
    }

    /// Write B embedding rows (one per stream) into the `(B, n)` row-major `x`
    /// buffer: stream b's row lands at `x[b*n .. b*n+n]`.
    fn write_rows_batch(buf: &PinnedBuffer, rows: &[&[f32]], n: usize) {
        let ptr = buf.contents() as *mut f32;
        for (b, row) in rows.iter().enumerate() {
            debug_assert_eq!(row.len(), n);
            unsafe { std::ptr::copy_nonoverlapping(row.as_ptr(), ptr.add(b * n), n) };
        }
    }

    /// Read B `vocab`-sized logit rows from the `(B, vocab)` row-major `logits`
    /// buffer: stream b's row is `logits[b*vocab .. b*vocab+vocab]`.
    fn read_rows_batch(buf: &PinnedBuffer, batch: usize, vocab: usize) -> Vec<Vec<f32>> {
        let ptr = buf.contents() as *const f32;
        (0..batch)
            .map(|b| unsafe { std::slice::from_raw_parts(ptr.add(b * vocab), vocab) }.to_vec())
            .collect()
    }

    /// B-STREAM (continuous-batch) RWKV-7 decode step on the GPU. Advances B
    /// INDEPENDENT recurrent states by one token each in ONE pass, mirroring
    /// [`forward_token_gpu`] op-for-op — so stream b is bit-for-bit its own
    /// single-stream GPU decode. The bandwidth win: every big projection + the LM
    /// head reads its weight ONCE across the B activation columns (`gemv_batched`
    /// → the Q4_K `gemm_q4_k_m_batched_v3w_predec` for r/k/v/o/channel-mix; the
    /// Q6_K LM head + f32 LoRA loop B vectors over the SINGLE resident weight).
    ///
    /// Buffers follow the [`RwkvDecodeArena::new_with_batch`] contract:
    /// activations are `(B, dim)` row-major, the `xs` lerp output is `(slot,B,n)`,
    /// and the three state planes are stream-major (per-stream/per-layer windows).
    /// `tokens.len()` must equal the arena batch. Returns B `vocab` logit rows.
    pub fn forward_token_gpu_multiseq(
        ctx: &MetalContext,
        g: &mut RwkvGpu,
        cfg: &RwkvConfig,
        embed: &[f32],
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        let n = cfg.n_embd;
        let eps = cfg.ln_eps;
        let b = g.arena.batch;
        if tokens.len() != b {
            return Err(Error::Model(format!(
                "rwkv7 gpu multiseq: tokens={} != arena batch={}",
                tokens.len(),
                b
            )));
        }
        let f32b = std::mem::size_of::<f32>();
        // Per-stream element strides into the stream-major state planes.
        let s_per_layer = cfg.head_count * cfg.head_size * cfg.head_size;
        let wkv_stream_stride = cfg.n_layer * s_per_layer; // elems / stream
        let shift_stream_stride = cfg.n_layer * n; // elems / stream

        // Embedding gather: write B rows into the (B, n) x buffer.
        let mut rows: Vec<&[f32]> = Vec::with_capacity(b);
        for &tok in tokens {
            let row = tok as usize * n;
            if row + n > embed.len() {
                return Err(Error::Model(format!("rwkv7: token {tok} out of vocab")));
            }
            rows.push(&embed[row..row + n]);
        }
        let a = &g.arena;
        write_rows_batch(&a.x, &rows, n);

        let mut tcb = TokenCommandBuffer::new(ctx);

        // LN0 over all B rows: x_norm = layernorm(x); x <- x_norm (copy B*n).
        rwkv7_layernorm_multiseq_tcb(
            &mut tcb,
            &a.x,
            0,
            &g.tok_norm_w,
            &g.tok_norm_b,
            &a.x_norm,
            0,
            n,
            b,
            eps,
        )?;
        rwkv7_copy_tcb(&mut tcb, &a.x_norm, &a.x, 0, b * n)?;

        let mut have_v_first = false;
        // byte offset of slot `s` in the (slot, B, n) xs buffer.
        let xs_slot = |s: usize| s * b * n * f32b;

        for li in 0..cfg.n_layer {
            let layer = &g.layers[li];
            // (stream 0, layer li) byte bases into the stream-major planes.
            let shift_base = a.shift_slot_layer_byte_offset(0, li);
            let wkv_layer_base = li * s_per_layer; // elems within a stream window

            // ── time-mix ──
            // att_in = layernorm(x, attn_norm)   over (B, n)
            rwkv7_layernorm_multiseq_tcb(
                &mut tcb,
                &a.x,
                0,
                &layer.attn_norm_w,
                &layer.attn_norm_b,
                &a.att_in,
                0,
                n,
                b,
                eps,
            )?;
            // token-shift lerp from the stream-major att_shift window → xs (slot,B,n).
            let n_slots = if layer.has_gate { 6 } else { 5 };
            rwkv7_token_shift_lerp_multiseq_tcb(
                &mut tcb,
                &a.att_in,
                &a.att_shift,
                shift_base,
                shift_stream_stride,
                &layer.lerp_fused,
                &a.xs,
                n,
                n_slots,
                b,
                g.fresh,
            )?;

            // r = Wr @ xr   (slot 0) — batched Q4_K projection
            layer
                .receptance
                .gemv_batched(&mut tcb, &a.xs, xs_slot(0), &a.r, b)?;

            // w = exp(-0.606531 * sigmoid(w0 + W2 @ tanh(W1 @ xw)))  (slot 1).
            // LoRA: per-stream f32 GEMVs into (B, decay_lora) then (B, n).
            for bi in 0..b {
                rwkv7_gemv_f32_xoff_yoff_tcb(
                    &mut tcb,
                    &layer.w1,
                    cfg.decay_lora,
                    n,
                    &a.xs,
                    xs_slot(1) + bi * n * f32b,
                    &a.w_lo,
                    bi * cfg.decay_lora * f32b,
                )?;
            }
            rwkv7_tanh_inplace_tcb(&mut tcb, &a.w_lo, b * cfg.decay_lora)?;
            for bi in 0..b {
                rwkv7_gemv_f32_xoff_yoff_tcb(
                    &mut tcb,
                    &layer.w2,
                    n,
                    cfg.decay_lora,
                    &a.w_lo,
                    bi * cfg.decay_lora * f32b,
                    &a.w_raw,
                    bi * n * f32b,
                )?;
            }
            rwkv7_decay_act_multiseq_tcb(&mut tcb, &a.w_raw, &layer.w0, &a.w, n, b)?;

            // k = Wk @ xk (slot 2) ; v = Wv @ xv (slot 3) — batched Q4_K projections
            layer
                .key
                .gemv_batched(&mut tcb, &a.xs, xs_slot(2), &a.k, b)?;
            layer
                .value
                .gemv_batched(&mut tcb, &a.xs, xs_slot(3), &a.v, b)?;

            // value-residual mix (skip on layer 0 where v_first is established).
            if !have_v_first {
                rwkv7_copy_tcb(&mut tcb, &a.v, &a.v_first, 0, b * n)?;
                have_v_first = true;
            } else {
                for bi in 0..b {
                    rwkv7_gemv_f32_xoff_yoff_tcb(
                        &mut tcb,
                        &layer.v1,
                        cfg.value_res_lora,
                        n,
                        &a.xs,
                        xs_slot(3) + bi * n * f32b,
                        &a.v_lo,
                        bi * cfg.value_res_lora * f32b,
                    )?;
                }
                for bi in 0..b {
                    rwkv7_gemv_f32_xoff_yoff_tcb(
                        &mut tcb,
                        &layer.v2,
                        n,
                        cfg.value_res_lora,
                        &a.v_lo,
                        bi * cfg.value_res_lora * f32b,
                        &a.v_mix,
                        bi * n * f32b,
                    )?;
                }
                rwkv7_value_residual_mix_multiseq_tcb(
                    &mut tcb, &a.v, &a.v_first, &a.v_mix, &layer.v0, n, b,
                )?;
            }

            // gate g = G2 @ sigmoid(G1 @ xg)   (slot 5; only if gated)
            if layer.has_gate {
                let g1 = layer.g1.as_ref().unwrap();
                let g2 = layer.g2.as_ref().unwrap();
                for bi in 0..b {
                    rwkv7_gemv_f32_xoff_yoff_tcb(
                        &mut tcb,
                        g1,
                        cfg.gate_lora,
                        n,
                        &a.xs,
                        xs_slot(5) + bi * n * f32b,
                        &a.g_lo,
                        bi * cfg.gate_lora * f32b,
                    )?;
                }
                rwkv7_sigmoid_inplace_tcb(&mut tcb, &a.g_lo, b * cfg.gate_lora)?;
                for bi in 0..b {
                    rwkv7_gemv_f32_xoff_yoff_tcb(
                        &mut tcb,
                        g2,
                        n,
                        cfg.gate_lora,
                        &a.g_lo,
                        bi * cfg.gate_lora * f32b,
                        &a.gate,
                        bi * n * f32b,
                    )?;
                }
            }

            // a = sigmoid(a0 + A2 @ (A1 @ xa))   (slot 4)
            for bi in 0..b {
                rwkv7_gemv_f32_xoff_yoff_tcb(
                    &mut tcb,
                    &layer.a1,
                    cfg.iclr_lora,
                    n,
                    &a.xs,
                    xs_slot(4) + bi * n * f32b,
                    &a.a_lo,
                    bi * cfg.iclr_lora * f32b,
                )?;
            }
            for bi in 0..b {
                rwkv7_gemv_f32_xoff_yoff_tcb(
                    &mut tcb,
                    &layer.a2,
                    n,
                    cfg.iclr_lora,
                    &a.a_lo,
                    bi * cfg.iclr_lora * f32b,
                    &a.a,
                    bi * n * f32b,
                )?;
            }
            rwkv7_sigmoid_bias_multiseq_tcb(&mut tcb, &a.a, &layer.a0, n, b)?;

            // kk = l2norm_per_head(k*k_k); k += (a-1)*(k*k_a); a_op=-kk; b_op=kk*a.
            rwkv7_kk_kmix_multiseq_tcb(
                &mut tcb,
                &a.k,
                &layer.k_k,
                &layer.k_a,
                &a.a,
                &a.a_op,
                &a.b_op,
                cfg.head_size,
                cfg.head_count,
                b,
            )?;

            // WKV-7 recurrence + per-head group-norm + bonus + gate, for B streams.
            rwkv7_wkv_decode_multiseq_tcb(
                &mut tcb,
                &a.wkv_state,
                wkv_stream_stride,
                wkv_layer_base,
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
                n,
                cfg.head_size,
                cfg.head_count,
                b,
                GN_EPS,
                layer.has_gate,
            )?;

            // y = Wo @ out → cur — batched Q4_K projection (x from out_wkv, off 0)
            layer
                .output
                .gemv_batched(&mut tcb, &a.out_wkv, 0, &a.cur, b)?;

            // store att_in as next token-shift for this layer (scatter into the
            // stream-major att_shift plane).
            rwkv7_shift_writeback_multiseq_tcb(
                &mut tcb,
                &a.att_in,
                &a.att_shift,
                shift_base,
                shift_stream_stride,
                n,
                b,
            )?;

            // ffn_inp = cur + x   (flat B*n)
            rwkv7_add_into_flat_tcb(&mut tcb, &a.cur, &a.x, &a.ffn_inp, b * n)?;

            // ── channel-mix ──
            // ffn_in = layernorm(ffn_inp, attn_norm_2)   over (B, n)
            rwkv7_layernorm_multiseq_tcb(
                &mut tcb,
                &a.ffn_inp,
                0,
                &layer.attn_norm2_w,
                &layer.attn_norm2_b,
                &a.ffn_in,
                0,
                n,
                b,
                eps,
            )?;
            // xk = ffn_in + (x_prev - ffn_in) * lerp_k   (stream-major ffn_shift)
            rwkv7_channel_mix_shift_multiseq_tcb(
                &mut tcb,
                &a.ffn_in,
                &a.ffn_shift,
                shift_base,
                shift_stream_stride,
                &layer.channel_mix_lerp_k,
                &a.xk_ffn,
                n,
                b,
                g.fresh,
            )?;
            // k = relu(Wk @ xk)^2 — batched Q4_K projection (B, n_ff)
            layer
                .channel_mix_key
                .gemv_batched(&mut tcb, &a.xk_ffn, 0, &a.ffn_k, b)?;
            rwkv7_relu_sq_inplace_tcb(&mut tcb, &a.ffn_k, b * cfg.n_ff)?;
            // cmix = Wv @ k — batched Q4_K projection (B, n)
            layer
                .channel_mix_value
                .gemv_batched(&mut tcb, &a.ffn_k, 0, &a.cmix, b)?;
            // store ffn_in as next token-shift (scatter into stream-major plane).
            rwkv7_shift_writeback_multiseq_tcb(
                &mut tcb,
                &a.ffn_in,
                &a.ffn_shift,
                shift_base,
                shift_stream_stride,
                n,
                b,
            )?;

            // x = cmix + ffn_inp   (flat B*n)
            rwkv7_add_into_flat_tcb(&mut tcb, &a.cmix, &a.ffn_inp, &a.x, b * n)?;
        }

        // final norm (B, n) + LM head (batched: Q6_K loops B over one weight).
        rwkv7_layernorm_multiseq_tcb(
            &mut tcb,
            &a.x,
            0,
            &g.output_norm_w,
            &g.output_norm_b,
            &a.x_norm,
            0,
            n,
            b,
            eps,
        )?;
        g.lm_head
            .gemv_batched(&mut tcb, &a.x_norm, 0, &a.logits, b)?;

        tcb.commit_and_wait()?;
        g.fresh = false;
        Ok(read_rows_batch(&a.logits, b, cfg.vocab_size))
    }
}
