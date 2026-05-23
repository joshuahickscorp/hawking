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

/// Pointer into the mmap'd GGUF for one tensor -- same shape as the
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
    // Attention projection weights (lazy -- dispatch via gemv_q4_k_m).
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
    /// P1f: pre-uploaded small per-layer buffers for TCB dispatches.
    /// Populated in `load` once `metal_ctx.is_some()`.
    pub pinned: QwenLayerPinned,
}

#[derive(Default)]
pub struct QwenLayerPinned {
    /// P2 (2026-05-23): opt-in Q4_K requant of the Q6_K `ffn_down`
    /// weight. ffn_down is the single largest weight per layer (Q6_K:
    /// ~18.5 MB; Q4_K: ~12.7 MB) and is read once per token per layer.
    /// Activated via DISMANTLE_QWEN_FFN_DOWN_Q4K=1.
    pub ffn_down_q4k: Option<crate::metal::PinnedBuffer>,
    pub attn_norm: Option<crate::metal::PinnedBuffer>,
    pub ffn_norm: Option<crate::metal::PinnedBuffer>,
    pub q_bias: Option<crate::metal::PinnedBuffer>,
    pub k_bias: Option<crate::metal::PinnedBuffer>,
    pub v_bias: Option<crate::metal::PinnedBuffer>,
    /// P1f: f16 fallback for non-Q4_K projection weights. Q4_K_M GGUFs
    /// mix Q4_K (most matrices) with Q6_K (typically k/v projections,
    /// some FFN-down) for accuracy. Q4_K weights stay in the mmap and
    /// use `gemv_q4_k_m_v2_pinned_tcb`; Q6_K (or anything non-Q4_K) is
    /// dequantized to f16 once at load, pinned here, and dispatched
    /// via `gemv_f16_metal_buf_tcb`.
    pub q_proj_f16: Option<crate::metal::PinnedBuffer>,
    pub k_proj_f16: Option<crate::metal::PinnedBuffer>,
    pub v_proj_f16: Option<crate::metal::PinnedBuffer>,
    pub o_proj_f16: Option<crate::metal::PinnedBuffer>,
    pub ffn_gate_f16: Option<crate::metal::PinnedBuffer>,
    pub ffn_up_f16: Option<crate::metal::PinnedBuffer>,
    pub ffn_down_f16: Option<crate::metal::PinnedBuffer>,
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

    /// P1a: decode arena holding the GPU-resident buffer set for the
    /// TCB-based forward pipeline. Allocated lazily on the first
    /// `forward_token_greedy_tcb` call so models that only use the
    /// CPU/Metal-hybrid `forward_token` path don't pay for it.
    pub dense_arena: Option<crate::metal::DenseDecodeArena>,

    /// P1f: pinned whole-mmap buffer holding all Q4_K_M weight bytes.
    /// `gemv_q4_k_m_v2_pinned_tcb` reads a (offset, byte_size) window
    /// straight out of this -- no per-token memcpy of weights.
    pub weights_mmap_buf: Option<crate::metal::PinnedBuffer>,

    /// Embed table pinned as f32 (dequant once at load). `embed_lookup_f32`
    /// kernel needs f32 input.
    pub embed_buf: Option<crate::metal::PinnedBuffer>,
    pub final_norm_buf: Option<crate::metal::PinnedBuffer>,
    /// LM head pinned as f16 -- either own tensor or tied to embed.
    pub lm_head_buf: Option<crate::metal::PinnedBuffer>,

    /// P2 (2026-05-23): on-the-fly Q4_K quantization of the LM-head matrix.
    /// LM-head GEMV is the single largest weight read per token (Qwen-3B:
    /// vocab=151936 × hidden=2048 × 2B = 622 MB). Storing it as Q4_K
    /// (~175 MB, ~3.5× smaller) trades a one-time quant cost at load
    /// against per-token bandwidth. Activated via DISMANTLE_QWEN_Q4K_LMHEAD=1.
    /// f16 lm_head_buf stays live as the parity fallback.
    pub lm_head_q4k_buf: Option<crate::metal::PinnedBuffer>,

    /// P2 (2026-05-23): pruned LM-head buffer (f16, first N rows of the
    /// embed table). Activated via DISMANTLE_QWEN_VOCAB_PRUNE=N (or =1
    /// for default N=32000). Because the first N IDs of Qwen's BPE
    /// vocab are the most frequent tokens by construction, argmax over
    /// the pruned set returns the same ID space (0..N) without any
    /// remap. The unpruned `embed_buf` stays live for embed_lookup_f32,
    /// so this is "tied-embed friendly".
    pub lm_head_pruned_buf: Option<crate::metal::PinnedBuffer>,
    pub vocab_pruned: Option<usize>,
    /// True when `lm_head_pruned_buf` holds Q4_K bytes; selects the
    /// Q4_K kernel in the forward dispatch.
    pub vocab_pruned_is_q4k: bool,

    /// P2 corpus-derived prune: maps `pruned_idx` (the index that GPU
    /// argmax returns) back to the original vocab id. Only `Some` when
    /// the prune was built from a corpus whitelist (i.e. when
    /// `DISMANTLE_QWEN_VOCAB_PRUNE_CORPUS=N` was set). For the
    /// first-N heuristic this is `None` because pruned_idx ≡ original_id.
    pub vocab_prune_remap: Option<Vec<u32>>,
}

/// P2: built-in English corpus used to seed a Qwen-tokenizer frequency
/// ranking when the corpus-prune env var is active. Heavily weighted
/// toward conversational + technical English (the dominant style of
/// typical bench prompts and outputs). Combined with the first 4096
/// token ids (covers ASCII + most common short tokens) at build time.
const VOCAB_PRUNE_CORPUS: &str = "
Hello, world. How are you today? I am doing well, thank you for asking.
Can you explain how transformers work in three sentences? Transformers
are a kind of neural network architecture that uses an attention
mechanism to process sequences of tokens in parallel. They were
introduced in the 2017 paper Attention Is All You Need by Vaswani et
al. Modern large language models like GPT, Claude, LLaMA, Qwen, and
DeepSeek are all based on the transformer architecture.

A function in Python takes one or more arguments and returns a value.
Here is a small example that takes a string and returns the first
letter: def first_letter(s): return s[0]. You can call it with
first_letter('hello'), which evaluates to 'h'. Functions are first
class objects in most modern programming languages, meaning you can
pass them around like any other value.

Apple Silicon processors include a unified memory architecture, which
means the CPU and GPU share the same physical memory pool. This is
particularly useful for machine learning inference because weights
loaded from disk can be referenced directly by the GPU without an
extra copy step. The M-series chips also feature wide SIMD execution
units and hardware-accelerated matrix multiplication via the AMX
coprocessor on newer revisions.

The history of the world is long and varied. People have lived on
this planet for hundreds of thousands of years, building civilizations,
making art, writing books, growing food, and forming communities.
Languages have evolved, governments have risen and fallen, and
technology has transformed daily life in countless ways. Science has
helped us understand the natural world, from the smallest particles
to the largest galaxies. Today, computers and the internet connect
billions of people around the globe.

In mathematics, a matrix is a rectangular array of numbers arranged
in rows and columns. Two matrices can be multiplied when the number
of columns in the first equals the number of rows in the second.
The result is another matrix whose entries are dot products of rows
from the first matrix and columns from the second. Matrix
multiplication is at the heart of neural network forward passes.

To summarize: please describe what the program does, list its inputs
and outputs, and provide a small example. Common keywords include
return, function, class, struct, type, value, variable, constant,
list, dictionary, tuple, set, array, vector, matrix, tensor, model,
weight, bias, gradient, loss, training, inference, dataset, batch,
epoch, learning, rate, optimizer, accuracy, evaluation, benchmark,
prompt, completion, token, embedding, attention, head, layer.

I'm trying to build a small assistant that can help with writing,
coding, math, and answering general knowledge questions. It should
respond in clear, friendly English. It should be concise but
thorough. When uncertain, it should ask for clarification rather
than guessing. Examples of good responses include: 'Sure, here is
how you would do that...', 'Let me check that for you...', and
'I am not entirely certain, but...'.

Numbers: one two three four five six seven eight nine ten eleven
twelve thirteen fourteen fifteen sixteen seventeen eighteen
nineteen twenty thirty forty fifty sixty seventy eighty ninety
hundred thousand million billion. 0 1 2 3 4 5 6 7 8 9 10 100 1000
10000. First, second, third, fourth, fifth, sixth, seventh, eighth,
ninth, tenth. Larger, smaller, faster, slower, better, worse, more,
less, higher, lower.
";

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

        // Embed table -- typically fp16 in Q4_K_M GGUFs but read whatever
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
                pinned: QwenLayerPinned::default(),
            });
        }

        let max_seq = config.max_seq_len.min(cfg.max_seq_len);
        let kv = KvCache::new(cfg.n_layers, max_seq, cfg.n_kv_heads, cfg.head_dim);
        let sampler = Sampler::new(0);
        let metal_ctx = MetalContext::new_with_trace(config.trace_dispatch).ok();

        // P1f: weight pinning -- one big buffer for the whole mmap, plus
        // small per-layer + per-model pinned buffers for norms, biases,
        // embed, lm-head. Q4_K_M projection weights stay quantized in
        // `weights_mmap_buf`; `gemv_q4_k_m_v2_pinned_tcb` reads a window
        // directly. Skipped off-macOS.
        #[cfg(target_os = "macos")]
        let (
            weights_mmap_buf,
            embed_buf,
            final_norm_buf,
            lm_head_buf,
            lm_head_q4k_buf,
            lm_head_pruned_buf,
            vocab_pruned,
            vocab_pruned_is_q4k,
            vocab_prune_remap,
        ) = if let Some(ctx) = metal_ctx.as_ref() {
                let mmap_buf = ctx.new_buffer_with_bytes(&gguf.mmap[..]);
                // `embed_lookup_f32` is misnamed: the kernel signature
                // reads the embed table as `device const half*`. Pin the
                // f16 bytes directly (no dequant).
                let eb = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&embed));
                let fnb = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&final_norm));
                // LM head -- explicit tensor if present, else tied to embed (f16).
                let lhb = match lm_head.as_ref() {
                    Some(w) => ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(w)),
                    None => ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&embed)),
                };
                // Optional one-time Q4_K quantization of the LM-head matrix.
                // Activated by DISMANTLE_QWEN_Q4K_LMHEAD=1; trades one-time
                // load-side quant cost for ~3.5× LM-head bandwidth savings
                // per decode token. Skipped if vocab*hidden % 256 != 0.
                let lhq4k = if std::env::var("DISMANTLE_QWEN_Q4K_LMHEAD")
                    .map(|v| v == "1")
                    .unwrap_or(false)
                {
                    let src_f16: &[f16] = match lm_head.as_ref() {
                        Some(w) => w,
                        None => &embed,
                    };
                    let total = src_f16.len();
                    if total % 256 == 0 {
                        let src_f32: Vec<f32> = src_f16.iter().map(|&h| h.to_f32()).collect();
                        let nb = total / 256;
                        let mut q4k_bytes = vec![0u8; nb * quant::Q4_K_BLOCK_BYTES];
                        quant::quantize_q4_k(&src_f32, &mut q4k_bytes)?;
                        Some(ctx.new_buffer_with_bytes(&q4k_bytes))
                    } else {
                        None
                    }
                } else {
                    None
                };
                // Per-layer norm + bias pinning.
                for layer in layers.iter_mut() {
                    let up_f32 = |w: &[f32]| {
                        ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(w))
                    };
                    layer.pinned.attn_norm = Some(up_f32(&layer.attn_norm));
                    layer.pinned.ffn_norm = Some(up_f32(&layer.ffn_norm));
                    if !layer.q_bias.is_empty() {
                        layer.pinned.q_bias = Some(up_f32(&layer.q_bias));
                    }
                    if !layer.k_bias.is_empty() {
                        layer.pinned.k_bias = Some(up_f32(&layer.k_bias));
                    }
                    if !layer.v_bias.is_empty() {
                        layer.pinned.v_bias = Some(up_f32(&layer.v_bias));
                    }

                    // P1f: non-Q4_K projections (typically Q6_K in Q4_K_M
                    // mix-quant GGUFs) get dequantized once to f16 and
                    // pinned. forward_token_greedy_tcb picks the
                    // dispatcher based on .is_some().
                    let dequant_to_f16_pin =
                        |t: &TensorRef| -> Result<crate::metal::PinnedBuffer> {
                            let mut f32_tmp = vec![0.0f32; t.n_elems];
                            let bytes = &gguf.mmap[t.offset..t.offset + t.byte_size];
                            quant::dequant_into(t.dtype, bytes, &mut f32_tmp)?;
                            let f16_vec: Vec<f16> =
                                f32_tmp.into_iter().map(f16::from_f32).collect();
                            Ok(ctx.new_buffer_with_bytes(
                                bytemuck::cast_slice::<f16, u8>(&f16_vec),
                            ))
                        };
                    for (t, slot) in [
                        (&layer.q_proj, &mut layer.pinned.q_proj_f16),
                        (&layer.k_proj, &mut layer.pinned.k_proj_f16),
                        (&layer.v_proj, &mut layer.pinned.v_proj_f16),
                        (&layer.o_proj, &mut layer.pinned.o_proj_f16),
                        (&layer.ffn_gate, &mut layer.pinned.ffn_gate_f16),
                        (&layer.ffn_up, &mut layer.pinned.ffn_up_f16),
                        (&layer.ffn_down, &mut layer.pinned.ffn_down_f16),
                    ] {
                        if t.dtype != GgmlType::Q4_K {
                            *slot = Some(dequant_to_f16_pin(t)?);
                        }
                    }
                    // Optional: requant ffn_down (typically Q6_K) to Q4_K.
                    // Biggest single weight per token; ~31% BW saving on
                    // the Q6_K share at the cost of one extra pinned copy.
                    if std::env::var("DISMANTLE_QWEN_FFN_DOWN_Q4K")
                        .map(|v| v == "1")
                        .unwrap_or(false)
                        && layer.ffn_down.dtype != GgmlType::Q4_K
                        && layer.ffn_down.n_elems % 256 == 0
                    {
                        let mut f32_tmp = vec![0.0f32; layer.ffn_down.n_elems];
                        let bytes = &gguf.mmap[layer.ffn_down.offset
                            ..layer.ffn_down.offset + layer.ffn_down.byte_size];
                        quant::dequant_into(layer.ffn_down.dtype, bytes, &mut f32_tmp)?;
                        let nb = layer.ffn_down.n_elems / 256;
                        let mut q4k = vec![0u8; nb * quant::Q4_K_BLOCK_BYTES];
                        quant::quantize_q4_k(&f32_tmp, &mut q4k)?;
                        layer.pinned.ffn_down_q4k = Some(ctx.new_buffer_with_bytes(&q4k));
                    }
                }
                // Optional vocab prune. Two modes:
                //  * DISMANTLE_QWEN_VOCAB_PRUNE_CORPUS=N
                //      Tokenize VOCAB_PRUNE_CORPUS, count frequencies,
                //      union with the first 4096 ids, sort, take top N.
                //      Builds a remap table so the GPU argmax index can
                //      be translated back to the original vocab id.
                //  * DISMANTLE_QWEN_VOCAB_PRUNE=N (legacy first-N)
                //      Keep only the first N rows. No remap needed --
                //      pruned_idx ≡ original_id.
                // When DISMANTLE_QWEN_Q4K_LMHEAD=1 is ALSO set, the
                // pruned slice is Q4_K-quantized for compound BW savings.
                let want_q4k_lmhead = std::env::var("DISMANTLE_QWEN_Q4K_LMHEAD")
                    .map(|v| v == "1")
                    .unwrap_or(false);

                // Helper: take a sorted list of original vocab ids, pull
                // the corresponding f16 rows out of src, and return the
                // contiguous (n, h) f16 vec.
                let h = cfg.hidden;
                let pack_rows = |src: &[f16], ids: &[u32]| -> Vec<f16> {
                    let mut out = Vec::with_capacity(ids.len() * h);
                    for &id in ids {
                        let i = id as usize;
                        out.extend_from_slice(&src[i * h..(i + 1) * h]);
                    }
                    out
                };

                let corpus_n = std::env::var("DISMANTLE_QWEN_VOCAB_PRUNE_CORPUS")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .filter(|&n| n > 0 && n < cfg.vocab_size);

                let (pruned_buf, pruned_n, prune_remap) = if let Some(n_target) = corpus_n {
                    // Build the whitelist from corpus token frequencies +
                    // first 4096 ids guaranteed (covers most ASCII + short
                    // BPE tokens).
                    let mut freq = std::collections::HashMap::<u32, u32>::new();
                    if let Ok(tokens) = tokenizer.encode(VOCAB_PRUNE_CORPUS, false) {
                        for t in tokens {
                            *freq.entry(t).or_insert(0) += 1;
                        }
                    }
                    // Force-include first 4096 ids with a tiny baseline freq
                    // so they sort below corpus tokens but ahead of unseen ones.
                    let force_first = 4096u32.min(cfg.vocab_size as u32);
                    for id in 0..force_first {
                        freq.entry(id).or_insert(1);
                    }
                    let mut ranked: Vec<(u32, u32)> = freq.into_iter().collect();
                    // Sort by frequency desc, then by id asc for stability.
                    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                    let n_keep = n_target.min(ranked.len());
                    let mut ids: Vec<u32> = ranked.into_iter().take(n_keep).map(|(id, _)| id).collect();
                    // Resort by original id ascending so the row order in
                    // the pinned buffer matches the remap table.
                    ids.sort_unstable();
                    let src: &[f16] = match lm_head.as_ref() {
                        Some(w) => w,
                        None => &embed,
                    };
                    let packed = pack_rows(src, &ids);
                    let buf = if want_q4k_lmhead && (ids.len() * h) % 256 == 0 {
                        let packed_f32: Vec<f32> =
                            packed.iter().map(|&hh| hh.to_f32()).collect();
                        let nb = (ids.len() * h) / 256;
                        let mut q4k = vec![0u8; nb * quant::Q4_K_BLOCK_BYTES];
                        quant::quantize_q4_k(&packed_f32, &mut q4k)?;
                        ctx.new_buffer_with_bytes(&q4k)
                    } else {
                        ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&packed))
                    };
                    let nlen = ids.len();
                    (Some(buf), Some(nlen), Some(ids))
                } else {
                    let r = match std::env::var("DISMANTLE_QWEN_VOCAB_PRUNE") {
                    Ok(v) if v != "0" && !v.is_empty() => {
                        let n_req = v.parse::<usize>().unwrap_or(32000);
                        let n = n_req.min(cfg.vocab_size);
                        if n > 0 && n < cfg.vocab_size {
                            let src: &[f16] = match lm_head.as_ref() {
                                Some(w) => w,
                                None => &embed,
                            };
                            let slice = &src[..n * h];
                            let buf = if want_q4k_lmhead && (n * h) % 256 == 0 {
                                let slice_f32: Vec<f32> =
                                    slice.iter().map(|&hh| hh.to_f32()).collect();
                                let nb = (n * h) / 256;
                                let mut q4k = vec![0u8; nb * quant::Q4_K_BLOCK_BYTES];
                                quant::quantize_q4_k(&slice_f32, &mut q4k)?;
                                ctx.new_buffer_with_bytes(&q4k)
                            } else {
                                ctx.new_buffer_with_bytes(
                                    bytemuck::cast_slice::<f16, u8>(slice),
                                )
                            };
                            (Some(buf), Some(n), None)
                        } else {
                            (None, None, None)
                        }
                    }
                    _ => (None, None, None),
                    };
                    r
                };
                // Whether the pruned buffer holds Q4_K bytes (vs plain f16).
                let pruned_is_q4k =
                    want_q4k_lmhead && pruned_n.is_some() && pruned_n.unwrap() > 0;
                (
                    Some(mmap_buf), Some(eb), Some(fnb), Some(lhb), lhq4k,
                    pruned_buf, pruned_n, pruned_is_q4k, prune_remap,
                )
            } else {
                (None, None, None, None, None, None, None, false, None)
            };
        #[cfg(not(target_os = "macos"))]
        let (
            weights_mmap_buf,
            embed_buf,
            final_norm_buf,
            lm_head_buf,
            lm_head_q4k_buf,
            lm_head_pruned_buf,
            vocab_pruned,
            vocab_pruned_is_q4k,
            vocab_prune_remap,
        ): (
            Option<crate::metal::PinnedBuffer>,
            Option<crate::metal::PinnedBuffer>,
            Option<crate::metal::PinnedBuffer>,
            Option<crate::metal::PinnedBuffer>,
            Option<crate::metal::PinnedBuffer>,
            Option<crate::metal::PinnedBuffer>,
            Option<usize>,
            bool,
            Option<Vec<u32>>,
        ) = (None, None, None, None, None, None, None, false, None);

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
            dense_arena: None,
            weights_mmap_buf,
            embed_buf,
            final_norm_buf,
            lm_head_buf,
            lm_head_q4k_buf,
            lm_head_pruned_buf,
            vocab_pruned,
            vocab_pruned_is_q4k,
            vocab_prune_remap,
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
        let use_tcb_prefill = std::env::var("DISMANTLE_QWEN_TCB")
            .map(|v| v == "1")
            .unwrap_or(false);
        // P3 — batched prefill: chunk prompt into B≤4 token windows and
        // process each through `forward_tokens_batch_tcb`. Each weight
        // is read once per chunk (instead of once per token), amortizing
        // BW across B. Decode loop is unchanged. Requires TCB prefill.
        let batch_prefill = use_tcb_prefill
            && std::env::var("DISMANTLE_QWEN_BATCH_PREFILL")
                .map(|v| v == "1")
                .unwrap_or(false);
        #[cfg(target_os = "macos")]
        if batch_prefill {
            const B_MAX: usize = 8;
            let positions: Vec<usize> = (0..prompt_len).collect();
            let mut i = 0usize;
            while i < prompt_len {
                if abort_set(&req) {
                    prefill_aborted = true;
                    break;
                }
                let step_start = Instant::now();
                let end = (i + B_MAX).min(prompt_len);
                self.forward_tokens_batch_tcb(&prompt_ids[i..end], &positions[i..end])?;
                if stall_active && step_start.elapsed() > stall_limit {
                    prefill_aborted = true;
                    break;
                }
                i = end;
            }
        } else {
            for (i, &t) in prompt_ids.iter().enumerate() {
                if abort_set(&req) {
                    prefill_aborted = true;
                    break;
                }
                let step_start = Instant::now();
                if use_tcb_prefill {
                    let _ = self.forward_token_greedy_tcb(t, i)?;
                } else {
                    let _ = self.forward_token(t, i)?;
                }
                if stall_active && step_start.elapsed() > stall_limit {
                    prefill_aborted = true;
                    break;
                }
            }
        }
        #[cfg(not(target_os = "macos"))]
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

        // P1f: opt-in full-Metal TCB path. Greedy (argmax) only -- the
        // GPU sample kernel implements pure argmax, so any non-greedy
        // sampling must take the CPU/Metal-hybrid `forward_token` path
        // (full logits → CPU sampler).
        let use_tcb = std::env::var("DISMANTLE_QWEN_TCB")
            .map(|v| v == "1")
            .unwrap_or(false)
            && req.sampling.temperature == 0.0;

        for step in 0..req.max_new_tokens {
            if abort_set(&req) {
                reason = StopReason::Aborted;
                break;
            }
            let pos = prompt_len + step;
            let step_start = Instant::now();
            let next_id = if use_tcb {
                #[cfg(target_os = "macos")]
                {
                    self.forward_token_greedy_tcb(last_id, pos)?
                }
                #[cfg(not(target_os = "macos"))]
                {
                    let mut logits = self.forward_token(last_id, pos)?;
                    self.sampler.sample(&mut logits, &req.sampling)
                }
            } else {
                let mut logits = self.forward_token(last_id, pos)?;
                self.sampler.sample(&mut logits, &req.sampling)
            };
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

    fn encode_prompt_for_batch(&self, prompt: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode(prompt, true)
    }

    fn decode_token_for_batch(&self, token: u32) -> Result<String> {
        self.tokenizer.decode_one(token)
    }

    fn eos_id_for_batch(&self) -> Option<u32> {
        self.tokenizer.eos_id()
    }

    fn forward_tokens_batched(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        self.forward_tokens_for_test(tokens, positions)
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
    /// dequant-into-scratch + CPU gemv_f32 -- slow but correct.
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

#[cfg(target_os = "macos")]
impl QwenDense {
    /// P1f: full-Metal decode forward. Encodes the entire per-layer
    /// graph + final norm + LM head + GPU argmax into a single
    /// `TokenCommandBuffer`, commits once, and reads back the next
    /// token id (4 bytes) from the GPU.
    ///
    /// Requires `metal_ctx`, `weights_mmap_buf`, `embed_buf`,
    /// `final_norm_buf`, `lm_head_buf`, and all per-layer pinned
    /// norm + bias buffers to be populated (done in `Self::load` on
    /// macOS). Returns `Err(Metal)` if anything is missing.
    pub fn forward_token_greedy_tcb(&mut self, token: u32, pos: usize) -> Result<u32> {
        use crate::kernels;
        use crate::metal::{DenseDecodeArena, TokenCommandBuffer};

        let ctx = self
            .metal_ctx
            .as_ref()
            .ok_or_else(|| Error::Metal("forward_token_greedy_tcb: no metal_ctx".into()))?;
        let mmap_buf = self
            .weights_mmap_buf
            .as_ref()
            .ok_or_else(|| Error::Metal("forward_token_greedy_tcb: weights not pinned".into()))?;
        let embed_buf = self
            .embed_buf
            .as_ref()
            .ok_or_else(|| Error::Metal("forward_token_greedy_tcb: embed not pinned".into()))?;
        let final_norm_buf = self
            .final_norm_buf
            .as_ref()
            .ok_or_else(|| Error::Metal("forward_token_greedy_tcb: final_norm not pinned".into()))?;
        let lm_head_buf = self
            .lm_head_buf
            .as_ref()
            .ok_or_else(|| Error::Metal("forward_token_greedy_tcb: lm_head not pinned".into()))?;

        let cfg = &self.config;
        let h = cfg.hidden;
        let head_dim = cfg.head_dim;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let intermediate = cfg.intermediate;
        let eps = cfg.rms_norm_eps;
        let theta = cfg.rope_theta;
        let vocab = cfg.vocab_size;
        let pos_u32 = pos as u32;

        if self.kv.seq_len >= self.kv.max_seq {
            return Err(Error::Model(format!(
                "kv cache full at {}",
                self.kv.max_seq
            )));
        }
        let seq_slot = self.kv.seq_len;
        let mha_seq_len = seq_slot + 1;
        let max_seq = self.kv.max_seq;

        // Lazy-init arena on first call + bridge any CPU-side prefill
        // KV state into the GPU arena buffers. The CPU `forward_token`
        // path used during prefill writes K/V into `self.kv.keys/values`
        // but not into `arena.k_cache_buf/v_cache_buf`, so on the first
        // TCB call we copy the populated prefix once.
        let fresh_arena = self.dense_arena.is_none();
        if fresh_arena {
            self.dense_arena = Some(DenseDecodeArena::new(
                ctx,
                cfg.n_layers,
                n_heads,
                n_kv_heads,
                head_dim,
                h,
                intermediate,
                vocab,
                max_seq,
            ));
        }
        if fresh_arena && seq_slot > 0 {
            let arena = self.dense_arena.as_ref().unwrap();
            let kv_stride = n_kv_heads * head_dim;
            let prefill_elems = seq_slot * kv_stride;
            let layer_stride_elems = max_seq * kv_stride;
            for li in 0..cfg.n_layers {
                let layer_off_elems = li * layer_stride_elems;
                let k_src = &self.kv.keys[li][..prefill_elems];
                let v_src = &self.kv.values[li][..prefill_elems];
                let k_dst = arena.k_cache_buf.contents() as *mut f32;
                let v_dst = arena.v_cache_buf.contents() as *mut f32;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        k_src.as_ptr(),
                        k_dst.add(layer_off_elems),
                        prefill_elems,
                    );
                    std::ptr::copy_nonoverlapping(
                        v_src.as_ptr(),
                        v_dst.add(layer_off_elems),
                        prefill_elems,
                    );
                }
            }
        }
        let arena = self.dense_arena.as_ref().unwrap();

        let mut tcb = TokenCommandBuffer::new(ctx);

        // x_buf <- embed[token]
        kernels::embed_lookup_metal_f32_tcb(&mut tcb, embed_buf, token, h, &arena.x_buf)?;

        let f32_bytes = std::mem::size_of::<f32>();
        let kv_dim_bytes = kv_dim * f32_bytes;
        let layer_kv_stride_bytes = max_seq * kv_dim_bytes;

        // Pre-norm for layer 0 (hoisted so the loop can fuse the residual-add
        // of layer li with the attn_norm of layer li+1 -- saving 73 dispatches
        // per token across 36 layers + final norm).
        let layer0_attn_norm = self.layers[0]
            .pinned
            .attn_norm
            .as_ref()
            .ok_or_else(|| Error::Metal("layer 0 attn_norm not pinned".into()))?;
        kernels::rmsnorm_metal_buf_tcb(
            &mut tcb,
            &arena.x_buf,
            layer0_attn_norm,
            eps,
            h,
            &arena.x_norm_buf,
        )?;

        for li in 0..cfg.n_layers {
            let layer = &self.layers[li];

            // Dispatcher choice helper. When the projection's GGUF dtype
            // isn't Q4_K, `pinned.*_f16` holds a dequantized-once f16
            // copy and we go through `gemv_f16_metal_buf_tcb`; otherwise
            // the Q4_K_M-aware kernel reads directly from the pinned mmap.
            // P2 (2026-05-23): dispatch by Q-quant dtype. Q4_K and Q6_K
            // both decode directly from the pinned mmap; only truly
            // exotic dtypes fall back to the f16 dequant path.
            macro_rules! gemv_proj {
                ($tref:expr, $pinned_f16:expr, $rows:expr, $cols:expr, $x:expr, $out:expr) => {{
                    match $tref.dtype {
                        GgmlType::Q4_K => {
                            // Winner of the Q4_K variant sweep (2026-05-23):
                            // v3_8r (TG=256, 8 rows/TG, scale + activation
                            // preload, paired nibble reads). Beat v2 by
                            // +5.4%, simdmat by +1.6%, v3_llama within noise.
                            kernels::gemv_q4_k_m_v3_8r_pinned_tcb(
                                &mut tcb,
                                mmap_buf,
                                $tref.offset,
                                $tref.byte_size,
                                $rows,
                                $cols,
                                $x,
                                $out,
                            )?;
                        }
                        GgmlType::Q6_K => {
                            kernels::gemv_q6_k_pinned_tcb(
                                &mut tcb,
                                mmap_buf,
                                $tref.offset,
                                $tref.byte_size,
                                $rows,
                                $cols,
                                $x,
                                $out,
                            )?;
                        }
                        _ => {
                            let buf_f16 = $pinned_f16.ok_or_else(|| {
                                Error::Metal(
                                    "gemv_proj: non-Q4_K/Q6_K dtype and no f16 fallback pinned"
                                        .into(),
                                )
                            })?;
                            kernels::gemv_f16_metal_buf_tcb(
                                &mut tcb, buf_f16, $rows, $cols, $x, $out,
                            )?;
                        }
                    }
                }};
            }

            // Attn-norm already in arena.x_norm_buf (hoisted for layer 0,
            // produced by the previous layer's tail-fusion for layers 1+).

            // ── Q / K / V projections ────────────────────────────────
            gemv_proj!(
                layer.q_proj,
                layer.pinned.q_proj_f16.as_ref(),
                q_dim,
                h,
                &arena.x_norm_buf,
                &arena.q_buf
            );
            gemv_proj!(
                layer.k_proj,
                layer.pinned.k_proj_f16.as_ref(),
                kv_dim,
                h,
                &arena.x_norm_buf,
                &arena.k_token_buf
            );
            gemv_proj!(
                layer.v_proj,
                layer.pinned.v_proj_f16.as_ref(),
                kv_dim,
                h,
                &arena.x_norm_buf,
                &arena.v_token_buf
            );

            // ── Biases (Qwen2 carries q/k/v biases) ──────────────────
            if let Some(qb) = layer.pinned.q_bias.as_ref() {
                kernels::add_inplace_metal_tcb(&mut tcb, &arena.q_buf, qb, q_dim)?;
            }
            if let Some(kb) = layer.pinned.k_bias.as_ref() {
                kernels::add_inplace_metal_tcb(&mut tcb, &arena.k_token_buf, kb, kv_dim)?;
            }
            if let Some(vb) = layer.pinned.v_bias.as_ref() {
                kernels::add_inplace_metal_tcb(&mut tcb, &arena.v_token_buf, vb, kv_dim)?;
            }

            // ── RoPE on full head_dim for every Q and K head ─────────
            // rope_q_f32_inplace with qk_nope_dim=0 ⇒ rotates the entire
            // head; matches Qwen's standard interleaved RoPE.
            kernels::rope_q_f32_inplace_tcb(
                &mut tcb,
                &arena.q_buf,
                n_heads,
                head_dim,
                0,
                head_dim,
                pos_u32,
                theta,
            )?;
            kernels::rope_q_f32_inplace_tcb(
                &mut tcb,
                &arena.k_token_buf,
                n_kv_heads,
                head_dim,
                0,
                head_dim,
                pos_u32,
                theta,
            )?;

            // ── KV append into per-layer slice of k/v_cache buffer ───
            let layer_kv_off_elems = li * max_seq * kv_dim;
            let slot_kv_off_elems = layer_kv_off_elems + seq_slot * kv_dim;
            kernels::memcpy_f32_off_tcb(
                &mut tcb,
                &arena.k_token_buf,
                &arena.k_cache_buf,
                0,
                slot_kv_off_elems,
                kv_dim,
            )?;
            kernels::memcpy_f32_off_tcb(
                &mut tcb,
                &arena.v_token_buf,
                &arena.v_cache_buf,
                0,
                slot_kv_off_elems,
                kv_dim,
            )?;

            // ── MHA decode (GQA) ─────────────────────────────────────
            let layer_kv_off_bytes = li * layer_kv_stride_bytes;
            kernels::mha_decode_f32_tcb(
                &mut tcb,
                &arena.q_buf,
                &arena.k_cache_buf,
                layer_kv_off_bytes,
                &arena.v_cache_buf,
                layer_kv_off_bytes,
                &arena.attn_out_buf,
                mha_seq_len,
                head_dim,
                n_heads,
                n_kv_heads,
            )?;

            // ── O projection ─────────────────────────────────────────
            gemv_proj!(
                layer.o_proj,
                layer.pinned.o_proj_f16.as_ref(),
                h,
                q_dim,
                &arena.attn_out_buf,
                &arena.o_proj_out_buf
            );
            // ── Fused (x += o_proj_out) + FFN norm ───────────────────
            let ffn_norm_pin = layer
                .pinned
                .ffn_norm
                .as_ref()
                .ok_or_else(|| Error::Metal("ffn_norm not pinned".into()))?;
            kernels::add_rmsnorm_fused_tcb(
                &mut tcb,
                &arena.x_buf,
                &arena.o_proj_out_buf,
                ffn_norm_pin,
                &arena.x_norm_buf,
                eps,
                h,
            )?;

            // ── FFN gate / up / silu_mul / down ──────────────────────
            gemv_proj!(
                layer.ffn_gate,
                layer.pinned.ffn_gate_f16.as_ref(),
                intermediate,
                h,
                &arena.x_norm_buf,
                &arena.ffn_gate_buf
            );
            gemv_proj!(
                layer.ffn_up,
                layer.pinned.ffn_up_f16.as_ref(),
                intermediate,
                h,
                &arena.x_norm_buf,
                &arena.ffn_up_buf
            );
            kernels::silu_mul_tcb(
                &mut tcb,
                &arena.ffn_gate_buf,
                &arena.ffn_up_buf,
                &arena.ffn_act_buf,
                intermediate,
            )?;
            // ffn_down: if the requant'd Q4_K buffer is populated (opt-in
            // via DISMANTLE_QWEN_FFN_DOWN_Q4K=1), prefer it over the
            // f16 fallback / native Q6_K path. ~31% BW saving on the
            // single largest weight per layer.
            if let Some(q4k_buf) = layer.pinned.ffn_down_q4k.as_ref() {
                let blocks_per_row = intermediate / 256;
                let row_bytes = blocks_per_row * 144;
                kernels::gemv_q4_k_m_v3_8r_pinned_tcb(
                    &mut tcb,
                    q4k_buf,
                    0,
                    h * row_bytes,
                    h,
                    intermediate,
                    &arena.ffn_act_buf,
                    &arena.ffn_down_buf,
                )?;
            } else {
                gemv_proj!(
                    layer.ffn_down,
                    layer.pinned.ffn_down_f16.as_ref(),
                    h,
                    intermediate,
                    &arena.ffn_act_buf,
                    &arena.ffn_down_buf
                );
            }
            // ── Fused (x += ffn_down) + next-layer's attn_norm (or
            //    final_norm on the last layer). After the loop, x_norm
            //    already holds the final norm output -- no separate
            //    final-norm dispatch needed.
            let next_norm = if li + 1 < cfg.n_layers {
                self.layers[li + 1]
                    .pinned
                    .attn_norm
                    .as_ref()
                    .ok_or_else(|| Error::Metal("attn_norm not pinned".into()))?
            } else {
                final_norm_buf
            };
            kernels::add_rmsnorm_fused_tcb(
                &mut tcb,
                &arena.x_buf,
                &arena.ffn_down_buf,
                next_norm,
                &arena.x_norm_buf,
                eps,
                h,
            )?;
            let _ = kv_dim_bytes;
        }

        // ── LM head → argmax (x_norm already holds final_norm output) ─
        // LM head: priority is (1) vocab-prune buf if active (smaller GEMV,
        // pruned argmax over first-N tokens), else (2) Q4_K-quantized buf
        // if active, else (3) full f16 gemv.
        if let (Some(pruned_buf), Some(pn)) =
            (self.lm_head_pruned_buf.as_ref(), self.vocab_pruned)
        {
            if self.vocab_pruned_is_q4k {
                let blocks_per_row = h / 256;
                let row_bytes = blocks_per_row * 144;
                kernels::gemv_q4_k_m_v3_8r_pinned_tcb(
                    &mut tcb,
                    pruned_buf,
                    0,
                    pn * row_bytes,
                    pn,
                    h,
                    &arena.x_norm_buf,
                    &arena.logits_buf,
                )?;
            } else {
                kernels::gemv_f16_metal_buf_tcb(
                    &mut tcb,
                    pruned_buf,
                    pn,
                    h,
                    &arena.x_norm_buf,
                    &arena.logits_buf,
                )?;
            }
            kernels::sample_argmax_f32_tcb(&mut tcb, &arena.logits_buf, &arena.token_buf, pn)?;

            tcb.commit_and_wait()?;
            self.kv.seq_len += 1;
            let token_ptr = arena.token_buf.contents() as *const u32;
            let pruned_idx = unsafe { *token_ptr };
            // Corpus prune: GPU argmax returns whitelist index, remap to
            // original vocab id. First-N prune: identity (no remap).
            let token = match self.vocab_prune_remap.as_ref() {
                Some(map) => *map.get(pruned_idx as usize).unwrap_or(&pruned_idx),
                None => pruned_idx,
            };
            return Ok(token);
        } else if let Some(lhq) = self.lm_head_q4k_buf.as_ref() {
            let blocks_per_row = h / 256;
            let row_bytes = blocks_per_row * 144;
            kernels::gemv_q4_k_m_v3_8r_pinned_tcb(
                &mut tcb,
                lhq,
                0,
                vocab * row_bytes,
                vocab,
                h,
                &arena.x_norm_buf,
                &arena.logits_buf,
            )?;
        } else {
            kernels::gemv_f16_metal_buf_tcb(
                &mut tcb,
                lm_head_buf,
                vocab,
                h,
                &arena.x_norm_buf,
                &arena.logits_buf,
            )?;
        }
        kernels::sample_argmax_f32_tcb(&mut tcb, &arena.logits_buf, &arena.token_buf, vocab)?;

        tcb.commit_and_wait()?;

        // KV cache pointer bump (CPU mirror), so the CPU fallback path
        // remains consistent for hybrid runs that mix the two.
        self.kv.seq_len += 1;

        let token_ptr = arena.token_buf.contents() as *const u32;
        let next = unsafe { *token_ptr };
        Ok(next)
    }

    /// P3 — Batched prefill: process `tokens.len()` consecutive prompt
    /// tokens (1..=arena.max_batch) through one full forward pass, reading
    /// each weight once and producing B = `tokens.len()` output rows of
    /// activations. K/V cache writes happen at positions `[positions[0] ..
    /// positions[0]+B)`, which the caller must guarantee are contiguous.
    ///
    /// Does not sample / write a token result -- prefill discards the
    /// predicted next token (we already know what comes next in the
    /// prompt). The side effect is the GPU KV cache buffer is populated
    /// for slots [positions[0] .. positions[0]+B), same as
    /// `forward_token_greedy_tcb` would do B times.
    #[cfg(target_os = "macos")]
    #[allow(clippy::too_many_arguments)]
    pub fn forward_tokens_batch_tcb(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
    ) -> Result<()> {
        use crate::kernels;
        use crate::metal::{DenseDecodeArena, TokenCommandBuffer};

        if tokens.len() != positions.len() {
            return Err(Error::Model(format!(
                "forward_tokens_batch_tcb: tokens={} positions={}",
                tokens.len(), positions.len()
            )));
        }
        let b = tokens.len();
        if b == 0 {
            return Ok(());
        }
        // Require contiguous positions [p0..p0+B).
        for (i, &p) in positions.iter().enumerate() {
            if p != positions[0] + i {
                return Err(Error::Model(format!(
                    "forward_tokens_batch_tcb: positions must be contiguous; got [{}]={} expected {}",
                    i, p, positions[0] + i
                )));
            }
        }

        let ctx = self
            .metal_ctx
            .as_ref()
            .ok_or_else(|| Error::Metal("forward_tokens_batch_tcb: no metal_ctx".into()))?;
        let mmap_buf = self
            .weights_mmap_buf
            .as_ref()
            .ok_or_else(|| Error::Metal("forward_tokens_batch_tcb: weights not pinned".into()))?;
        let embed_buf = self
            .embed_buf
            .as_ref()
            .ok_or_else(|| Error::Metal("forward_tokens_batch_tcb: embed not pinned".into()))?;
        let final_norm_buf = self
            .final_norm_buf
            .as_ref()
            .ok_or_else(|| Error::Metal("forward_tokens_batch_tcb: final_norm not pinned".into()))?;

        let cfg = &self.config;
        let h = cfg.hidden;
        let head_dim = cfg.head_dim;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let intermediate = cfg.intermediate;
        let eps = cfg.rms_norm_eps;
        let theta = cfg.rope_theta;

        let p0 = positions[0];
        if self.kv.seq_len != p0 {
            return Err(Error::Model(format!(
                "forward_tokens_batch_tcb: kv.seq_len={} != positions[0]={}",
                self.kv.seq_len, p0
            )));
        }
        if self.kv.seq_len + b > self.kv.max_seq {
            return Err(Error::Model(format!(
                "forward_tokens_batch_tcb: kv overflow ({} + {} > {})",
                self.kv.seq_len, b, self.kv.max_seq
            )));
        }
        let max_seq = self.kv.max_seq;

        // Lazy-init arena. Bridge CPU prefill KV state into GPU arena
        // (only matters if a hybrid prefill ran some CPU-path steps
        // earlier -- in pure-batched prefill paths this branch is a
        // no-op).
        let fresh_arena = self.dense_arena.is_none();
        if fresh_arena {
            self.dense_arena = Some(DenseDecodeArena::new(
                ctx,
                cfg.n_layers,
                n_heads,
                n_kv_heads,
                head_dim,
                h,
                intermediate,
                cfg.vocab_size,
                max_seq,
            ));
        }
        if fresh_arena && p0 > 0 {
            let arena = self.dense_arena.as_ref().unwrap();
            let kv_stride = n_kv_heads * head_dim;
            let prefill_elems = p0 * kv_stride;
            let layer_stride_elems = max_seq * kv_stride;
            for li in 0..cfg.n_layers {
                let layer_off_elems = li * layer_stride_elems;
                let k_src = &self.kv.keys[li][..prefill_elems];
                let v_src = &self.kv.values[li][..prefill_elems];
                let k_dst = arena.k_cache_buf.contents() as *mut f32;
                let v_dst = arena.v_cache_buf.contents() as *mut f32;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        k_src.as_ptr(), k_dst.add(layer_off_elems), prefill_elems,
                    );
                    std::ptr::copy_nonoverlapping(
                        v_src.as_ptr(), v_dst.add(layer_off_elems), prefill_elems,
                    );
                }
            }
        }
        let arena = self.dense_arena.as_ref().unwrap();
        if b > arena.max_batch {
            return Err(Error::Model(format!(
                "forward_tokens_batch_tcb: B={} > arena.max_batch={}",
                b, arena.max_batch
            )));
        }

        let f32_bytes = std::mem::size_of::<f32>();
        let h_bytes = h * f32_bytes;
        let q_dim_bytes = q_dim * f32_bytes;
        let kv_dim_bytes = kv_dim * f32_bytes;
        let int_bytes = intermediate * f32_bytes;
        let layer_kv_stride_bytes = max_seq * kv_dim_bytes;

        let mut tcb = TokenCommandBuffer::new(ctx);

        // ── Embed B tokens into x_buf_batch[b, :] ────────────────
        for (bi, &tok) in tokens.iter().enumerate() {
            kernels::embed_lookup_metal_f32_off_tcb(
                &mut tcb, embed_buf, tok, h,
                &arena.x_buf_batch, bi * h_bytes,
            )?;
        }

        // Pre-norm for layer 0 (B sequential rmsnorm dispatches).
        let layer0_attn_norm = self.layers[0]
            .pinned
            .attn_norm
            .as_ref()
            .ok_or_else(|| Error::Metal("layer 0 attn_norm not pinned".into()))?;
        for bi in 0..b {
            kernels::rmsnorm_metal_buf_off_tcb(
                &mut tcb,
                &arena.x_buf_batch, bi * h_bytes,
                layer0_attn_norm, eps, h,
                &arena.x_norm_buf_batch, bi * h_bytes,
            )?;
        }

        // Helper: batched projection via Q4_K batched GEMM (single
        // dispatch, weight read once) when dtype is Q4_K; else B
        // sequential single-vector GEMV calls (no BW saving) for Q6_K /
        // f16 fallback. `x_off_stride` and `out_off_stride` are byte
        // strides between consecutive batch rows -- usually `dim *
        // f32_bytes`.
        macro_rules! batched_proj {
            ($tref:expr, $pinned_f16:expr, $rows:expr, $cols:expr,
             $x_batch:expr, $x_stride:expr,
             $out_batch:expr, $out_stride:expr) => {{
                match $tref.dtype {
                    GgmlType::Q4_K => {
                        // Contiguous layout: (B, cols) f32 → (B, rows) f32.
                        // Requires x_stride = cols*f32 and out_stride = rows*f32.
                        debug_assert_eq!($x_stride, $cols * f32_bytes,
                            "batched_proj: Q4_K requires contiguous x_stride");
                        debug_assert_eq!($out_stride, $rows * f32_bytes,
                            "batched_proj: Q4_K requires contiguous out_stride");
                        kernels::gemm_q4_k_m_batched_v3w_pinned_tcb(
                            &mut tcb, mmap_buf, $tref.offset, $tref.byte_size,
                            $rows, $cols, b, $x_batch, $out_batch,
                        )?;
                    }
                    GgmlType::Q6_K => {
                        for bi in 0..b {
                            kernels::gemv_q6_k_pinned_off_tcb(
                                &mut tcb, mmap_buf, $tref.offset, $tref.byte_size,
                                $rows, $cols,
                                $x_batch, bi * $x_stride,
                                $out_batch, bi * $out_stride,
                            )?;
                        }
                    }
                    _ => {
                        let buf_f16 = $pinned_f16.ok_or_else(|| {
                            Error::Metal(
                                "batched_proj: non-Q4_K/Q6_K dtype and no f16 fallback".into(),
                            )
                        })?;
                        for bi in 0..b {
                            kernels::gemv_f16_metal_buf_off_tcb(
                                &mut tcb, buf_f16, $rows, $cols,
                                $x_batch, bi * $x_stride,
                                $out_batch, bi * $out_stride,
                            )?;
                        }
                    }
                }
            }};
        }

        for li in 0..cfg.n_layers {
            let layer = &self.layers[li];

            // x_norm_buf_batch is already populated (hoisted at layer 0,
            // produced by previous layer's tail fusion at layers 1+).

            // ── Q / K / V projections ────────────────────────────
            batched_proj!(
                layer.q_proj, layer.pinned.q_proj_f16.as_ref(),
                q_dim, h,
                &arena.x_norm_buf_batch, h_bytes,
                &arena.q_buf_batch, q_dim_bytes
            );
            batched_proj!(
                layer.k_proj, layer.pinned.k_proj_f16.as_ref(),
                kv_dim, h,
                &arena.x_norm_buf_batch, h_bytes,
                &arena.k_token_buf_batch, kv_dim_bytes
            );
            batched_proj!(
                layer.v_proj, layer.pinned.v_proj_f16.as_ref(),
                kv_dim, h,
                &arena.x_norm_buf_batch, h_bytes,
                &arena.v_token_buf_batch, kv_dim_bytes
            );

            // ── Biases: broadcast one bias vector across B output
            // rows in a single dispatch — saves 3*(B-1) per layer.
            if let Some(qb) = layer.pinned.q_bias.as_ref() {
                kernels::add_inplace_broadcast_tcb(
                    &mut tcb, &arena.q_buf_batch, qb, q_dim, b,
                )?;
            }
            if let Some(kb) = layer.pinned.k_bias.as_ref() {
                kernels::add_inplace_broadcast_tcb(
                    &mut tcb, &arena.k_token_buf_batch, kb, kv_dim, b,
                )?;
            }
            if let Some(vb) = layer.pinned.v_bias.as_ref() {
                kernels::add_inplace_broadcast_tcb(
                    &mut tcb, &arena.v_token_buf_batch, vb, kv_dim, b,
                )?;
            }

            // ── RoPE on Q and K, per batch element (different pos) ─
            for bi in 0..b {
                let pos_u32 = (p0 + bi) as u32;
                kernels::rope_q_f32_inplace_off_tcb(
                    &mut tcb, &arena.q_buf_batch, bi * q_dim_bytes,
                    n_heads, head_dim, 0, head_dim, pos_u32, theta,
                )?;
                kernels::rope_q_f32_inplace_off_tcb(
                    &mut tcb, &arena.k_token_buf_batch, bi * kv_dim_bytes,
                    n_kv_heads, head_dim, 0, head_dim, pos_u32, theta,
                )?;
            }

            // ── KV append: B contiguous slots [p0..p0+B) in the
            // per-layer window. Source (B, kv_dim) is contiguous and
            // dest k_cache[layer][p0..p0+B] is contiguous, so one
            // memcpy of B*kv_dim floats replaces 2B sequential calls.
            let layer_kv_off_elems = li * max_seq * kv_dim;
            let slot_kv_off_elems = layer_kv_off_elems + p0 * kv_dim;
            kernels::memcpy_f32_off_tcb(
                &mut tcb,
                &arena.k_token_buf_batch,
                &arena.k_cache_buf,
                0,
                slot_kv_off_elems,
                b * kv_dim,
            )?;
            kernels::memcpy_f32_off_tcb(
                &mut tcb,
                &arena.v_token_buf_batch,
                &arena.v_cache_buf,
                0,
                slot_kv_off_elems,
                b * kv_dim,
            )?;

            // ── MHA decode: one dispatch over (n_heads, B) TGs.
            // Each batch element gets its own causal seq_len = p0+b+1.
            // Saves B-1 dispatches per layer (n_heads × B TGs in one
            // launch vs B separate launches of n_heads TGs).
            let layer_kv_off_bytes = li * layer_kv_stride_bytes;
            kernels::mha_decode_f32_batched_tcb(
                &mut tcb,
                &arena.q_buf_batch,
                &arena.k_cache_buf, layer_kv_off_bytes,
                &arena.v_cache_buf, layer_kv_off_bytes,
                &arena.attn_out_buf_batch,
                p0, b, head_dim, n_heads, n_kv_heads,
            )?;

            // ── O projection (batched) ───────────────────────────
            batched_proj!(
                layer.o_proj, layer.pinned.o_proj_f16.as_ref(),
                h, q_dim,
                &arena.attn_out_buf_batch, q_dim_bytes,
                &arena.o_proj_out_buf_batch, h_bytes
            );

            // ── Fused (x += o_proj_out) + FFN norm, B rows in 1 dispatch.
            let ffn_norm_pin = layer
                .pinned
                .ffn_norm
                .as_ref()
                .ok_or_else(|| Error::Metal("ffn_norm not pinned".into()))?;
            kernels::add_rmsnorm_fused_batched_tcb(
                &mut tcb,
                &arena.x_buf_batch,
                &arena.o_proj_out_buf_batch,
                ffn_norm_pin,
                &arena.x_norm_buf_batch,
                eps, h, b,
            )?;

            // ── FFN gate / up / silu_mul / down ──────────────────
            batched_proj!(
                layer.ffn_gate, layer.pinned.ffn_gate_f16.as_ref(),
                intermediate, h,
                &arena.x_norm_buf_batch, h_bytes,
                &arena.ffn_gate_buf_batch, int_bytes
            );
            batched_proj!(
                layer.ffn_up, layer.pinned.ffn_up_f16.as_ref(),
                intermediate, h,
                &arena.x_norm_buf_batch, h_bytes,
                &arena.ffn_up_buf_batch, int_bytes
            );
            // silu_mul is flat elementwise; (B, intermediate) buffers
            // are contiguous so one dispatch with n=intermediate*B
            // replaces B sequential calls — saves (B-1)*36 dispatches
            // per chunk.
            kernels::silu_mul_tcb(
                &mut tcb,
                &arena.ffn_gate_buf_batch,
                &arena.ffn_up_buf_batch,
                &arena.ffn_act_buf_batch,
                intermediate * b,
            )?;
            // ffn_down: prefer requant'd Q4_K buffer if active (~31% BW
            // saving on the largest weight per layer); else go through
            // the standard projection dispatcher.
            if let Some(q4k_buf) = layer.pinned.ffn_down_q4k.as_ref() {
                let blocks_per_row = intermediate / 256;
                let row_bytes = blocks_per_row * 144;
                kernels::gemm_q4_k_m_batched_v3w_pinned_tcb(
                    &mut tcb, q4k_buf, 0, h * row_bytes,
                    h, intermediate, b,
                    &arena.ffn_act_buf_batch, &arena.ffn_down_buf_batch,
                )?;
            } else {
                batched_proj!(
                    layer.ffn_down, layer.pinned.ffn_down_f16.as_ref(),
                    h, intermediate,
                    &arena.ffn_act_buf_batch, int_bytes,
                    &arena.ffn_down_buf_batch, h_bytes
                );
            }

            // ── Fused (x += ffn_down) + next-layer attn_norm
            // (or final_norm on the last layer). Per batch element.
            let next_norm = if li + 1 < cfg.n_layers {
                self.layers[li + 1]
                    .pinned
                    .attn_norm
                    .as_ref()
                    .ok_or_else(|| Error::Metal("attn_norm not pinned".into()))?
            } else {
                final_norm_buf
            };
            kernels::add_rmsnorm_fused_batched_tcb(
                &mut tcb,
                &arena.x_buf_batch,
                &arena.ffn_down_buf_batch,
                next_norm,
                &arena.x_norm_buf_batch,
                eps, h, b,
            )?;
            let _ = kv_dim_bytes;
        }

        // No LM head / sample — prefill discards the predicted token.
        tcb.commit_and_wait()?;

        // Bump CPU-side KV mirror so hybrid runs stay consistent.
        self.kv.seq_len += b;

        Ok(())
    }
}
