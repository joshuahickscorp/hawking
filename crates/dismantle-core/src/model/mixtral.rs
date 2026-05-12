//! Mixtral 8x7B loader and forward-path scaffold.
//!
//! Mixtral GGUFs in the target Q4_K_M export use split per-expert tensors:
//! `blk.N.ffn_gate.E.weight`, `blk.N.ffn_up.E.weight`, and
//! `blk.N.ffn_down.E.weight`. This module intentionally does not look for the
//! fused `*_exps.weight` layout used by the DeepSeek path.

use crate::engine::{Engine, EngineConfig, GenStats, GenerateRequest, StreamEvent};
use crate::gguf::{GgmlType, GgufFile};
use crate::metal::{MetalContext, PinnedBuffer};
use crate::quant;
use crate::tokenizer::Tokenizer;
use crate::{Error, Result};
use half::f16;
use std::path::{Path, PathBuf};

const WEIGHT_CHUNK_STRIDE: usize = 128 * 1024 * 1024;
const WEIGHT_CHUNK_OVERLAP: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq)]
pub struct MixtralConfig {
    pub n_layers: usize,
    pub hidden: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
}

impl MixtralConfig {
    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let get_u32 = |k: &str| g.metadata.get(k).and_then(|v| v.as_u32());
        let get_f32 = |k: &str| g.metadata.get(k).and_then(|v| v.as_f32());

        let n_layers = get_u32("llama.block_count")
            .ok_or_else(|| Error::Model("missing llama.block_count".into()))?
            as usize;
        let hidden = get_u32("llama.embedding_length")
            .ok_or_else(|| Error::Model("missing llama.embedding_length".into()))?
            as usize;
        let n_heads = get_u32("llama.attention.head_count")
            .ok_or_else(|| Error::Model("missing llama.attention.head_count".into()))?
            as usize;
        let n_kv_heads =
            get_u32("llama.attention.head_count_kv").unwrap_or(n_heads as u32) as usize;
        let intermediate = get_u32("llama.feed_forward_length")
            .ok_or_else(|| Error::Model("missing llama.feed_forward_length".into()))?
            as usize;
        let n_experts = get_u32("llama.expert_count")
            .ok_or_else(|| Error::Model("missing llama.expert_count".into()))?
            as usize;
        let top_k = get_u32("llama.expert_used_count").unwrap_or(2) as usize;
        let vocab_size = get_u32("llama.vocab_size")
            .map(|v| v as usize)
            .or_else(|| {
                g.tensor("token_embd.weight")
                    .and_then(|t| t.dims.iter().copied().max())
                    .map(|v| v as usize)
            })
            .ok_or_else(|| Error::Model("missing llama vocab size".into()))?;

        if hidden % n_heads != 0 {
            return Err(Error::Model(format!(
                "Mixtral hidden/head mismatch: hidden={hidden} n_heads={n_heads}"
            )));
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
            rope_theta: get_f32("llama.rope.freq_base").unwrap_or(1_000_000.0),
            rms_norm_eps: get_f32("llama.attention.layer_norm_rms_epsilon").unwrap_or(1e-5),
            max_seq_len: get_u32("llama.context_length").unwrap_or(32768) as usize,
        })
    }

    pub fn synthetic_for_test() -> Self {
        Self {
            n_layers: 2,
            hidden: 256,
            n_heads: 8,
            n_kv_heads: 2,
            head_dim: 32,
            intermediate: 512,
            n_experts: 8,
            top_k: 2,
            vocab_size: 128,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-5,
            max_seq_len: 256,
        }
    }
}

/// Pointer into the mmap'd GGUF for one quantized tensor.
#[derive(Debug, Clone)]
pub struct TensorRef {
    pub offset: usize,
    pub byte_size: usize,
    pub dtype: GgmlType,
    pub n_elems: usize,
    pub rows: usize,
    pub cols: usize,
    pub chunk_index: usize,
    pub chunk_offset: usize,
}

pub struct MixtralLayer {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub attn_q: TensorRef,
    pub attn_output: TensorRef,
    pub ffn_gate: Vec<TensorRef>,
    pub ffn_up: Vec<TensorRef>,
    pub ffn_down: Vec<TensorRef>,
    pub pinned: MixtralLayerPinned,
}

#[derive(Default)]
pub struct MixtralLayerPinned {
    pub attn_norm: Option<PinnedBuffer>,
    pub ffn_norm: Option<PinnedBuffer>,
    pub attn_k: Option<PinnedBuffer>,
    pub attn_v: Option<PinnedBuffer>,
    pub gate_inp: Option<PinnedBuffer>,
}

pub struct MixtralEngine {
    pub config: MixtralConfig,
    pub tokenizer: Tokenizer,
    pub model_id: String,
    pub gguf: GgufFile,
    pub embed: Vec<f16>,
    pub final_norm: Vec<f32>,
    pub lm_head: Option<Vec<f16>>,
    pub layers: Vec<MixtralLayer>,
    pub _weights_path: PathBuf,
    pub metal_ctx: Option<MetalContext>,
    pub weights_mmap_buf: Option<Vec<PinnedBuffer>>,
    pub embed_buf: Option<PinnedBuffer>,
    pub final_norm_buf: Option<PinnedBuffer>,
    pub lm_head_buf: Option<PinnedBuffer>,
    pub decode_arena: Option<MixtralDecodeArena>,
}

#[cfg(target_os = "macos")]
pub struct MixtralDecodeArena {
    pub x_buf: PinnedBuffer,
    pub x_norm_buf: PinnedBuffer,
    pub q_buf: PinnedBuffer,
    pub k_buf: PinnedBuffer,
    pub v_buf: PinnedBuffer,
    pub attn_out_buf: PinnedBuffer,
    pub proj_out_buf: PinnedBuffer,
    pub gate_logits_buf: PinnedBuffer,
    pub expert_gate_bufs: Vec<PinnedBuffer>,
    pub expert_up_bufs: Vec<PinnedBuffer>,
    pub expert_act_bufs: Vec<PinnedBuffer>,
    pub expert_out_bufs: Vec<PinnedBuffer>,
    pub ffn_out_buf: PinnedBuffer,
    pub logits_buf: PinnedBuffer,
}

#[cfg(target_os = "macos")]
impl MixtralDecodeArena {
    fn new(ctx: &MetalContext, cfg: &MixtralConfig) -> Self {
        let f32_bytes = std::mem::size_of::<f32>();
        let kv_hidden = cfg.n_kv_heads * cfg.head_dim;
        let scratch = |len: usize| ctx.new_buffer(len * f32_bytes);
        Self {
            x_buf: scratch(cfg.hidden),
            x_norm_buf: scratch(cfg.hidden),
            q_buf: scratch(cfg.hidden),
            k_buf: scratch(kv_hidden),
            v_buf: scratch(kv_hidden),
            attn_out_buf: scratch(cfg.hidden),
            proj_out_buf: scratch(cfg.hidden),
            gate_logits_buf: scratch(cfg.n_experts),
            expert_gate_bufs: (0..cfg.top_k).map(|_| scratch(cfg.intermediate)).collect(),
            expert_up_bufs: (0..cfg.top_k).map(|_| scratch(cfg.intermediate)).collect(),
            expert_act_bufs: (0..cfg.top_k).map(|_| scratch(cfg.intermediate)).collect(),
            expert_out_bufs: (0..cfg.top_k).map(|_| scratch(cfg.hidden)).collect(),
            ffn_out_buf: scratch(cfg.hidden),
            logits_buf: scratch(cfg.vocab_size),
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub struct MixtralDecodeArena;

impl MixtralEngine {
    pub fn load_tokenizer_preview(weights: &Path, gguf: &GgufFile) -> Result<Tokenizer> {
        let sidecar = weights
            .parent()
            .map(|d| d.join("tokenizer.json"))
            .filter(|p| p.exists());
        if let Some(path) = sidecar {
            Tokenizer::from_file(path)
        } else {
            Tokenizer::from_gguf(gguf)
        }
    }

    pub fn synthetic_forward_shape_for_test(config: &MixtralConfig, token: u32) -> Vec<f32> {
        (0..config.vocab_size)
            .map(|i| ((i as u32 ^ token) as f32) / config.vocab_size as f32)
            .collect()
    }

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

    fn dequant_f32_expected(
        g: &GgufFile,
        name: &str,
        dtype: GgmlType,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>> {
        let t = Self::tensor_ref_expected(g, name, dtype, rows, cols)?;
        let bytes = &g.mmap[t.offset..t.offset + t.byte_size];
        let mut out = vec![0.0f32; t.n_elems];
        quant::dequant_into(t.dtype, bytes, &mut out)?;
        Ok(out)
    }

    fn tensor_ref_q4(g: &GgufFile, name: &str, rows: usize, cols: usize) -> Result<TensorRef> {
        Self::tensor_ref_expected(g, name, GgmlType::Q4_K, rows, cols)
    }

    fn tensor_ref_expected(
        g: &GgufFile,
        name: &str,
        dtype: GgmlType,
        rows: usize,
        cols: usize,
    ) -> Result<TensorRef> {
        let info = g
            .tensor(name)
            .ok_or_else(|| Error::Model(format!("missing tensor `{name}`")))?;
        let n_elems: usize = info.dims.iter().product::<u64>() as usize;
        let expected = rows
            .checked_mul(cols)
            .ok_or_else(|| Error::Model(format!("tensor `{name}` shape overflow")))?;
        if info.dtype != dtype || n_elems != expected {
            return Err(Error::Model(format!(
                "tensor `{name}` expected {:?} {rows}x{cols} ({expected} elems), got {:?} dims {:?}",
                dtype, info.dtype, info.dims
            )));
        }
        let offset = info.data_offset as usize;
        let byte_size = info.byte_size as usize;
        let chunk_offset = offset % WEIGHT_CHUNK_STRIDE;
        if chunk_offset + byte_size > WEIGHT_CHUNK_STRIDE + WEIGHT_CHUNK_OVERLAP {
            return Err(Error::Model(format!(
                "tensor `{name}` crosses Mixtral weight chunk coverage: chunk_offset={chunk_offset} byte_size={byte_size}"
            )));
        }
        Ok(TensorRef {
            offset,
            byte_size,
            dtype: info.dtype,
            n_elems,
            rows,
            cols,
            chunk_index: offset / WEIGHT_CHUNK_STRIDE,
            chunk_offset,
        })
    }

    #[cfg(target_os = "macos")]
    fn pin_weight_chunks(ctx: &MetalContext, gguf: &GgufFile) -> Result<Vec<PinnedBuffer>> {
        let mut chunks = Vec::new();
        let mut start = 0usize;
        while start < gguf.mmap.len() {
            let end = (start + WEIGHT_CHUNK_STRIDE + WEIGHT_CHUNK_OVERLAP).min(gguf.mmap.len());
            let requested = end - start;
            let buf = unsafe { ctx.new_buffer_no_copy(&gguf.mmap[start..end]) };
            if buf.length() < requested as u64 {
                return Err(Error::Metal(format!(
                    "Mixtral weight chunk no-copy failed at bytes [{start}, {end}) requested={requested} actual={}",
                    buf.length()
                )));
            }
            chunks.push(buf);
            start += WEIGHT_CHUNK_STRIDE;
        }
        Ok(chunks)
    }
}

impl Engine for MixtralEngine {
    fn load(weights: &Path, config: EngineConfig) -> Result<Self> {
        let gguf = GgufFile::open(weights)?;
        if !is_mixtral_gguf(&gguf) {
            return Err(Error::Model(
                "not a Mixtral split-expert MoE GGUF".into(),
            ));
        }
        let cfg = MixtralConfig::from_gguf(&gguf)?;
        let model_id = gguf.name().unwrap_or("mixtral-8x7b").to_string();
        let tokenizer = Self::load_tokenizer_preview(weights, &gguf)?;
        let embed = Self::dequant_f16(&gguf, "token_embd.weight")?;
        let final_norm = Self::dequant_f32(&gguf, "output_norm.weight")?;
        let lm_head = if gguf.tensor("output.weight").is_some() {
            Some(Self::dequant_f16(&gguf, "output.weight")?)
        } else {
            None
        };

        let metal_ctx = MetalContext::new_with_trace(config.trace_dispatch).ok();
        let kv_hidden = cfg.n_kv_heads * cfg.head_dim;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for li in 0..cfg.n_layers {
            let lp = |suf: &str| format!("blk.{li}.{suf}");
            let attn_norm = Self::dequant_f32(&gguf, &lp("attn_norm.weight"))?;
            let ffn_norm = Self::dequant_f32(&gguf, &lp("ffn_norm.weight"))?;
            let attn_q = Self::tensor_ref_q4(&gguf, &lp("attn_q.weight"), cfg.hidden, cfg.hidden)?;
            let attn_output =
                Self::tensor_ref_q4(&gguf, &lp("attn_output.weight"), cfg.hidden, cfg.hidden)?;

            let mut ffn_gate = Vec::with_capacity(cfg.n_experts);
            let mut ffn_up = Vec::with_capacity(cfg.n_experts);
            let mut ffn_down = Vec::with_capacity(cfg.n_experts);
            for eid in 0..cfg.n_experts {
                ffn_gate.push(Self::tensor_ref_q4(
                    &gguf,
                    &lp(&format!("ffn_gate.{eid}.weight")),
                    cfg.intermediate,
                    cfg.hidden,
                )?);
                ffn_up.push(Self::tensor_ref_q4(
                    &gguf,
                    &lp(&format!("ffn_up.{eid}.weight")),
                    cfg.intermediate,
                    cfg.hidden,
                )?);
                ffn_down.push(Self::tensor_ref_q4(
                    &gguf,
                    &lp(&format!("ffn_down.{eid}.weight")),
                    cfg.hidden,
                    cfg.intermediate,
                )?);
            }

            let mut pinned = MixtralLayerPinned::default();
            #[cfg(target_os = "macos")]
            if let Some(ctx) = metal_ctx.as_ref() {
                let upload_f32 =
                    |w: &[f32]| ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(w));
                pinned.attn_norm = Some(upload_f32(&attn_norm));
                pinned.ffn_norm = Some(upload_f32(&ffn_norm));
                pinned.attn_k = Some(upload_f32(&Self::dequant_f32_expected(
                    &gguf,
                    &lp("attn_k.weight"),
                    GgmlType::Q8_0,
                    kv_hidden,
                    cfg.hidden,
                )?));
                pinned.attn_v = Some(upload_f32(&Self::dequant_f32_expected(
                    &gguf,
                    &lp("attn_v.weight"),
                    GgmlType::Q8_0,
                    kv_hidden,
                    cfg.hidden,
                )?));
                pinned.gate_inp = Some(upload_f32(&Self::dequant_f32_expected(
                    &gguf,
                    &lp("ffn_gate_inp.weight"),
                    GgmlType::F16,
                    cfg.n_experts,
                    cfg.hidden,
                )?));
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = kv_hidden;
            }

            layers.push(MixtralLayer {
                attn_norm,
                ffn_norm,
                attn_q,
                attn_output,
                ffn_gate,
                ffn_up,
                ffn_down,
                pinned,
            });
        }

        #[cfg(target_os = "macos")]
        let weights_mmap_buf = if let Some(ctx) = metal_ctx.as_ref() {
            Some(Self::pin_weight_chunks(ctx, &gguf)?)
        } else {
            None
        };
        #[cfg(not(target_os = "macos"))]
        let weights_mmap_buf: Option<Vec<PinnedBuffer>> = None;

        #[cfg(target_os = "macos")]
        let (embed_buf, final_norm_buf, lm_head_buf, decode_arena) =
            if let Some(ctx) = metal_ctx.as_ref() {
                let lm: &[f16] = lm_head.as_deref().unwrap_or(&embed);
                (
                    Some(ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&embed))),
                    Some(ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(
                        &final_norm,
                    ))),
                    Some(ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(lm))),
                    Some(MixtralDecodeArena::new(ctx, &cfg)),
                )
            } else {
                (None, None, None, None)
            };
        #[cfg(not(target_os = "macos"))]
        let (embed_buf, final_norm_buf, lm_head_buf, decode_arena): (
            Option<PinnedBuffer>,
            Option<PinnedBuffer>,
            Option<PinnedBuffer>,
            Option<MixtralDecodeArena>,
        ) = (None, None, None, None);

        let engine = Self {
            config: cfg,
            tokenizer,
            model_id,
            gguf,
            embed,
            final_norm,
            lm_head,
            layers,
            _weights_path: weights.to_owned(),
            metal_ctx,
            weights_mmap_buf,
            embed_buf,
            final_norm_buf,
            lm_head_buf,
            decode_arena,
        };
        eprintln!(
            "[mixtral] metal_ctx: {:?}, weights_mmap_buf: {:?}, embed_buf: {:?}, decode_arena: {:?}",
            engine.metal_ctx.is_some(),
            engine.weights_mmap_buf.is_some(),
            engine.embed_buf.is_some(),
            engine.decode_arena.is_some()
        );
        Ok(engine)
    }

    fn generate(
        &mut self,
        _req: GenerateRequest,
        _sink: &mut dyn FnMut(StreamEvent),
    ) -> Result<GenStats> {
        Err(Error::Unimplemented(
            "MixtralEngine::forward_token is not implemented yet (v1.0.3 Step 2)",
        ))
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn model_arch(&self) -> &str {
        "mixtral"
    }

    fn forward_tokens_for_test(
        &mut self,
        _tokens: &[u32],
        _positions: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        Err(Error::Unimplemented(
            "MixtralEngine::forward_tokens_for_test",
        ))
    }
}

pub fn is_mixtral_gguf(gguf: &GgufFile) -> bool {
    gguf.architecture() == Some("llama") && gguf.tensor("blk.0.ffn_gate.0.weight").is_some()
}
