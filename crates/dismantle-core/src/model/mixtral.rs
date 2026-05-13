//! Mixtral 8x7B loader and forward-path scaffold.
//!
//! Mixtral GGUFs in the target Q4_K_M export use split per-expert tensors:
//! `blk.N.ffn_gate.E.weight`, `blk.N.ffn_up.E.weight`, and
//! `blk.N.ffn_down.E.weight`. This module intentionally does not look for the
//! fused `*_exps.weight` layout used by the DeepSeek path.

use crate::attn::mha_decode_step;
use crate::cache::KvCache;
use crate::engine::{Engine, EngineConfig, GenStats, GenerateRequest, StopReason, StreamEvent};
use crate::gguf::{GgmlType, GgufFile};
use crate::metal::{MetalContext, PinnedBuffer};
use crate::moe::topk_gate;
use crate::quant;
use crate::sample::Sampler;
use crate::tokenizer::Tokenizer;
use crate::{Error, Result};
use half::f16;
use std::path::{Path, PathBuf};
use std::time::Instant;

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
    pub kv: KvCache,
    pub sampler: Sampler,
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

    fn madvise_willneed_mmap(gguf: &GgufFile) {
        platform_madvise_willneed(gguf.mmap.as_ptr() as usize, gguf.mmap.len());
    }
}

#[cfg(unix)]
fn platform_madvise_willneed(addr: usize, len: usize) {
    use core::ffi::{c_int, c_void};
    const POSIX_MADV_WILLNEED: c_int = 3;
    unsafe extern "C" {
        fn posix_madvise(addr: *mut c_void, len: usize, advice: c_int) -> c_int;
    }
    let _ = unsafe { posix_madvise(addr as *mut c_void, len, POSIX_MADV_WILLNEED) };
}

#[cfg(not(unix))]
fn platform_madvise_willneed(_addr: usize, _len: usize) {}

impl Engine for MixtralEngine {
    fn load(weights: &Path, config: EngineConfig) -> Result<Self> {
        let gguf = GgufFile::open(weights)?;
        Self::madvise_willneed_mmap(&gguf);
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

        let max_seq = config.max_seq_len.min(cfg.max_seq_len);
        let kv = KvCache::new(cfg.n_layers, max_seq, cfg.n_kv_heads, cfg.head_dim);
        let sampler = Sampler::new(0);
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
            kv,
            sampler,
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

        let decode_start = Instant::now();
        let mut last_id = *prompt_ids
            .last()
            .ok_or_else(|| Error::Model("empty prompt after tokenization".into()))?;
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

    fn model_arch(&self) -> &str {
        "mixtral"
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
        self.kv.reset();
    }
}

impl MixtralEngine {
    fn forward_token(&mut self, token: u32, pos: usize) -> Result<Vec<f32>> {
        #[cfg(target_os = "macos")]
        {
            self.forward_token_tcb(token, pos)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (token, pos);
            Err(Error::Unimplemented(
                "MixtralEngine::forward_token requires the Metal TCB path",
            ))
        }
    }

    #[cfg(target_os = "macos")]
    fn read_f32_buffer(buf: &PinnedBuffer, out: &mut [f32]) -> Result<()> {
        let bytes = out.len() * std::mem::size_of::<f32>();
        if buf.length() < bytes as u64 {
            return Err(Error::Kernel(format!(
                "read_f32_buffer: buffer too small: got {} expected {bytes}",
                buf.length()
            )));
        }
        let ptr = buf.contents() as *const f32;
        let src = unsafe { std::slice::from_raw_parts(ptr, out.len()) };
        out.copy_from_slice(src);
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn write_f32_buffer(buf: &PinnedBuffer, xs: &[f32]) -> Result<()> {
        let bytes = xs.len() * std::mem::size_of::<f32>();
        if buf.length() < bytes as u64 {
            return Err(Error::Kernel(format!(
                "write_f32_buffer: buffer too small: got {} expected {bytes}",
                buf.length()
            )));
        }
        MetalContext::write_buffer_bytes(buf, bytemuck::cast_slice::<f32, u8>(xs));
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn l2_norm(xs: &[f32]) -> f32 {
        xs.iter()
            .map(|v| (*v as f64) * (*v as f64))
            .sum::<f64>()
            .sqrt() as f32
    }

    #[cfg(target_os = "macos")]
    fn debug_first(xs: &[f32], n: usize) -> Vec<f32> {
        xs.iter().take(n).copied().collect()
    }

    #[cfg(target_os = "macos")]
    fn debug_buffer(label: &str, buf: &PinnedBuffer, len: usize, first: usize) -> Result<()> {
        let mut xs = vec![0.0f32; len];
        Self::read_f32_buffer(buf, &mut xs)?;
        eprintln!(
            "[mixtral-debug] {label} l2={:.6} first{}={:?}",
            Self::l2_norm(&xs),
            first,
            Self::debug_first(&xs, first)
        );
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn debug_slice(label: &str, xs: &[f32], first: usize) {
        eprintln!(
            "[mixtral-debug] {label} l2={:.6} first{}={:?}",
            Self::l2_norm(xs),
            first,
            Self::debug_first(xs, first)
        );
    }

    #[cfg(target_os = "macos")]
    fn debug_top_logits(&self, logits: &[f32]) {
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
        let top: Vec<(usize, f32)> = idx.iter().take(10).map(|&i| (i, logits[i])).collect();
        let decoded: Vec<(usize, String)> = idx
            .iter()
            .take(3)
            .map(|&i| (i, self.tokenizer.decode_one(i as u32).unwrap_or_default()))
            .collect();
        eprintln!("[mixtral-debug] lm_head top10={top:?} top3_decoded={decoded:?}");
    }

    #[cfg(target_os = "macos")]
    fn encode_q4_tcb(
        &self,
        tcb: &mut crate::metal::TokenCommandBuffer<'_>,
        t: &TensorRef,
        x_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
    ) -> Result<()> {
        let chunks = self
            .weights_mmap_buf
            .as_ref()
            .ok_or_else(|| Error::Model("Mixtral Q4 path missing weight chunks".into()))?;
        let chunk = chunks.get(t.chunk_index).ok_or_else(|| {
            Error::Model(format!(
                "Mixtral tensor chunk {} missing for offset {}",
                t.chunk_index, t.offset
            ))
        })?;
        crate::kernels::gemv_q4_k_m_v2_pinned_tcb(
            tcb,
            chunk,
            t.chunk_offset,
            t.byte_size,
            t.rows,
            t.cols,
            x_buf,
            out_buf,
        )
    }

    #[cfg(target_os = "macos")]
    fn missing_buf(name: &str) -> Error {
        Error::Model(format!("Mixtral GPU path missing {name}"))
    }

    #[cfg(target_os = "macos")]
    fn forward_token_tcb(&mut self, token: u32, pos: usize) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let h = cfg.hidden;
        let head_dim = cfg.head_dim;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let kv_hidden = n_kv_heads * head_dim;
        let stride = kv_hidden;
        if self.kv.seq_len >= self.kv.max_seq {
            return Err(Error::Model(format!("kv cache full at {}", self.kv.max_seq)));
        }
        let kv_off = self.kv.seq_len * stride;
        let mha_seq_len = self.kv.seq_len + 1;

        let ctx = self
            .metal_ctx
            .as_ref()
            .ok_or_else(|| Self::missing_buf("metal_ctx"))?;
        let arena = self
            .decode_arena
            .as_ref()
            .ok_or_else(|| Self::missing_buf("decode_arena"))?;
        let embed_buf = self
            .embed_buf
            .as_ref()
            .ok_or_else(|| Self::missing_buf("embed_buf"))?;
        let final_norm_buf = self
            .final_norm_buf
            .as_ref()
            .ok_or_else(|| Self::missing_buf("final_norm_buf"))?;
        let lm_head_buf = self
            .lm_head_buf
            .as_ref()
            .ok_or_else(|| Self::missing_buf("lm_head_buf"))?;
        let debug = std::env::var("DISMANTLE_MIXTRAL_DEBUG").is_ok();
        if debug {
            eprintln!(
                "[mixtral-debug] token={token} pos={pos} seq_len={} rope_theta={} hidden={} heads={} kv_heads={}",
                self.kv.seq_len, cfg.rope_theta, h, n_heads, n_kv_heads
            );
        }

        for li in 0..cfg.n_layers {
            {
                let layer = &self.layers[li];
                let attn_norm = layer
                    .pinned
                    .attn_norm
                    .as_ref()
                    .ok_or_else(|| Self::missing_buf("layer.attn_norm"))?;
                let attn_k = layer
                    .pinned
                    .attn_k
                    .as_ref()
                    .ok_or_else(|| Self::missing_buf("layer.attn_k"))?;
                let attn_v = layer
                    .pinned
                    .attn_v
                    .as_ref()
                    .ok_or_else(|| Self::missing_buf("layer.attn_v"))?;
                let debug_layer = debug && li < 3;

                let mut tcb = crate::metal::TokenCommandBuffer::new(ctx);
                if li == 0 {
                    crate::kernels::embed_lookup_metal_f32_tcb(
                        &mut tcb,
                        embed_buf,
                        token,
                        h,
                        &arena.x_buf,
                    )?;
                }
                crate::kernels::rmsnorm_metal_buf_tcb(
                    &mut tcb,
                    &arena.x_buf,
                    attn_norm,
                    cfg.rms_norm_eps,
                    h,
                    &arena.x_norm_buf,
                )?;
                self.encode_q4_tcb(&mut tcb, &layer.attn_q, &arena.x_norm_buf, &arena.q_buf)?;
                crate::kernels::gemv_f32_attn_pinned_buf_tcb(
                    &mut tcb,
                    attn_k,
                    kv_hidden,
                    h,
                    &arena.x_norm_buf,
                    &arena.k_buf,
                )?;
                crate::kernels::gemv_f32_attn_pinned_buf_tcb(
                    &mut tcb,
                    attn_v,
                    kv_hidden,
                    h,
                    &arena.x_norm_buf,
                    &arena.v_buf,
                )?;
                if debug_layer {
                    tcb.commit_and_wait()?;
                    if li == 0 {
                        Self::debug_buffer("embed", &arena.x_buf, h, 5)?;
                    }
                    Self::debug_buffer(&format!("layer{li}.attn_norm"), &arena.x_norm_buf, h, 5)?;
                    Self::debug_buffer(&format!("layer{li}.q_pre_rope"), &arena.q_buf, h, 5)?;

                    let mut rope_tcb = crate::metal::TokenCommandBuffer::new(ctx);
                    crate::kernels::rope_q_f32_inplace_tcb(
                        &mut rope_tcb,
                        &arena.q_buf,
                        n_heads,
                        head_dim,
                        0,
                        head_dim,
                        pos as u32,
                        cfg.rope_theta,
                    )?;
                    crate::kernels::rope_q_f32_inplace_tcb(
                        &mut rope_tcb,
                        &arena.k_buf,
                        n_kv_heads,
                        head_dim,
                        0,
                        head_dim,
                        pos as u32,
                        cfg.rope_theta,
                    )?;
                    rope_tcb.commit_and_wait()?;
                    Self::debug_buffer(&format!("layer{li}.q_post_rope_head0"), &arena.q_buf, head_dim, 8)?;
                } else {
                    crate::kernels::rope_q_f32_inplace_tcb(
                        &mut tcb,
                        &arena.q_buf,
                        n_heads,
                        head_dim,
                        0,
                        head_dim,
                        pos as u32,
                        cfg.rope_theta,
                    )?;
                    crate::kernels::rope_q_f32_inplace_tcb(
                        &mut tcb,
                        &arena.k_buf,
                        n_kv_heads,
                        head_dim,
                        0,
                        head_dim,
                        pos as u32,
                        cfg.rope_theta,
                    )?;
                    tcb.commit_and_wait()?;
                }
            }

            let mut q_full = vec![0.0f32; h];
            let mut k_token = vec![0.0f32; kv_hidden];
            let mut v_token = vec![0.0f32; kv_hidden];
            Self::read_f32_buffer(&arena.q_buf, &mut q_full)?;
            Self::read_f32_buffer(&arena.k_buf, &mut k_token)?;
            Self::read_f32_buffer(&arena.v_buf, &mut v_token)?;
            self.kv.keys[li][kv_off..kv_off + stride].copy_from_slice(&k_token);
            self.kv.values[li][kv_off..kv_off + stride].copy_from_slice(&v_token);

            let kv_size = mha_seq_len * stride;
            let keys = &self.kv.keys[li][..kv_size];
            let values = &self.kv.values[li][..kv_size];
            let mut attn_out = vec![0.0f32; h];
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
            Self::write_f32_buffer(&arena.attn_out_buf, &attn_out)?;
            if debug && li < 3 {
                Self::debug_slice(&format!("layer{li}.attn_out"), &attn_out, 5);
            }

            let mut gate_logits = vec![0.0f32; cfg.n_experts];
            {
                let layer = &self.layers[li];
                let ffn_norm = layer
                    .pinned
                    .ffn_norm
                    .as_ref()
                    .ok_or_else(|| Self::missing_buf("layer.ffn_norm"))?;
                let gate_inp = layer
                    .pinned
                    .gate_inp
                    .as_ref()
                    .ok_or_else(|| Self::missing_buf("layer.gate_inp"))?;
                let mut tcb = crate::metal::TokenCommandBuffer::new(ctx);
                self.encode_q4_tcb(
                    &mut tcb,
                    &layer.attn_output,
                    &arena.attn_out_buf,
                    &arena.proj_out_buf,
                )?;
                crate::kernels::add_inplace_metal_tcb(
                    &mut tcb,
                    &arena.x_buf,
                    &arena.proj_out_buf,
                    h,
                )?;
                crate::kernels::rmsnorm_metal_buf_tcb(
                    &mut tcb,
                    &arena.x_buf,
                    ffn_norm,
                    cfg.rms_norm_eps,
                    h,
                    &arena.x_norm_buf,
                )?;
                crate::kernels::gemv_f32_attn_pinned_buf_tcb(
                    &mut tcb,
                    gate_inp,
                    cfg.n_experts,
                    h,
                    &arena.x_norm_buf,
                    &arena.gate_logits_buf,
                )?;
                tcb.commit_and_wait()?;
            }
            Self::read_f32_buffer(&arena.gate_logits_buf, &mut gate_logits)?;
            let raw_gate_logits = gate_logits.clone();
            let routes = topk_gate(&mut gate_logits, cfg.top_k, true);
            if debug && li < 3 {
                eprintln!(
                    "[mixtral-debug] layer{li}.gate raw={raw_gate_logits:?} routes={routes:?}"
                );
            }

            {
                let layer = &self.layers[li];
                let mut tcb = crate::metal::TokenCommandBuffer::new(ctx);
                for (route_i, (eid, _weight)) in routes.iter().enumerate() {
                    self.encode_q4_tcb(
                        &mut tcb,
                        &layer.ffn_gate[*eid],
                        &arena.x_norm_buf,
                        &arena.expert_gate_bufs[route_i],
                    )?;
                    self.encode_q4_tcb(
                        &mut tcb,
                        &layer.ffn_up[*eid],
                        &arena.x_norm_buf,
                        &arena.expert_up_bufs[route_i],
                    )?;
                    crate::kernels::silu_mul_tcb(
                        &mut tcb,
                        &arena.expert_gate_bufs[route_i],
                        &arena.expert_up_bufs[route_i],
                        &arena.expert_act_bufs[route_i],
                        cfg.intermediate,
                    )?;
                    self.encode_q4_tcb(
                        &mut tcb,
                        &layer.ffn_down[*eid],
                        &arena.expert_act_bufs[route_i],
                        &arena.expert_out_bufs[route_i],
                    )?;
                }
                tcb.commit_and_wait()?;
            }

            let mut ffn_out = vec![0.0f32; h];
            let mut expert_out = vec![0.0f32; h];
            for (route_i, (_eid, weight)) in routes.iter().enumerate() {
                Self::read_f32_buffer(&arena.expert_out_bufs[route_i], &mut expert_out)?;
                for i in 0..h {
                    ffn_out[i] += *weight * expert_out[i];
                }
            }
            if debug && li < 3 {
                Self::write_f32_buffer(&arena.ffn_out_buf, &ffn_out)?;
                Self::debug_buffer(&format!("layer{li}.moe_out"), &arena.ffn_out_buf, h, 5)?;
            }
            let mut x_cpu = vec![0.0f32; h];
            Self::read_f32_buffer(&arena.x_buf, &mut x_cpu)?;
            for i in 0..h {
                x_cpu[i] += ffn_out[i];
            }
            Self::write_f32_buffer(&arena.x_buf, &x_cpu)?;
        }

        self.kv.seq_len += 1;

        let mut tcb = crate::metal::TokenCommandBuffer::new(ctx);
        crate::kernels::rmsnorm_metal_buf_tcb(
            &mut tcb,
            &arena.x_buf,
            final_norm_buf,
            cfg.rms_norm_eps,
            h,
            &arena.x_norm_buf,
        )?;
        crate::kernels::gemv_f16_metal_buf_tcb(
            &mut tcb,
            lm_head_buf,
            cfg.vocab_size,
            h,
            &arena.x_norm_buf,
            &arena.logits_buf,
        )?;
        tcb.commit_and_wait()?;
        if debug {
            Self::debug_buffer("final_norm", &arena.x_norm_buf, h, 5)?;
        }

        let mut logits = vec![0.0f32; cfg.vocab_size];
        Self::read_f32_buffer(&arena.logits_buf, &mut logits)?;
        if debug {
            self.debug_top_logits(&logits);
        }
        Ok(logits)
    }
}

pub fn is_mixtral_gguf(gguf: &GgufFile) -> bool {
    gguf.architecture() == Some("llama") && gguf.tensor("blk.0.ffn_gate.0.weight").is_some()
}
