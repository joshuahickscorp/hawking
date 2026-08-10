//! Layer-major BF16 SOURCE activation capture for Qwen3-Coder-30B-A3B.
//!
//! Resource contract (intentionally distinct from the co-resident memory gate):
//!
//! * Capture does **not** resident-load the 56.9 GiB BF16 source.
//! * Loop is inverted: for each layer, range-read that layer's weights, push
//!   every probe token through the layer, record routes + retained hiddens,
//!   then free the layer weights.
//! * Working set is ~one layer of MoE (~1.1–2.2 GiB depending on widen) plus
//!   the full residual stream for all tokens (0.67 GiB at the sealed corpus).
//!
//! Captured router inputs are the post-attention RMSNorm vectors (same surface
//! as the existing complete-binary all-layer route capture). Output layout is
//! schema-compatible with that capture so the activation-weighted SVD repack
//! can consume it without changes.

use crate::artifact::widen_native;
use crate::attn::mha_decode_step;
use crate::kernels::{add_inplace, argmax_f32, rmsnorm, silu_mul, softmax_inplace};
use crate::{Error, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub const QWEN30_LAYERS: usize = 48;
pub const QWEN30_HIDDEN: usize = 2048;
pub const QWEN30_HEADS: usize = 32;
pub const QWEN30_KV_HEADS: usize = 4;
pub const QWEN30_HEAD_DIM: usize = 128;
pub const QWEN30_EXPERTS: usize = 128;
pub const QWEN30_TOP_K: usize = 8;
pub const QWEN30_MOE_INTERMEDIATE: usize = 768;
pub const QWEN30_VOCAB: usize = 151_936;
pub const QWEN30_ROPE_THETA: f32 = 10_000_000.0;
pub const QWEN30_RMS_EPS: f32 = 1e-6;
/// Soft upper bound declared by this contract (single-digit GiB). Exceeding
/// this means the implementation has effectively resident-loaded the source.
pub const STREAMED_PEAK_RSS_HARD_CAP_BYTES: u64 = 12 * 1024 * 1024 * 1024;
/// Approximate per-layer MoE BF16 payload (attention + 128 experts).
pub const PER_LAYER_MOE_BF16_BYTES: u64 = 1_200_000_000;

fn model_err(msg: impl Into<String>) -> Error {
    Error::Model(msg.into())
}

#[inline]
fn bf16_le_to_f32(bytes: &[u8]) -> f32 {
    debug_assert!(bytes.len() >= 2);
    f32::from_bits((u16::from_le_bytes([bytes[0], bytes[1]]) as u32) << 16)
}

/// Row-major GEMV: `out = W @ x` with W stored as little-endian BF16 rows.
pub fn gemv_bf16(weight_le: &[u8], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
    if x.len() != cols || out.len() != rows {
        return Err(model_err(format!(
            "gemv_bf16 geometry: x={} out={} rows={rows} cols={cols}",
            x.len(),
            out.len()
        )));
    }
    let expect = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| model_err("gemv_bf16 size overflow"))?;
    if weight_le.len() < expect {
        return Err(model_err(format!(
            "gemv_bf16 weight bytes {} < {expect}",
            weight_le.len()
        )));
    }
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 16);
    let chunk = rows.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (t, out_chunk) in out.chunks_mut(chunk).enumerate() {
            let row0 = t * chunk;
            let w = weight_le;
            let x = x;
            scope.spawn(move || {
                for (i, o) in out_chunk.iter_mut().enumerate() {
                    let row = row0 + i;
                    if row >= rows {
                        break;
                    }
                    let base = row * cols * 2;
                    let mut acc = 0.0f32;
                    // Unroll a little for throughput without pulling in BLAS.
                    let mut c = 0usize;
                    while c + 4 <= cols {
                        acc += bf16_le_to_f32(&w[base + c * 2..]) * x[c]
                            + bf16_le_to_f32(&w[base + (c + 1) * 2..]) * x[c + 1]
                            + bf16_le_to_f32(&w[base + (c + 2) * 2..]) * x[c + 2]
                            + bf16_le_to_f32(&w[base + (c + 3) * 2..]) * x[c + 3];
                        c += 4;
                    }
                    while c < cols {
                        acc += bf16_le_to_f32(&w[base + c * 2..]) * x[c];
                        c += 1;
                    }
                    *o = acc;
                }
            });
        }
    });
    Ok(())
}

/// Row-major GEMV with f32 weights (after a one-shot BF16 widen of a layer tensor).
pub fn gemv_f32_rows(w: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
    if x.len() != cols || out.len() != rows || w.len() < rows * cols {
        return Err(model_err(format!(
            "gemv_f32 geometry: w={} x={} out={} rows={rows} cols={cols}",
            w.len(),
            x.len(),
            out.len()
        )));
    }
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 16);
    let chunk = rows.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (t, out_chunk) in out.chunks_mut(chunk).enumerate() {
            let row0 = t * chunk;
            let w = w;
            let x = x;
            scope.spawn(move || {
                for (i, o) in out_chunk.iter_mut().enumerate() {
                    let row = row0 + i;
                    if row >= rows {
                        break;
                    }
                    let base = row * cols;
                    let mut acc = 0.0f32;
                    let row_w = &w[base..base + cols];
                    for c in 0..cols {
                        acc += row_w[c] * x[c];
                    }
                    *o = acc;
                }
            });
        }
    });
    Ok(())
}

#[derive(Clone, Debug)]
struct TensorLoc {
    shard: PathBuf,
    /// Absolute byte offset of tensor payload inside the shard file.
    data_offset: u64,
    nbytes: usize,
    shape: Vec<usize>,
    dtype: String,
}

/// Index over the source BF16 safetensors shards. Opens shard headers once;
/// tensor payloads are range-read on demand and never bulk-resident.
pub struct SourceBf16Index {
    pub model_dir: PathBuf,
    map: HashMap<String, TensorLoc>,
    /// Opened shard handles for positioned reads (not full-file loads).
    handles: Mutex<HashMap<PathBuf, File>>,
}

impl SourceBf16Index {
    pub fn open(model_dir: &Path) -> Result<Self> {
        let index_path = model_dir.join("model.safetensors.index.json");
        let index_bytes = std::fs::read(&index_path).map_err(|e| {
            model_err(format!(
                "cannot read safetensors index {}: {e}",
                index_path.display()
            ))
        })?;
        let index: Value = serde_json::from_slice(&index_bytes)
            .map_err(|e| model_err(format!("safetensors index is not JSON: {e}")))?;
        let weight_map = index
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or_else(|| model_err("safetensors index lacks weight_map"))?;

        let mut by_shard: HashMap<String, Vec<String>> = HashMap::new();
        for (name, shard_v) in weight_map {
            let shard = shard_v
                .as_str()
                .ok_or_else(|| model_err(format!("weight_map entry {name} is not a string")))?;
            by_shard.entry(shard.to_string()).or_default().push(name.clone());
        }

        let mut map = HashMap::new();
        for (shard_name, names) in by_shard {
            let shard_path = model_dir.join(&shard_name);
            let header = read_safetensors_header(&shard_path)?;
            let header_len = header.header_nbytes;
            for name in names {
                let info = header
                    .tensors
                    .get(&name)
                    .ok_or_else(|| model_err(format!("shard {shard_name} lacks tensor {name}")))?;
                if info.dtype != "BF16" && info.dtype != "BFLOAT16" {
                    return Err(model_err(format!(
                        "tensor {name} dtype {} is not BF16",
                        info.dtype
                    )));
                }
                let (begin, end) = info.data_offsets;
                if end < begin {
                    return Err(model_err(format!("tensor {name} has inverted data_offsets")));
                }
                let nbytes = (end - begin) as usize;
                map.insert(
                    name,
                    TensorLoc {
                        shard: shard_path.clone(),
                        data_offset: 8 + header_len + begin,
                        nbytes,
                        shape: info.shape.clone(),
                        dtype: info.dtype.clone(),
                    },
                );
            }
        }
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            map,
            handles: Mutex::new(HashMap::new()),
        })
    }

    pub fn tensor_count(&self) -> usize {
        self.map.len()
    }

    pub fn require(&self, name: &str) -> Result<&TensorLoc> {
        self.map
            .get(name)
            .ok_or_else(|| model_err(format!("source index lacks tensor {name}")))
    }

    /// Range-read a tensor's raw BF16 payload. Does not keep other tensors resident.
    pub fn read_raw(&self, name: &str) -> Result<Vec<u8>> {
        let loc = self.require(name)?;
        let mut handles = self
            .handles
            .lock()
            .map_err(|_| model_err("source shard handle map poisoned"))?;
        let file = if let Some(f) = handles.get_mut(&loc.shard) {
            f
        } else {
            let f = File::open(&loc.shard).map_err(|e| {
                model_err(format!("cannot open shard {}: {e}", loc.shard.display()))
            })?;
            handles.insert(loc.shard.clone(), f);
            handles.get_mut(&loc.shard).unwrap()
        };
        file.seek(SeekFrom::Start(loc.data_offset)).map_err(|e| {
            model_err(format!(
                "seek {} @ {}: {e}",
                loc.shard.display(),
                loc.data_offset
            ))
        })?;
        let mut buf = vec![0u8; loc.nbytes];
        file.read_exact(&mut buf).map_err(|e| {
            model_err(format!(
                "range-read {} ({} bytes) from {}: {e}",
                name,
                loc.nbytes,
                loc.shard.display()
            ))
        })?;
        Ok(buf)
    }

    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>> {
        let raw = self.read_raw(name)?;
        widen_native("native.bf16", &raw)
    }

    /// Read a single embedding row without loading the full embedding table.
    pub fn embed_row(&self, token: u32) -> Result<Vec<f32>> {
        if token as usize >= QWEN30_VOCAB {
            return Err(model_err(format!("token {token} outside vocabulary")));
        }
        let loc = self.require("model.embed_tokens.weight")?;
        if loc.shape != [QWEN30_VOCAB, QWEN30_HIDDEN] {
            return Err(model_err(format!(
                "embed_tokens shape {:?} is not [{QWEN30_VOCAB}, {QWEN30_HIDDEN}]",
                loc.shape
            )));
        }
        let row_bytes = QWEN30_HIDDEN * 2;
        let offset = loc
            .data_offset
            .checked_add((token as u64).checked_mul(row_bytes as u64).ok_or_else(|| {
                model_err("embed row offset overflow")
            })?)
            .ok_or_else(|| model_err("embed absolute offset overflow"))?;
        let mut handles = self
            .handles
            .lock()
            .map_err(|_| model_err("source shard handle map poisoned"))?;
        let file = if let Some(f) = handles.get_mut(&loc.shard) {
            f
        } else {
            let f = File::open(&loc.shard).map_err(|e| {
                model_err(format!("cannot open shard {}: {e}", loc.shard.display()))
            })?;
            handles.insert(loc.shard.clone(), f);
            handles.get_mut(&loc.shard).unwrap()
        };
        file.seek(SeekFrom::Start(offset)).map_err(|e| {
            model_err(format!("seek embed row {token}: {e}"))
        })?;
        let mut buf = vec![0u8; row_bytes];
        file.read_exact(&mut buf)
            .map_err(|e| model_err(format!("read embed row {token}: {e}")))?;
        widen_native("native.bf16", &buf)
    }
}

struct SafetensorsHeader {
    header_nbytes: u64,
    tensors: HashMap<String, SafetensorsTensorInfo>,
}

struct SafetensorsTensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (u64, u64),
}

fn read_safetensors_header(path: &Path) -> Result<SafetensorsHeader> {
    let mut file = File::open(path)
        .map_err(|e| model_err(format!("cannot open {}: {e}", path.display())))?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)
        .map_err(|e| model_err(format!("cannot read header length of {}: {e}", path.display())))?;
    let header_nbytes = u64::from_le_bytes(len_buf);
    if header_nbytes == 0 || header_nbytes > 64 * 1024 * 1024 {
        return Err(model_err(format!(
            "implausible safetensors header length {header_nbytes} in {}",
            path.display()
        )));
    }
    let mut raw = vec![0u8; header_nbytes as usize];
    file.read_exact(&mut raw)
        .map_err(|e| model_err(format!("cannot read header of {}: {e}", path.display())))?;
    let value: Value = serde_json::from_slice(&raw)
        .map_err(|e| model_err(format!("safetensors header JSON invalid in {}: {e}", path.display())))?;
    let object = value
        .as_object()
        .ok_or_else(|| model_err(format!("safetensors header is not an object in {}", path.display())))?;
    let mut tensors = HashMap::new();
    for (name, info_v) in object {
        if name == "__metadata__" {
            continue;
        }
        let info = info_v
            .as_object()
            .ok_or_else(|| model_err(format!("tensor {name} header is not an object")))?;
        let dtype = info
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| model_err(format!("tensor {name} lacks dtype")))?
            .to_string();
        let shape = info
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| model_err(format!("tensor {name} lacks shape")))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(|| model_err(format!("tensor {name} has non-integer shape")))
            })
            .collect::<Result<Vec<_>>>()?;
        let offsets = info
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| model_err(format!("tensor {name} lacks data_offsets")))?;
        if offsets.len() != 2 {
            return Err(model_err(format!("tensor {name} data_offsets is not a pair")));
        }
        let begin = offsets[0]
            .as_u64()
            .ok_or_else(|| model_err(format!("tensor {name} data_offsets[0] invalid")))?;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| model_err(format!("tensor {name} data_offsets[1] invalid")))?;
        tensors.insert(
            name.clone(),
            SafetensorsTensorInfo {
                dtype,
                shape,
                data_offsets: (begin, end),
            },
        );
    }
    Ok(SafetensorsHeader {
        header_nbytes,
        tensors,
    })
}

fn layer_name(layer: usize, suffix: &str) -> String {
    format!("model.layers.{layer}.{suffix}")
}

fn expert_name(layer: usize, expert: usize, role: &str) -> String {
    format!("model.layers.{layer}.mlp.experts.{expert}.{role}.weight")
}

/// One loaded transformer layer, held only for the duration of that layer's
/// pass over the corpus, then dropped.
pub struct LoadedLayer {
    pub layer: usize,
    pub input_layernorm: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
    pub q_proj: Vec<u8>,
    pub k_proj: Vec<u8>,
    pub v_proj: Vec<u8>,
    pub o_proj: Vec<u8>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub router: Vec<u8>,
    /// gate/up/down raw BF16 payloads per expert.
    pub experts: Vec<ExpertWeights>,
    pub resident_bytes: u64,
}

pub struct ExpertWeights {
    pub gate: Vec<u8>,
    pub up: Vec<u8>,
    pub down: Vec<u8>,
}

impl LoadedLayer {
    pub fn load(index: &SourceBf16Index, layer: usize) -> Result<Self> {
        if layer >= QWEN30_LAYERS {
            return Err(model_err(format!("layer {layer} out of range")));
        }
        let input_layernorm = index.read_f32(&layer_name(layer, "input_layernorm.weight"))?;
        let post_attention_layernorm =
            index.read_f32(&layer_name(layer, "post_attention_layernorm.weight"))?;
        let q_norm = index.read_f32(&layer_name(layer, "self_attn.q_norm.weight"))?;
        let k_norm = index.read_f32(&layer_name(layer, "self_attn.k_norm.weight"))?;
        let q_proj = index.read_raw(&layer_name(layer, "self_attn.q_proj.weight"))?;
        let k_proj = index.read_raw(&layer_name(layer, "self_attn.k_proj.weight"))?;
        let v_proj = index.read_raw(&layer_name(layer, "self_attn.v_proj.weight"))?;
        let o_proj = index.read_raw(&layer_name(layer, "self_attn.o_proj.weight"))?;
        let router = index.read_raw(&layer_name(layer, "mlp.gate.weight"))?;
        let mut experts = Vec::with_capacity(QWEN30_EXPERTS);
        let mut resident = (input_layernorm.len()
            + post_attention_layernorm.len()
            + q_norm.len()
            + k_norm.len())
            * 4
            + q_proj.len()
            + k_proj.len()
            + v_proj.len()
            + o_proj.len()
            + router.len();
        for expert in 0..QWEN30_EXPERTS {
            let gate = index.read_raw(&expert_name(layer, expert, "gate_proj"))?;
            let up = index.read_raw(&expert_name(layer, expert, "up_proj"))?;
            let down = index.read_raw(&expert_name(layer, expert, "down_proj"))?;
            resident += gate.len() + up.len() + down.len();
            experts.push(ExpertWeights { gate, up, down });
        }
        Ok(Self {
            layer,
            input_layernorm,
            post_attention_layernorm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            router,
            experts,
            resident_bytes: resident as u64,
        })
    }
}

/// Apply Qwen3 NeoX / rotate_half RoPE to one head vector in place.
pub fn rope_neox_inplace(x: &mut [f32], pos: u32, base: f32) {
    let head_dim = x.len();
    let half = head_dim / 2;
    for i in 0..half {
        let theta = (pos as f32) / base.powf(2.0 * i as f32 / head_dim as f32);
        let (sin, cos) = theta.sin_cos();
        let x0 = x[i];
        let x1 = x[i + half];
        x[i] = x0 * cos - x1 * sin;
        x[i + half] = x0 * sin + x1 * cos;
    }
}

fn rmsnorm_rows(x: &mut [f32], weight: &[f32], n_heads: usize, head_dim: usize) -> Result<()> {
    if x.len() != n_heads * head_dim || weight.len() != head_dim {
        return Err(model_err("rmsnorm_rows geometry mismatch"));
    }
    let mut tmp = vec![0.0f32; head_dim];
    for h in 0..n_heads {
        let start = h * head_dim;
        let row = &x[start..start + head_dim];
        rmsnorm(row, weight, QWEN30_RMS_EPS, &mut tmp);
        x[start..start + head_dim].copy_from_slice(&tmp);
    }
    Ok(())
}

/// Top-k over softmax with `norm_topk_prob=true` (renormalize selected weights).
pub fn router_topk_norm(
    logits: &[f32],
    top_k: usize,
) -> Result<(Vec<u32>, Vec<f32>)> {
    if logits.len() != QWEN30_EXPERTS {
        return Err(model_err(format!(
            "router logits len {} != {QWEN30_EXPERTS}",
            logits.len()
        )));
    }
    let mut probs = logits.to_vec();
    softmax_inplace(&mut probs);
    let mut ids = Vec::with_capacity(top_k);
    let mut weights = Vec::with_capacity(top_k);
    let mut work = probs;
    for _ in 0..top_k {
        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in work.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best_i = i;
            }
        }
        ids.push(best_i as u32);
        weights.push(best_v);
        work[best_i] = f32::NEG_INFINITY;
    }
    let sum: f32 = weights.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(model_err("router top-k weight sum is non-positive"));
    }
    for w in &mut weights {
        *w /= sum;
    }
    Ok((ids, weights))
}

/// Per-token capture surface for one layer (matches complete-binary capture).
#[derive(Clone, Debug)]
pub struct LayerTokenCapture {
    pub layer: usize,
    pub selected_expert_ids: Vec<u32>,
    pub normalized_route_weights: Vec<f32>,
    pub router_input_hidden: Vec<f32>,
}

/// Residuals for every token of one probe: length `seq * hidden`.
pub type ProbeHidden = Vec<f32>;

/// Run one loaded layer over one probe sequence (causal attention within the probe).
///
/// Returns updated residuals and per-token route/hidden captures for this layer.
pub fn forward_layer_probe(
    layer: &LoadedLayer,
    hidden: &mut ProbeHidden,
    seq_len: usize,
) -> Result<Vec<LayerTokenCapture>> {
    if seq_len == 0 {
        return Ok(Vec::new());
    }
    if hidden.len() != seq_len * QWEN30_HIDDEN {
        return Err(model_err(format!(
            "probe hidden len {} != seq {seq_len} * {QWEN30_HIDDEN}",
            hidden.len()
        )));
    }

    let q_dim = QWEN30_HEADS * QWEN30_HEAD_DIM;
    let kv_dim = QWEN30_KV_HEADS * QWEN30_HEAD_DIM;
    let mut k_cache = vec![0.0f32; seq_len * kv_dim];
    let mut v_cache = vec![0.0f32; seq_len * kv_dim];
    let mut captures = Vec::with_capacity(seq_len);

    let mut x_norm = vec![0.0f32; QWEN30_HIDDEN];
    let mut q = vec![0.0f32; q_dim];
    let mut k = vec![0.0f32; kv_dim];
    let mut v = vec![0.0f32; kv_dim];
    let mut attn = vec![0.0f32; q_dim];
    let mut attn_proj = vec![0.0f32; QWEN30_HIDDEN];
    let mut router_logits = vec![0.0f32; QWEN30_EXPERTS];
    let mut gate = vec![0.0f32; QWEN30_MOE_INTERMEDIATE];
    let mut up = vec![0.0f32; QWEN30_MOE_INTERMEDIATE];
    let mut act = vec![0.0f32; QWEN30_MOE_INTERMEDIATE];
    let mut down = vec![0.0f32; QWEN30_HIDDEN];
    let mut moe_combined = vec![0.0f32; QWEN30_HIDDEN];

    for pos in 0..seq_len {
        let x = &mut hidden[pos * QWEN30_HIDDEN..(pos + 1) * QWEN30_HIDDEN];
        rmsnorm(x, &layer.input_layernorm, QWEN30_RMS_EPS, &mut x_norm);

        gemv_bf16(&layer.q_proj, q_dim, QWEN30_HIDDEN, &x_norm, &mut q)?;
        gemv_bf16(&layer.k_proj, kv_dim, QWEN30_HIDDEN, &x_norm, &mut k)?;
        gemv_bf16(&layer.v_proj, kv_dim, QWEN30_HIDDEN, &x_norm, &mut v)?;

        rmsnorm_rows(&mut q, &layer.q_norm, QWEN30_HEADS, QWEN30_HEAD_DIM)?;
        rmsnorm_rows(&mut k, &layer.k_norm, QWEN30_KV_HEADS, QWEN30_HEAD_DIM)?;

        for h in 0..QWEN30_HEADS {
            let start = h * QWEN30_HEAD_DIM;
            rope_neox_inplace(
                &mut q[start..start + QWEN30_HEAD_DIM],
                pos as u32,
                QWEN30_ROPE_THETA,
            );
        }
        for h in 0..QWEN30_KV_HEADS {
            let start = h * QWEN30_HEAD_DIM;
            rope_neox_inplace(
                &mut k[start..start + QWEN30_HEAD_DIM],
                pos as u32,
                QWEN30_ROPE_THETA,
            );
        }

        let kv_off = pos * kv_dim;
        k_cache[kv_off..kv_off + kv_dim].copy_from_slice(&k);
        v_cache[kv_off..kv_off + kv_dim].copy_from_slice(&v);

        mha_decode_step(
            &q,
            &k_cache[..(pos + 1) * kv_dim],
            &v_cache[..(pos + 1) * kv_dim],
            QWEN30_HEADS,
            QWEN30_KV_HEADS,
            QWEN30_HEAD_DIM,
            pos + 1,
            &mut attn,
        )?;

        gemv_bf16(&layer.o_proj, QWEN30_HIDDEN, q_dim, &attn, &mut attn_proj)?;
        add_inplace(x, &attn_proj);

        // Router input = post-attention RMSNorm(x). This is the activation
        // surface the repack fits against.
        rmsnorm(x, &layer.post_attention_layernorm, QWEN30_RMS_EPS, &mut x_norm);
        gemv_bf16(
            &layer.router,
            QWEN30_EXPERTS,
            QWEN30_HIDDEN,
            &x_norm,
            &mut router_logits,
        )?;
        let (ids, weights) = router_topk_norm(&router_logits, QWEN30_TOP_K)?;

        moe_combined.fill(0.0);
        for (slot, (&expert, &w)) in ids.iter().zip(weights.iter()).enumerate() {
            let expert_w = layer
                .experts
                .get(expert as usize)
                .ok_or_else(|| model_err(format!("route expert {expert} out of range")))?;
            gemv_bf16(
                &expert_w.gate,
                QWEN30_MOE_INTERMEDIATE,
                QWEN30_HIDDEN,
                &x_norm,
                &mut gate,
            )?;
            gemv_bf16(
                &expert_w.up,
                QWEN30_MOE_INTERMEDIATE,
                QWEN30_HIDDEN,
                &x_norm,
                &mut up,
            )?;
            silu_mul(&gate, &up, &mut act);
            gemv_bf16(
                &expert_w.down,
                QWEN30_HIDDEN,
                QWEN30_MOE_INTERMEDIATE,
                &act,
                &mut down,
            )?;
            for i in 0..QWEN30_HIDDEN {
                moe_combined[i] += down[i] * w;
            }
            let _ = slot;
        }
        add_inplace(x, &moe_combined);

        captures.push(LayerTokenCapture {
            layer: layer.layer,
            selected_expert_ids: ids,
            normalized_route_weights: weights,
            router_input_hidden: x_norm.clone(),
        });
    }
    Ok(captures)
}

/// Embed every probe's tokens into residual streams (range-read rows only).
pub fn embed_probes(
    index: &SourceBf16Index,
    probes: &[(String, Vec<u32>)],
) -> Result<Vec<ProbeHidden>> {
    let mut out = Vec::with_capacity(probes.len());
    for (_, tokens) in probes {
        let mut h = Vec::with_capacity(tokens.len() * QWEN30_HIDDEN);
        for &tok in tokens {
            let row = index.embed_row(tok)?;
            h.extend_from_slice(&row);
        }
        out.push(h);
    }
    Ok(out)
}

/// Final RMSNorm + lm_head logits for the last residual of a sequence.
pub fn logits_from_final_hidden(
    index: &SourceBf16Index,
    hidden: &[f32],
) -> Result<Vec<f32>> {
    if hidden.len() != QWEN30_HIDDEN {
        return Err(model_err("final hidden width mismatch"));
    }
    let norm_w = index.read_f32("model.norm.weight")?;
    let mut normed = vec![0.0f32; QWEN30_HIDDEN];
    rmsnorm(hidden, &norm_w, QWEN30_RMS_EPS, &mut normed);
    // lm_head is large (~593 MiB BF16). Load, matvec, free.
    let lm_head = index.read_raw("lm_head.weight")?;
    let mut logits = vec![0.0f32; QWEN30_VOCAB];
    gemv_bf16(&lm_head, QWEN30_VOCAB, QWEN30_HIDDEN, &normed, &mut logits)?;
    drop(lm_head);
    Ok(logits)
}

/// Layer-major full forward over all probes: returns per-probe per-layer per-token captures
/// and leaves `hiddens` as the final residuals.
pub fn capture_all_layers(
    index: &SourceBf16Index,
    probes: &[(String, Vec<u32>)],
    hiddens: &mut [ProbeHidden],
    mut on_layer: Option<&mut dyn FnMut(usize, u64)>,
) -> Result<Vec<Vec<Vec<LayerTokenCapture>>>> {
    if hiddens.len() != probes.len() {
        return Err(model_err("hiddens/probes length mismatch"));
    }
    // captures[probe][token][layer]
    let mut captures: Vec<Vec<Vec<LayerTokenCapture>>> = probes
        .iter()
        .map(|(_, toks)| (0..toks.len()).map(|_| Vec::with_capacity(QWEN30_LAYERS)).collect())
        .collect();

    for layer_idx in 0..QWEN30_LAYERS {
        let layer = LoadedLayer::load(index, layer_idx)?;
        let resident = layer.resident_bytes;
        if let Some(cb) = on_layer.as_mut() {
            cb(layer_idx, resident);
        }
        for (pi, (_, tokens)) in probes.iter().enumerate() {
            let seq_len = tokens.len();
            let layer_caps = forward_layer_probe(&layer, &mut hiddens[pi], seq_len)?;
            if layer_caps.len() != seq_len {
                return Err(model_err(format!(
                    "layer {layer_idx} probe {pi}: capture count {} != {seq_len}",
                    layer_caps.len()
                )));
            }
            for (pos, cap) in layer_caps.into_iter().enumerate() {
                captures[pi][pos].push(cap);
            }
        }
        // Explicit free before next layer load.
        drop(layer);
    }
    Ok(captures)
}

/// One layer's KV cache (f32), layout `(seq, kv_heads * head_dim)`.
struct LayerKv {
    k: Vec<f32>,
    v: Vec<f32>,
    seq: usize,
}

impl LayerKv {
    fn new() -> Self {
        Self {
            k: Vec::new(),
            v: Vec::new(),
            seq: 0,
        }
    }
}

/// Run one position through a loaded layer, appending to that layer's KV cache.
/// Updates `x` (length `hidden`) in place to the post-MoE residual.
fn forward_layer_token(
    layer: &LoadedLayer,
    x: &mut [f32],
    pos: usize,
    kv: &mut LayerKv,
) -> Result<LayerTokenCapture> {
    if x.len() != QWEN30_HIDDEN {
        return Err(model_err("forward_layer_token residual width mismatch"));
    }
    let q_dim = QWEN30_HEADS * QWEN30_HEAD_DIM;
    let kv_dim = QWEN30_KV_HEADS * QWEN30_HEAD_DIM;
    if pos != kv.seq {
        return Err(model_err(format!(
            "KV position mismatch: pos={pos} kv.seq={}",
            kv.seq
        )));
    }
    let mut x_norm = vec![0.0f32; QWEN30_HIDDEN];
    let mut q = vec![0.0f32; q_dim];
    let mut k = vec![0.0f32; kv_dim];
    let mut v = vec![0.0f32; kv_dim];
    let mut attn = vec![0.0f32; q_dim];
    let mut attn_proj = vec![0.0f32; QWEN30_HIDDEN];
    let mut router_logits = vec![0.0f32; QWEN30_EXPERTS];
    let mut gate = vec![0.0f32; QWEN30_MOE_INTERMEDIATE];
    let mut up = vec![0.0f32; QWEN30_MOE_INTERMEDIATE];
    let mut act = vec![0.0f32; QWEN30_MOE_INTERMEDIATE];
    let mut down = vec![0.0f32; QWEN30_HIDDEN];
    let mut moe_combined = vec![0.0f32; QWEN30_HIDDEN];

    rmsnorm(x, &layer.input_layernorm, QWEN30_RMS_EPS, &mut x_norm);
    gemv_bf16(&layer.q_proj, q_dim, QWEN30_HIDDEN, &x_norm, &mut q)?;
    gemv_bf16(&layer.k_proj, kv_dim, QWEN30_HIDDEN, &x_norm, &mut k)?;
    gemv_bf16(&layer.v_proj, kv_dim, QWEN30_HIDDEN, &x_norm, &mut v)?;
    rmsnorm_rows(&mut q, &layer.q_norm, QWEN30_HEADS, QWEN30_HEAD_DIM)?;
    rmsnorm_rows(&mut k, &layer.k_norm, QWEN30_KV_HEADS, QWEN30_HEAD_DIM)?;
    for h in 0..QWEN30_HEADS {
        let start = h * QWEN30_HEAD_DIM;
        rope_neox_inplace(
            &mut q[start..start + QWEN30_HEAD_DIM],
            pos as u32,
            QWEN30_ROPE_THETA,
        );
    }
    for h in 0..QWEN30_KV_HEADS {
        let start = h * QWEN30_HEAD_DIM;
        rope_neox_inplace(
            &mut k[start..start + QWEN30_HEAD_DIM],
            pos as u32,
            QWEN30_ROPE_THETA,
        );
    }
    kv.k.extend_from_slice(&k);
    kv.v.extend_from_slice(&v);
    kv.seq += 1;
    mha_decode_step(
        &q,
        &kv.k,
        &kv.v,
        QWEN30_HEADS,
        QWEN30_KV_HEADS,
        QWEN30_HEAD_DIM,
        kv.seq,
        &mut attn,
    )?;
    gemv_bf16(&layer.o_proj, QWEN30_HIDDEN, q_dim, &attn, &mut attn_proj)?;
    add_inplace(x, &attn_proj);
    rmsnorm(x, &layer.post_attention_layernorm, QWEN30_RMS_EPS, &mut x_norm);
    gemv_bf16(
        &layer.router,
        QWEN30_EXPERTS,
        QWEN30_HIDDEN,
        &x_norm,
        &mut router_logits,
    )?;
    let (ids, weights) = router_topk_norm(&router_logits, QWEN30_TOP_K)?;
    moe_combined.fill(0.0);
    for (&expert, &w) in ids.iter().zip(weights.iter()) {
        let expert_w = layer
            .experts
            .get(expert as usize)
            .ok_or_else(|| model_err(format!("route expert {expert} out of range")))?;
        gemv_bf16(
            &expert_w.gate,
            QWEN30_MOE_INTERMEDIATE,
            QWEN30_HIDDEN,
            &x_norm,
            &mut gate,
        )?;
        gemv_bf16(
            &expert_w.up,
            QWEN30_MOE_INTERMEDIATE,
            QWEN30_HIDDEN,
            &x_norm,
            &mut up,
        )?;
        silu_mul(&gate, &up, &mut act);
        gemv_bf16(
            &expert_w.down,
            QWEN30_HIDDEN,
            QWEN30_MOE_INTERMEDIATE,
            &act,
            &mut down,
        )?;
        for i in 0..QWEN30_HIDDEN {
            moe_combined[i] += down[i] * w;
        }
    }
    add_inplace(x, &moe_combined);
    Ok(LayerTokenCapture {
        layer: layer.layer,
        selected_expert_ids: ids,
        normalized_route_weights: weights,
        router_input_hidden: x_norm,
    })
}

/// Greedy decode with the source one-user chat template.
///
/// Layer-major: prefill streams each layer once over the prompt (keeping a
/// small per-layer KV), then each new token re-streams the 48 layers once
/// with only the new position. Never co-resident-loads the full source.
pub fn greedy_decode_user_prompt(
    index: &SourceBf16Index,
    tokenizer_path: &Path,
    user_text: &str,
    max_new_tokens: usize,
) -> Result<GreedyDecodeResult> {
    use tokenizers::Tokenizer;

    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| {
        model_err(format!(
            "cannot load tokenizer {}: {e}",
            tokenizer_path.display()
        ))
    })?;
    let rendered = format!("<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n");
    let encoding = tokenizer
        .encode(rendered.as_str(), false)
        .map_err(|e| model_err(format!("tokenizer encode failed: {e}")))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    if prompt_ids.is_empty() {
        return Err(model_err("chat-template encoding produced no tokens"));
    }

    // Residuals for the current sequence (grows with generation).
    let mut residuals: Vec<Vec<f32>> = Vec::with_capacity(prompt_ids.len() + max_new_tokens);
    for &tok in &prompt_ids {
        residuals.push(index.embed_row(tok)?);
    }
    // Per-layer KV across the whole decode session.
    let mut kvs: Vec<LayerKv> = (0..QWEN30_LAYERS).map(|_| LayerKv::new()).collect();

    // Prefill: for each layer, push every prompt position through it.
    for layer_idx in 0..QWEN30_LAYERS {
        let layer = LoadedLayer::load(index, layer_idx)?;
        for pos in 0..prompt_ids.len() {
            forward_layer_token(&layer, &mut residuals[pos], pos, &mut kvs[layer_idx])?;
        }
        drop(layer);
    }

    let mut generated = Vec::new();
    let mut first_token_top10 = Vec::new();
    let eos = [151645u32, 151643u32];

    for _step in 0..max_new_tokens {
        let last = residuals.last().ok_or_else(|| model_err("empty residual"))?;
        let logits = logits_from_final_hidden(index, last)?;
        let next = argmax_f32(&logits);
        if generated.is_empty() {
            let mut ranked: Vec<(u32, f32)> = logits
                .iter()
                .enumerate()
                .map(|(i, &v)| (i as u32, v))
                .collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            ranked.truncate(10);
            first_token_top10 = ranked;
        }
        generated.push(next);
        if eos.contains(&next) {
            break;
        }
        // Append embed of new token and stream it through every layer.
        let pos = residuals.len();
        residuals.push(index.embed_row(next)?);
        for layer_idx in 0..QWEN30_LAYERS {
            let layer = LoadedLayer::load(index, layer_idx)?;
            forward_layer_token(&layer, &mut residuals[pos], pos, &mut kvs[layer_idx])?;
            drop(layer);
        }
    }

    let cont_text = tokenizer
        .decode(&generated, true)
        .map_err(|e| model_err(format!("tokenizer decode failed: {e}")))?;
    Ok(GreedyDecodeResult {
        prompt_token_count: prompt_ids.len(),
        prompt_token_ids: prompt_ids,
        generated_token_ids: generated,
        continuation_text: cont_text,
        rendered_prompt: rendered,
        first_token_top10,
    })
}

#[derive(Clone, Debug)]
pub struct GreedyDecodeResult {
    pub prompt_token_count: usize,
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub continuation_text: String,
    pub rendered_prompt: String,
    pub first_token_top10: Vec<(u32, f32)>,
}

/// True when the continuation is a coherent capital-of-France answer.
///
/// Accepts top-1 `" Paris"` / `"Paris"` and multi-token variants such as
/// `"The capital of France is Paris"`. Rejects `"Wien swiper"` degeneration.
pub fn is_coherent_paris_continuation(text: &str) -> bool {
    let t = text.trim_start();
    if t.is_empty() {
        return false;
    }
    let lower = t.to_ascii_lowercase();
    if lower.contains("wien") || lower.contains("swiper") {
        return false;
    }
    // Degenerate pure repetition (same word thrice) is not coherent.
    let words: Vec<&str> = lower.split_whitespace().collect();
    if words.len() >= 3 && words[0] == words[1] && words[1] == words[2] {
        return false;
    }
    if lower.starts_with("paris")
        || lower.starts_with("**paris")
        || lower.starts_with("*paris")
    {
        return true;
    }
    // Multi-token correct answers.
    if lower.contains("paris")
        && (lower.contains("france")
            || lower.starts_with("the capital")
            || lower.starts_with("it's paris")
            || lower.starts_with("it is paris"))
    {
        return true;
    }
    false
}

/// Process peak RSS in bytes (macOS: `ru_maxrss` is already bytes).
pub fn peak_rss_bytes() -> u64 {
    #[cfg(unix)]
    {
        // Layout matches cost_ledger::sample_page_faults (Darwin timeval padding).
        #[cfg(target_os = "macos")]
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct TimeVal {
            tv_sec: i64,
            tv_usec: i32,
            _pad: i32,
        }
        #[cfg(not(target_os = "macos"))]
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct TimeVal {
            tv_sec: i64,
            tv_usec: i64,
        }
        #[repr(C)]
        struct Rusage {
            ru_utime: TimeVal,
            ru_stime: TimeVal,
            ru_maxrss: i64,
            ru_ixrss: i64,
            ru_idrss: i64,
            ru_isrss: i64,
            ru_minflt: i64,
            ru_majflt: i64,
            _pad: [i64; 8],
        }
        extern "C" {
            fn getrusage(who: i32, usage: *mut Rusage) -> i32;
        }
        const RUSAGE_SELF: i32 = 0;
        #[cfg(target_os = "macos")]
        let zero_tv = TimeVal {
            tv_sec: 0,
            tv_usec: 0,
            _pad: 0,
        };
        #[cfg(not(target_os = "macos"))]
        let zero_tv = TimeVal {
            tv_sec: 0,
            tv_usec: 0,
        };
        let mut u = Rusage {
            ru_utime: zero_tv,
            ru_stime: zero_tv,
            ru_maxrss: 0,
            ru_ixrss: 0,
            ru_idrss: 0,
            ru_isrss: 0,
            ru_minflt: 0,
            ru_majflt: 0,
            _pad: [0; 8],
        };
        let rc = unsafe { getrusage(RUSAGE_SELF, &mut u) };
        if rc != 0 {
            return 0;
        }
        u.ru_maxrss.max(0) as u64
    }
    #[cfg(not(unix))]
    {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn router_topk_renormalizes() {
        let mut logits = vec![-10.0f32; QWEN30_EXPERTS];
        logits[3] = 5.0;
        logits[7] = 4.0;
        logits[11] = 3.0;
        logits[13] = 2.0;
        logits[17] = 1.0;
        logits[19] = 0.5;
        logits[23] = 0.25;
        logits[29] = 0.1;
        let (ids, w) = router_topk_norm(&logits, 8).unwrap();
        assert_eq!(ids.len(), 8);
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert_eq!(ids[0], 3);
    }

    #[test]
    fn paris_coherence_accepts_variants() {
        assert!(is_coherent_paris_continuation(" Paris"));
        assert!(is_coherent_paris_continuation("Paris is the capital"));
        assert!(!is_coherent_paris_continuation(" Wien swiper swiper"));
        assert!(!is_coherent_paris_continuation("swiper Wien"));
    }
}
