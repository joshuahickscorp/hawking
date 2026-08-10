//! Layer-major BF16 SOURCE forward + activation capture for Qwen3-Coder-Next (Q80).
//!
//! Resource contract:
//! * Does **not** resident-load the ~148 GiB BF16 source.
//! * Generation and capture both stream one layer's weights at a time via
//!   safetensors range-reads, then free them.
//! * Capture inverts the loop (layer-major): load layer N, push ALL corpus
//!   tokens through it, write routes + retained hiddens, free weights.
//!
//! Operator semantics are **reused** from `qwen80_complete_runtime`:
//! residual RMSNorm `(1+w)`, Gated DeltaNet recurrence, GQA q/k norm + partial
//! RoPE + causal attention + sigmoid gate, top-10 `norm_topk_prob` router,
//! SwiGLU experts, shared expert + sigmoid gate combine. Only the weight
//! backend differs (BF16 GEMV vs packed complete-binary).

use crate::artifact::widen_native;
use crate::kernels::{add_inplace, argmax_f32, silu_mul};
use crate::model::qwen80_complete_runtime::{
    qwen80_gqa_apply_sigmoid_gate, qwen80_gqa_causal_attention,
    qwen80_gqa_query_from_interleaved_q_projection, qwen80_gqa_source_norm_rope,
    source_qwen80_ba_to_decay_beta, source_qwen80_causal_conv_step_dense,
    source_qwen80_gated_rms_norm, source_qwen80_l2_normalize, source_qwen80_recurrent_deltanet,
    source_qwen80_residual_rms_norm, source_qwen80_split_linear_qkvz, source_qwen80_topk_router,
    Qwen80CanonicalGqaLayout, Qwen80CanonicalLinearDeltaNetLayout,
};
use crate::{Error, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

pub const QWEN80_LAYERS: usize = 48;
pub const QWEN80_HIDDEN: usize = 2048;
pub const QWEN80_FULL_ATTN_HEADS: usize = 16;
pub const QWEN80_FULL_ATTN_KV_HEADS: usize = 2;
pub const QWEN80_FULL_ATTN_HEAD_DIM: usize = 256;
pub const QWEN80_EXPERTS: usize = 512;
pub const QWEN80_TOP_K: usize = 10;
pub const QWEN80_MOE_INTERMEDIATE: usize = 512;
pub const QWEN80_SHARED_EXPERT_INTERMEDIATE: usize = 512;
pub const QWEN80_VOCAB: usize = 151_936;
pub const QWEN80_TOKENIZER_VOCAB: usize = 151_669;
pub const QWEN80_FULL_ATTENTION_INTERVAL: usize = 4;
pub const QWEN80_RMS_EPS: f32 = 1.0e-6;
/// Soft upper bound: single-digit GiB. Approaching 148 GiB means resident load.
pub const STREAMED_PEAK_RSS_HARD_CAP_BYTES: u64 = 16 * 1024 * 1024 * 1024;
/// Contract estimate for per-layer expert BF16 payload (~3 GiB).
pub const PER_LAYER_EXPERT_BF16_BYTES: u64 = 3 * 1024 * 1024 * 1024;

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

/// Row-major GEMV with f32 rows (after a one-shot BF16 widen of a small tensor).
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
    data_offset: u64,
    nbytes: usize,
    shape: Vec<usize>,
    dtype: String,
}

/// Index over the source BF16 safetensors shards. Headers only; payloads range-read.
pub struct SourceBf16Index {
    pub model_dir: PathBuf,
    map: HashMap<String, TensorLoc>,
    handles: Mutex<HashMap<PathBuf, File>>,
    /// Cumulative payload bytes successfully range-read.
    pub bytes_read: Mutex<u64>,
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
            by_shard
                .entry(shard.to_string())
                .or_default()
                .push(name.clone());
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
            bytes_read: Mutex::new(0),
        })
    }

    pub fn tensor_count(&self) -> usize {
        self.map.len()
    }

    pub fn bytes_read_total(&self) -> u64 {
        *self.bytes_read.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn require(&self, name: &str) -> Result<&TensorLoc> {
        self.map
            .get(name)
            .ok_or_else(|| model_err(format!("source index lacks tensor {name}")))
    }

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
        if let Ok(mut br) = self.bytes_read.lock() {
            *br = br.saturating_add(loc.nbytes as u64);
        }
        Ok(buf)
    }

    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>> {
        let raw = self.read_raw(name)?;
        widen_native("native.bf16", &raw)
    }

    /// Read a single embedding row without loading the full embedding table.
    pub fn embed_row(&self, token: u32) -> Result<Vec<f32>> {
        if token as usize >= QWEN80_VOCAB {
            return Err(model_err(format!("token {token} outside vocabulary")));
        }
        let loc = self.require("model.embed_tokens.weight")?;
        if loc.shape != [QWEN80_VOCAB, QWEN80_HIDDEN] {
            return Err(model_err(format!(
                "embed_tokens shape {:?} is not [{QWEN80_VOCAB}, {QWEN80_HIDDEN}]",
                loc.shape
            )));
        }
        let row_bytes = QWEN80_HIDDEN * 2;
        let offset = loc
            .data_offset
            .checked_add(
                (token as u64)
                    .checked_mul(row_bytes as u64)
                    .ok_or_else(|| model_err("embed row offset overflow"))?,
            )
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
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| model_err(format!("seek embed row {token}: {e}")))?;
        let mut buf = vec![0u8; row_bytes];
        file.read_exact(&mut buf)
            .map_err(|e| model_err(format!("read embed row {token}: {e}")))?;
        if let Ok(mut br) = self.bytes_read.lock() {
            *br = br.saturating_add(row_bytes as u64);
        }
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
    file.read_exact(&mut len_buf).map_err(|e| {
        model_err(format!(
            "cannot read header length of {}: {e}",
            path.display()
        ))
    })?;
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
    let value: Value = serde_json::from_slice(&raw).map_err(|e| {
        model_err(format!(
            "safetensors header JSON invalid in {}: {e}",
            path.display()
        ))
    })?;
    let object = value.as_object().ok_or_else(|| {
        model_err(format!(
            "safetensors header is not an object in {}",
            path.display()
        ))
    })?;
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
            return Err(model_err(format!(
                "tensor {name} data_offsets is not a pair"
            )));
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayerKind {
    LinearDeltaNet,
    FullAttentionGqa,
}

pub fn layer_kind(layer: usize) -> Result<LayerKind> {
    if layer >= QWEN80_LAYERS {
        return Err(model_err(format!("layer {layer} out of range")));
    }
    // Source: (layer + 1) % full_attention_interval == 0 => GQA
    if (layer + 1) % QWEN80_FULL_ATTENTION_INTERVAL == 0 {
        Ok(LayerKind::FullAttentionGqa)
    } else {
        Ok(LayerKind::LinearDeltaNet)
    }
}

pub struct ExpertWeights {
    pub gate: Vec<u8>,
    pub up: Vec<u8>,
    pub down: Vec<u8>,
}

/// Mixer + MoE weights for one layer, held only for that layer's corpus pass.
pub struct LoadedLayer {
    pub layer: usize,
    pub kind: LayerKind,
    pub input_layernorm: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
    // DeltaNet
    pub in_proj_qkvz: Option<Vec<u8>>,
    pub in_proj_ba: Option<Vec<u8>>,
    pub conv1d: Option<Vec<f32>>,
    pub a_log: Option<Vec<f32>>,
    pub dt_bias: Option<Vec<f32>>,
    pub gated_rms_norm: Option<Vec<f32>>,
    pub out_proj_linear: Option<Vec<u8>>,
    // GQA
    pub q_proj: Option<Vec<u8>>,
    pub k_proj: Option<Vec<u8>>,
    pub v_proj: Option<Vec<u8>>,
    pub o_proj: Option<Vec<u8>>,
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    // MoE
    pub router: Vec<u8>,
    pub shared_gate: Vec<u8>,
    pub shared_up: Vec<u8>,
    pub shared_down: Vec<u8>,
    pub shared_expert_gate: Vec<u8>,
    pub experts: Vec<ExpertWeights>,
    pub resident_bytes: u64,
    pub load_secs: f64,
}

impl LoadedLayer {
    pub fn load(index: &SourceBf16Index, layer: usize) -> Result<Self> {
        let t0 = Instant::now();
        let kind = layer_kind(layer)?;
        let input_layernorm = index.read_f32(&layer_name(layer, "input_layernorm.weight"))?;
        let post_attention_layernorm =
            index.read_f32(&layer_name(layer, "post_attention_layernorm.weight"))?;
        let router = index.read_raw(&layer_name(layer, "mlp.gate.weight"))?;
        let shared_gate = index.read_raw(&layer_name(layer, "mlp.shared_expert.gate_proj.weight"))?;
        let shared_up = index.read_raw(&layer_name(layer, "mlp.shared_expert.up_proj.weight"))?;
        let shared_down = index.read_raw(&layer_name(layer, "mlp.shared_expert.down_proj.weight"))?;
        let shared_expert_gate =
            index.read_raw(&layer_name(layer, "mlp.shared_expert_gate.weight"))?;

        let mut resident = (input_layernorm.len() + post_attention_layernorm.len()) * 4
            + router.len()
            + shared_gate.len()
            + shared_up.len()
            + shared_down.len()
            + shared_expert_gate.len();

        let mut in_proj_qkvz = None;
        let mut in_proj_ba = None;
        let mut conv1d = None;
        let mut a_log = None;
        let mut dt_bias = None;
        let mut gated_rms_norm = None;
        let mut out_proj_linear = None;
        let mut q_proj = None;
        let mut k_proj = None;
        let mut v_proj = None;
        let mut o_proj = None;
        let mut q_norm = None;
        let mut k_norm = None;

        match kind {
            LayerKind::LinearDeltaNet => {
                let qkvz = index.read_raw(&layer_name(layer, "linear_attn.in_proj_qkvz.weight"))?;
                let ba = index.read_raw(&layer_name(layer, "linear_attn.in_proj_ba.weight"))?;
                let conv = index.read_f32(&layer_name(layer, "linear_attn.conv1d.weight"))?;
                let al = index.read_f32(&layer_name(layer, "linear_attn.A_log"))?;
                let dt = index.read_f32(&layer_name(layer, "linear_attn.dt_bias"))?;
                let gn = index.read_f32(&layer_name(layer, "linear_attn.norm.weight"))?;
                let op = index.read_raw(&layer_name(layer, "linear_attn.out_proj.weight"))?;
                resident += qkvz.len()
                    + ba.len()
                    + conv.len() * 4
                    + al.len() * 4
                    + dt.len() * 4
                    + gn.len() * 4
                    + op.len();
                in_proj_qkvz = Some(qkvz);
                in_proj_ba = Some(ba);
                conv1d = Some(conv);
                a_log = Some(al);
                dt_bias = Some(dt);
                gated_rms_norm = Some(gn);
                out_proj_linear = Some(op);
            }
            LayerKind::FullAttentionGqa => {
                let q = index.read_raw(&layer_name(layer, "self_attn.q_proj.weight"))?;
                let k = index.read_raw(&layer_name(layer, "self_attn.k_proj.weight"))?;
                let v = index.read_raw(&layer_name(layer, "self_attn.v_proj.weight"))?;
                let o = index.read_raw(&layer_name(layer, "self_attn.o_proj.weight"))?;
                let qn = index.read_f32(&layer_name(layer, "self_attn.q_norm.weight"))?;
                let kn = index.read_f32(&layer_name(layer, "self_attn.k_norm.weight"))?;
                resident += q.len() + k.len() + v.len() + o.len() + qn.len() * 4 + kn.len() * 4;
                q_proj = Some(q);
                k_proj = Some(k);
                v_proj = Some(v);
                o_proj = Some(o);
                q_norm = Some(qn);
                k_norm = Some(kn);
            }
        }

        let mut experts = Vec::with_capacity(QWEN80_EXPERTS);
        for expert in 0..QWEN80_EXPERTS {
            let gate = index.read_raw(&expert_name(layer, expert, "gate_proj"))?;
            let up = index.read_raw(&expert_name(layer, expert, "up_proj"))?;
            let down = index.read_raw(&expert_name(layer, expert, "down_proj"))?;
            resident += gate.len() + up.len() + down.len();
            experts.push(ExpertWeights { gate, up, down });
        }

        Ok(Self {
            layer,
            kind,
            input_layernorm,
            post_attention_layernorm,
            in_proj_qkvz,
            in_proj_ba,
            conv1d,
            a_log,
            dt_bias,
            gated_rms_norm,
            out_proj_linear,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            router,
            shared_gate,
            shared_up,
            shared_down,
            shared_expert_gate,
            experts,
            resident_bytes: resident as u64,
            load_secs: t0.elapsed().as_secs_f64(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct DeltaNetState {
    pub conv_state: Vec<f32>,
    pub recurrent_state: Vec<f32>,
}

impl DeltaNetState {
    pub fn zero(layout: &Qwen80CanonicalLinearDeltaNetLayout) -> Result<Self> {
        Ok(Self {
            conv_state: vec![0.0; layout.conv_state_elements()?],
            recurrent_state: vec![0.0; layout.recurrent_state_elements()?],
        })
    }
}

#[derive(Clone, Debug)]
pub struct GqaState {
    pub key_cache: Vec<f32>,
    pub value_cache: Vec<f32>,
    pub max_seq: usize,
}

impl GqaState {
    pub fn new(max_seq: usize, layout: &Qwen80CanonicalGqaLayout) -> Self {
        Self {
            key_cache: vec![0.0; max_seq * layout.kv_dim],
            value_cache: vec![0.0; max_seq * layout.kv_dim],
            max_seq,
        }
    }
}

/// Per-token capture surface for one layer (matches complete-binary capture).
#[derive(Clone, Debug)]
pub struct LayerTokenCapture {
    pub layer: usize,
    pub selected_expert_ids: Vec<u32>,
    pub normalized_route_weights: Vec<f32>,
    pub router_input_hidden: Vec<f32>,
}

pub type ProbeHidden = Vec<f32>;

fn swiglu_mlp_bf16(
    gate_w: &[u8],
    up_w: &[u8],
    down_w: &[u8],
    x: &[f32],
    intermediate: usize,
    gate_buf: &mut [f32],
    up_buf: &mut [f32],
    act_buf: &mut [f32],
    down_buf: &mut [f32],
) -> Result<()> {
    gemv_bf16(gate_w, intermediate, QWEN80_HIDDEN, x, gate_buf)?;
    gemv_bf16(up_w, intermediate, QWEN80_HIDDEN, x, up_buf)?;
    silu_mul(gate_buf, up_buf, act_buf);
    gemv_bf16(down_w, QWEN80_HIDDEN, intermediate, act_buf, down_buf)?;
    Ok(())
}

fn moe_combine(
    layer: &LoadedLayer,
    router_input: &[f32],
    moe_combined: &mut [f32],
    router_logits: &mut [f32],
    gate: &mut [f32],
    up: &mut [f32],
    act: &mut [f32],
    down: &mut [f32],
    shared_gate: &mut [f32],
    shared_up: &mut [f32],
    shared_act: &mut [f32],
    shared_down: &mut [f32],
    shared_gate_logit: &mut [f32],
) -> Result<(Vec<u32>, Vec<f32>)> {
    // Shared MLP first (source SparseMoeBlock order), then router + routed.
    swiglu_mlp_bf16(
        &layer.shared_gate,
        &layer.shared_up,
        &layer.shared_down,
        router_input,
        QWEN80_SHARED_EXPERT_INTERMEDIATE,
        shared_gate,
        shared_up,
        shared_act,
        shared_down,
    )?;
    gemv_bf16(
        &layer.router,
        QWEN80_EXPERTS,
        QWEN80_HIDDEN,
        router_input,
        router_logits,
    )?;
    let route = source_qwen80_topk_router(router_logits)?;
    moe_combined.fill(0.0);
    for (slot, (&eid, &w)) in route.ids.iter().zip(route.weights.iter()).enumerate() {
        let expert = layer
            .experts
            .get(eid as usize)
            .ok_or_else(|| model_err(format!("route expert {eid} out of range")))?;
        swiglu_mlp_bf16(
            &expert.gate,
            &expert.up,
            &expert.down,
            router_input,
            QWEN80_MOE_INTERMEDIATE,
            gate,
            up,
            act,
            down,
        )?;
        for i in 0..QWEN80_HIDDEN {
            moe_combined[i] += down[i] * w;
        }
        let _ = slot;
    }
    gemv_bf16(
        &layer.shared_expert_gate,
        1,
        QWEN80_HIDDEN,
        router_input,
        shared_gate_logit,
    )?;
    let gate_val = 1.0 / (1.0 + (-shared_gate_logit[0]).exp());
    if !gate_val.is_finite() || !(0.0..=1.0).contains(&gate_val) {
        return Err(model_err("shared expert gate sigmoid invalid"));
    }
    for i in 0..QWEN80_HIDDEN {
        moe_combined[i] += shared_down[i] * gate_val;
    }
    let ids = route.ids.iter().map(|&id| id as u32).collect();
    let weights = route.weights.to_vec();
    Ok((ids, weights))
}

/// One DeltaNet mixer step through first residual (reuses packed-oracle maths).
fn deltanet_mixer_step(
    layer: &LoadedLayer,
    hidden: &[f32],
    state: &mut DeltaNetState,
    layout: &Qwen80CanonicalLinearDeltaNetLayout,
) -> Result<Vec<f32>> {
    let qkvz_w = layer
        .in_proj_qkvz
        .as_ref()
        .ok_or_else(|| model_err("missing in_proj_qkvz"))?;
    let ba_w = layer
        .in_proj_ba
        .as_ref()
        .ok_or_else(|| model_err("missing in_proj_ba"))?;
    let conv_w = layer
        .conv1d
        .as_ref()
        .ok_or_else(|| model_err("missing conv1d"))?;
    let a_log = layer.a_log.as_ref().ok_or_else(|| model_err("missing A_log"))?;
    let dt_bias = layer
        .dt_bias
        .as_ref()
        .ok_or_else(|| model_err("missing dt_bias"))?;
    let gated_norm = layer
        .gated_rms_norm
        .as_ref()
        .ok_or_else(|| model_err("missing gated_rms_norm"))?;
    let out_proj = layer
        .out_proj_linear
        .as_ref()
        .ok_or_else(|| model_err("missing linear out_proj"))?;

    let input_rms = source_qwen80_residual_rms_norm(hidden, &layer.input_layernorm)?;
    let qkvz_rows = layout.qkvz_projection_elements()?;
    let ba_rows = layout.ba_projection_elements()?;
    let mut projected_qkvz = vec![0.0f32; qkvz_rows];
    let mut projected_ba = vec![0.0f32; ba_rows];
    gemv_bf16(qkvz_w, qkvz_rows, QWEN80_HIDDEN, &input_rms, &mut projected_qkvz)?;
    gemv_bf16(ba_w, ba_rows, QWEN80_HIDDEN, &input_rms, &mut projected_ba)?;

    let (raw_query, raw_key, raw_value, z) =
        source_qwen80_split_linear_qkvz(&projected_qkvz, layout)?;
    let mut mixed_qkv = Vec::with_capacity(layout.conv_channels);
    mixed_qkv.extend_from_slice(&raw_query);
    mixed_qkv.extend_from_slice(&raw_key);
    mixed_qkv.extend_from_slice(&raw_value);
    let (convolved_qkv, next_conv) =
        source_qwen80_causal_conv_step_dense(&mixed_qkv, &state.conv_state, conv_w, layout)?;
    let raw_query_len = layout.key_elements()?;
    let raw_key_len = raw_query_len;
    let raw_value_len = layout.value_elements()?;
    let convolved_query = &convolved_qkv[..raw_query_len];
    let convolved_key = &convolved_qkv[raw_query_len..raw_query_len + raw_key_len];
    let convolved_value = &convolved_qkv[raw_query_len + raw_key_len..];
    if convolved_value.len() != raw_value_len {
        return Err(model_err("convolution value geometry broken"));
    }
    let convolved_value = convolved_value.to_vec();

    let mut repeated_query = vec![0.0f32; raw_value_len];
    let mut repeated_key = vec![0.0f32; raw_value_len];
    for value_head in 0..layout.value_heads {
        let key_head = value_head / layout.value_heads_per_key_head;
        let mut query_head = convolved_query
            [key_head * layout.key_head_dim..(key_head + 1) * layout.key_head_dim]
            .to_vec();
        let mut key_head_values = convolved_key
            [key_head * layout.key_head_dim..(key_head + 1) * layout.key_head_dim]
            .to_vec();
        source_qwen80_l2_normalize(
            &mut query_head,
            (layout.key_head_dim as f32).sqrt().recip(),
        )?;
        source_qwen80_l2_normalize(&mut key_head_values, 1.0)?;
        let destination = value_head * layout.key_head_dim;
        repeated_query[destination..destination + layout.key_head_dim]
            .copy_from_slice(&query_head);
        repeated_key[destination..destination + layout.key_head_dim]
            .copy_from_slice(&key_head_values);
    }
    let (decay, beta) =
        source_qwen80_ba_to_decay_beta(&projected_ba, a_log, dt_bias, layout)?;
    let recurrent_output = source_qwen80_recurrent_deltanet(
        &mut state.recurrent_state,
        &repeated_query,
        &repeated_key,
        &convolved_value,
        &decay,
        &beta,
        layout,
    )?;
    state.conv_state = next_conv;
    let repeated_gated_norm_weight = (0..layout.value_heads)
        .flat_map(|_| gated_norm.iter().copied())
        .collect::<Vec<_>>();
    let gated_output = source_qwen80_gated_rms_norm(
        &recurrent_output,
        &z,
        &repeated_gated_norm_weight,
        layout.value_heads,
        layout.value_head_dim,
    )?;
    let mut mixer_output = vec![0.0f32; QWEN80_HIDDEN];
    gemv_bf16(
        out_proj,
        QWEN80_HIDDEN,
        raw_value_len,
        &gated_output,
        &mut mixer_output,
    )?;
    let mut residual = hidden.to_vec();
    add_inplace(&mut residual, &mixer_output);
    if residual.iter().any(|v| !v.is_finite()) {
        return Err(model_err("DeltaNet residual non-finite"));
    }
    Ok(residual)
}

/// One GQA mixer step through first residual.
fn gqa_mixer_step(
    layer: &LoadedLayer,
    hidden: &[f32],
    state: &mut GqaState,
    position: usize,
    layout: &Qwen80CanonicalGqaLayout,
) -> Result<Vec<f32>> {
    if position >= state.max_seq {
        return Err(model_err(format!(
            "GQA position {position} exceeds max_seq {}",
            state.max_seq
        )));
    }
    let q_proj = layer.q_proj.as_ref().ok_or_else(|| model_err("missing q_proj"))?;
    let k_proj = layer.k_proj.as_ref().ok_or_else(|| model_err("missing k_proj"))?;
    let v_proj = layer.v_proj.as_ref().ok_or_else(|| model_err("missing v_proj"))?;
    let o_proj = layer.o_proj.as_ref().ok_or_else(|| model_err("missing o_proj"))?;
    let q_norm = layer.q_norm.as_ref().ok_or_else(|| model_err("missing q_norm"))?;
    let k_norm = layer.k_norm.as_ref().ok_or_else(|| model_err("missing k_norm"))?;

    let input_rms = source_qwen80_residual_rms_norm(hidden, &layer.input_layernorm)?;
    let mut q_projection = vec![0.0f32; layout.q_proj_rows];
    let mut k_projection = vec![0.0f32; layout.kv_dim];
    let mut v_projection = vec![0.0f32; layout.kv_dim];
    gemv_bf16(
        q_proj,
        layout.q_proj_rows,
        QWEN80_HIDDEN,
        &input_rms,
        &mut q_projection,
    )?;
    gemv_bf16(
        k_proj,
        layout.kv_dim,
        QWEN80_HIDDEN,
        &input_rms,
        &mut k_projection,
    )?;
    gemv_bf16(
        v_proj,
        layout.kv_dim,
        QWEN80_HIDDEN,
        &input_rms,
        &mut v_projection,
    )?;

    let query_raw = qwen80_gqa_query_from_interleaved_q_projection(&q_projection, layout)?;
    let query = qwen80_gqa_source_norm_rope(
        &query_raw,
        q_norm,
        layout.query_heads,
        layout.head_dim,
        layout.rotary_dim,
        position,
        "GQA q_norm + partial RoPE",
    )?;
    let key_row = qwen80_gqa_source_norm_rope(
        &k_projection,
        k_norm,
        layout.key_value_heads,
        layout.head_dim,
        layout.rotary_dim,
        position,
        "GQA k_norm + partial RoPE",
    )?;
    let cache_start = position
        .checked_mul(layout.kv_dim)
        .ok_or_else(|| model_err("GQA cache start overflow"))?;
    let cache_end = cache_start
        .checked_add(layout.kv_dim)
        .ok_or_else(|| model_err("GQA cache end overflow"))?;
    state.key_cache[cache_start..cache_end].copy_from_slice(&key_row);
    state.value_cache[cache_start..cache_end].copy_from_slice(&v_projection);
    let sequence_length = position + 1;
    let attention = qwen80_gqa_causal_attention(
        &query,
        &state.key_cache,
        &state.value_cache,
        sequence_length,
        layout,
    )?;
    let gated = qwen80_gqa_apply_sigmoid_gate(&attention, &q_projection, layout)?;
    let mut mixer_output = vec![0.0f32; QWEN80_HIDDEN];
    gemv_bf16(
        o_proj,
        QWEN80_HIDDEN,
        layout.query_dim,
        &gated,
        &mut mixer_output,
    )?;
    let mut residual = hidden.to_vec();
    add_inplace(&mut residual, &mixer_output);
    if residual.iter().any(|v| !v.is_finite()) {
        return Err(model_err("GQA residual non-finite"));
    }
    Ok(residual)
}

/// Run one loaded layer over one probe sequence (causal within the probe).
pub fn forward_layer_probe(
    layer: &LoadedLayer,
    hidden: &mut ProbeHidden,
    seq_len: usize,
) -> Result<Vec<LayerTokenCapture>> {
    if seq_len == 0 {
        return Ok(Vec::new());
    }
    if hidden.len() != seq_len * QWEN80_HIDDEN {
        return Err(model_err(format!(
            "probe hidden len {} != seq {seq_len} * {QWEN80_HIDDEN}",
            hidden.len()
        )));
    }

    let linear_layout = Qwen80CanonicalLinearDeltaNetLayout::source_exact();
    linear_layout.validate()?;
    let gqa_layout = Qwen80CanonicalGqaLayout::source_exact();
    gqa_layout.validate()?;

    let mut delta_state = DeltaNetState::zero(&linear_layout)?;
    let mut gqa_state = GqaState::new(seq_len, &gqa_layout);
    let mut captures = Vec::with_capacity(seq_len);

    let mut router_logits = vec![0.0f32; QWEN80_EXPERTS];
    let mut gate = vec![0.0f32; QWEN80_MOE_INTERMEDIATE];
    let mut up = vec![0.0f32; QWEN80_MOE_INTERMEDIATE];
    let mut act = vec![0.0f32; QWEN80_MOE_INTERMEDIATE];
    let mut down = vec![0.0f32; QWEN80_HIDDEN];
    let mut shared_gate = vec![0.0f32; QWEN80_SHARED_EXPERT_INTERMEDIATE];
    let mut shared_up = vec![0.0f32; QWEN80_SHARED_EXPERT_INTERMEDIATE];
    let mut shared_act = vec![0.0f32; QWEN80_SHARED_EXPERT_INTERMEDIATE];
    let mut shared_down = vec![0.0f32; QWEN80_HIDDEN];
    let mut shared_gate_logit = vec![0.0f32; 1];
    let mut moe_combined = vec![0.0f32; QWEN80_HIDDEN];

    for pos in 0..seq_len {
        let x_slice = &hidden[pos * QWEN80_HIDDEN..(pos + 1) * QWEN80_HIDDEN];
        let x_in = x_slice.to_vec();
        let first_residual = match layer.kind {
            LayerKind::LinearDeltaNet => {
                deltanet_mixer_step(layer, &x_in, &mut delta_state, &linear_layout)?
            }
            LayerKind::FullAttentionGqa => {
                gqa_mixer_step(layer, &x_in, &mut gqa_state, pos, &gqa_layout)?
            }
        };
        let router_input =
            source_qwen80_residual_rms_norm(&first_residual, &layer.post_attention_layernorm)?;
        let (ids, weights) = moe_combine(
            layer,
            &router_input,
            &mut moe_combined,
            &mut router_logits,
            &mut gate,
            &mut up,
            &mut act,
            &mut down,
            &mut shared_gate,
            &mut shared_up,
            &mut shared_act,
            &mut shared_down,
            &mut shared_gate_logit,
        )?;
        let mut out = first_residual;
        add_inplace(&mut out, &moe_combined);
        if out.iter().any(|v| !v.is_finite()) {
            return Err(model_err(format!(
                "layer {} pos {pos} second residual non-finite",
                layer.layer
            )));
        }
        hidden[pos * QWEN80_HIDDEN..(pos + 1) * QWEN80_HIDDEN].copy_from_slice(&out);
        captures.push(LayerTokenCapture {
            layer: layer.layer,
            selected_expert_ids: ids,
            normalized_route_weights: weights,
            router_input_hidden: router_input,
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
        let mut h = Vec::with_capacity(tokens.len() * QWEN80_HIDDEN);
        for &tok in tokens {
            let row = index.embed_row(tok)?;
            h.extend_from_slice(&row);
        }
        out.push(h);
    }
    Ok(out)
}

/// Final RMSNorm + lm_head logits for the last residual of a sequence.
pub fn logits_from_final_hidden(index: &SourceBf16Index, hidden: &[f32]) -> Result<Vec<f32>> {
    if hidden.len() != QWEN80_HIDDEN {
        return Err(model_err("final hidden width mismatch"));
    }
    let norm_w = index.read_f32("model.norm.weight")?;
    let normed = source_qwen80_residual_rms_norm(hidden, &norm_w)?;
    // lm_head ~593 MiB BF16 — load, matvec, free.
    let lm_head = index.read_raw("lm_head.weight")?;
    let mut logits = vec![0.0f32; QWEN80_VOCAB];
    gemv_bf16(&lm_head, QWEN80_VOCAB, QWEN80_HIDDEN, &normed, &mut logits)?;
    drop(lm_head);
    // Mask source-reserved tail (tokenizer vocab 151669 .. 151935).
    for logit in logits.iter_mut().skip(QWEN80_TOKENIZER_VOCAB) {
        *logit = f32::NEG_INFINITY;
    }
    Ok(logits)
}

/// Timing / bandwidth telemetry for one full layer-major pass.
#[derive(Clone, Debug, Default)]
pub struct StreamTelemetry {
    pub layers: usize,
    pub tokens: usize,
    pub weight_bytes_read: u64,
    pub load_secs: f64,
    pub compute_secs: f64,
    pub wall_secs: f64,
    pub max_layer_resident_bytes: u64,
    pub peak_rss_bytes: u64,
}

impl StreamTelemetry {
    pub fn stream_gib_per_s(&self) -> f64 {
        if self.load_secs <= 0.0 {
            return 0.0;
        }
        (self.weight_bytes_read as f64) / self.load_secs / (1024.0 * 1024.0 * 1024.0)
    }

    /// Tokens for which compute wall would equal load wall at this measured rate.
    /// Corpus sizes up to this are "free" (I/O bound).
    pub fn free_corpus_crossover_tokens(&self) -> f64 {
        if self.compute_secs <= 0.0 || self.tokens == 0 {
            return f64::INFINITY;
        }
        // At fixed stream: load_secs is independent of tokens; compute scales linearly.
        // Crossover when compute_secs(t) = load_secs => t = load_secs / (compute_secs/tokens)
        let compute_per_token = self.compute_secs / self.tokens as f64;
        if compute_per_token <= 0.0 {
            return f64::INFINITY;
        }
        self.load_secs / compute_per_token
    }
}

/// Layer-major full forward over all probes.
pub fn capture_all_layers(
    index: &SourceBf16Index,
    probes: &[(String, Vec<u32>)],
    hiddens: &mut [ProbeHidden],
    mut on_layer: Option<&mut dyn FnMut(usize, &LoadedLayer, &StreamTelemetry)>,
) -> Result<(Vec<Vec<Vec<LayerTokenCapture>>>, StreamTelemetry)> {
    if hiddens.len() != probes.len() {
        return Err(model_err("hiddens/probes length mismatch"));
    }
    let total_tokens: usize = probes.iter().map(|(_, t)| t.len()).sum();
    let mut captures: Vec<Vec<Vec<LayerTokenCapture>>> = probes
        .iter()
        .map(|(_, toks)| {
            (0..toks.len())
                .map(|_| Vec::with_capacity(QWEN80_LAYERS))
                .collect()
        })
        .collect();

    let wall0 = Instant::now();
    let bytes0 = index.bytes_read_total();
    let mut telem = StreamTelemetry {
        layers: QWEN80_LAYERS,
        tokens: total_tokens,
        ..Default::default()
    };

    for layer_idx in 0..QWEN80_LAYERS {
        let load_t0 = Instant::now();
        let layer = LoadedLayer::load(index, layer_idx)?;
        telem.load_secs += load_t0.elapsed().as_secs_f64();
        telem.max_layer_resident_bytes = telem.max_layer_resident_bytes.max(layer.resident_bytes);
        if let Some(cb) = on_layer.as_mut() {
            cb(layer_idx, &layer, &telem);
        }
        let comp_t0 = Instant::now();
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
        telem.compute_secs += comp_t0.elapsed().as_secs_f64();
        drop(layer);
        telem.peak_rss_bytes = telem.peak_rss_bytes.max(peak_rss_bytes());
        if telem.peak_rss_bytes > STREAMED_PEAK_RSS_HARD_CAP_BYTES {
            return Err(model_err(format!(
                "peak RSS {} exceeds streamed hard cap {}; refuse (looks like resident load)",
                telem.peak_rss_bytes, STREAMED_PEAK_RSS_HARD_CAP_BYTES
            )));
        }
    }
    telem.weight_bytes_read = index.bytes_read_total().saturating_sub(bytes0);
    telem.wall_secs = wall0.elapsed().as_secs_f64();
    telem.peak_rss_bytes = telem.peak_rss_bytes.max(peak_rss_bytes());
    Ok((captures, telem))
}

/// Stateful single-token step through all 48 layers (one full weight stream).
/// Used by the autoregressive generation loop after prompt prefill.
fn decode_one_token_stream(
    index: &SourceBf16Index,
    mut hidden: Vec<f32>,
    position: usize,
    max_seq: usize,
    delta_states: &mut [Option<DeltaNetState>],
    gqa_states: &mut [Option<GqaState>],
) -> Result<Vec<f32>> {
    if hidden.len() != QWEN80_HIDDEN {
        return Err(model_err("decode hidden width mismatch"));
    }
    let linear_layout = Qwen80CanonicalLinearDeltaNetLayout::source_exact();
    linear_layout.validate()?;
    let gqa_layout = Qwen80CanonicalGqaLayout::source_exact();
    gqa_layout.validate()?;

    let mut router_logits = vec![0.0f32; QWEN80_EXPERTS];
    let mut gate = vec![0.0f32; QWEN80_MOE_INTERMEDIATE];
    let mut up = vec![0.0f32; QWEN80_MOE_INTERMEDIATE];
    let mut act = vec![0.0f32; QWEN80_MOE_INTERMEDIATE];
    let mut down = vec![0.0f32; QWEN80_HIDDEN];
    let mut shared_gate = vec![0.0f32; QWEN80_SHARED_EXPERT_INTERMEDIATE];
    let mut shared_up = vec![0.0f32; QWEN80_SHARED_EXPERT_INTERMEDIATE];
    let mut shared_act = vec![0.0f32; QWEN80_SHARED_EXPERT_INTERMEDIATE];
    let mut shared_down = vec![0.0f32; QWEN80_HIDDEN];
    let mut shared_gate_logit = vec![0.0f32; 1];
    let mut moe_combined = vec![0.0f32; QWEN80_HIDDEN];

    for layer_idx in 0..QWEN80_LAYERS {
        let layer = LoadedLayer::load(index, layer_idx)?;
        let first_residual = match layer.kind {
            LayerKind::LinearDeltaNet => {
                let state = delta_states[layer_idx]
                    .get_or_insert_with(|| DeltaNetState::zero(&linear_layout).expect("layout ok"));
                deltanet_mixer_step(&layer, &hidden, state, &linear_layout)?
            }
            LayerKind::FullAttentionGqa => {
                let state = gqa_states[layer_idx]
                    .get_or_insert_with(|| GqaState::new(max_seq, &gqa_layout));
                gqa_mixer_step(&layer, &hidden, state, position, &gqa_layout)?
            }
        };
        let router_input =
            source_qwen80_residual_rms_norm(&first_residual, &layer.post_attention_layernorm)?;
        let _ = moe_combine(
            &layer,
            &router_input,
            &mut moe_combined,
            &mut router_logits,
            &mut gate,
            &mut up,
            &mut act,
            &mut down,
            &mut shared_gate,
            &mut shared_up,
            &mut shared_act,
            &mut shared_down,
            &mut shared_gate_logit,
        )?;
        hidden = first_residual;
        add_inplace(&mut hidden, &moe_combined);
        drop(layer);
        if peak_rss_bytes() > STREAMED_PEAK_RSS_HARD_CAP_BYTES {
            return Err(model_err(format!(
                "peak RSS {} exceeds streamed hard cap during decode",
                peak_rss_bytes()
            )));
        }
    }
    Ok(hidden)
}

/// Prefill a prompt by processing every position layer-major (one weight stream).
fn prefill_prompt_stream(
    index: &SourceBf16Index,
    token_ids: &[u32],
    max_seq: usize,
    delta_states: &mut [Option<DeltaNetState>],
    gqa_states: &mut [Option<GqaState>],
) -> Result<Vec<f32>> {
    let probes = vec![("prefill".to_string(), token_ids.to_vec())];
    let mut hiddens = embed_probes(index, &probes)?;
    let linear_layout = Qwen80CanonicalLinearDeltaNetLayout::source_exact();
    linear_layout.validate()?;
    let gqa_layout = Qwen80CanonicalGqaLayout::source_exact();
    gqa_layout.validate()?;

    for layer_idx in 0..QWEN80_LAYERS {
        let layer = LoadedLayer::load(index, layer_idx)?;
        // Process the single probe sequence, capturing states for decode.
        let seq_len = token_ids.len();
        let hidden = &mut hiddens[0];
        match layer.kind {
            LayerKind::LinearDeltaNet => {
                let state = delta_states[layer_idx]
                    .get_or_insert_with(|| DeltaNetState::zero(&linear_layout).expect("layout ok"));
                // Re-run with state we keep (forward_layer_probe uses local state).
                let mut router_logits = vec![0.0f32; QWEN80_EXPERTS];
                let mut gate = vec![0.0f32; QWEN80_MOE_INTERMEDIATE];
                let mut up = vec![0.0f32; QWEN80_MOE_INTERMEDIATE];
                let mut act = vec![0.0f32; QWEN80_MOE_INTERMEDIATE];
                let mut down = vec![0.0f32; QWEN80_HIDDEN];
                let mut shared_gate = vec![0.0f32; QWEN80_SHARED_EXPERT_INTERMEDIATE];
                let mut shared_up = vec![0.0f32; QWEN80_SHARED_EXPERT_INTERMEDIATE];
                let mut shared_act = vec![0.0f32; QWEN80_SHARED_EXPERT_INTERMEDIATE];
                let mut shared_down = vec![0.0f32; QWEN80_HIDDEN];
                let mut shared_gate_logit = vec![0.0f32; 1];
                let mut moe_combined = vec![0.0f32; QWEN80_HIDDEN];
                for pos in 0..seq_len {
                    let x_in = hidden[pos * QWEN80_HIDDEN..(pos + 1) * QWEN80_HIDDEN].to_vec();
                    let first = deltanet_mixer_step(&layer, &x_in, state, &linear_layout)?;
                    let rin =
                        source_qwen80_residual_rms_norm(&first, &layer.post_attention_layernorm)?;
                    let _ = moe_combine(
                        &layer,
                        &rin,
                        &mut moe_combined,
                        &mut router_logits,
                        &mut gate,
                        &mut up,
                        &mut act,
                        &mut down,
                        &mut shared_gate,
                        &mut shared_up,
                        &mut shared_act,
                        &mut shared_down,
                        &mut shared_gate_logit,
                    )?;
                    let mut out = first;
                    add_inplace(&mut out, &moe_combined);
                    hidden[pos * QWEN80_HIDDEN..(pos + 1) * QWEN80_HIDDEN].copy_from_slice(&out);
                }
            }
            LayerKind::FullAttentionGqa => {
                let state = gqa_states[layer_idx]
                    .get_or_insert_with(|| GqaState::new(max_seq, &gqa_layout));
                let mut router_logits = vec![0.0f32; QWEN80_EXPERTS];
                let mut gate = vec![0.0f32; QWEN80_MOE_INTERMEDIATE];
                let mut up = vec![0.0f32; QWEN80_MOE_INTERMEDIATE];
                let mut act = vec![0.0f32; QWEN80_MOE_INTERMEDIATE];
                let mut down = vec![0.0f32; QWEN80_HIDDEN];
                let mut shared_gate = vec![0.0f32; QWEN80_SHARED_EXPERT_INTERMEDIATE];
                let mut shared_up = vec![0.0f32; QWEN80_SHARED_EXPERT_INTERMEDIATE];
                let mut shared_act = vec![0.0f32; QWEN80_SHARED_EXPERT_INTERMEDIATE];
                let mut shared_down = vec![0.0f32; QWEN80_HIDDEN];
                let mut shared_gate_logit = vec![0.0f32; 1];
                let mut moe_combined = vec![0.0f32; QWEN80_HIDDEN];
                for pos in 0..seq_len {
                    let x_in = hidden[pos * QWEN80_HIDDEN..(pos + 1) * QWEN80_HIDDEN].to_vec();
                    let first = gqa_mixer_step(&layer, &x_in, state, pos, &gqa_layout)?;
                    let rin =
                        source_qwen80_residual_rms_norm(&first, &layer.post_attention_layernorm)?;
                    let _ = moe_combine(
                        &layer,
                        &rin,
                        &mut moe_combined,
                        &mut router_logits,
                        &mut gate,
                        &mut up,
                        &mut act,
                        &mut down,
                        &mut shared_gate,
                        &mut shared_up,
                        &mut shared_act,
                        &mut shared_down,
                        &mut shared_gate_logit,
                    )?;
                    let mut out = first;
                    add_inplace(&mut out, &moe_combined);
                    hidden[pos * QWEN80_HIDDEN..(pos + 1) * QWEN80_HIDDEN].copy_from_slice(&out);
                }
            }
        }
        drop(layer);
        if peak_rss_bytes() > STREAMED_PEAK_RSS_HARD_CAP_BYTES {
            return Err(model_err(format!(
                "peak RSS {} exceeds streamed hard cap during prefill",
                peak_rss_bytes()
            )));
        }
    }
    let h = &hiddens[0];
    let n = h.len() / QWEN80_HIDDEN;
    Ok(h[(n - 1) * QWEN80_HIDDEN..n * QWEN80_HIDDEN].to_vec())
}

/// Greedy decode with the source one-user chat template.
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
    // Source chat template for a single user turn without tools/system:
    // <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n
    let rendered =
        format!("<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n");
    let encoding = tokenizer
        .encode(rendered.as_str(), false)
        .map_err(|e| model_err(format!("tokenizer encode failed: {e}")))?;
    let mut token_ids: Vec<u32> = encoding.get_ids().to_vec();
    if token_ids.is_empty() {
        return Err(model_err("chat-template encoding produced no tokens"));
    }
    let prompt_len = token_ids.len();
    let max_seq = prompt_len + max_new_tokens + 8;
    let mut delta_states: Vec<Option<DeltaNetState>> = (0..QWEN80_LAYERS).map(|_| None).collect();
    let mut gqa_states: Vec<Option<GqaState>> = (0..QWEN80_LAYERS).map(|_| None).collect();

    let mut last_hidden =
        prefill_prompt_stream(index, &token_ids, max_seq, &mut delta_states, &mut gqa_states)?;
    let mut generated = Vec::new();
    let eos = [151645u32, 151643u32];

    for step in 0..max_new_tokens {
        let logits = logits_from_final_hidden(index, &last_hidden)?;
        let next = argmax_f32(&logits);
        generated.push(next);
        if eos.contains(&next) {
            break;
        }
        token_ids.push(next);
        let position = prompt_len + step;
        let emb = index.embed_row(next)?;
        last_hidden = decode_one_token_stream(
            index,
            emb,
            position,
            max_seq,
            &mut delta_states,
            &mut gqa_states,
        )?;
    }

    let cont_text = tokenizer
        .decode(&generated, true)
        .map_err(|e| model_err(format!("tokenizer decode failed: {e}")))?;
    Ok(GreedyDecodeResult {
        prompt_token_count: prompt_len,
        generated_token_ids: generated,
        continuation_text: cont_text,
        rendered_prompt: rendered,
        peak_rss_bytes: peak_rss_bytes(),
        weight_bytes_read: index.bytes_read_total(),
    })
}

#[derive(Clone, Debug)]
pub struct GreedyDecodeResult {
    pub prompt_token_count: usize,
    pub generated_token_ids: Vec<u32>,
    pub continuation_text: String,
    pub rendered_prompt: String,
    pub peak_rss_bytes: u64,
    pub weight_bytes_read: u64,
}

/// True when the top-1 continuation is a coherent capital-of-France answer.
pub fn is_coherent_paris_continuation(text: &str) -> bool {
    let t = text.trim_start();
    if t.contains("Wien") || t.to_ascii_lowercase().contains("swiper") {
        return false;
    }
    // Degenerate single-token loops / pure punctuation
    if t.len() > 8 {
        let first = t.chars().next().unwrap_or('\0');
        if t.chars().filter(|c| *c == first).count() > t.len() * 3 / 4 {
            return false;
        }
    }
    let lower = t.to_ascii_lowercase();
    lower.starts_with("paris")
        || lower.starts_with(" paris")
        || lower.starts_with("the capital of france is paris")
        || lower.starts_with("**paris")
        || lower.contains("paris is the capital")
}

/// Process peak RSS in bytes (macOS: `ru_maxrss` is already bytes).
pub fn peak_rss_bytes() -> u64 {
    #[cfg(unix)]
    {
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
    fn layer_kinds_match_source_interval() {
        assert_eq!(layer_kind(0).unwrap(), LayerKind::LinearDeltaNet);
        assert_eq!(layer_kind(2).unwrap(), LayerKind::LinearDeltaNet);
        assert_eq!(layer_kind(3).unwrap(), LayerKind::FullAttentionGqa);
        assert_eq!(layer_kind(7).unwrap(), LayerKind::FullAttentionGqa);
        assert_eq!(layer_kind(47).unwrap(), LayerKind::FullAttentionGqa);
        let gqa: Vec<_> = (0..48)
            .filter(|&l| layer_kind(l).unwrap() == LayerKind::FullAttentionGqa)
            .collect();
        assert_eq!(
            gqa,
            vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]
        );
    }

    #[test]
    fn paris_coherence_accepts_variants() {
        assert!(is_coherent_paris_continuation(" Paris"));
        assert!(is_coherent_paris_continuation("Paris is the capital"));
        assert!(!is_coherent_paris_continuation(" Wien swiper swiper"));
        assert!(!is_coherent_paris_continuation("swiper Wien"));
    }

    #[test]
    fn layouts_validate() {
        Qwen80CanonicalLinearDeltaNetLayout::source_exact()
            .validate()
            .unwrap();
        Qwen80CanonicalGqaLayout::source_exact().validate().unwrap();
    }
}
