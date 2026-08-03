//! Llama-family forward pass executed directly out of a `.gravity` shard.
//!
//! This is the production counterpart of the numpy oracle in
//! `tools/condense/gravity_llama_reference.py`, and it is graded against
//! that oracle's frozen logits (`tests/fixtures/gravity_llama/`) rather
//! than against the BF16 parent. A sub-bit artifact is lossy by
//! construction; "correct" means the runtime computes what the *artifact*
//! encodes, so the oracle reads the same container through the same codec
//! and the two must agree.
//!
//! No dense reconstruction anywhere: every projection is a matvec over the
//! packed `gravity-pq` payload, and the embedding row is decoded from its
//! own chunk codes rather than by materializing the `[vocab, hidden]`
//! matrix. No source weights are consulted.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::attn::mha_decode_step;
use crate::gguf::GgmlType;
use crate::gravity::{widen_native, GravityShard, GravityWeights};
use crate::kernels::{add_inplace, rmsnorm, rope_inplace_scaled, silu_mul, Llama3RopeScaling};
use crate::numeric_parity::{
    matvec_dense_f64_authority, matvec_ggml_quant_f64_authority, rmsnorm_f64,
    row_ggml_quant_f64_authority, silu_mul_f64_authority,
};
use crate::{Error, Result};

/// The architecture fields the forward pass needs, read from the shard
/// header's `architecture` object. Absent or malformed fields are an
/// error: a runtime that guesses `rope_theta` produces plausible garbage.
#[derive(Debug, Clone)]
pub struct GravityLlamaArch {
    /// Source family carried by the artifact.  Llama, Mistral and Qwen2 use
    /// the same dense projection tensor ABI here; Mistral's sliding-window
    /// attention contract remains explicit in the runtime.
    pub model_type: String,
    pub n_layers: usize,
    pub hidden: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub rope_scaling: Option<Llama3RopeScaling>,
    /// Explicit source layout.  Historical Gravity PQ artifacts used the
    /// split-half (NeoX) convention; source-preserving Llama/Mistral shards
    /// carry `"interleaved"` because their GGUF `rope_type=0` pairs adjacent
    /// coordinates.
    pub rope_interleaved: bool,
    /// Optional resolved per-frequency divisors from GGUF
    /// `rope_freqs.weight`.  These take precedence over metadata scaling.
    pub rope_freq_factors: Option<Vec<f32>>,
    pub sliding_window: Option<usize>,
}

fn arch_u64(v: &serde_json::Value, key: &str) -> Result<u64> {
    v.get(key)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| Error::Gravity(format!("architecture.{key} missing or not an integer")))
}

fn arch_f64(v: &serde_json::Value, key: &str) -> Result<f64> {
    v.get(key)
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| Error::Gravity(format!("architecture.{key} missing or not a number")))
}

impl GravityLlamaArch {
    pub fn from_header(extra: &serde_json::Value) -> Result<GravityLlamaArch> {
        let a = extra
            .get("architecture")
            .ok_or_else(|| Error::Gravity("shard header has no `architecture`".into()))?;

        let model_type = a.get("model_type").and_then(serde_json::Value::as_str);
        if !matches!(model_type, Some("llama") | Some("mistral") | Some("qwen2")) {
            return Err(Error::Gravity(format!(
                "gravity_llama: architecture.model_type is {model_type:?}, expected \"llama\", \"mistral\" or \"qwen2\""
            )));
        }

        let sliding_window = a
            .get("sliding_window")
            .and_then(serde_json::Value::as_u64)
            .map(|value| value as usize)
            .filter(|value| *value > 0);

        let rope_interleaved = matches!(
            a.get("rope_layout").and_then(serde_json::Value::as_str),
            Some("interleaved")
        );

        let hidden = arch_u64(a, "hidden_size")? as usize;
        let n_heads = arch_u64(a, "num_attention_heads")? as usize;
        // `head_dim` is explicit in recent configs; older ones imply it.
        let head_dim = match a.get("head_dim").and_then(serde_json::Value::as_u64) {
            Some(hd) => hd as usize,
            None => {
                if n_heads == 0 || hidden % n_heads != 0 {
                    return Err(Error::Gravity(format!(
                        "cannot infer head_dim from hidden_size {hidden} / num_attention_heads {n_heads}"
                    )));
                }
                hidden / n_heads
            }
        };

        let rope_freq_factors = match a.get("rope_freq_factors") {
            None | Some(serde_json::Value::Null) => None,
            Some(value) => {
                let values = value.as_array().ok_or_else(|| {
                    Error::Gravity("architecture.rope_freq_factors must be an array".into())
                })?;
                let factors = values
                    .iter()
                    .map(|item| {
                        item.as_f64().map(|v| v as f32).ok_or_else(|| {
                            Error::Gravity(
                                "architecture.rope_freq_factors contains a non-number".into(),
                            )
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                if factors.len() != head_dim / 2
                    || factors
                        .iter()
                        .any(|factor| !factor.is_finite() || *factor <= 0.0)
                {
                    return Err(Error::Gravity(format!(
                        "architecture.rope_freq_factors must contain {} finite positive values, found {}",
                        head_dim / 2,
                        factors.len()
                    )));
                }
                Some(factors)
            }
        };

        // Only `rope_type == "llama3"` is a scaling this decoder implements.
        // Any other declared rope_type is refused rather than silently
        // executed as unscaled RoPE, which would be a wrong model that runs.
        let rope_scaling = match a.get("rope_scaling") {
            None | Some(serde_json::Value::Null) => None,
            Some(rs) => {
                let ty = rs.get("rope_type").and_then(serde_json::Value::as_str);
                if ty != Some("llama3") {
                    return Err(Error::Gravity(format!(
                        "unsupported architecture.rope_scaling.rope_type {ty:?}"
                    )));
                }
                Some(Llama3RopeScaling {
                    factor: arch_f64(rs, "factor")? as f32,
                    low_freq_factor: arch_f64(rs, "low_freq_factor")? as f32,
                    high_freq_factor: arch_f64(rs, "high_freq_factor")? as f32,
                    original_max_position_embeddings: arch_u64(
                        rs,
                        "original_max_position_embeddings",
                    )? as u32,
                })
            }
        };

        Ok(GravityLlamaArch {
            model_type: model_type.expect("model_type matched above").to_string(),
            n_layers: arch_u64(a, "num_hidden_layers")? as usize,
            hidden,
            n_heads,
            n_kv_heads: arch_u64(a, "num_key_value_heads")? as usize,
            head_dim,
            vocab_size: arch_u64(a, "vocab_size")? as usize,
            rope_theta: arch_f64(a, "rope_theta")? as f32,
            rms_norm_eps: arch_f64(a, "rms_norm_eps")? as f32,
            rope_scaling,
            rope_interleaved,
            rope_freq_factors,
            sliding_window,
        })
    }
}

#[inline]
fn attention_window(seq_len: usize, sliding_window: Option<usize>) -> (usize, usize) {
    let start = sliding_window
        .map(|window| seq_len.saturating_sub(window))
        .unwrap_or(0);
    (start, seq_len - start)
}

/// One immutable artifact tensor assigned to a future layer-window transfer.
/// This is an admission plan, not a loaded GPU buffer and not a performance
/// receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GravityWindowTensor {
    pub name: String,
    /// Absolute byte offset in the immutable `.gravity` file.  The window
    /// reader uses this range directly rather than reconstructing or reading
    /// the artifact body as a whole.
    pub file_offset: u64,
    pub payload_bytes: usize,
    pub sha256: String,
}

/// The exact tensor set consumed while evaluating one transformer layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GravityLayerWindow {
    pub layer: usize,
    pub tensors: Vec<GravityWindowTensor>,
    pub payload_bytes: usize,
}

/// A conservative dependency plan for an executable, future layer-windowed
/// decoder. It proves descriptor coverage only: a caller must still prove
/// hash verification, Metal transfer, completion, parity, and whole-token
/// timing before it can claim resident execution or a TG rung.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GravityLayerWindowPlan {
    pub global_tensors: Vec<GravityWindowTensor>,
    pub global_payload_bytes: usize,
    pub layers: Vec<GravityLayerWindow>,
    pub artifact_payload_bytes: usize,
    pub unassigned_tensors: Vec<String>,
    pub maximum_dependency_complete_window_bytes: usize,
    pub planning_only: bool,
}

/// One tensor body fetched through its exact immutable file range and checked
/// against the descriptor's SHA-256.  It is deliberately still codec-neutral:
/// a later resident executor owns decoding and GPU transfer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedGravityWindowTensor {
    pub descriptor: GravityWindowTensor,
    pub payload: Vec<u8>,
}

/// A dependency-complete payload set for one decoder layer.  Global tensors
/// are included because a single token cannot safely run from a layer alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedGravityLayerWindow {
    pub layer: usize,
    pub global_tensors: Vec<VerifiedGravityWindowTensor>,
    pub layer_tensors: Vec<VerifiedGravityWindowTensor>,
    pub bytes_read: usize,
}

fn layer_tensor_names(layer: usize) -> Vec<String> {
    let p = format!("model.layers.{layer}.");
    [
        "input_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]
    .into_iter()
    .map(|suffix| format!("{p}{suffix}"))
    .collect()
}

/// Build a conservative full-token dependency plan directly from immutable
/// Gravity descriptors. This is the precondition a streamed/layer-windowed
/// runtime must pass before fetching any model body range: every tensor the
/// current decoder reads is assigned exactly once to a global or layer group.
pub fn plan_layer_windows(path: &Path) -> Result<GravityLayerWindowPlan> {
    let shard = GravityShard::open(path)?;
    let arch = GravityLlamaArch::from_header(&shard.extra)?;
    let mut assigned = std::collections::HashSet::<String>::new();

    let mut take = |name: String| -> Result<GravityWindowTensor> {
        let descriptor = shard.descriptor(&name).ok_or_else(|| {
            Error::Gravity(format!(
                "layer-window plan missing required tensor {name:?}"
            ))
        })?;
        if !assigned.insert(name.clone()) {
            return Err(Error::Gravity(format!(
                "layer-window plan assigns tensor {name:?} more than once"
            )));
        }
        Ok(GravityWindowTensor {
            file_offset: shard.payload_file_offset(&name)?,
            name,
            payload_bytes: usize::try_from(descriptor.bytes)
                .map_err(|_| Error::Gravity("tensor payload bytes overflow usize".into()))?,
            sha256: descriptor.sha256.clone(),
        })
    };

    let mut global_tensors = vec![
        take("model.embed_tokens.weight".into())?,
        take("model.norm.weight".into())?,
    ];
    if shard.descriptor("lm_head.weight").is_some() {
        global_tensors.push(take("lm_head.weight".into())?);
    }
    let global_payload_bytes = global_tensors
        .iter()
        .try_fold(0usize, |total, tensor| {
            total.checked_add(tensor.payload_bytes)
        })
        .ok_or_else(|| Error::Gravity("global layer-window bytes overflow".into()))?;

    let mut layers = Vec::with_capacity(arch.n_layers);
    for layer in 0..arch.n_layers {
        let mut tensors = Vec::new();
        for name in layer_tensor_names(layer) {
            tensors.push(take(name)?);
        }
        // Qwen2 may carry attention biases; Llama/Mistral do not. Optional
        // means descriptor-present, never an inferred zero vector.
        for suffix in [
            "self_attn.q_proj.bias",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.bias",
        ] {
            let name = format!("model.layers.{layer}.{suffix}");
            if shard.descriptor(&name).is_some() {
                tensors.push(take(name)?);
            }
        }
        let payload_bytes = tensors
            .iter()
            .try_fold(0usize, |total, tensor| {
                total.checked_add(tensor.payload_bytes)
            })
            .ok_or_else(|| Error::Gravity("layer-window bytes overflow".into()))?;
        layers.push(GravityLayerWindow {
            layer,
            tensors,
            payload_bytes,
        });
    }

    let mut unassigned_tensors: Vec<String> = shard
        .tensor_names()
        .filter(|name| !assigned.contains(*name))
        .map(str::to_string)
        .collect();
    unassigned_tensors.sort();
    let artifact_payload_bytes = shard.tensor_names().try_fold(0usize, |total, name| {
        let bytes = usize::try_from(
            shard
                .descriptor(name)
                .expect("tensor name came from shard")
                .bytes,
        )
        .map_err(|_| Error::Gravity("tensor payload bytes overflow usize".into()))?;
        total
            .checked_add(bytes)
            .ok_or_else(|| Error::Gravity("artifact payload bytes overflow".into()))
    })?;
    let maximum_dependency_complete_window_bytes = layers
        .iter()
        .map(|layer| {
            global_payload_bytes
                .checked_add(layer.payload_bytes)
                .ok_or_else(|| Error::Gravity("dependency window bytes overflow".into()))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .max()
        .unwrap_or(global_payload_bytes);

    Ok(GravityLayerWindowPlan {
        global_tensors,
        global_payload_bytes,
        layers,
        artifact_payload_bytes,
        unassigned_tensors,
        maximum_dependency_complete_window_bytes,
        planning_only: true,
    })
}

fn read_verified_window_tensor(
    file: &mut File,
    tensor: &GravityWindowTensor,
) -> Result<VerifiedGravityWindowTensor> {
    file.seek(SeekFrom::Start(tensor.file_offset))?;
    let mut payload = vec![0_u8; tensor.payload_bytes];
    file.read_exact(&mut payload)?;
    let actual = format!("{:x}", Sha256::digest(&payload));
    if actual != tensor.sha256 {
        return Err(Error::Gravity(format!(
            "layer-window range hash mismatch for {:?}: expected {}, got {actual}",
            tensor.name, tensor.sha256
        )));
    }
    Ok(VerifiedGravityWindowTensor {
        descriptor: tensor.clone(),
        payload,
    })
}

/// Read precisely the global dependency set plus one layer's tensors, verify
/// every payload hash, and return only those bytes.  This is intentionally a
/// bounded *range primitive*, not a decoder: it provides the verifiable input
/// boundary that a resident, direct-execution runtime will consume.  In
/// particular it does not call `GravityWeights::open`, which eagerly decodes
/// every artifact tensor.
pub fn read_verified_layer_window(
    path: &Path,
    plan: &GravityLayerWindowPlan,
    layer: usize,
) -> Result<VerifiedGravityLayerWindow> {
    let window = plan.layers.get(layer).ok_or_else(|| {
        Error::Gravity(format!(
            "layer-window {layer} unavailable; plan contains {} layers",
            plan.layers.len()
        ))
    })?;
    if window.layer != layer {
        return Err(Error::Gravity(format!(
            "layer-window plan index {layer} names layer {}",
            window.layer
        )));
    }
    let expected_bytes = plan
        .global_payload_bytes
        .checked_add(window.payload_bytes)
        .ok_or_else(|| Error::Gravity("layer-window read bytes overflow".into()))?;
    if expected_bytes > plan.maximum_dependency_complete_window_bytes {
        return Err(Error::Gravity(format!(
            "layer-window {layer} exceeds declared maximum dependency window"
        )));
    }

    let mut file = File::open(path)?;
    let mut global_tensors = Vec::with_capacity(plan.global_tensors.len());
    for tensor in &plan.global_tensors {
        global_tensors.push(read_verified_window_tensor(&mut file, tensor)?);
    }
    let mut layer_tensors = Vec::with_capacity(window.tensors.len());
    for tensor in &window.tensors {
        layer_tensors.push(read_verified_window_tensor(&mut file, tensor)?);
    }
    let bytes_read = global_tensors
        .iter()
        .chain(&layer_tensors)
        .try_fold(0usize, |total, tensor| {
            total.checked_add(tensor.payload.len())
        })
        .ok_or_else(|| Error::Gravity("layer-window read bytes overflow".into()))?;
    if bytes_read != expected_bytes {
        return Err(Error::Gravity(format!(
            "layer-window {layer} read {bytes_read} bytes, expected {expected_bytes}"
        )));
    }
    Ok(VerifiedGravityLayerWindow {
        layer,
        global_tensors,
        layer_tensors,
        bytes_read,
    })
}

/// A `.gravity` shard loaded as an executable Llama model.
pub struct GravityLlama {
    pub arch: GravityLlamaArch,
    weights: GravityWeights,
    /// `lm_head.weight` when the artifact carries one, otherwise the tied
    /// `model.embed_tokens.weight`.
    head_name: String,
    pub tied_head: bool,
}

fn raw_ggml_kind(codec: &str) -> Result<Option<GgmlType>> {
    Ok(match codec {
        "ggml.q4_k" => Some(GgmlType::Q4_K),
        "ggml.q5_k" => Some(GgmlType::Q5_K),
        "ggml.q5_0" => Some(GgmlType::Q5_0),
        "ggml.q6_k" => Some(GgmlType::Q6_K),
        "ggml.q8_0" => Some(GgmlType::Q8_0),
        codec if codec.starts_with("native.") => None,
        _ => {
            return Err(Error::Gravity(format!(
                "f64 authority does not admit tensor codec {codec:?}"
            )))
        }
    })
}

fn rope_inplace_f64(x: &mut [f64], pos: usize, a: &GravityLlamaArch) {
    let half = x.len() / 2;
    for i in 0..half {
        let base_freq = 1.0 / (a.rope_theta as f64).powf(2.0 * i as f64 / x.len() as f64);
        let freq = if let Some(factors) = a.rope_freq_factors.as_deref() {
            base_freq / factors[i] as f64
        } else if let Some(scale) = a.rope_scaling {
            let wave_len = std::f64::consts::TAU / base_freq;
            let low = scale.original_max_position_embeddings as f64 / scale.low_freq_factor as f64;
            let high =
                scale.original_max_position_embeddings as f64 / scale.high_freq_factor as f64;
            if wave_len < high {
                base_freq
            } else if wave_len > low {
                base_freq / scale.factor as f64
            } else {
                let smooth = (scale.original_max_position_embeddings as f64 / wave_len
                    - scale.low_freq_factor as f64)
                    / (scale.high_freq_factor as f64 - scale.low_freq_factor as f64);
                (1.0 - smooth) * (base_freq / scale.factor as f64) + smooth * base_freq
            }
        } else {
            base_freq
        };
        let (sin, cos) = (pos as f64 * freq).sin_cos();
        if a.rope_interleaved {
            let left = 2 * i;
            let x0 = x[left];
            let x1 = x[left + 1];
            x[left] = x0 * cos - x1 * sin;
            x[left + 1] = x0 * sin + x1 * cos;
        } else {
            let x0 = x[i];
            let x1 = x[i + half];
            x[i] = x0 * cos - x1 * sin;
            x[i + half] = x0 * sin + x1 * cos;
        }
    }
}

fn mha_decode_step_f64(
    q: &[f64],
    keys: &[f64],
    values: &[f64],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Result<Vec<f64>> {
    if n_heads == 0
        || n_kv_heads == 0
        || n_heads % n_kv_heads != 0
        || q.len() != n_heads * head_dim
        || keys.len() != seq_len * n_kv_heads * head_dim
        || values.len() != keys.len()
    {
        return Err(Error::Gravity("f64 attention geometry is invalid".into()));
    }
    let group = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f64).sqrt();
    let width = n_kv_heads * head_dim;
    let mut out = vec![0.0; q.len()];
    for head in 0..n_heads {
        let qh = &q[head * head_dim..(head + 1) * head_dim];
        let kv_head = head / group;
        let mut scores = Vec::with_capacity(seq_len);
        let mut best = f64::NEG_INFINITY;
        for position in 0..seq_len {
            let key_start = position * width + kv_head * head_dim;
            let dot = qh
                .iter()
                .zip(&keys[key_start..key_start + head_dim])
                .map(|(left, right)| left * right)
                .sum::<f64>()
                * scale;
            best = best.max(dot);
            scores.push(dot);
        }
        let total = scores.iter().map(|score| (score - best).exp()).sum::<f64>();
        for (position, score) in scores.into_iter().enumerate() {
            let probability = (score - best).exp() / total;
            let value_start = position * width + kv_head * head_dim;
            for (dst, value) in out[head * head_dim..(head + 1) * head_dim]
                .iter_mut()
                .zip(&values[value_start..value_start + head_dim])
            {
                *dst += probability * value;
            }
        }
    }
    Ok(out)
}

impl GravityLlama {
    /// Open the descriptor directory without eagerly decoding every tensor.
    /// `verify_hash` checks each payload against its descriptor SHA-256 at
    /// first use; native tensors are bounded by the Gravity dense memo rather
    /// than being reconstructed into a full-model resident image.
    pub fn open(path: &Path, verify_hash: bool) -> Result<GravityLlama> {
        let weights = GravityWeights::open_lazy_file(path, verify_hash)?;
        let arch = GravityLlamaArch::from_header(&weights.header)?;
        let tied_head = !weights.contains("lm_head.weight");
        let head_name = if tied_head {
            "model.embed_tokens.weight".to_string()
        } else {
            "lm_head.weight".to_string()
        };
        if !weights.contains(&head_name) {
            return Err(Error::Gravity(
                "artifact has neither lm_head.weight nor model.embed_tokens.weight".into(),
            ));
        }
        Ok(GravityLlama {
            arch,
            weights,
            head_name,
            tied_head,
        })
    }

    fn authority_dense_f64(&self, name: &str) -> Result<Vec<f64>> {
        let (codec, payload, _shape) = self.weights.raw_payload_with_shape(name)?;
        if raw_ggml_kind(&codec)?.is_some() {
            return Err(Error::Gravity(format!(
                "f64 authority dense tensor {name:?} unexpectedly uses raw quant {codec}"
            )));
        }
        Ok(widen_native(&codec, &payload)?
            .into_iter()
            .map(f64::from)
            .collect())
    }

    fn authority_row_f64(&self, name: &str, row: usize, cols: usize) -> Result<Vec<f64>> {
        let (codec, payload, shape) = self.weights.raw_payload_with_shape(name)?;
        match raw_ggml_kind(&codec)? {
            Some(dtype) => {
                if shape.len() != 2 || shape[0] > usize::MAX as u64 || shape[1] != cols as u64 {
                    return Err(Error::Gravity(format!(
                        "f64 authority row {name:?} has invalid shape {shape:?}"
                    )));
                }
                row_ggml_quant_f64_authority(dtype, &payload, shape[0] as usize, cols, row)
                    .map_err(Error::Gravity)
            }
            None => Ok(self
                .weights
                .row(name, row, cols)?
                .into_iter()
                .map(f64::from)
                .collect()),
        }
    }

    fn authority_matvec_f64(&self, name: &str, x: &[f64]) -> Result<Vec<f64>> {
        let (codec, payload, shape) = self.weights.raw_payload_with_shape(name)?;
        if shape.len() != 2 || shape[0] > usize::MAX as u64 || shape[1] > usize::MAX as u64 {
            return Err(Error::Gravity(format!(
                "f64 authority matrix {name:?} has invalid shape {shape:?}"
            )));
        }
        let rows = shape[0] as usize;
        let cols = shape[1] as usize;
        match raw_ggml_kind(&codec)? {
            Some(dtype) => matvec_ggml_quant_f64_authority(dtype, &payload, rows, cols, x)
                .map_err(Error::Gravity),
            None => matvec_dense_f64_authority(&widen_native(&codec, &payload)?, cols, x)
                .map_err(Error::Gravity),
        }
    }

    /// Complete source-artifact transformer forward accumulated in f64.
    ///
    /// This is Numeric Parity V2.1 authority code only. It independently
    /// decodes raw packed projection bytes and accumulates every residual,
    /// attention reduction, normalization, activation and head dot product in
    /// f64. It is deliberately not a serving path and never reports TPS.
    pub fn forward_f64_authority(&self, tokens: &[u32]) -> Result<Vec<f64>> {
        if tokens.is_empty() {
            return Err(Error::Gravity("f64 forward: no tokens".into()));
        }
        let a = &self.arch;
        let kv_width = a.n_kv_heads * a.head_dim;
        let mut k_cache = vec![Vec::<f64>::new(); a.n_layers];
        let mut v_cache = vec![Vec::<f64>::new(); a.n_layers];
        let mut logits = Vec::new();
        for (pos, &token) in tokens.iter().enumerate() {
            if token as usize >= a.vocab_size {
                return Err(Error::Gravity(format!(
                    "f64 forward token {token} out of range for vocab_size {}",
                    a.vocab_size
                )));
            }
            let mut x =
                self.authority_row_f64("model.embed_tokens.weight", token as usize, a.hidden)?;
            for layer in 0..a.n_layers {
                let prefix = format!("model.layers.{layer}.");
                let attn_norm =
                    self.authority_dense_f64(&format!("{prefix}input_layernorm.weight"))?;
                let h =
                    rmsnorm_f64(&x, &attn_norm, a.rms_norm_eps as f64).map_err(Error::Gravity)?;
                let mut q =
                    self.authority_matvec_f64(&format!("{prefix}self_attn.q_proj.weight"), &h)?;
                let mut k =
                    self.authority_matvec_f64(&format!("{prefix}self_attn.k_proj.weight"), &h)?;
                let mut v =
                    self.authority_matvec_f64(&format!("{prefix}self_attn.v_proj.weight"), &h)?;
                for (values, projection) in
                    [(&mut q, "q_proj"), (&mut k, "k_proj"), (&mut v, "v_proj")]
                {
                    let name = format!("{prefix}self_attn.{projection}.bias");
                    if self.weights.contains(&name) {
                        let bias = self.authority_dense_f64(&name)?;
                        if bias.len() != values.len() {
                            return Err(Error::Gravity(format!(
                                "f64 Qwen bias {name:?} has wrong width"
                            )));
                        }
                        for (value, bias) in values.iter_mut().zip(bias) {
                            *value += bias;
                        }
                    }
                }
                for head in 0..a.n_heads {
                    rope_inplace_f64(&mut q[head * a.head_dim..(head + 1) * a.head_dim], pos, a);
                }
                for head in 0..a.n_kv_heads {
                    rope_inplace_f64(&mut k[head * a.head_dim..(head + 1) * a.head_dim], pos, a);
                }
                k_cache[layer].extend_from_slice(&k);
                v_cache[layer].extend_from_slice(&v);
                let full_len = k_cache[layer].len() / kv_width;
                let (window_start, seq_len) = attention_window(full_len, a.sliding_window);
                let attn = mha_decode_step_f64(
                    &q,
                    &k_cache[layer][window_start * kv_width..],
                    &v_cache[layer][window_start * kv_width..],
                    a.n_heads,
                    a.n_kv_heads,
                    a.head_dim,
                    seq_len,
                )?;
                let attention_out =
                    self.authority_matvec_f64(&format!("{prefix}self_attn.o_proj.weight"), &attn)?;
                for (value, residual) in x.iter_mut().zip(attention_out) {
                    *value += residual;
                }
                let ffn_norm =
                    self.authority_dense_f64(&format!("{prefix}post_attention_layernorm.weight"))?;
                let h =
                    rmsnorm_f64(&x, &ffn_norm, a.rms_norm_eps as f64).map_err(Error::Gravity)?;
                let gate =
                    self.authority_matvec_f64(&format!("{prefix}mlp.gate_proj.weight"), &h)?;
                let up = self.authority_matvec_f64(&format!("{prefix}mlp.up_proj.weight"), &h)?;
                let act = silu_mul_f64_authority(&gate, &up).map_err(Error::Gravity)?;
                let down =
                    self.authority_matvec_f64(&format!("{prefix}mlp.down_proj.weight"), &act)?;
                for (value, residual) in x.iter_mut().zip(down) {
                    *value += residual;
                }
            }
            let norm = self.authority_dense_f64("model.norm.weight")?;
            let final_hidden =
                rmsnorm_f64(&x, &norm, a.rms_norm_eps as f64).map_err(Error::Gravity)?;
            logits = self.authority_matvec_f64(&self.head_name, &final_hidden)?;
        }
        Ok(logits)
    }

    /// Run `tokens` through the model from an empty KV cache and return the
    /// logits after the final token — `vocab_size` values.
    pub fn forward(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(Error::Gravity("forward: no tokens".into()));
        }
        let a = &self.arch;
        let kv_width = a.n_kv_heads * a.head_dim;

        // KV cache per layer, appended one position per token, laid out
        // [pos][kv_head][head_dim] to match `mha_decode_step`.
        let mut k_cache: Vec<Vec<f32>> = vec![Vec::new(); a.n_layers];
        let mut v_cache: Vec<Vec<f32>> = vec![Vec::new(); a.n_layers];

        let mut scratch = vec![0f32; a.hidden];
        let mut logits = Vec::new();
        // CPU/GPU source-artifact parity bisection. This is deliberately an
        // opt-in diagnostic surface: it records bounded scalar summaries at
        // every layer without changing the executable math or pretending a
        // traced run is a throughput measurement.
        let trace_path = std::env::var_os("HAWKING_GRAVITY_CPU_TRACE_PATH");
        let trace_position = std::env::var("HAWKING_GRAVITY_TRACE_POSITION")
            .ok()
            .and_then(|value| value.parse::<usize>().ok());
        let mut trace_rows = Vec::<serde_json::Value>::new();

        for (pos, &token) in tokens.iter().enumerate() {
            let trace_enabled =
                trace_path.is_some() && trace_position.map(|wanted| wanted == pos).unwrap_or(true);
            if token as usize >= a.vocab_size {
                return Err(Error::Gravity(format!(
                    "token {token} out of range for vocab_size {}",
                    a.vocab_size
                )));
            }
            let mut x = self
                .weights
                .row("model.embed_tokens.weight", token as usize, a.hidden)?;
            if x.len() != a.hidden {
                return Err(Error::Gravity(format!(
                    "embedding row is {} wide, expected hidden_size {}",
                    x.len(),
                    a.hidden
                )));
            }

            for layer in 0..a.n_layers {
                let p = format!("model.layers.{layer}.");

                rmsnorm(
                    &x,
                    &self.weights.dense(&format!("{p}input_layernorm.weight"))?,
                    a.rms_norm_eps,
                    &mut scratch,
                );
                let attn_norm_sum = if trace_enabled {
                    scratch.iter().map(|value| *value as f64).sum::<f64>()
                } else {
                    0.0
                };
                let attn_norm_head = if trace_enabled {
                    scratch[..8.min(scratch.len())].to_vec()
                } else {
                    Vec::new()
                };
                let mut q = self
                    .weights
                    .matvec(&format!("{p}self_attn.q_proj.weight"), &scratch)?;
                let mut k = self
                    .weights
                    .matvec(&format!("{p}self_attn.k_proj.weight"), &scratch)?;
                let mut v = self
                    .weights
                    .matvec(&format!("{p}self_attn.v_proj.weight"), &scratch)?;

                // Qwen2 carries explicit Q/K/V biases while Llama/Mistral
                // normally do not. Their presence is descriptor-authoritative:
                // never infer a zero vector, but also never drop a copied
                // source bias just because the dense decoder ABI is shared.
                for (values, projection) in
                    [(&mut q, "q_proj"), (&mut k, "k_proj"), (&mut v, "v_proj")]
                {
                    let bias_name = format!("{p}self_attn.{projection}.bias");
                    if self.weights.contains(&bias_name) {
                        let bias = self.weights.dense(&bias_name)?;
                        add_inplace(values, &bias);
                    }
                }
                let (q_raw_sum, k_raw_sum, v_raw_sum) = if trace_enabled {
                    (
                        q.iter().map(|value| *value as f64).sum::<f64>(),
                        k.iter().map(|value| *value as f64).sum::<f64>(),
                        v.iter().map(|value| *value as f64).sum::<f64>(),
                    )
                } else {
                    (0.0, 0.0, 0.0)
                };

                for h in 0..a.n_heads {
                    rope_inplace_scaled(
                        &mut q[h * a.head_dim..(h + 1) * a.head_dim],
                        pos as u32,
                        a.rope_theta,
                        a.rope_scaling,
                    );
                }
                for h in 0..a.n_kv_heads {
                    rope_inplace_scaled(
                        &mut k[h * a.head_dim..(h + 1) * a.head_dim],
                        pos as u32,
                        a.rope_theta,
                        a.rope_scaling,
                    );
                }
                let (q_rope_sum, k_rope_sum) = if trace_enabled {
                    (
                        q.iter().map(|value| *value as f64).sum::<f64>(),
                        k.iter().map(|value| *value as f64).sum::<f64>(),
                    )
                } else {
                    (0.0, 0.0)
                };

                k_cache[layer].extend_from_slice(&k);
                v_cache[layer].extend_from_slice(&v);
                let full_seq_len = k_cache[layer].len() / kv_width;
                let (window_start, seq_len) = attention_window(full_seq_len, a.sliding_window);
                let k_window = &k_cache[layer][window_start * kv_width..];
                let v_window = &v_cache[layer][window_start * kv_width..];

                let mut attn = vec![0f32; a.n_heads * a.head_dim];
                mha_decode_step(
                    &q,
                    k_window,
                    v_window,
                    a.n_heads,
                    a.n_kv_heads,
                    a.head_dim,
                    seq_len,
                    &mut attn,
                )?;

                let o = self
                    .weights
                    .matvec(&format!("{p}self_attn.o_proj.weight"), &attn)?;
                add_inplace(&mut x, &o);
                let ffn_input_sum = if trace_enabled {
                    x.iter().map(|value| *value as f64).sum::<f64>()
                } else {
                    0.0
                };

                rmsnorm(
                    &x,
                    &self
                        .weights
                        .dense(&format!("{p}post_attention_layernorm.weight"))?,
                    a.rms_norm_eps,
                    &mut scratch,
                );
                let ffn_norm_sum = if trace_enabled {
                    scratch.iter().map(|value| *value as f64).sum::<f64>()
                } else {
                    0.0
                };
                let gate = self
                    .weights
                    .matvec(&format!("{p}mlp.gate_proj.weight"), &scratch)?;
                let up = self
                    .weights
                    .matvec(&format!("{p}mlp.up_proj.weight"), &scratch)?;
                let mut act = vec![0f32; gate.len()];
                silu_mul(&gate, &up, &mut act);
                let down = self
                    .weights
                    .matvec(&format!("{p}mlp.down_proj.weight"), &act)?;
                add_inplace(&mut x, &down);

                if trace_enabled {
                    trace_rows.push(serde_json::json!({
                        "position": pos,
                        "token_id": token,
                        "layer": layer,
                        "qkv_trace_stage": "post_projection_post_bias",
                        "q_raw_sum": q_raw_sum,
                        "q_raw_head": &q[..8.min(q.len())],
                        "k_raw_sum": k_raw_sum,
                        "k_raw_head": &k[..8.min(k.len())],
                        "v_raw_sum": v_raw_sum,
                        "v_raw_head": &v[..8.min(v.len())],
                        "q_rope_sum": q_rope_sum,
                        "k_rope_sum": k_rope_sum,
                        "attn_norm_sum": attn_norm_sum,
                        "attn_norm_head": attn_norm_head,
                        "ffn_input_sum": ffn_input_sum,
                        "ffn_norm_sum": ffn_norm_sum,
                        "ffn_gate_sum": gate.iter().map(|value| *value as f64).sum::<f64>(),
                        "ffn_gate_head": &gate[..8.min(gate.len())],
                        "ffn_up_sum": up.iter().map(|value| *value as f64).sum::<f64>(),
                        "ffn_up_head": &up[..8.min(up.len())],
                        "ffn_swiglu_sum": act.iter().map(|value| *value as f64).sum::<f64>(),
                        "ffn_out_sum": down.iter().map(|value| *value as f64).sum::<f64>(),
                        "layer_out_sum": x.iter().map(|value| *value as f64).sum::<f64>(),
                    }));
                }
            }

            rmsnorm(
                &x,
                &self.weights.dense("model.norm.weight")?,
                a.rms_norm_eps,
                &mut scratch,
            );
            logits = self.weights.matvec(&self.head_name.clone(), &scratch)?;
        }

        if let Some(path) = trace_path.filter(|_| !trace_rows.is_empty()) {
            let trace = serde_json::json!({
                "schema": "hawking.gravity.llama_cpu_layer_trace.v1",
                "artifact_architecture": {
                    "model_type": a.model_type,
                    "rope_interleaved": a.rope_interleaved,
                    "rope_freq_factors": a.rope_freq_factors.as_ref().map(Vec::len),
                },
                "rows": trace_rows,
                "note": "scalar source-artifact diagnostic; not a throughput path"
            });
            let bytes = serde_json::to_vec_pretty(&trace)
                .map_err(|err| Error::Gravity(format!("serialize CPU trace: {err}")))?;
            std::fs::write(path, bytes)?;
        }

        Ok(logits)
    }
}

// ---------------------------------------------------------------------
// Resident GPU path.
// ---------------------------------------------------------------------

/// The same model with its packed weights resident in device memory.
///
/// The distinction that matters for throughput is *resident*:
/// [`crate::gravity::pq_matvec_metal`] uploads codebooks and codes on every
/// call, which is right for a parity test and useless for a runtime, where
/// each tensor is read once per token forever. Here every payload is
/// uploaded once at load, verbatim — the kernel reads `half` codebooks and
/// walks the packed index stream itself, so nothing is widened or unpacked
/// on the way in and device bytes equal artifact bytes.
#[cfg(target_os = "macos")]
pub mod gpu {
    use super::*;
    use crate::gravity::{
        parse_pq_header, parse_residual_pq_header, pq_row, pq_sections, residual_pq_sections,
        PqHeader, ResidualPqHeader, ResidualPqTensor,
    };
    use crate::metal::{MetalContext, TokenCommandBuffer};
    use metal::Buffer;
    use std::sync::Mutex;
    use std::time::Instant;

    /// Mirror of `GravityPQParams` in `shaders/gravity_pq.metal`: eight
    /// `uint`s in declaration order, `#[repr(C)]` so a pointer cast is a
    /// valid `set_bytes` payload.
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    struct PqParams {
        dim: u32,
        subspaces: u32,
        sub: u32,
        card: u32,
        rows: u32,
        cols: u32,
        nchunk: u32,
        bits: u32,
    }

    impl PqParams {
        fn from_header(h: &PqHeader) -> PqParams {
            PqParams {
                dim: h.d as u32,
                subspaces: h.s as u32,
                sub: h.sub as u32,
                card: h.card as u32,
                rows: h.rows,
                cols: h.cols,
                nchunk: h.nchunk,
                bits: h.bits as u32,
            }
        }
    }

    /// One packed tensor resident on the device.
    struct GpuPq {
        codebooks: Buffer,
        codes: Buffer,
        params: PqParams,
    }

    /// Wire mirror of `GravityResidualPQParams`.  Keep this eight-u32 shape
    /// stable: the Metal kernel reads it directly out of `set_bytes`.
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    struct ResidualPqParams {
        dim: u32,
        stages: u32,
        card: u32,
        rows: u32,
        cols: u32,
        nchunk: u32,
        bits: u32,
        reserved: u32,
    }

    impl ResidualPqParams {
        fn from_header(h: &ResidualPqHeader) -> Self {
            Self {
                dim: h.d as u32,
                stages: h.stages as u32,
                card: h.card as u32,
                rows: h.rows,
                cols: h.cols,
                nchunk: h.nchunk,
                bits: h.bits as u32,
                reserved: 0,
            }
        }
    }

    struct GpuResidualPq {
        codebooks: Buffer,
        codes: Buffer,
        params: ResidualPqParams,
    }

    /// Source-preserving GGML K-quant tensor carried by a Gravity shard. The
    /// payload is the same row-major block stream as the parent GGUF; the
    /// strict Llama b9430 kernels consume it directly without widening.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum RawQuantKind {
        Q4K,
        Q5K,
        Q6K,
        /// GGML's 32-element legacy blocks occur in Qwen2 embeddings and
        /// selected projections.  They remain packed end-to-end; this is not
        /// a dense reconstruction escape hatch.
        Q5_0,
        Q8_0,
    }

    /// Measured Q4_K execution grammar.  The strict b9430 port is the
    /// authority default; alternate geometries are loaded only when an
    /// explicit preregistered schedule is requested for a same-model sweep.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Q4Schedule {
        B9430,
        V3_8R,
        V3Dual,
        V3Llama,
        Simdmat,
        Predec2R,
        FusedV2,
    }

    fn q4_schedule_from_env() -> Q4Schedule {
        match std::env::var("HAWKING_GRAVITY_Q4_SCHEDULE")
            .ok()
            .as_deref()
        {
            Some("v3_8r") => Q4Schedule::V3_8R,
            Some("v3_dual") => Q4Schedule::V3Dual,
            Some("v3_llama") => Q4Schedule::V3Llama,
            Some("simdmat") => Q4Schedule::Simdmat,
            Some("predec_2r") => Q4Schedule::Predec2R,
            Some("v2") => Q4Schedule::FusedV2,
            Some("b9430") | None => {
                if crate::env_on("HAWKING_GRAVITY_Q4_V3") {
                    Q4Schedule::V3_8R
                } else {
                    Q4Schedule::B9430
                }
            }
            Some(other) => {
                eprintln!(
                    "unknown HAWKING_GRAVITY_Q4_SCHEDULE={other:?}; using strict b9430"
                );
                Q4Schedule::B9430
            }
        }
    }

    struct GpuRawQuant {
        data: Buffer,
        /// Optional predecoded (ds,dm) table used only by the opt-in fused
        /// Q/K/V source kernel. Keeping this optional preserves the compact
        /// default resident image and makes the extra bytes visible in the
        /// candidate receipt.
        predec_scales: Option<Buffer>,
        rows: usize,
        cols: usize,
        bytes: usize,
        kind: RawQuantKind,
    }

    /// Norm weights live on the device too: leaving them on the host would
    /// force a round trip per layer purely to hand a 2048-float vector to a
    /// kernel that runs for microseconds.
    enum GpuWeight {
        Pq(GpuPq),
        ResidualPq(GpuResidualPq),
        RawQuant(GpuRawQuant),
        Dense(Buffer),
    }

    /// Reusable device-side activation vectors, allocated once per role and
    /// reused for every token thereafter. Keyed by role rather than by size:
    /// two roles that happen to share a width (`k` and `v`, `gate` and `up`)
    /// must still be two buffers, and a size key would silently alias them
    /// into one.
    struct BufPool {
        buffers: Mutex<HashMap<&'static str, Buffer>>,
    }

    impl BufPool {
        fn new() -> BufPool {
            BufPool {
                buffers: Mutex::new(HashMap::new()),
            }
        }

        fn get(&self, ctx: &MetalContext, role: &'static str, elems: usize) -> Buffer {
            self.buffers
                .lock()
                .expect("buffer pool mutex")
                .entry(role)
                .or_insert_with(|| ctx.new_buffer(elems * std::mem::size_of::<f32>()))
                .clone()
        }
    }

    fn write_f32(buf: &Buffer, src: &[f32]) {
        // Safety: shared-storage buffer sized for at least `src.len()` f32s
        // by construction (`BufPool::get` is called with that element count).
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), buf.contents() as *mut f32, src.len());
        }
    }

    fn read_f32(buf: &Buffer, n: usize) -> Vec<f32> {
        // Safety: same sizing contract as `write_f32`.
        unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, n) }.to_vec()
    }

    fn read_f32_at(buf: &Buffer, offset: usize, n: usize) -> Vec<f32> {
        // Safety: callers pass an offset/count within a shared-storage buffer
        // allocated for the corresponding KV cache capacity.
        unsafe {
            std::slice::from_raw_parts((buf.contents() as *const f32).add(offset), n).to_vec()
        }
    }

    /// What one `forward` call cost, so throughput reporting never has to
    /// infer timings from a wall clock wrapped around the whole thing.
    ///
    /// `per_token_ms` is what separates prefill from decode: both phases run
    /// the same code here (one token at a time against a growing cache), so
    /// the only honest way to report them separately is to time each token
    /// and split the series, rather than to run two passes and pretend the
    /// second started cold.
    #[derive(Debug, Clone, Default)]
    pub struct ForwardStats {
        pub tokens: usize,
        pub command_buffers: usize,
        pub dispatches: usize,
        /// Number of layer executions that physically selected the fused
        /// source Q/K/V + RoPE + KV-append grammar.  This is distinct from
        /// the environment request: an unsupported tensor mix must remain a
        /// safe decomposed fallback and must not be reported as fused work.
        pub fused_qkv_dispatches: usize,
        /// Number of layer executions that physically selected a paired
        /// source gate/up projection wave. Requested-but-ineligible source
        /// grammars are deliberately excluded.
        pub fused_gate_up_dispatches: usize,
        pub first_token_ms: f64,
        pub total_ms: f64,
        pub per_token_ms: Vec<f64>,
    }

    pub struct GravityLlamaGpu {
        ctx: MetalContext,
        pub arch: GravityLlamaArch,
        weights: HashMap<String, GpuWeight>,
        /// Raw `gravity-pq` payload of the embedding table, kept for row
        /// lookup. One row per token is decoded from it directly; the same
        /// tensor is also resident on the device for the tied LM head.
        embed_payload: Vec<u8>,
        embed_is_residual_pq: bool,
        embed_raw_quant: Option<RawQuantKind>,
        head_name: String,
        pub tied_head: bool,
        pub load_ms: f64,
        pub device_bytes: usize,
        /// Explicitly selected Q4_K execution grammar.  The value is sealed
        /// into benchmark receipts so a kernel result cannot be mistaken for
        /// the strict source-control path.
        q4_schedule: Q4Schedule,
        /// Opt-in source-QKV candidate. It is enabled only for Llama-family
        /// shards whose RoPE can be represented by the fused kernel's scalar
        /// base; artifacts with resolved per-frequency factors stay on the
        /// parity-proven baseline until the factor buffer is wired.
        fused_source_qkv: bool,
        pool: BufPool,
        /// Per-layer K and V caches resident on the device, grown to fit the
        /// longest run seen so far. Attention reads them in place, so the
        /// only thing crossing the bus per layer is q/k/v and the attention
        /// output -- not the O(seq_len) cache.
        kv: Mutex<KvBuffers>,
    }

    #[derive(Default)]
    struct KvBuffers {
        k: Vec<Buffer>,
        v: Vec<Buffer>,
        /// Exact f16 SET_ROWS images used by the source-style flash attention
        /// candidate.  The f32 images remain available for the Llama/Mistral
        /// default path and diagnostics; Qwen2 selects the source-compatible
        /// f16 path automatically, and `HAWKING_GRAVITY_F16_KV=1` enables it
        /// for the other families.
        k_f16: Vec<Buffer>,
        v_f16: Vec<Buffer>,
        capacity_tokens: usize,
    }

    fn raw_quant_shape(
        descriptor: &crate::artifact::TensorDescriptor,
        bytes: usize,
        kind: RawQuantKind,
        name: &str,
    ) -> Result<(usize, usize)> {
        if descriptor.shape.len() != 2 {
            return Err(Error::Gravity(format!(
                "tensor {name:?}: raw K-quant shape must be [rows, cols], got {:?}",
                descriptor.shape
            )));
        }
        let rows = usize::try_from(descriptor.shape[0])
            .map_err(|_| Error::Gravity(format!("tensor {name:?}: rows overflow")))?;
        let cols = usize::try_from(descriptor.shape[1])
            .map_err(|_| Error::Gravity(format!("tensor {name:?}: cols overflow")))?;
        let (block_elems, block_bytes) = match kind {
            RawQuantKind::Q4K => (256usize, 144usize),
            RawQuantKind::Q5K => (256usize, 176usize),
            RawQuantKind::Q6K => (256usize, 210usize),
            RawQuantKind::Q5_0 => (32usize, 22usize),
            RawQuantKind::Q8_0 => (32usize, 34usize),
        };
        if rows == 0 || cols == 0 || cols % block_elems != 0 {
            return Err(Error::Gravity(format!(
                "tensor {name:?}: raw quant geometry {rows}x{cols} is not a positive {block_elems}-wide matrix"
            )));
        }
        let expected = rows
            .checked_mul(cols / block_elems)
            .and_then(|blocks| blocks.checked_mul(block_bytes))
            .ok_or_else(|| Error::Gravity(format!("tensor {name:?}: raw K-quant size overflow")))?;
        if expected != bytes {
            return Err(Error::Gravity(format!(
                "tensor {name:?}: raw K-quant payload {bytes} B != expected {expected} B"
            )));
        }
        Ok((rows, cols))
    }

    fn raw_quant_row(
        payload: &[u8],
        row: usize,
        rows: usize,
        cols: usize,
        kind: RawQuantKind,
    ) -> Result<Vec<f32>> {
        if row >= rows {
            return Err(Error::Gravity(format!(
                "raw K-quant embedding row {row} outside 0..{rows}"
            )));
        }
        let (block_elems, block_bytes, dtype) = match kind {
            RawQuantKind::Q4K => (256usize, 144usize, crate::gguf::GgmlType::Q4_K),
            RawQuantKind::Q5K => (256usize, 176usize, crate::gguf::GgmlType::Q5_K),
            RawQuantKind::Q6K => (256usize, 210usize, crate::gguf::GgmlType::Q6_K),
            RawQuantKind::Q5_0 => (32usize, 22usize, crate::gguf::GgmlType::Q5_0),
            RawQuantKind::Q8_0 => (32usize, 34usize, crate::gguf::GgmlType::Q8_0),
        };
        let row_bytes = cols / block_elems * block_bytes;
        let start = row
            .checked_mul(row_bytes)
            .ok_or_else(|| Error::Gravity("raw K-quant row offset overflow".into()))?;
        let end = start
            .checked_add(row_bytes)
            .ok_or_else(|| Error::Gravity("raw K-quant row end overflow".into()))?;
        let bytes = payload
            .get(start..end)
            .ok_or_else(|| Error::Gravity("raw K-quant embedding row outside payload".into()))?;
        let mut out = vec![0.0f32; cols];
        crate::quant::dequant_into(dtype, bytes, &mut out)?;
        Ok(out)
    }

    impl GravityLlamaGpu {
        pub fn device_name(&self) -> String {
            self.ctx.device_name()
        }

        /// Open with a context this model owns. An `Engine` must be `Send +
        /// Sync`, and a borrowed context makes that impossible to express, so
        /// the model holds its own rather than living inside someone's scope.
        pub fn open(path: &Path, verify_hash: bool) -> Result<GravityLlamaGpu> {
            Self::open_with(MetalContext::new()?, path, verify_hash)
        }

        pub fn open_with(
            ctx: MetalContext,
            path: &Path,
            verify_hash: bool,
        ) -> Result<GravityLlamaGpu> {
            let t0 = Instant::now();
            let shard = GravityShard::open(path)?;
            let arch = GravityLlamaArch::from_header(&shard.extra)?;
            // This is deliberately opt-in: predecoded scale tables add
            // ~44% to Q4_K projection residency. They are only justified for
            // the fused source-QKV timing candidate, never for the compact
            // baseline artifact.
            let fused_source_qkv = crate::env_on("HAWKING_GRAVITY_FUSED_QKV")
                && matches!(arch.model_type.as_str(), "llama" | "mistral" | "qwen2");

            let names: Vec<String> = shard.tensor_names().map(str::to_string).collect();
            let mut weights = HashMap::with_capacity(names.len());
            let mut embed_payload = Vec::new();
            let mut embed_is_residual_pq = false;
            let mut embed_raw_quant = None;
            let mut device_bytes = 0usize;
            for name in &names {
                let descriptor = shard.descriptor(name).expect("name came from tensor_names");
                let codec = descriptor.codec.clone();
                let blob = shard.read_tensor(name, verify_hash)?;
                if codec == "gravity-pq" {
                    let h = parse_pq_header(&blob)?;
                    let (cb, codes) = pq_sections(&blob)?;
                    // Four bytes of tail padding so the kernel's whole-word
                    // read at the last index's byte offset stays in bounds.
                    let mut codes_padded = Vec::with_capacity(codes.len() + 4);
                    codes_padded.extend_from_slice(codes);
                    codes_padded.extend_from_slice(&[0u8; 4]);
                    device_bytes += cb.len() + codes_padded.len();
                    weights.insert(
                        name.clone(),
                        GpuWeight::Pq(GpuPq {
                            codebooks: ctx.new_buffer_with_bytes_checked(cb)?,
                            codes: ctx.new_buffer_with_bytes_checked(&codes_padded)?,
                            params: PqParams::from_header(&h),
                        }),
                    );
                    if name == "model.embed_tokens.weight" {
                        embed_payload = blob;
                    }
                } else if codec == "llama.residual-pq.v1" {
                    let h = parse_residual_pq_header(&blob)?;
                    let (cb, codes) = residual_pq_sections(&blob)?;
                    let mut codes_padded = Vec::with_capacity(codes.len() + 4);
                    codes_padded.extend_from_slice(codes);
                    codes_padded.extend_from_slice(&[0u8; 4]);
                    device_bytes += cb.len() + codes_padded.len();
                    weights.insert(
                        name.clone(),
                        GpuWeight::ResidualPq(GpuResidualPq {
                            codebooks: ctx.new_buffer_with_bytes_checked(cb)?,
                            codes: ctx.new_buffer_with_bytes_checked(&codes_padded)?,
                            params: ResidualPqParams::from_header(&h),
                        }),
                    );
                    if name == "model.embed_tokens.weight" {
                        embed_payload = blob;
                        embed_is_residual_pq = true;
                    }
                } else if matches!(
                    codec.as_str(),
                    "ggml.q4_k"
                        | "ggml.q5_k"
                        | "ggml.q6_k"
                        | "ggml.q5_0"
                        | "ggml.q8_0"
                ) {
                    let kind = if codec == "ggml.q4_k" {
                        RawQuantKind::Q4K
                    } else if codec == "ggml.q5_k" {
                        RawQuantKind::Q5K
                    } else if codec == "ggml.q6_k" {
                        RawQuantKind::Q6K
                    } else if codec == "ggml.q5_0" {
                        RawQuantKind::Q5_0
                    } else {
                        RawQuantKind::Q8_0
                    };
                    let (rows, cols) = raw_quant_shape(descriptor, blob.len(), kind, name)?;
                    device_bytes += blob.len();
                    let predecode_all_q4 = crate::env_on("HAWKING_GRAVITY_Q4_PREDEC");
                    let predec_scales = if kind == RawQuantKind::Q4K
                        && (predecode_all_q4
                            || (fused_source_qkv
                                && (name.ends_with(".self_attn.q_proj.weight")
                                    || name.ends_with(".self_attn.k_proj.weight")
                                    || name.ends_with(".self_attn.v_proj.weight"))))
                    {
                        let scales = crate::kernels::predecode_q4_k_scale_table(&blob);
                        let bytes = bytemuck::cast_slice::<f32, u8>(&scales);
                        device_bytes += bytes.len();
                        Some(ctx.new_buffer_with_bytes_checked(bytes)?)
                    } else {
                        None
                    };
                    weights.insert(
                        name.clone(),
                        GpuWeight::RawQuant(GpuRawQuant {
                            data: ctx.new_buffer_with_bytes_checked(&blob)?,
                            predec_scales,
                            rows,
                            cols,
                            bytes: blob.len(),
                            kind,
                        }),
                    );
                    if name == "model.embed_tokens.weight" {
                        embed_payload = blob;
                        embed_raw_quant = Some(kind);
                    }
                } else if codec.starts_with("native.") {
                    let widened = widen_native(&codec, &blob)?;
                    device_bytes += widened.len() * std::mem::size_of::<f32>();
                    weights.insert(
                        name.clone(),
                        GpuWeight::Dense(ctx.new_buffer_with_bytes_checked(
                            bytemuck::cast_slice::<f32, u8>(&widened),
                        )?),
                    );
                } else {
                    return Err(Error::Gravity(format!(
                        "tensor {name}: unsupported codec {codec:?}"
                    )));
                }
            }

            let tied_head = !weights.contains_key("lm_head.weight");
            let head_name = if tied_head {
                "model.embed_tokens.weight".to_string()
            } else {
                "lm_head.weight".to_string()
            };
            if embed_payload.is_empty() {
                return Err(Error::Gravity(
                    "artifact has no packed model.embed_tokens.weight to look rows up in".into(),
                ));
            }
            let fused_source_qkv_enabled =
                fused_source_qkv && arch.rope_freq_factors.is_none() && arch.rope_scaling.is_none();

            Ok(GravityLlamaGpu {
                ctx,
                arch,
                weights,
                embed_payload,
                embed_is_residual_pq,
                embed_raw_quant,
                head_name,
                tied_head,
                load_ms: t0.elapsed().as_secs_f64() * 1e3,
                device_bytes,
                q4_schedule: q4_schedule_from_env(),
                fused_source_qkv: fused_source_qkv_enabled,
                pool: BufPool::new(),
                kv: Mutex::new(KvBuffers::default()),
            })
        }

        /// Exact KV bytes for `tokens` positions -- two f32 caches of
        /// `n_kv_heads * head_dim` per layer per position.
        pub fn kv_bytes_for(&self, tokens: usize) -> usize {
            2 * self.arch.n_layers * self.arch.n_kv_heads * self.arch.head_dim * 4 * tokens
        }

        /// Grow the device KV caches to hold at least `tokens` positions,
        /// preserving whatever is already cached.
        ///
        /// The copy is what makes an incremental session safe: a session that
        /// outgrows its capacity must not silently forget its own prefix, and
        /// a forgotten prefix produces fluent, confident, wrong continuations
        /// that nothing downstream would flag.
        fn reserve_kv(&self, tokens: usize) -> Result<()> {
            let mut kv = self.kv.lock().expect("kv mutex");
            if kv.capacity_tokens >= tokens && !kv.k.is_empty() {
                return Ok(());
            }
            let kv_elems = tokens * self.arch.n_kv_heads * self.arch.head_dim;
            let per_layer = kv_elems * std::mem::size_of::<f32>();
            let per_layer_f16 = kv_elems * std::mem::size_of::<half::f16>();
            let carry = kv.capacity_tokens
                * self.arch.n_kv_heads
                * self.arch.head_dim
                * std::mem::size_of::<f32>();
            let carry_f16 = kv.capacity_tokens
                * self.arch.n_kv_heads
                * self.arch.head_dim
                * std::mem::size_of::<half::f16>();
            let mut k = Vec::with_capacity(self.arch.n_layers);
            let mut v = Vec::with_capacity(self.arch.n_layers);
            let mut k_f16 = Vec::with_capacity(self.arch.n_layers);
            let mut v_f16 = Vec::with_capacity(self.arch.n_layers);
            for layer in 0..self.arch.n_layers {
                let nk = self.ctx.new_buffer_checked(per_layer)?;
                let nv = self.ctx.new_buffer_checked(per_layer)?;
                let nk_f16 = self.ctx.new_buffer_checked(per_layer_f16)?;
                let nv_f16 = self.ctx.new_buffer_checked(per_layer_f16)?;
                if carry > 0 {
                    // Safety: both buffers are shared storage and `carry` is the
                    // old capacity in bytes, which the new buffers exceed.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            kv.k[layer].contents() as *const u8,
                            nk.contents() as *mut u8,
                            carry,
                        );
                        std::ptr::copy_nonoverlapping(
                            kv.v[layer].contents() as *const u8,
                            nv.contents() as *mut u8,
                            carry,
                        );
                    }
                }
                if carry_f16 > 0 {
                    // Safety: the f16 buffers use the same row-major cache
                    // layout at half the element width and the new buffers
                    // exceed the old capacity.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            kv.k_f16[layer].contents() as *const u8,
                            nk_f16.contents() as *mut u8,
                            carry_f16,
                        );
                        std::ptr::copy_nonoverlapping(
                            kv.v_f16[layer].contents() as *const u8,
                            nv_f16.contents() as *mut u8,
                            carry_f16,
                        );
                    }
                }
                k.push(nk);
                v.push(nv);
                k_f16.push(nk_f16);
                v_f16.push(nv_f16);
            }
            *kv = KvBuffers {
                k,
                v,
                k_f16,
                v_f16,
                capacity_tokens: tokens,
            };
            Ok(())
        }

        fn packed_rows(&self, name: &str) -> Result<usize> {
            match self
                .weights
                .get(name)
                .ok_or_else(|| Error::Gravity(format!("artifact has no tensor {name:?}")))?
            {
                GpuWeight::Pq(t) => Ok(t.params.rows as usize),
                GpuWeight::ResidualPq(t) => Ok(t.params.rows as usize),
                GpuWeight::RawQuant(t) => Ok(t.rows),
                GpuWeight::Dense(_) => Err(Error::Gravity(format!(
                    "tensor {name:?} is dense; expected a packed projection"
                ))),
            }
        }

        fn dense(&self, name: &str) -> Result<&Buffer> {
            match self
                .weights
                .get(name)
                .ok_or_else(|| Error::Gravity(format!("artifact has no tensor {name:?}")))?
            {
                GpuWeight::Dense(b) => Ok(b),
                GpuWeight::Pq(_) => Err(Error::Gravity(format!(
                    "tensor {name:?} is packed; expected a natively-carried dense tensor"
                ))),
                GpuWeight::ResidualPq(_) => Err(Error::Gravity(format!(
                    "tensor {name:?} is packed; expected a natively-carried dense tensor"
                ))),
                GpuWeight::RawQuant(_) => Err(Error::Gravity(format!(
                    "tensor {name:?} is raw K-quantized; expected a natively-carried dense tensor"
                ))),
            }
        }

        /// Optional native bias vector used by Qwen2's attention projections.
        /// Llama/Mistral source shards simply omit these tensors.  Refuse a
        /// packed or raw value under a bias name instead of treating it as a
        /// compatible dense vector.
        fn optional_bias(&self, name: &str) -> Result<Option<&Buffer>> {
            match self.weights.get(name) {
                None => Ok(None),
                Some(GpuWeight::Dense(b)) => Ok(Some(b)),
                Some(_) => Err(Error::Gravity(format!(
                    "tensor {name:?} is not a native attention bias"
                ))),
            }
        }

        fn add_optional_bias(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            target: &Buffer,
            name: &str,
            n: usize,
        ) -> Result<()> {
            if let Some(bias) = self.optional_bias(name)? {
                crate::kernels::add_inplace_metal_tcb(tcb, target, bias, n)?;
            }
            Ok(())
        }

        /// Encode one raw Q4_K matvec using the selected measured geometry.
        /// K/V projections pass an element offset so the same helper can
        /// write directly into the resident cache slot.
        fn encode_q4_matvec(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            w: &GpuRawQuant,
            x: &Buffer,
            y: &Buffer,
            y_offset: usize,
        ) -> Result<()> {
            let off = y_offset * std::mem::size_of::<f32>();
            match (self.q4_schedule, y_offset == 0) {
                (Q4Schedule::B9430, true) => crate::kernels::gemv_q4_k_m_llama_b9430_pinned_tcb(
                    tcb, &w.data, 0, w.bytes, w.rows, w.cols, x, y,
                ),
                (Q4Schedule::B9430, false) => {
                    crate::kernels::gemv_q4_k_m_llama_b9430_pinned_off_tcb(
                        tcb, &w.data, 0, w.bytes, w.rows, w.cols, x, y, off,
                    )
                }
                (Q4Schedule::V3_8R, true) => crate::kernels::gemv_q4_k_m_v3_8r_pinned_tcb(
                    tcb, &w.data, 0, w.bytes, w.rows, w.cols, x, y,
                ),
                (Q4Schedule::V3_8R, false) => {
                    crate::kernels::gemv_q4_k_m_v3_8r_pinned_off_tcb(
                        tcb, &w.data, 0, w.bytes, w.rows, w.cols, x, y, off,
                    )
                }
                (Q4Schedule::V3Dual, true) => crate::kernels::gemv_q4_k_m_v3_dual_pinned_tcb(
                    tcb, &w.data, 0, w.bytes, w.rows, w.cols, x, y,
                ),
                (Q4Schedule::V3Dual, false) => {
                    crate::kernels::gemv_q4_k_m_v3_dual_pinned_off_tcb(
                        tcb, &w.data, 0, w.bytes, w.rows, w.cols, x, y, off,
                    )
                }
                (Q4Schedule::V3Llama, true) => {
                    crate::kernels::gemv_q4_k_m_v3_llama_pinned_tcb(
                        tcb, &w.data, 0, w.bytes, w.rows, w.cols, x, y,
                    )
                }
                (Q4Schedule::V3Llama, false) => {
                    crate::kernels::gemv_q4_k_m_v3_llama_pinned_off_tcb(
                        tcb, &w.data, 0, w.bytes, w.rows, w.cols, x, y, off,
                    )
                }
                (Q4Schedule::Simdmat, true) => crate::kernels::gemv_q4_k_m_simdmat_pinned_tcb(
                    tcb, &w.data, 0, w.bytes, w.rows, w.cols, x, y,
                ),
                (Q4Schedule::Simdmat, false) => {
                    crate::kernels::gemv_q4_k_m_simdmat_pinned_off_tcb(
                        tcb, &w.data, 0, w.bytes, w.rows, w.cols, x, y, off,
                    )
                }
                (Q4Schedule::Predec2R, true) => {
                    let scales = w.predec_scales.as_ref().ok_or_else(|| {
                        Error::Gravity("Q4 predecode schedule selected without scale table".into())
                    })?;
                    crate::kernels::gemv_q4_k_v4_predec_pinned_tcb(
                        tcb, &w.data, 0, w.bytes, scales, 0, w.rows, w.cols, x, y,
                    )
                }
                (Q4Schedule::Predec2R, false) => {
                    // The predecoded ABI currently has no output-offset form;
                    // keep direct KV-cache writes on the exact b9430 grammar
                    // rather than introducing a copy or hidden host readback.
                    crate::kernels::gemv_q4_k_m_llama_b9430_pinned_off_tcb(
                        tcb, &w.data, 0, w.bytes, w.rows, w.cols, x, y, off,
                    )
                }
                (Q4Schedule::FusedV2, true) => crate::kernels::gemv_q4_k_m_v2_pinned_tcb(
                    tcb, &w.data, 0, w.bytes, w.rows, w.cols, x, y,
                ),
                (Q4Schedule::FusedV2, false) => {
                    // The v2 shader has no output-offset entry point; keep
                    // direct K/V cache writes on the exact b9430 ABI.
                    crate::kernels::gemv_q4_k_m_llama_b9430_pinned_off_tcb(
                        tcb, &w.data, 0, w.bytes, w.rows, w.cols, x, y, off,
                    )
                }
            }
        }

        /// The RoPE cos/sin table for one position: `head_dim/2` cosines then
        /// `head_dim/2` sines. Computed in f64 from the header's declared
        /// scaling, matching the oracle's frequency math exactly, so the
        /// kernel applies a rotation it does not need to understand.
        fn rope_table(&self, pos: usize) -> Vec<f32> {
            let a = &self.arch;
            let half = a.head_dim / 2;
            let mut out = vec![0f32; a.head_dim];
            let base = a.rope_theta as f64;
            for i in 0..half {
                let inv = 1.0 / base.powf(2.0 * i as f64 / a.head_dim as f64);
                let freq = if let Some(factors) = a.rope_freq_factors.as_deref() {
                    // GGUF's resolved tensor is authoritative over optional
                    // metadata scaling, matching the source Llama loader.
                    inv / factors[i] as f64
                } else {
                    match a.rope_scaling {
                        None => inv,
                        Some(s) => {
                            let orig = s.original_max_position_embeddings as f64;
                            let (low, high) = (s.low_freq_factor as f64, s.high_freq_factor as f64);
                            let wavelen = std::f64::consts::TAU / inv;
                            if wavelen < orig / high {
                                inv
                            } else if wavelen > orig / low {
                                inv / s.factor as f64
                            } else {
                                let smooth = (orig / wavelen - low) / (high - low);
                                (1.0 - smooth) * (inv / s.factor as f64) + smooth * inv
                            }
                        }
                    }
                };
                let theta = pos as f64 * freq;
                out[i] = theta.cos() as f32;
                out[half + i] = theta.sin() as f32;
            }
            out
        }

        /// Encode one packed matvec into an open command buffer, writing at
        /// `y_offset` f32 elements into `y`. The offset is what lets the K
        /// and V projections write straight into their layer's KV cache
        /// slot: no append kernel, no copy, no round trip through the host.
        fn encode_matvec_at(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            name: &str,
            x: &Buffer,
            y: &Buffer,
            y_offset: usize,
        ) -> Result<()> {
            const TG: u32 = 256;
            let byte_off = (y_offset * std::mem::size_of::<f32>()) as u64;
            match self
                .weights
                .get(name)
                .ok_or_else(|| Error::Gravity(format!("artifact has no tensor {name:?}")))?
            {
                GpuWeight::Pq(w) => {
                    let n_tg = w.params.rows.div_ceil(8);
                    let params = w.params;
                    tcb.dispatch_threads(
                        "gravity_pq_matvec",
                        (n_tg * TG, 1, 1),
                        (TG, 1, 1),
                        |enc| {
                            enc.set_buffer(0, Some(&w.codebooks), 0);
                            enc.set_buffer(1, Some(&w.codes), 0);
                            enc.set_buffer(2, Some(x), 0);
                            enc.set_buffer(3, Some(y), byte_off);
                            enc.set_bytes(
                                4,
                                std::mem::size_of::<PqParams>() as u64,
                                &params as *const PqParams as *const _,
                            );
                        },
                    )
                }
                GpuWeight::ResidualPq(w) => {
                    let n_tg = w.params.rows.div_ceil(8);
                    let params = w.params;
                    tcb.dispatch_threads(
                        "gravity_residual_pq_matvec",
                        (n_tg * TG, 1, 1),
                        (TG, 1, 1),
                        |enc| {
                            enc.set_buffer(0, Some(&w.codebooks), 0);
                            enc.set_buffer(1, Some(&w.codes), 0);
                            enc.set_buffer(2, Some(x), 0);
                            enc.set_buffer(3, Some(y), byte_off);
                            enc.set_bytes(
                                4,
                                std::mem::size_of::<ResidualPqParams>() as u64,
                                &params as *const ResidualPqParams as *const _,
                            );
                        },
                    )
                }
                GpuWeight::RawQuant(w) => match w.kind {
                    RawQuantKind::Q4K => self.encode_q4_matvec(tcb, w, x, y, y_offset),
                    RawQuantKind::Q5K =>
                        crate::kernels::gemv_q5_k_serial_authority_pinned_off_tcb(
                            tcb,
                            &w.data,
                            0,
                            w.bytes,
                            w.rows,
                            w.cols,
                            x,
                            0,
                            y,
                            y_offset * std::mem::size_of::<f32>(),
                        ),
                    RawQuantKind::Q6K => crate::kernels::gemv_q6_k_llama_b9430_pinned_off_tcb(
                        tcb,
                        &w.data,
                        0,
                        w.bytes,
                        w.rows,
                        w.cols,
                        x,
                        0,
                        y,
                        y_offset * std::mem::size_of::<f32>(),
                    ),
                    RawQuantKind::Q5_0 | RawQuantKind::Q8_0 => {
                        Self::encode_raw32_matvec(tcb, w, x, y, y_offset)
                    }
                },
                GpuWeight::Dense(_) => Err(Error::Gravity(format!(
                    "tensor {name:?} is dense; expected a packed projection"
                ))),
            }
        }

        /// Direct source-packed 32-element Q5_0/Q8_0 matvec.  These codecs
        /// occur in Qwen2 source-compatible artifacts; executing them here
        /// keeps their source bytes and the activation on the device instead
        /// of widening a tensor or silently switching to CPU.
        fn encode_raw32_matvec(
            tcb: &mut TokenCommandBuffer<'_>,
            w: &GpuRawQuant,
            x: &Buffer,
            y: &Buffer,
            y_offset: usize,
        ) -> Result<()> {
            #[repr(C)]
            #[derive(Clone, Copy)]
            struct Raw32Params {
                rows: u32,
                cols: u32,
            }
            let kernel = match w.kind {
                RawQuantKind::Q5_0 => "gravity_raw_q5_0_matvec",
                RawQuantKind::Q8_0 => "gravity_raw_q8_0_matvec",
                RawQuantKind::Q4K | RawQuantKind::Q5K | RawQuantKind::Q6K => {
                    return Err(Error::Gravity(
                        "raw32 dispatcher received a K-quant tensor".into(),
                    ));
                }
            };
            let rows = u32::try_from(w.rows)
                .map_err(|_| Error::Gravity("raw32 rows exceed u32".into()))?;
            let cols = u32::try_from(w.cols)
                .map_err(|_| Error::Gravity("raw32 cols exceed u32".into()))?;
            let params = Raw32Params { rows, cols };
            const TG: u32 = 256;
            let n_tg = rows.div_ceil(8);
            let y_byte_offset = y_offset
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| Error::Gravity("raw32 output offset overflow".into()))?;
            tcb.dispatch_threads(kernel, (n_tg * TG, 1, 1), (TG, 1, 1), |enc| {
                enc.set_buffer(0, Some(&w.data), 0);
                enc.set_buffer(1, Some(x), 0);
                enc.set_buffer(2, Some(y), y_byte_offset as u64);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<Raw32Params>() as u64,
                    &params as *const Raw32Params as *const _,
                );
            })
        }

        /// Encode `gravity_rope_table_f32` over `n_heads` heads starting at
        /// f32 element `offset` of `x`.
        fn encode_rope(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            x: &Buffer,
            offset: usize,
            n_heads: usize,
            table: &Buffer,
        ) -> Result<()> {
            #[repr(C)]
            #[derive(Clone, Copy)]
            struct RopeParams {
                offset: u32,
                n_heads: u32,
                head_dim: u32,
                interleaved: u32,
            }
            let head_dim = self.arch.head_dim;
            let total = (n_heads * head_dim / 2) as u32;
            let tg = 64u32.min(total.max(1));
            let p = RopeParams {
                offset: offset as u32,
                n_heads: n_heads as u32,
                head_dim: head_dim as u32,
                interleaved: self.arch.rope_interleaved as u32,
            };
            tcb.dispatch_threads(
                "gravity_rope_table_f32",
                (total.div_ceil(tg) * tg, 1, 1),
                (tg, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(x), 0);
                    enc.set_buffer(1, Some(table), 0);
                    enc.set_bytes(
                        2,
                        std::mem::size_of::<RopeParams>() as u64,
                        &p as *const RopeParams as *const _,
                    );
                },
            )
        }

        fn encode_silu_mul(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            gate: &Buffer,
            up: &Buffer,
            out: &Buffer,
            n: usize,
        ) -> Result<()> {
            const TG: u32 = 256;
            let n_u32 = n as u32;
            tcb.dispatch_threads(
                "gravity_silu_mul_f32",
                (n_u32.div_ceil(TG) * TG, 1, 1),
                (TG, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(gate), 0);
                    enc.set_buffer(1, Some(up), 0);
                    enc.set_buffer(2, Some(out), 0);
                    enc.set_bytes(3, 4, &n_u32 as *const u32 as *const _);
                },
            )
        }

        /// Fuse the two raw-Q4 FFN projections when the source geometry
        /// admits the strict paired kernel.  Other Gravity grammars keep the
        /// generic two-dispatch path, so this is a capability-gated topology
        /// optimization rather than a silent representation substitution.
        ///
        /// The pair kernel is deliberately opt-in at runtime.  It is bitwise
        /// source-compatible, but on the current Metal device its extra
        /// register pressure is slower than two independent strict Q4 GEMVs;
        /// the default therefore preserves measured TPS, while
        /// `HAWKING_GRAVITY_FUSE_GATE_UP=1` remains available for comparison.
        fn encode_raw_q4_pair(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            gate_name: &str,
            up_name: &str,
            x: &Buffer,
            gate_out: &Buffer,
            up_out: &Buffer,
        ) -> Result<bool> {
            let gate = self.weights.get(gate_name);
            let up = self.weights.get(up_name);
            let (Some(GpuWeight::RawQuant(gate)), Some(GpuWeight::RawQuant(up))) = (gate, up)
            else {
                return Ok(false);
            };
            if gate.kind != RawQuantKind::Q4K
                || up.kind != RawQuantKind::Q4K
                || gate.rows != up.rows
                || gate.cols != up.cols
            {
                return Ok(false);
            }
            crate::kernels::gemv_q4_k_m_llama_b9430_pair_pinned_tcb(
                tcb, &gate.data, 0, gate.bytes, &up.data, 0, up.bytes, gate.rows, gate.cols, x,
                gate_out, up_out,
            )?;
            Ok(true)
        }

        /// Pair exact source Q5_0 gate/up projections. Qwen-family GGUFs
        /// commonly retain this legacy 32-element grammar, so widening or
        /// rewriting the weights would make a false Gravity win. Ineligible
        /// pairs return false and retain the regular source-packed path.
        fn encode_raw_q5_pair(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            gate_name: &str,
            up_name: &str,
            x: &Buffer,
            gate_out: &Buffer,
            up_out: &Buffer,
        ) -> Result<bool> {
            let gate = self.weights.get(gate_name);
            let up = self.weights.get(up_name);
            let (Some(GpuWeight::RawQuant(gate)), Some(GpuWeight::RawQuant(up))) = (gate, up)
            else {
                return Ok(false);
            };
            if gate.kind != RawQuantKind::Q5_0
                || up.kind != RawQuantKind::Q5_0
                || gate.rows != up.rows
                || gate.cols != up.cols
            {
                return Ok(false);
            }
            #[repr(C)]
            #[derive(Clone, Copy)]
            struct RawQ5PairParams {
                rows: u32,
                cols: u32,
            }
            let params = RawQ5PairParams {
                rows: u32::try_from(gate.rows)
                    .map_err(|_| Error::Gravity("raw Q5 pair rows exceed u32".into()))?,
                cols: u32::try_from(gate.cols)
                    .map_err(|_| Error::Gravity("raw Q5 pair cols exceed u32".into()))?,
            };
            let groups_per_wave = gate.rows.div_ceil(8);
            let total_groups = groups_per_wave
                .checked_mul(2)
                .ok_or_else(|| Error::Gravity("raw Q5 pair threadgroup count overflow".into()))?;
            const TG: u32 = 256;
            let grid = u32::try_from(total_groups)
                .map_err(|_| Error::Gravity("raw Q5 pair grid exceeds u32".into()))?
                .checked_mul(TG)
                .ok_or_else(|| Error::Gravity("raw Q5 pair grid overflow".into()))?;
            tcb.dispatch_threads(
                "gravity_raw_q5_0_pair_matvec",
                (grid, 1, 1),
                (TG, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&gate.data), 0);
                    enc.set_buffer(1, Some(&up.data), 0);
                    enc.set_buffer(2, Some(x), 0);
                    enc.set_buffer(3, Some(gate_out), 0);
                    enc.set_buffer(4, Some(up_out), 0);
                    enc.set_bytes(
                        5,
                        std::mem::size_of::<RawQ5PairParams>() as u64,
                        &params as *const RawQ5PairParams as *const _,
                    );
                },
            )?;
            Ok(true)
        }

        /// Encode source Q/K/V Q4_K projections, RoPE, and the current
        /// position's KV append in one Metal dispatch. This candidate is
        /// deliberately narrow: all three projections must be raw Q4_K with
        /// f32 predecoded scale tables, and the artifact must use a scalar
        /// RoPE base. A caller that cannot satisfy those conditions falls back
        /// to the already-parity-gated path instead of silently changing the
        /// model graph.
        #[allow(clippy::too_many_arguments)]
        fn encode_source_qkv_rope_append(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
            x: &Buffer,
            rope: &Buffer,
            q_buf: &Buffer,
            k_cache: &Buffer,
            v_cache: &Buffer,
            pos: u32,
            kv_off: usize,
        ) -> Result<bool> {
            if !self.fused_source_qkv {
                return Ok(false);
            }
            let p = format!("model.layers.{layer}.self_attn.");
            let q_name = format!("{p}q_proj.weight");
            let k_name = format!("{p}k_proj.weight");
            let v_name = format!("{p}v_proj.weight");
            let (
                Some(GpuWeight::RawQuant(q)),
                Some(GpuWeight::RawQuant(k)),
                Some(GpuWeight::RawQuant(v)),
            ) = (
                self.weights.get(&q_name),
                self.weights.get(&k_name),
                self.weights.get(&v_name),
            )
            else {
                return Ok(false);
            };
            // Qwen2's source Q/K are Q5_0 and its V is Q5_0 or Q8_0 in
            // this family of GGUFs. Execute that exact mixed grammar in one
            // projection/RoPE/KV wave; every other combination keeps the
            // established source-Q4 path or decomposed fallback below.
            if q.kind == RawQuantKind::Q5_0
                && k.kind == RawQuantKind::Q5_0
                && matches!(v.kind, RawQuantKind::Q5_0 | RawQuantKind::Q8_0)
            {
                return self.encode_raw_q5q5qv_rope_append(
                    tcb, layer, q, k, v, x, rope, q_buf, k_cache, v_cache, kv_off,
                );
            }
            if q.kind != RawQuantKind::Q4K
                || k.kind != RawQuantKind::Q4K
                || q.rows != self.arch.n_heads * self.arch.head_dim
                || k.rows != self.arch.n_kv_heads * self.arch.head_dim
                || v.rows != self.arch.n_kv_heads * self.arch.head_dim
                || q.cols != self.arch.hidden
                || k.cols != self.arch.hidden
                || v.cols != self.arch.hidden
            {
                return Ok(false);
            }
            let (Some(q_scales), Some(k_scales)) =
                (q.predec_scales.as_ref(), k.predec_scales.as_ref())
            else {
                return Ok(false);
            };
            // Qwen2 carries projection biases while Llama and Mistral do
            // not.  The fused grammar takes nullable bias buffers, so retain
            // those source values in the fused path rather than silently
            // dropping them or requiring Qwen to stay decomposed forever.
            let q_bias = self.optional_bias(&format!("{p}q_proj.bias"))?;
            let k_bias = self.optional_bias(&format!("{p}k_proj.bias"))?;
            let v_bias = self.optional_bias(&format!("{p}v_proj.bias"))?;
            if v.kind == RawQuantKind::Q6K {
                crate::kernels::gemv_q4k_q4k_q6k_rope_append_buffers_pinned_tcb(
                    tcb,
                    &q.data,
                    q.bytes,
                    q_scales,
                    &k.data,
                    k.bytes,
                    k_scales,
                    &v.data,
                    v.bytes,
                    q.rows,
                    k.rows,
                    q.cols,
                    self.arch.n_heads,
                    self.arch.n_kv_heads,
                    self.arch.head_dim,
                    pos,
                    self.arch.rope_theta,
                    kv_off,
                    x,
                    q_buf,
                    q_bias,
                    k_bias,
                    v_bias,
                    k_cache,
                    v_cache,
                    self.arch.rope_interleaved,
                )?;
            } else if v.kind == RawQuantKind::Q4K {
                let Some(v_scales) = v.predec_scales.as_ref() else {
                    return Ok(false);
                };
                crate::kernels::gemv_q4k_predec_qkv_rope_append_4r_buffers_pinned_tcb(
                    tcb,
                    &q.data,
                    q.bytes,
                    q_scales,
                    &k.data,
                    k.bytes,
                    k_scales,
                    &v.data,
                    v.bytes,
                    v_scales,
                    q.rows,
                    k.rows,
                    q.cols,
                    self.arch.n_heads,
                    self.arch.n_kv_heads,
                    self.arch.head_dim,
                    pos,
                    self.arch.rope_theta,
                    kv_off,
                    x,
                    q_buf,
                    q_bias,
                    k_bias,
                    v_bias,
                    k_cache,
                    v_cache,
                    self.arch.rope_interleaved,
                )?;
            } else {
                return Ok(false);
            }
            Ok(true)
        }

        /// Exact source Q5_0/Q5_0/(Q5_0|Q8_0) Qwen projection wave. It is
        /// intentionally narrow: source layout, bias buffers and precomputed
        /// f64-derived RoPE table all remain the authority, so this is a
        /// command-topology change rather than a quantization conversion.
        #[allow(clippy::too_many_arguments)]
        fn encode_raw_q5q5qv_rope_append(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
            q: &GpuRawQuant,
            k: &GpuRawQuant,
            v: &GpuRawQuant,
            x: &Buffer,
            rope: &Buffer,
            q_buf: &Buffer,
            k_cache: &Buffer,
            v_cache: &Buffer,
            kv_off: usize,
        ) -> Result<bool> {
            if q.rows != self.arch.n_heads * self.arch.head_dim
                || k.rows != self.arch.n_kv_heads * self.arch.head_dim
                || v.rows != self.arch.n_kv_heads * self.arch.head_dim
                || q.cols != self.arch.hidden
                || k.cols != self.arch.hidden
                || v.cols != self.arch.hidden
                || self.arch.head_dim == 0
                || self.arch.head_dim % 2 != 0
                || kv_off > u32::MAX as usize
            {
                return Ok(false);
            }
            #[repr(C)]
            #[derive(Clone, Copy)]
            struct RawQ5Q5QvParams {
                q_rows: u32,
                kv_rows: u32,
                cols: u32,
                kv_off: u32,
                head_dim: u32,
                has_q_bias: u32,
                has_k_bias: u32,
                has_v_bias: u32,
                v_is_q8: u32,
            }
            let prefix = format!("model.layers.{layer}.self_attn.");
            let q_bias = self.optional_bias(&format!("{prefix}q_proj.bias"))?;
            let k_bias = self.optional_bias(&format!("{prefix}k_proj.bias"))?;
            let v_bias = self.optional_bias(&format!("{prefix}v_proj.bias"))?;
            let params = RawQ5Q5QvParams {
                q_rows: u32::try_from(q.rows)
                    .map_err(|_| Error::Gravity("raw Q5 Q rows exceed u32".into()))?,
                kv_rows: u32::try_from(k.rows)
                    .map_err(|_| Error::Gravity("raw Q5 KV rows exceed u32".into()))?,
                cols: u32::try_from(q.cols)
                    .map_err(|_| Error::Gravity("raw Q5 cols exceed u32".into()))?,
                kv_off: kv_off as u32,
                head_dim: u32::try_from(self.arch.head_dim)
                    .map_err(|_| Error::Gravity("raw Q5 head dim exceeds u32".into()))?,
                has_q_bias: q_bias.is_some() as u32,
                has_k_bias: k_bias.is_some() as u32,
                has_v_bias: v_bias.is_some() as u32,
                v_is_q8: (v.kind == RawQuantKind::Q8_0) as u32,
            };
            let q_bias_buffer = q_bias.unwrap_or(q_buf);
            let k_bias_buffer = k_bias.unwrap_or(q_buf);
            let v_bias_buffer = v_bias.unwrap_or(q_buf);
            let q_pairs = q.rows / 2;
            let k_pairs = k.rows / 2;
            let q_tg = q_pairs.div_ceil(8);
            let k_tg = k_pairs.div_ceil(8);
            let v_tg = v.rows.div_ceil(8);
            let total_tg = q_tg
                .checked_add(k_tg)
                .and_then(|value| value.checked_add(v_tg))
                .ok_or_else(|| Error::Gravity("raw Q5 QKV threadgroup count overflow".into()))?;
            const TG: u32 = 256;
            tcb.dispatch_threads(
                "gravity_raw_q5q5qv_rope_append",
                (
                    u32::try_from(total_tg)
                        .map_err(|_| Error::Gravity("raw Q5 QKV grid exceeds u32".into()))?
                        * TG,
                    1,
                    1,
                ),
                (TG, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&q.data), 0);
                    enc.set_buffer(1, Some(&k.data), 0);
                    enc.set_buffer(2, Some(&v.data), 0);
                    enc.set_buffer(3, Some(x), 0);
                    enc.set_buffer(4, Some(q_buf), 0);
                    enc.set_buffer(5, Some(k_cache), 0);
                    enc.set_buffer(6, Some(v_cache), 0);
                    enc.set_buffer(7, Some(rope), 0);
                    enc.set_buffer(8, Some(q_bias_buffer), 0);
                    enc.set_buffer(9, Some(k_bias_buffer), 0);
                    enc.set_buffer(10, Some(v_bias_buffer), 0);
                    enc.set_bytes(
                        11,
                        std::mem::size_of::<RawQ5Q5QvParams>() as u64,
                        &params as *const RawQ5Q5QvParams as *const _,
                    );
                },
            )?;
            Ok(true)
        }

        /// Encode one packed matvec into an open command buffer. Dispatches
        /// encoded into the same buffer must write disjoint outputs.
        fn encode_matvec(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            name: &str,
            x: &Buffer,
            y: &Buffer,
        ) -> Result<()> {
            // One SIMD group (32 lanes) per output row, 8 SIMD groups (256
            // threads) per threadgroup; the kernel guards the tail.
            const TG: u32 = 256;
            match self
                .weights
                .get(name)
                .ok_or_else(|| Error::Gravity(format!("artifact has no tensor {name:?}")))?
            {
                GpuWeight::Pq(w) => {
                    let n_tg = w.params.rows.div_ceil(8);
                    let params = w.params;
                    tcb.dispatch_threads(
                        "gravity_pq_matvec",
                        (n_tg * TG, 1, 1),
                        (TG, 1, 1),
                        |enc| {
                            enc.set_buffer(0, Some(&w.codebooks), 0);
                            enc.set_buffer(1, Some(&w.codes), 0);
                            enc.set_buffer(2, Some(x), 0);
                            enc.set_buffer(3, Some(y), 0);
                            enc.set_bytes(
                                4,
                                std::mem::size_of::<PqParams>() as u64,
                                &params as *const PqParams as *const _,
                            );
                        },
                    )
                }
                GpuWeight::ResidualPq(w) => {
                    let n_tg = w.params.rows.div_ceil(8);
                    let params = w.params;
                    tcb.dispatch_threads(
                        "gravity_residual_pq_matvec",
                        (n_tg * TG, 1, 1),
                        (TG, 1, 1),
                        |enc| {
                            enc.set_buffer(0, Some(&w.codebooks), 0);
                            enc.set_buffer(1, Some(&w.codes), 0);
                            enc.set_buffer(2, Some(x), 0);
                            enc.set_buffer(3, Some(y), 0);
                            enc.set_bytes(
                                4,
                                std::mem::size_of::<ResidualPqParams>() as u64,
                                &params as *const ResidualPqParams as *const _,
                            );
                        },
                    )
                }
                GpuWeight::RawQuant(w) => match w.kind {
                    RawQuantKind::Q4K => self.encode_q4_matvec(tcb, w, x, y, 0),
                    RawQuantKind::Q5K => crate::kernels::gemv_q5_k_serial_authority_pinned_tcb(
                        tcb, &w.data, 0, w.bytes, w.rows, w.cols, x, y,
                    ),
                    RawQuantKind::Q6K => crate::kernels::gemv_q6_k_llama_b9430_pinned_tcb(
                        tcb, &w.data, 0, w.bytes, w.rows, w.cols, x, y,
                    ),
                    RawQuantKind::Q5_0 | RawQuantKind::Q8_0 => {
                        Self::encode_raw32_matvec(tcb, w, x, y, 0)
                    }
                },
                GpuWeight::Dense(_) => Err(Error::Gravity(format!(
                    "tensor {name:?} is dense; expected a packed projection"
                ))),
            }
        }

        /// Run `tokens` from an empty KV cache; returns the logits after the
        /// final token, plus what the run cost.
        pub fn forward(&self, tokens: &[u32]) -> Result<(Vec<f32>, ForwardStats)> {
            self.forward_at(tokens, 0)
        }

        /// Run `tokens` starting at position `start_pos`, keeping whatever the
        /// cache already holds below it.
        ///
        /// This is what makes generation linear instead of quadratic. Each
        /// position writes its own cache slot, so continuing a sequence is a
        /// matter of not resetting: replaying the prefix would recompute
        /// identical keys and values and reach identical logits, just slower.
        pub fn forward_at(
            &self,
            tokens: &[u32],
            start_pos: usize,
        ) -> Result<(Vec<f32>, ForwardStats)> {
            if tokens.is_empty() {
                return Err(Error::Gravity("forward: no tokens".into()));
            }
            let a = &self.arch;
            let kv_width = a.n_kv_heads * a.head_dim;
            let inter = self.packed_rows("model.layers.0.mlp.gate_proj.weight")?;
            // KV precision is a representation choice, not an architecture
            // default.  The direct GGUF exact lane keeps f32 K/V unless it is
            // explicitly configured otherwise; silently forcing Qwen2 into
            // f16 here makes an artifact/source comparison diverge after the
            // first attention-dependent token.  An explicit environment value
            // selects f16 or f32; absent a request, retain f32 source parity.
            let use_f16_kv = std::env::var_os("HAWKING_GRAVITY_F16_KV")
                .map(|_| crate::env_on("HAWKING_GRAVITY_F16_KV"))
                .unwrap_or(false);

            let x_buf = self.pool.get(&self.ctx, "x", a.hidden);
            let x_norm_buf = self.pool.get(&self.ctx, "x_norm", a.hidden);
            let q_buf = self.pool.get(&self.ctx, "q", a.n_heads * a.head_dim);
            let attn_buf = self.pool.get(&self.ctx, "attn", a.n_heads * a.head_dim);
            let gate_buf = self.pool.get(&self.ctx, "gate", inter);
            let up_buf = self.pool.get(&self.ctx, "up", inter);
            let act_buf = self.pool.get(&self.ctx, "act", inter);
            let o_buf = self.pool.get(&self.ctx, "o", a.hidden);
            let rope_buf = self.pool.get(&self.ctx, "rope", a.head_dim);
            let logits_buf = self.pool.get(&self.ctx, "logits", a.vocab_size);

            self.reserve_kv(start_pos + tokens.len())?;
            let kv = self.kv.lock().expect("kv mutex");
            let zeros = vec![0f32; a.hidden];
            let mut logits = Vec::new();
            let mut stats = ForwardStats {
                tokens: tokens.len(),
                ..Default::default()
            };
            // Diagnostic-only layer trace. It commits one command buffer per
            // layer and reads a bounded set of scalar surfaces so a source
            // parity failure can be localized without changing the normal
            // throughput path. Never set this in production measurements.
            let trace_path = std::env::var_os("HAWKING_GRAVITY_TRACE_PATH");
            // An optional position filter prevents every prefill token from
            // paying the per-layer diagnostic commits when localizing one
            // source/Gravity divergence.  The trace remains one file with the
            // selected rows only; unset retains the historical all-position
            // diagnostic behavior.
            let trace_position = std::env::var("HAWKING_GRAVITY_TRACE_POSITION")
                .ok()
                .and_then(|value| value.parse::<usize>().ok());
            let mut trace_rows = Vec::<serde_json::Value>::new();

            let t_start = Instant::now();
            let mut t_token = t_start;
            for (step, &token) in tokens.iter().enumerate() {
                let pos = start_pos + step;
                let trace_enabled = trace_path.is_some()
                    && trace_position.map(|wanted| wanted == pos).unwrap_or(true);
                if token as usize >= a.vocab_size {
                    return Err(Error::Gravity(format!(
                        "token {token} out of range for vocab_size {}",
                        a.vocab_size
                    )));
                }
                // The only host work per token: decode the embedding row and
                // the position's rotation table. Everything after this is one
                // command buffer.
                let embed = if let Some(kind) = self.embed_raw_quant {
                    raw_quant_row(
                        &self.embed_payload,
                        token as usize,
                        a.vocab_size,
                        a.hidden,
                        kind,
                    )?
                } else if self.embed_is_residual_pq {
                    ResidualPqTensor::from_payload(&self.embed_payload)?.row(token as usize)?
                } else {
                    pq_row(&self.embed_payload, token as usize)?
                };
                write_f32(&x_buf, &embed);
                write_f32(&rope_buf, &self.rope_table(pos));
                // Layer 0's residual add has nothing to add yet, so it adds
                // zero rather than needing a separate un-fused norm.
                write_f32(&o_buf, &zeros);
                let seq_len = pos + 1;

                let mut tcb = TokenCommandBuffer::new(&self.ctx);
                for layer in 0..a.n_layers {
                    let p = format!("model.layers.{layer}.");

                    // x += previous residual output, then normalize into
                    // x_norm. One dispatch, one DRAM pass over x.
                    crate::kernels::add_rmsnorm_fused_tcb(
                        &mut tcb,
                        &x_buf,
                        &o_buf,
                        self.dense(&format!("{p}input_layernorm.weight"))?,
                        &x_norm_buf,
                        a.rms_norm_eps,
                        a.hidden,
                    )?;

                    // The opt-in source-QKV candidate writes Q and the
                    // RoPE'd K/V cache slot in one dispatch. Trace mode stays
                    // on the decomposed path so its raw projection surfaces
                    // remain meaningful.
                    let fused_qkv = if trace_enabled {
                        false
                    } else {
                        self.encode_source_qkv_rope_append(
                            &mut tcb,
                            layer,
                            &x_norm_buf,
                            &rope_buf,
                            &q_buf,
                            &kv.k[layer],
                            &kv.v[layer],
                            pos as u32,
                            pos * kv_width,
                        )?
                    };
                    if fused_qkv {
                        stats.fused_qkv_dispatches = stats.fused_qkv_dispatches.saturating_add(1);
                    }
                    if !fused_qkv {
                        // K and V project straight into this layer's cache
                        // slot: no append kernel, no copy.
                        self.encode_matvec(
                            &mut tcb,
                            &format!("{p}self_attn.q_proj.weight"),
                            &x_norm_buf,
                            &q_buf,
                        )?;
                        self.add_optional_bias(
                            &mut tcb,
                            &q_buf,
                            &format!("{p}self_attn.q_proj.bias"),
                            a.n_heads * a.head_dim,
                        )?;
                        self.encode_matvec_at(
                            &mut tcb,
                            &format!("{p}self_attn.k_proj.weight"),
                            &x_norm_buf,
                            &kv.k[layer],
                            pos * kv_width,
                        )?;
                        if let Some(bias) =
                            self.optional_bias(&format!("{p}self_attn.k_proj.bias"))?
                        {
                            // K is written directly into the current absolute
                            // cache slot, just like V.  A non-offset add would
                            // repeatedly perturb row zero and leave later
                            // positions unbiased.
                            crate::kernels::add_inplace_metal_off_tcb(
                                &mut tcb,
                                &kv.k[layer],
                                bias,
                                pos * kv_width,
                                kv_width,
                            )?;
                        }
                        self.encode_matvec_at(
                            &mut tcb,
                            &format!("{p}self_attn.v_proj.weight"),
                            &x_norm_buf,
                            &kv.v[layer],
                            pos * kv_width,
                        )?;
                        if let Some(bias) =
                            self.optional_bias(&format!("{p}self_attn.v_proj.bias"))?
                        {
                            // V is written at the current absolute cache slot, so
                            // use the offset-capable add kernel rather than adding
                            // a bias to row zero.
                            crate::kernels::add_inplace_metal_off_tcb(
                                &mut tcb,
                                &kv.v[layer],
                                bias,
                                pos * kv_width,
                                kv_width,
                            )?;
                        }
                    }

                    // In the diagnostic trace only, split the projection
                    // surface from RoPE.  Qwen-family biases are applied by
                    // the generic artifact path immediately after each GEMV,
                    // so these values are explicitly *post-bias*; do not
                    // compare them to a pre-bias trace surface.  This extra
                    // commit is never used by the throughput path.
                    let (
                        q_raw_sum,
                        k_raw_sum,
                        v_raw_sum,
                        attn_norm_sum,
                        q_raw_head,
                        k_raw_head,
                        v_raw_head,
                        attn_norm_head,
                    ) = if trace_enabled {
                        let projection_dispatches = tcb.dispatch_count();
                        tcb.commit_and_wait()?;
                        stats.dispatches += projection_dispatches;
                        stats.command_buffers += 1;
                        let q_raw = read_f32(&q_buf, a.n_heads * a.head_dim);
                        let k_raw = read_f32_at(&kv.k[layer], pos * kv_width, kv_width);
                        let v_raw = read_f32_at(&kv.v[layer], pos * kv_width, kv_width);
                        let values = (
                            q_raw.iter().map(|v| *v as f64).sum::<f64>(),
                            k_raw.iter().map(|v| *v as f64).sum::<f64>(),
                            v_raw.iter().map(|v| *v as f64).sum::<f64>(),
                            read_f32(&x_norm_buf, a.hidden)
                                .iter()
                                .map(|v| *v as f64)
                                .sum::<f64>(),
                            q_raw[..8.min(q_raw.len())].to_vec(),
                            k_raw[..8.min(k_raw.len())].to_vec(),
                            v_raw[..8.min(v_raw.len())].to_vec(),
                            read_f32(&x_norm_buf, 8),
                        );
                        tcb = TokenCommandBuffer::new(&self.ctx);
                        values
                    } else {
                        (
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            Vec::new(),
                            Vec::new(),
                            Vec::new(),
                            Vec::new(),
                        )
                    };

                    if !fused_qkv {
                        self.encode_rope(&mut tcb, &q_buf, 0, a.n_heads, &rope_buf)?;
                        self.encode_rope(
                            &mut tcb,
                            &kv.k[layer],
                            pos * kv_width,
                            a.n_kv_heads,
                            &rope_buf,
                        )?;
                    }

                    // Mistral's sliding-window cache remains physically
                    // contiguous and absolute-positioned; attention simply
                    // starts at the oldest visible token.  Llama leaves the
                    // start at zero.  The K/V writes above always use the
                    // absolute `pos` slot, so state advance is unchanged.
                    let (window_start, attention_len) = attention_window(seq_len, a.sliding_window);
                    if use_f16_kv {
                        // The f16 cache is the source SET_ROWS image.  The
                        // append rounds both newly RoPE'd vectors once, then
                        // flash attention widens them in-register.  Keep the
                        // f32 image above for diagnostics and the Llama/Mistral
                        // default path; Qwen2 uses this source-compatible
                        // image automatically.
                        crate::kernels::llama_b9430_cache_append_kv_f16_off_tcb(
                            &mut tcb,
                            &kv.k[layer],
                            &kv.v[layer],
                            &kv.k_f16[layer],
                            &kv.v_f16[layer],
                            pos * kv_width,
                            pos * kv_width,
                            kv_width,
                        )?;
                        let cache_offset_bytes =
                            window_start * kv_width * std::mem::size_of::<half::f16>();
                        crate::kernels::mha_decode_flash_f16kv_tcb(
                            &mut tcb,
                            &q_buf,
                            &kv.k_f16[layer],
                            cache_offset_bytes,
                            &kv.v_f16[layer],
                            cache_offset_bytes,
                            &attn_buf,
                            attention_len,
                            a.head_dim,
                            a.n_heads,
                            a.n_kv_heads,
                        )?;
                    } else {
                        let cache_offset_bytes =
                            window_start * kv_width * std::mem::size_of::<f32>();
                        crate::kernels::mha_decode_flash_f32_tcb(
                            &mut tcb,
                            &q_buf,
                            &kv.k[layer],
                            cache_offset_bytes,
                            &kv.v[layer],
                            cache_offset_bytes,
                            &attn_buf,
                            attention_len,
                            a.head_dim,
                            a.n_heads,
                            a.n_kv_heads,
                        )?;
                    }
                    self.encode_matvec(
                        &mut tcb,
                        &format!("{p}self_attn.o_proj.weight"),
                        &attn_buf,
                        &o_buf,
                    )?;

                    crate::kernels::add_rmsnorm_fused_tcb(
                        &mut tcb,
                        &x_buf,
                        &o_buf,
                        self.dense(&format!("{p}post_attention_layernorm.weight"))?,
                        &x_norm_buf,
                        a.rms_norm_eps,
                        a.hidden,
                    )?;
                    let gate_name = format!("{p}mlp.gate_proj.weight");
                    let up_name = format!("{p}mlp.up_proj.weight");
                    let fused_gate_up = crate::env_on("HAWKING_GRAVITY_FUSE_GATE_UP")
                        && (self.encode_raw_q5_pair(
                            &mut tcb,
                            &gate_name,
                            &up_name,
                            &x_norm_buf,
                            &gate_buf,
                            &up_buf,
                        )? || self.encode_raw_q4_pair(
                            &mut tcb,
                            &gate_name,
                            &up_name,
                            &x_norm_buf,
                            &gate_buf,
                            &up_buf,
                        )?);
                    if fused_gate_up {
                        stats.fused_gate_up_dispatches =
                            stats.fused_gate_up_dispatches.saturating_add(1);
                    } else {
                        self.encode_matvec(&mut tcb, &gate_name, &x_norm_buf, &gate_buf)?;
                        self.encode_matvec(&mut tcb, &up_name, &x_norm_buf, &up_buf)?;
                    }
                    self.encode_silu_mul(&mut tcb, &gate_buf, &up_buf, &act_buf, inter)?;
                    self.encode_matvec(
                        &mut tcb,
                        &format!("{p}mlp.down_proj.weight"),
                        &act_buf,
                        &o_buf,
                    )?;

                    if trace_enabled {
                        stats.dispatches += tcb.dispatch_count();
                        tcb.commit_and_wait()?;
                        stats.command_buffers += 1;
                        let cache_offset = pos * kv_width;
                        let k = read_f32_at(&kv.k[layer], cache_offset, kv_width);
                        let x = read_f32(&x_buf, a.hidden);
                        let ffn_out = read_f32(&o_buf, a.hidden);
                        let gate_values = read_f32(&gate_buf, inter);
                        let up_values = read_f32(&up_buf, inter);
                        let act_values = read_f32(&act_buf, inter);
                        let ffn_norm_values = read_f32(&x_norm_buf, a.hidden);
                        trace_rows.push(serde_json::json!({
                            "position": pos,
                            "token_id": token,
                            "layer": layer,
                            "qkv_trace_stage": "post_projection_post_bias",
                            "q_rope_sum": read_f32(&q_buf, a.n_heads * a.head_dim)
                                .iter().map(|v| *v as f64).sum::<f64>(),
                            "q_raw_sum": q_raw_sum,
                            "q_raw_head": q_raw_head,
                            "k_raw_sum": k_raw_sum,
                            "k_raw_head": k_raw_head,
                            "k_rope_sum": k.iter().map(|v| *v as f64).sum::<f64>(),
                            "v_raw_sum": v_raw_sum,
                            "v_raw_head": v_raw_head,
                            "attn_norm_sum": attn_norm_sum,
                            "attn_norm_head": attn_norm_head,
                            "ffn_input_sum": x.iter().map(|v| *v as f64).sum::<f64>(),
                            "ffn_input_head": &x[..8.min(x.len())],
                            "ffn_norm_sum": ffn_norm_values.iter().map(|v| *v as f64).sum::<f64>(),
                            "ffn_norm_head": &ffn_norm_values[..8.min(ffn_norm_values.len())],
                            "ffn_gate_sum": gate_values.iter().map(|v| *v as f64).sum::<f64>(),
                            "ffn_gate_head": &gate_values[..8.min(gate_values.len())],
                            "ffn_up_sum": up_values.iter().map(|v| *v as f64).sum::<f64>(),
                            "ffn_up_head": &up_values[..8.min(up_values.len())],
                            "ffn_swiglu_sum": act_values.iter().map(|v| *v as f64).sum::<f64>(),
                            "ffn_swiglu_head": &act_values[..8.min(act_values.len())],
                            "ffn_out_sum": ffn_out.iter().map(|v| *v as f64).sum::<f64>(),
                            "ffn_out_head": &ffn_out[..8.min(ffn_out.len())],
                            "layer_out_sum": x.iter().zip(&ffn_out)
                                .map(|(x, y)| *x as f64 + *y as f64).sum::<f64>(),
                            "layer_out_head": x.iter().zip(&ffn_out)
                                .take(8).map(|(x, y)| *x as f32 + *y as f32).collect::<Vec<_>>(),
                        }));
                        tcb = TokenCommandBuffer::new(&self.ctx);
                    }
                }

                crate::kernels::add_rmsnorm_fused_tcb(
                    &mut tcb,
                    &x_buf,
                    &o_buf,
                    self.dense("model.norm.weight")?,
                    &x_norm_buf,
                    a.rms_norm_eps,
                    a.hidden,
                )?;
                self.encode_matvec(&mut tcb, &self.head_name.clone(), &x_norm_buf, &logits_buf)?;

                stats.dispatches += tcb.dispatch_count();
                tcb.commit_and_wait()?;
                stats.command_buffers += 1;
                logits = read_f32(&logits_buf, a.vocab_size);

                let now = Instant::now();
                stats
                    .per_token_ms
                    .push(now.duration_since(t_token).as_secs_f64() * 1e3);
                t_token = now;
                if step == 0 {
                    stats.first_token_ms = t_start.elapsed().as_secs_f64() * 1e3;
                }
            }
            stats.total_ms = t_start.elapsed().as_secs_f64() * 1e3;
            // `GravityEngine::generate` invokes this once for prefill and
            // again for each decode token.  With a position filter, later
            // nonmatching calls must not overwrite the selected prefill trace
            // with an empty document.
            if let Some(path) = trace_path.filter(|_| !trace_rows.is_empty()) {
                let trace = serde_json::json!({
                    "schema": "hawking.gravity.llama_gpu_layer_trace.v1",
                    "artifact_architecture": {
                        "model_type": a.model_type,
                        "rope_interleaved": a.rope_interleaved,
                        "rope_freq_factors": a.rope_freq_factors.as_ref().map(Vec::len),
                    },
                    "rows": trace_rows,
                    "note": "diagnostic per-layer commits; not a throughput path"
                });
                let bytes = serde_json::to_vec_pretty(&trace)
                    .map_err(|err| Error::Gravity(format!("serialize GPU trace: {err}")))?;
                std::fs::write(path, bytes)?;
            }
            Ok((logits, stats))
        }
    }
}

/// `GravityLlamaGpu` must be `Send + Sync` to be served behind the `Engine`
/// trait. This fails to compile the moment that stops being true, which is
/// the only way to notice: nothing else in the crate would.
#[cfg(all(test, target_os = "macos"))]
mod gpu_bounds {
    fn _assert_send_sync<T: Send + Sync>() {}
    #[test]
    fn gravity_llama_gpu_is_send_and_sync() {
        _assert_send_sync::<super::gpu::GravityLlamaGpu>();
    }
}

#[cfg(test)]
mod tests {
    use super::{
        layer_tensor_names, plan_layer_windows, read_verified_layer_window, GravityLlama,
        GravityLlamaArch,
    };
    use crate::numeric_parity::{score_against_f64, Bounds};
    use sha2::{Digest, Sha256};
    use std::path::Path;

    fn header(model_type: &str, sliding_window: Option<u64>) -> serde_json::Value {
        let mut architecture = serde_json::json!({
            "model_type": model_type,
            "num_hidden_layers": 2,
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "vocab_size": 32,
            "rope_theta": 10000.0,
            "rms_norm_eps": 0.00001
        });
        if let Some(window) = sliding_window {
            architecture["sliding_window"] = serde_json::json!(window);
        }
        serde_json::json!({"architecture": architecture})
    }

    #[test]
    fn mistral_header_is_admitted_with_explicit_window() {
        let arch = GravityLlamaArch::from_header(&header("mistral", Some(4096))).unwrap();
        assert_eq!(arch.model_type, "mistral");
        assert_eq!(arch.sliding_window, Some(4096));
    }

    #[test]
    fn llama_header_keeps_full_attention_default() {
        let arch = GravityLlamaArch::from_header(&header("llama", None)).unwrap();
        assert_eq!(arch.model_type, "llama");
        assert_eq!(arch.sliding_window, None);
    }

    #[test]
    fn qwen2_header_reuses_dense_decoder_adapter() {
        let arch = GravityLlamaArch::from_header(&header("qwen2", None)).unwrap();
        assert_eq!(arch.model_type, "qwen2");
        assert_eq!(arch.sliding_window, None);
    }

    #[test]
    fn sliding_window_keeps_recent_tokens_and_full_prefix_when_short() {
        assert_eq!(super::attention_window(8192, Some(4096)), (4096, 4096));
        assert_eq!(super::attention_window(128, Some(4096)), (0, 128));
        assert_eq!(super::attention_window(128, None), (0, 128));
    }

    #[test]
    fn unrelated_architecture_is_refused() {
        let error = GravityLlamaArch::from_header(&header("deepseek2", None)).unwrap_err();
        assert!(error
            .to_string()
            .contains("expected \"llama\", \"mistral\" or \"qwen2\""));
    }

    #[test]
    fn layer_window_plan_assigns_every_runtime_tensor_once_and_marks_extras() {
        let dir = tempfile::tempdir().expect("temporary Gravity artifact");
        let path = dir.path().join("one-layer.gravity");
        let mut names = vec![
            "model.embed_tokens.weight".to_string(),
            "model.norm.weight".to_string(),
        ];
        names.extend(layer_tensor_names(0));
        names.push("model.layers.0.unused.adapter.weight".to_string());
        let payload: Vec<u8> = (0..names.len()).map(|offset| offset as u8).collect();
        let tensors: Vec<_> = names
            .iter()
            .enumerate()
            .map(|(offset, name)| {
                serde_json::json!({
                    "name": name,
                    "codec": "native.f32",
                    "offset": offset,
                    "bytes": 1,
                    "sha256": format!("{:x}", Sha256::digest([offset as u8])),
                    "shape": [1],
                    "elements": 1,
                })
            })
            .collect();
        let mut architecture = header("qwen2", None)["architecture"].clone();
        architecture["num_hidden_layers"] = serde_json::json!(1);
        let header = serde_json::json!({
            "schema": "hawking.gravity.shard_header.v1",
            "format_version": 1,
            "architecture": architecture,
            "tensors": tensors,
        });
        let header_bytes = serde_json::to_vec(&header).expect("serialize header");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GRAVITY\0");
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header_bytes);
        bytes.extend_from_slice(&payload);
        std::fs::write(&path, bytes).expect("write artifact");

        let plan = plan_layer_windows(Path::new(&path)).expect("plan valid descriptors");
        assert!(plan.planning_only);
        assert_eq!(plan.global_tensors.len(), 2, "tied head reuses embedding");
        assert_eq!(plan.layers.len(), 1);
        assert_eq!(plan.layers[0].tensors.len(), 9);
        assert_eq!(plan.layers[0].payload_bytes, 9);
        assert_eq!(plan.global_payload_bytes, 2);
        assert_eq!(plan.maximum_dependency_complete_window_bytes, 11);
        assert_eq!(
            plan.unassigned_tensors,
            ["model.layers.0.unused.adapter.weight"]
        );

        let window = read_verified_layer_window(Path::new(&path), &plan, 0)
            .expect("exact global plus layer ranges verify");
        assert_eq!(window.bytes_read, 11);
        assert_eq!(window.global_tensors.len(), 2);
        assert_eq!(window.layer_tensors.len(), 9);
        assert_eq!(window.global_tensors[0].payload, [0]);
        assert_eq!(window.global_tensors[1].payload, [1]);
        assert_eq!(window.layer_tensors[0].payload, [2]);

        let first_layer_offset = plan.layers[0].tensors[0].file_offset as usize;
        let mut corrupted = std::fs::read(&path).expect("read fixture for corruption");
        corrupted[first_layer_offset] ^= 0xff;
        std::fs::write(&path, corrupted).expect("corrupt one selected payload byte");
        let error = read_verified_layer_window(Path::new(&path), &plan, 0)
            .expect_err("corrupted selected payload must fail hash verification");
        assert!(error.to_string().contains("range hash mismatch"));
    }

    fn write_minimal_qwen_bias_fixture(path: &Path, value_bias: [f32; 2]) {
        use sha2::{Digest, Sha256};

        let mut architecture = header("qwen2", None)["architecture"].clone();
        architecture["num_hidden_layers"] = serde_json::json!(1);
        architecture["hidden_size"] = serde_json::json!(2);
        architecture["num_attention_heads"] = serde_json::json!(1);
        architecture["num_key_value_heads"] = serde_json::json!(1);
        architecture["head_dim"] = serde_json::json!(2);
        architecture["vocab_size"] = serde_json::json!(3);

        let identity = [1.0_f32, 0.0, 0.0, 1.0];
        let zero_matrix = [0.0_f32; 4];
        let mut tensors: Vec<(String, Vec<f32>, Vec<u64>)> = vec![
            (
                "model.embed_tokens.weight".into(),
                vec![1.0, 0.0, 0.0, 1.0, -1.0, -1.0],
                vec![3, 2],
            ),
            ("model.norm.weight".into(), vec![1.0, 1.0], vec![2]),
            (
                "model.layers.0.input_layernorm.weight".into(),
                vec![1.0, 1.0],
                vec![2],
            ),
            (
                "model.layers.0.self_attn.q_proj.weight".into(),
                identity.to_vec(),
                vec![2, 2],
            ),
            (
                "model.layers.0.self_attn.k_proj.weight".into(),
                identity.to_vec(),
                vec![2, 2],
            ),
            (
                "model.layers.0.self_attn.v_proj.weight".into(),
                identity.to_vec(),
                vec![2, 2],
            ),
            (
                "model.layers.0.self_attn.o_proj.weight".into(),
                identity.to_vec(),
                vec![2, 2],
            ),
            (
                "model.layers.0.post_attention_layernorm.weight".into(),
                vec![1.0, 1.0],
                vec![2],
            ),
            (
                "model.layers.0.mlp.gate_proj.weight".into(),
                zero_matrix.to_vec(),
                vec![2, 2],
            ),
            (
                "model.layers.0.mlp.up_proj.weight".into(),
                zero_matrix.to_vec(),
                vec![2, 2],
            ),
            (
                "model.layers.0.mlp.down_proj.weight".into(),
                zero_matrix.to_vec(),
                vec![2, 2],
            ),
            (
                "model.layers.0.self_attn.q_proj.bias".into(),
                vec![0.0, 0.0],
                vec![2],
            ),
            (
                "model.layers.0.self_attn.k_proj.bias".into(),
                vec![0.0, 0.0],
                vec![2],
            ),
            (
                "model.layers.0.self_attn.v_proj.bias".into(),
                value_bias.to_vec(),
                vec![2],
            ),
        ];
        let mut body = Vec::new();
        let descriptors: Vec<_> = tensors
            .drain(..)
            .map(|(name, values, shape)| {
                let blob: Vec<u8> = values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect();
                let offset = body.len();
                body.extend_from_slice(&blob);
                serde_json::json!({
                    "name": name,
                    "codec": "native.f32",
                    "offset": offset,
                    "bytes": blob.len(),
                    "sha256": format!("{:x}", Sha256::digest(&blob)),
                    "shape": shape,
                    "elements": values.len(),
                })
            })
            .collect();
        let header = serde_json::json!({
            "schema": "hawking.gravity.shard_header.v1",
            "format_version": 1,
            "architecture": architecture,
            "tensors": descriptors,
        });
        let header_bytes = serde_json::to_vec(&header).expect("serialize Qwen bias fixture");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GRAVITY\0");
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header_bytes);
        bytes.extend_from_slice(&body);
        std::fs::write(path, bytes).expect("write Qwen bias fixture");
    }

    /// A compact raw-Q8_0 artifact deliberately shaped like the bounded Qwen
    /// source lane.  It exercises the f64 authority's direct source-byte row
    /// lookup and all packed projection matvecs; native-f32 fixtures cannot.
    fn write_q8_0_qwen_authority_fixture(path: &Path) {
        use sha2::{Digest, Sha256};

        let architecture = serde_json::json!({
            "model_type": "qwen2",
            "num_hidden_layers": 1,
            "hidden_size": 32,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "vocab_size": 32,
            "rope_theta": 10000.0,
            "rms_norm_eps": 0.00001,
        });
        let q8_matrix = |seed: i8| {
            let mut bytes = Vec::with_capacity(32 * 34);
            for row in 0..32_i16 {
                bytes.extend_from_slice(&half::f16::from_f32(0.03125).to_bits().to_le_bytes());
                for col in 0..32_i16 {
                    // Nonzero, bounded codes keep the fixture well-conditioned
                    // while making each source row and projection distinguishable.
                    let code = (row * 3 + col * 5 + i16::from(seed)).rem_euclid(15) - 7;
                    bytes.push(code as u8);
                }
            }
            bytes
        };
        let native_ones = || {
            (0..32)
                .flat_map(|_| 1.0_f32.to_le_bytes())
                .collect::<Vec<_>>()
        };
        let mut tensors: Vec<(String, String, Vec<u8>, Vec<u64>)> = vec![
            (
                "model.embed_tokens.weight".into(),
                "ggml.q8_0".into(),
                q8_matrix(1),
                vec![32, 32],
            ),
            (
                "model.norm.weight".into(),
                "native.f32".into(),
                native_ones(),
                vec![32],
            ),
        ];
        for (index, suffix) in [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ]
        .into_iter()
        .enumerate()
        {
            let name = format!("model.layers.0.{suffix}");
            if suffix.ends_with("layernorm.weight") {
                tensors.push((name, "native.f32".into(), native_ones(), vec![32]));
            } else {
                tensors.push((
                    name,
                    "ggml.q8_0".into(),
                    q8_matrix(index as i8 + 2),
                    vec![32, 32],
                ));
            }
        }

        let mut body = Vec::new();
        let descriptors: Vec<_> = tensors
            .drain(..)
            .map(|(name, codec, payload, shape)| {
                let offset = body.len();
                body.extend_from_slice(&payload);
                serde_json::json!({
                    "name": name,
                    "codec": codec,
                    "offset": offset,
                    "bytes": payload.len(),
                    "sha256": format!("{:x}", Sha256::digest(&payload)),
                    "shape": shape,
                    "elements": 32 * shape.iter().skip(1).product::<u64>(),
                })
            })
            .collect();
        let header = serde_json::json!({
            "schema": "hawking.gravity.shard_header.v1",
            "format_version": 1,
            "architecture": architecture,
            "tensors": descriptors,
        });
        let header_bytes = serde_json::to_vec(&header).expect("serialize Q8_0 fixture header");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GRAVITY\0");
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header_bytes);
        bytes.extend_from_slice(&body);
        std::fs::write(path, bytes).expect("write Q8_0 authority fixture");
    }

    #[test]
    fn qwen_qkv_biases_change_the_executable_forward() {
        let dir = tempfile::tempdir().expect("temporary Qwen bias artifact");
        let without_bias = dir.path().join("without-bias.gravity");
        let with_bias = dir.path().join("with-bias.gravity");
        write_minimal_qwen_bias_fixture(&without_bias, [0.0, 0.0]);
        write_minimal_qwen_bias_fixture(&with_bias, [0.0, 1.0]);

        let no_bias_logits = GravityLlama::open(&without_bias, true)
            .expect("open zero-bias Qwen fixture")
            .forward(&[0])
            .expect("forward zero-bias Qwen fixture");
        let biased_logits = GravityLlama::open(&with_bias, true)
            .expect("open biased Qwen fixture")
            .forward(&[0])
            .expect("forward biased Qwen fixture");
        assert!(no_bias_logits.iter().all(|value| value.is_finite()));
        assert!(biased_logits.iter().all(|value| value.is_finite()));
        assert!(
            no_bias_logits
                .iter()
                .zip(&biased_logits)
                .any(|(without, with)| (without - with).abs() > 1e-5),
            "Qwen Q/K/V biases were copied but had no effect on the executable forward"
        );
    }

    #[test]
    fn qwen_raw_q8_0_full_forward_f32_passes_independent_f64_authority() {
        let dir = tempfile::tempdir().expect("temporary Q8_0 Qwen artifact");
        let path = dir.path().join("qwen-q8_0-authority.gravity");
        write_q8_0_qwen_authority_fixture(&path);
        let model = GravityLlama::open(&path, true).expect("open raw Q8_0 Qwen artifact");
        let f32_logits = model.forward(&[5, 6]).expect("f32 raw-Q8_0 full forward");
        let f64_logits = model
            .forward_f64_authority(&[5, 6])
            .expect("independent f64 raw-Q8_0 full forward");
        let score = score_against_f64(
            &f32_logits,
            &f64_logits,
            &Bounds::full_forward_logits(),
            "qwen_raw_q8_0_cpu_f32",
        );
        assert!(
            score.pass,
            "raw-Q8_0 f32 forward failed independent f64 authority: {:?}",
            score.failures
        );
    }
}
