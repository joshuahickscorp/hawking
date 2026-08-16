//! Language-only Qwen3.8 packer: skip vision, fuse split in_proj at pack
//! time, store large GEMVs as HQ30UQ4 group-64 and small vectors as f32.
//!
//! One source tensor (or fused pair) is resident at a time. Does not generate
//! tokens or touch the Q80/Q30 admission seals.

use super::qwen38_geometry::{
    fuse_in_proj_ba, fuse_in_proj_qkvz, qwen38_accept_config, qwen38_layer_name, qwen38_mixer_kind,
    Qwen38MixerKind, QWEN38_HIDDEN, QWEN38_IN_PROJ_A_ROWS, QWEN38_IN_PROJ_B_ROWS,
    QWEN38_IN_PROJ_QKV_ROWS, QWEN38_IN_PROJ_Z_ROWS, QWEN38_LANGUAGE_PREFIX, QWEN38_LAYERS,
    QWEN38_VISION_PREFIX,
};
use super::qwen_complete_binary::{
    pack_uniform_q4_group64, parse_uniform_q4_header, UNIFORM_Q4_GROUP_SIZE, UNIFORM_Q4_NOMINAL_BPW,
};
use crate::artifact::widen_native;
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

pub const QWEN38_UNIFORM_Q4_SCHEMA: &str = "hawking.ascent.qwen38_language_uniform_q4.v1";
pub const QWEN38_PACK_STATUS: &str = "CANDIDATE_QWEN38_LANGUAGE_Q4_FUSED_INPROJ";
pub const QWEN38_Q4_EXT: &str = "hq30uq4";
pub const QWEN38_F32_EXT: &str = "f32v2";
pub const QWEN38_EXPECTED_Q4_TENSORS: usize = 402;
pub const QWEN38_EXPECTED_F32_TENSORS: usize = 353;
pub const QWEN38_EXPECTED_CATALOG_TENSORS: usize =
    QWEN38_EXPECTED_Q4_TENSORS + QWEN38_EXPECTED_F32_TENSORS;

#[derive(Clone, Debug)]
pub struct Qwen38PackRequest {
    pub source_dir: PathBuf,
    pub output_root: PathBuf,
}

#[derive(Clone, Debug, Serialize)]
pub struct Qwen38PackReport {
    pub manifest_path: PathBuf,
    pub tensor_count: usize,
    pub q4_tensors: usize,
    pub f32_tensors: usize,
    pub source_weight_elements: u64,
    pub tensor_payload_bytes: u64,
    pub complete_physical_bpw: f64,
    pub fused_in_proj_layers: usize,
    pub skipped_vision_tensors: usize,
    pub min_q4_cosine: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Qwen38CatalogRow {
    pub name: String,
    pub kind: String,
    pub shape: Vec<usize>,
    pub elements: u64,
    pub artifact: String,
    pub bytes: u64,
    #[serde(default)]
    pub cosine: Option<f64>,
}

struct TensorLoc {
    shard: PathBuf,
    data_offset: u64,
    nbytes: usize,
    shape: Vec<usize>,
}

struct SourceIndex {
    map: HashMap<String, TensorLoc>,
    vision_tensors: usize,
}

impl SourceIndex {
    fn open(source_dir: &Path) -> Result<Self> {
        let config_path = source_dir.join("config.json");
        let config_raw = fs::read(&config_path).map_err(|error| {
            Error::Model(format!("cannot read {}: {error}", config_path.display()))
        })?;
        let config: Value = serde_json::from_slice(&config_raw)
            .map_err(|error| Error::Model(format!("qwen38 config JSON: {error}")))?;
        qwen38_accept_config(&config)?;

        let index_path = source_dir.join("model.safetensors.index.json");
        let index_raw = fs::read(&index_path).map_err(|error| {
            Error::Model(format!("cannot read {}: {error}", index_path.display()))
        })?;
        let index: Value = serde_json::from_slice(&index_raw)
            .map_err(|error| Error::Model(format!("qwen38 index JSON: {error}")))?;
        let weight_map = index
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or_else(|| Error::Model("qwen38 index lacks weight_map".into()))?;

        let mut by_shard: HashMap<String, Vec<String>> = HashMap::new();
        let mut vision_tensors = 0usize;
        for (name, shard_v) in weight_map {
            if name.starts_with(QWEN38_VISION_PREFIX) {
                vision_tensors += 1;
                continue;
            }
            if !name.starts_with(QWEN38_LANGUAGE_PREFIX) {
                return Err(Error::Model(format!(
                    "qwen38 unexpected tensor root {name}"
                )));
            }
            let shard = shard_v
                .as_str()
                .ok_or_else(|| Error::Model(format!("weight_map {name} is not a string")))?;
            by_shard
                .entry(shard.to_string())
                .or_default()
                .push(name.clone());
        }

        let mut map = HashMap::new();
        for (shard_name, names) in by_shard {
            let shard_path = source_dir.join(&shard_name);
            let header = read_safetensors_header(&shard_path)?;
            for name in names {
                let info = header.tensors.get(&name).ok_or_else(|| {
                    Error::Model(format!("shard {shard_name} lacks tensor {name}"))
                })?;
                if info.dtype != "BF16" && info.dtype != "BFLOAT16" {
                    return Err(Error::Model(format!(
                        "tensor {name} dtype {} is not BF16",
                        info.dtype
                    )));
                }
                let (begin, end) = info.data_offsets;
                if end < begin {
                    return Err(Error::Model(format!(
                        "tensor {name} has inverted data_offsets"
                    )));
                }
                map.insert(
                    name,
                    TensorLoc {
                        shard: shard_path.clone(),
                        data_offset: 8 + header.header_nbytes + begin,
                        nbytes: (end - begin) as usize,
                        shape: info.shape.clone(),
                    },
                );
            }
        }
        Ok(Self {
            map,
            vision_tensors,
        })
    }

    fn read_f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let loc = self
            .map
            .get(name)
            .ok_or_else(|| Error::Model(format!("source lacks {name}")))?;
        let mut raw = vec![0u8; loc.nbytes];
        let file = File::open(&loc.shard).map_err(|error| {
            Error::Model(format!("open {}: {error}", loc.shard.display()))
        })?;
        #[cfg(unix)]
        {
            file.read_exact_at(&mut raw, loc.data_offset).map_err(|error| {
                Error::Model(format!("pread {name}: {error}"))
            })?;
        }
        #[cfg(not(unix))]
        {
            use std::io::{Seek, SeekFrom};
            let mut file = file;
            file.seek(SeekFrom::Start(loc.data_offset))?;
            file.read_exact(&mut raw)?;
        }
        let values = widen_native("native.bf16", &raw)?;
        Ok((values, loc.shape.clone()))
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
        .map_err(|error| Error::Model(format!("cannot open {}: {error}", path.display())))?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let header_nbytes = u64::from_le_bytes(len_buf);
    if header_nbytes == 0 || header_nbytes > 64 * 1024 * 1024 {
        return Err(Error::Model(format!(
            "implausible safetensors header {header_nbytes} in {}",
            path.display()
        )));
    }
    let mut raw = vec![0u8; header_nbytes as usize];
    file.read_exact(&mut raw)?;
    let value: Value = serde_json::from_slice(&raw)
        .map_err(|error| Error::Model(format!("safetensors header JSON: {error}")))?;
    let object = value
        .as_object()
        .ok_or_else(|| Error::Model("safetensors header is not an object".into()))?;
    let mut tensors = HashMap::new();
    for (name, info_v) in object {
        if name == "__metadata__" {
            continue;
        }
        let info = info_v
            .as_object()
            .ok_or_else(|| Error::Model(format!("tensor {name} header is not an object")))?;
        let dtype = info
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::Model(format!("tensor {name} lacks dtype")))?
            .to_string();
        let shape = info
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::Model(format!("tensor {name} lacks shape")))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(|| Error::Model(format!("tensor {name} has non-integer shape")))
            })
            .collect::<Result<Vec<_>>>()?;
        let offsets = info
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::Model(format!("tensor {name} lacks data_offsets")))?;
        if offsets.len() != 2 {
            return Err(Error::Model(format!(
                "tensor {name} data_offsets is not a pair"
            )));
        }
        let begin = offsets[0]
            .as_u64()
            .ok_or_else(|| Error::Model(format!("tensor {name} data_offsets[0] invalid")))?;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| Error::Model(format!("tensor {name} data_offsets[1] invalid")))?;
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

fn artifact_filename(name: &str, ext: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(name.as_bytes());
    format!("{:x}.{ext}", digest)
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("tmp");
    {
        let mut file = File::create(&tmp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    fs::rename(&tmp, path)?;
    Ok(())
}

fn try_reuse_q4(
    tensors_dir: &Path,
    name: &str,
    expected_shape: &[usize],
) -> Result<Option<Qwen38CatalogRow>> {
    let artifact = artifact_filename(name, QWEN38_Q4_EXT);
    let path = tensors_dir.join(&artifact);
    if !path.is_file() {
        return Ok(None);
    }
    let payload = fs::read(&path)?;
    let header = parse_uniform_q4_header(&payload)?;
    if header.shape != expected_shape {
        return Ok(None);
    }
    Ok(Some(Qwen38CatalogRow {
        name: name.to_owned(),
        kind: "q4".into(),
        shape: expected_shape.to_vec(),
        elements: header.elements as u64,
        artifact,
        bytes: payload.len() as u64,
        cosine: None,
    }))
}

fn try_reuse_f32(
    tensors_dir: &Path,
    name: &str,
    expected_shape: &[usize],
) -> Result<Option<Qwen38CatalogRow>> {
    let artifact = artifact_filename(name, QWEN38_F32_EXT);
    let path = tensors_dir.join(&artifact);
    if !path.is_file() {
        return Ok(None);
    }
    let payload = fs::read(&path)?;
    let values = read_qwen38_f32_payload(&payload)?;
    let elements = expected_shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| Error::Model(format!("{name} shape overflow")))
    })?;
    if values.len() != elements {
        return Ok(None);
    }
    Ok(Some(Qwen38CatalogRow {
        name: name.to_owned(),
        kind: "f32".into(),
        shape: expected_shape.to_vec(),
        elements: values.len() as u64,
        artifact,
        bytes: payload.len() as u64,
        cosine: Some(1.0),
    }))
}

fn pack_q4_from_source(
    source: &SourceIndex,
    tensors_dir: &Path,
    name: &str,
    expected_shape: &[usize],
) -> Result<Qwen38CatalogRow> {
    if let Some(row) = try_reuse_q4(tensors_dir, name, expected_shape)? {
        return Ok(row);
    }
    eprintln!("qwen38-pack q4 {name}");
    let (values, shape) = source.read_f32(name)?;
    if shape != expected_shape {
        return Err(Error::Model(format!(
            "{name} shape {shape:?} != {expected_shape:?}"
        )));
    }
    pack_q4_named(tensors_dir, name, &values, &shape)
}

fn pack_f32_from_source(
    source: &SourceIndex,
    tensors_dir: &Path,
    name: &str,
    expected_shape: &[usize],
) -> Result<Qwen38CatalogRow> {
    if let Some(row) = try_reuse_f32(tensors_dir, name, expected_shape)? {
        return Ok(row);
    }
    let (values, shape) = source.read_f32(name)?;
    if shape != expected_shape {
        return Err(Error::Model(format!(
            "{name} shape {shape:?} != {expected_shape:?}"
        )));
    }
    let stored = mlx_residual_norm_to_delta(name, &values).unwrap_or(values);
    pack_f32_named(tensors_dir, name, &stored, &shape)
}

fn pack_q4_named(
    tensors_dir: &Path,
    name: &str,
    values: &[f32],
    shape: &[usize],
) -> Result<Qwen38CatalogRow> {
    let artifact = artifact_filename(name, QWEN38_Q4_EXT);
    let path = tensors_dir.join(&artifact);
    if path.is_file() {
        let payload = fs::read(&path)?;
        let header = parse_uniform_q4_header(&payload)?;
        if header.shape == shape {
            return Ok(Qwen38CatalogRow {
                name: name.to_owned(),
                kind: "q4".into(),
                shape: shape.to_vec(),
                elements: values.len() as u64,
                artifact,
                bytes: payload.len() as u64,
                cosine: None,
            });
        }
    }
    let (payload, quality) = pack_uniform_q4_group64(values, shape)?;
    let header = parse_uniform_q4_header(&payload)?;
    if header.shape != shape {
        return Err(Error::Model(format!(
            "{name} Q4 header shape {:?} != {:?}",
            header.shape, shape
        )));
    }
    write_atomic(&path, &payload)?;
    Ok(Qwen38CatalogRow {
        name: name.to_owned(),
        kind: "q4".into(),
        shape: shape.to_vec(),
        elements: values.len() as u64,
        artifact,
        bytes: payload.len() as u64,
        cosine: Some(quality.cosine),
    })
}

/// MLX stores residual / q_norm / k_norm as already-materialized scales
/// (mean ~1). Q80 kernels apply `(1+w)` expecting HF delta-from-one. Convert
/// at pack time. Gated DeltaNet norm stays conventional (ones-init, no +1).
fn mlx_residual_norm_to_delta(name: &str, values: &[f32]) -> Option<Vec<f32>> {
    let is_residual = name.ends_with("input_layernorm.weight")
        || name.ends_with("post_attention_layernorm.weight")
        || name.ends_with("model.norm.weight")
        || name.ends_with("q_norm.weight")
        || name.ends_with("k_norm.weight");
    if !is_residual {
        return None;
    }
    Some(values.iter().map(|v| v - 1.0).collect())
}

fn pack_f32_named(
    tensors_dir: &Path,
    name: &str,
    values: &[f32],
    shape: &[usize],
) -> Result<Qwen38CatalogRow> {
    let mut payload = Vec::with_capacity(8 + values.len() * 4);
    payload.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    let artifact = artifact_filename(name, QWEN38_F32_EXT);
    write_atomic(&tensors_dir.join(&artifact), &payload)?;
    Ok(Qwen38CatalogRow {
        name: name.to_owned(),
        kind: "f32".into(),
        shape: shape.to_vec(),
        elements: values.len() as u64,
        artifact,
        bytes: payload.len() as u64,
        cosine: Some(1.0),
    })
}

pub fn pack_qwen38_language_uniform_q4(request: &Qwen38PackRequest) -> Result<Qwen38PackReport> {
    let source = SourceIndex::open(&request.source_dir)?;
    let tensors_dir = request.output_root.join("tensors");
    fs::create_dir_all(&tensors_dir)?;

    let mut rows: Vec<Qwen38CatalogRow> = Vec::new();
    let mut fused_layers = 0usize;

    rows.push(pack_q4_from_source(
        &source,
        &tensors_dir,
        "language_model.model.embed_tokens.weight",
        &[crate::model::qwen38_geometry::QWEN38_VOCAB, QWEN38_HIDDEN],
    )?);

    for layer in 0..QWEN38_LAYERS {
        let mixer = qwen38_mixer_kind(layer)?;
        eprintln!("qwen38-pack layer {layer} {}", mixer.as_str());
        rows.push(pack_f32_from_source(
            &source,
            &tensors_dir,
            &qwen38_layer_name(layer, "input_layernorm.weight"),
            &[QWEN38_HIDDEN],
        )?);
        rows.push(pack_f32_from_source(
            &source,
            &tensors_dir,
            &qwen38_layer_name(layer, "post_attention_layernorm.weight"),
            &[QWEN38_HIDDEN],
        )?);

        match mixer {
            Qwen38MixerKind::DeltaNet => {
                let fused_name = qwen38_layer_name(layer, "linear_attn.in_proj_qkvz.weight");
                let fused_shape = [
                    crate::model::qwen38_geometry::QWEN38_QKVZ_ROWS,
                    QWEN38_HIDDEN,
                ];
                if let Some(row) = try_reuse_q4(&tensors_dir, &fused_name, &fused_shape)? {
                    rows.push(row);
                } else {
                    let (qkv, qkv_shape) = source
                        .read_f32(&qwen38_layer_name(layer, "linear_attn.in_proj_qkv.weight"))?;
                    let (z, z_shape) = source
                        .read_f32(&qwen38_layer_name(layer, "linear_attn.in_proj_z.weight"))?;
                    if qkv_shape != [QWEN38_IN_PROJ_QKV_ROWS, QWEN38_HIDDEN]
                        || z_shape != [QWEN38_IN_PROJ_Z_ROWS, QWEN38_HIDDEN]
                    {
                        return Err(Error::Model(format!(
                            "layer {layer} in_proj_qkv/z shapes {qkv_shape:?}/{z_shape:?}"
                        )));
                    }
                    let fused = fuse_in_proj_qkvz(&qkv, &z, QWEN38_HIDDEN)?;
                    drop(qkv);
                    drop(z);
                    rows.push(pack_q4_named(&tensors_dir, &fused_name, &fused, &fused_shape)?);
                }

                let ba_name = qwen38_layer_name(layer, "linear_attn.in_proj_ba.weight");
                let ba_shape = [crate::model::qwen38_geometry::QWEN38_BA_ROWS, QWEN38_HIDDEN];
                if let Some(row) = try_reuse_q4(&tensors_dir, &ba_name, &ba_shape)? {
                    rows.push(row);
                } else {
                    let (b, b_shape) = source
                        .read_f32(&qwen38_layer_name(layer, "linear_attn.in_proj_b.weight"))?;
                    let (a, a_shape) = source
                        .read_f32(&qwen38_layer_name(layer, "linear_attn.in_proj_a.weight"))?;
                    if b_shape != [QWEN38_IN_PROJ_B_ROWS, QWEN38_HIDDEN]
                        || a_shape != [QWEN38_IN_PROJ_A_ROWS, QWEN38_HIDDEN]
                    {
                        return Err(Error::Model(format!(
                            "layer {layer} in_proj_b/a shapes {b_shape:?}/{a_shape:?}"
                        )));
                    }
                    let ba = fuse_in_proj_ba(&b, &a, QWEN38_HIDDEN)?;
                    drop(a);
                    drop(b);
                    rows.push(pack_q4_named(&tensors_dir, &ba_name, &ba, &ba_shape)?);
                }
                fused_layers += 1;

                rows.push(pack_q4_from_source(
                    &source,
                    &tensors_dir,
                    &qwen38_layer_name(layer, "linear_attn.out_proj.weight"),
                    &[
                        crate::model::qwen38_geometry::QWEN38_O_PROJ_ROWS,
                        crate::model::qwen38_geometry::QWEN38_O_PROJ_COLS,
                    ],
                )?);
                rows.push(pack_f32_from_source(
                    &source,
                    &tensors_dir,
                    &qwen38_layer_name(layer, "linear_attn.conv1d.weight"),
                    &[QWEN38_IN_PROJ_QKV_ROWS, 4, 1],
                )?);
                rows.push(pack_f32_from_source(
                    &source,
                    &tensors_dir,
                    &qwen38_layer_name(layer, "linear_attn.A_log"),
                    &[48],
                )?);
                rows.push(pack_f32_from_source(
                    &source,
                    &tensors_dir,
                    &qwen38_layer_name(layer, "linear_attn.dt_bias"),
                    &[48],
                )?);
                rows.push(pack_f32_from_source(
                    &source,
                    &tensors_dir,
                    &qwen38_layer_name(layer, "linear_attn.norm.weight"),
                    &[128],
                )?);
            }
            Qwen38MixerKind::Gqa => {
                rows.push(pack_q4_from_source(
                    &source,
                    &tensors_dir,
                    &qwen38_layer_name(layer, "self_attn.q_proj.weight"),
                    &[
                        crate::model::qwen38_geometry::QWEN38_Q_PROJ_ROWS,
                        QWEN38_HIDDEN,
                    ],
                )?);
                rows.push(pack_q4_from_source(
                    &source,
                    &tensors_dir,
                    &qwen38_layer_name(layer, "self_attn.k_proj.weight"),
                    &[
                        crate::model::qwen38_geometry::QWEN38_KV_PROJ_ROWS,
                        QWEN38_HIDDEN,
                    ],
                )?);
                rows.push(pack_q4_from_source(
                    &source,
                    &tensors_dir,
                    &qwen38_layer_name(layer, "self_attn.v_proj.weight"),
                    &[
                        crate::model::qwen38_geometry::QWEN38_KV_PROJ_ROWS,
                        QWEN38_HIDDEN,
                    ],
                )?);
                rows.push(pack_q4_from_source(
                    &source,
                    &tensors_dir,
                    &qwen38_layer_name(layer, "self_attn.o_proj.weight"),
                    &[
                        crate::model::qwen38_geometry::QWEN38_O_PROJ_ROWS,
                        crate::model::qwen38_geometry::QWEN38_O_PROJ_COLS,
                    ],
                )?);
                rows.push(pack_f32_from_source(
                    &source,
                    &tensors_dir,
                    &qwen38_layer_name(layer, "self_attn.q_norm.weight"),
                    &[256],
                )?);
                rows.push(pack_f32_from_source(
                    &source,
                    &tensors_dir,
                    &qwen38_layer_name(layer, "self_attn.k_norm.weight"),
                    &[256],
                )?);
            }
        }

        rows.push(pack_q4_from_source(
            &source,
            &tensors_dir,
            &qwen38_layer_name(layer, "mlp.gate_proj.weight"),
            &[crate::model::qwen38_geometry::QWEN38_INTERMEDIATE, QWEN38_HIDDEN],
        )?);
        rows.push(pack_q4_from_source(
            &source,
            &tensors_dir,
            &qwen38_layer_name(layer, "mlp.up_proj.weight"),
            &[crate::model::qwen38_geometry::QWEN38_INTERMEDIATE, QWEN38_HIDDEN],
        )?);
        rows.push(pack_q4_from_source(
            &source,
            &tensors_dir,
            &qwen38_layer_name(layer, "mlp.down_proj.weight"),
            &[QWEN38_HIDDEN, crate::model::qwen38_geometry::QWEN38_INTERMEDIATE],
        )?);
    }

    rows.push(pack_f32_from_source(
        &source,
        &tensors_dir,
        "language_model.model.norm.weight",
        &[QWEN38_HIDDEN],
    )?);
    rows.push(pack_q4_from_source(
        &source,
        &tensors_dir,
        "language_model.lm_head.weight",
        &[crate::model::qwen38_geometry::QWEN38_VOCAB, QWEN38_HIDDEN],
    )?);

    let q4_tensors = rows.iter().filter(|row| row.kind == "q4").count();
    let f32_tensors = rows.iter().filter(|row| row.kind == "f32").count();
    if q4_tensors != QWEN38_EXPECTED_Q4_TENSORS || f32_tensors != QWEN38_EXPECTED_F32_TENSORS {
        return Err(Error::Model(format!(
            "qwen38 catalog counts q4={q4_tensors} f32={f32_tensors}, expected q4={QWEN38_EXPECTED_Q4_TENSORS} f32={QWEN38_EXPECTED_F32_TENSORS}"
        )));
    }
    let source_weight_elements: u64 = rows.iter().map(|row| row.elements).sum();
    let tensor_payload_bytes: u64 = rows.iter().map(|row| row.bytes).sum();
    let complete_physical_bpw = if source_weight_elements == 0 {
        0.0
    } else {
        (tensor_payload_bytes as f64 * 8.0) / source_weight_elements as f64
    };
    let min_q4_cosine = rows
        .iter()
        .filter(|row| row.kind == "q4")
        .filter_map(|row| row.cosine)
        .fold(1.0f64, f64::min);

    let manifest = json!({
        "schema": QWEN38_UNIFORM_Q4_SCHEMA,
        "status": QWEN38_PACK_STATUS,
        "source_dir": request.source_dir,
        "skipped_vision_tensors": source.vision_tensors,
        "fused_in_proj_layers": fused_layers,
        "q4_group_size": UNIFORM_Q4_GROUP_SIZE,
        "nominal_codec_bpw": UNIFORM_Q4_NOMINAL_BPW,
        "complete_physical_bpw": complete_physical_bpw,
        "source_weight_elements": source_weight_elements,
        "tensor_payload_bytes": tensor_payload_bytes,
        "tensor_count": rows.len(),
        "q4_tensors": q4_tensors,
        "f32_tensors": f32_tensors,
        "min_q4_cosine": min_q4_cosine,
        "tensors": rows,
    });
    let manifest_path = request.output_root.join("manifest.json");
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| Error::Model(format!("qwen38 manifest encode: {error}")))?;
    write_atomic(&manifest_path, &manifest_bytes)?;

    Ok(Qwen38PackReport {
        manifest_path,
        tensor_count: rows.len(),
        q4_tensors,
        f32_tensors,
        source_weight_elements,
        tensor_payload_bytes,
        complete_physical_bpw,
        fused_in_proj_layers: fused_layers,
        skipped_vision_tensors: source.vision_tensors,
        min_q4_cosine,
    })
}

pub fn load_qwen38_manifest(root: impl AsRef<Path>) -> Result<(PathBuf, Vec<Qwen38CatalogRow>)> {
    let manifest_path = root.as_ref().join("manifest.json");
    let raw = fs::read(&manifest_path).map_err(|error| {
        Error::Model(format!(
            "cannot read {}: {error}",
            manifest_path.display()
        ))
    })?;
    let value: Value = serde_json::from_slice(&raw)
        .map_err(|error| Error::Model(format!("qwen38 manifest JSON: {error}")))?;
    let schema = value
        .get("schema")
        .and_then(Value::as_str)
        .unwrap_or("");
    if schema != QWEN38_UNIFORM_Q4_SCHEMA {
        return Err(Error::Model(format!(
            "qwen38 manifest schema {schema:?} != {QWEN38_UNIFORM_Q4_SCHEMA}"
        )));
    }
    let rows: Vec<Qwen38CatalogRow> = serde_json::from_value(
        value
            .get("tensors")
            .cloned()
            .ok_or_else(|| Error::Model("qwen38 manifest lacks tensors".into()))?,
    )
    .map_err(|error| Error::Model(format!("qwen38 manifest tensors: {error}")))?;
    Ok((manifest_path, rows))
}

pub fn read_qwen38_f32_payload(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() < 8 {
        return Err(Error::Model("qwen38 f32 payload truncated".into()));
    }
    let n = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let need = 8 + n * 4;
    if bytes.len() != need {
        return Err(Error::Model(format!(
            "qwen38 f32 payload {} != {need}",
            bytes.len()
        )));
    }
    let mut values = Vec::with_capacity(n);
    for chunk in bytes[8..].chunks_exact(4) {
        values.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlx_norms_convert_to_hf_delta_from_one() {
        let residual = mlx_residual_norm_to_delta(
            "language_model.model.layers.0.input_layernorm.weight",
            &[0.97, 1.05],
        )
        .unwrap();
        assert!((residual[0] + 0.03).abs() < 1e-6);
        assert!((residual[1] - 0.05).abs() < 1e-6);
        assert!(mlx_residual_norm_to_delta(
            "language_model.model.layers.0.linear_attn.norm.weight",
            &[0.87],
        )
        .is_none());
        assert!(mlx_residual_norm_to_delta(
            "language_model.model.layers.0.linear_attn.A_log",
            &[-3.2],
        )
        .is_none());
        let qn = mlx_residual_norm_to_delta(
            "language_model.model.layers.3.self_attn.q_norm.weight",
            &[1.23],
        )
        .unwrap();
        assert!((qn[0] - 0.23).abs() < 1e-6);
    }
}
