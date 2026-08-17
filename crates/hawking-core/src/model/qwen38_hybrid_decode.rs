//! Native Qwen3.8 hybrid token graph: Q4 GEMVs + Q80 f32 activation
//! kernels + Q38-forked rearrange/GQA. Dense SwiGLU suffix. Zero fallbacks.
//!
//! Mixed catalogs (`catalog.hq38m20`) bind HGRAVB01 / HGRAVR02 / HGRAVS01
//! and pack-declared HGRAVU01 (including MLP) through the existing Q80
//! occupancy tiles. Packed bytes stay packed. A missing codec fails the
//! run; there is no reconstruct-to-Q4 path.

use super::qwen38_64_layer_execution_schedule::qwen38_assert_schedule_intact;
use super::qwen38_geometry::{
    qwen38_deltanet_state_slot, qwen38_gqa_state_slot, qwen38_layer_name, qwen38_mixer_kind,
    Qwen38DeltaNetLayout, Qwen38MixerKind, QWEN38_GQA_HEAD_DIM, QWEN38_GQA_HEADS,
    QWEN38_GQA_KV_HEADS, QWEN38_GQA_LAYERS, QWEN38_GQA_ROTARY_DIM, QWEN38_HIDDEN,
    QWEN38_INTERMEDIATE, QWEN38_LAYERS, QWEN38_RMS_EPS, QWEN38_ROPE_THETA, QWEN38_VOCAB,
};
use super::qwen38_pack::{
    load_qwen38_manifest, read_qwen38_f32_payload, QWEN38_EXPECTED_CATALOG_TENSORS,
};
use super::qwen_complete_binary::{
    expand_rice_indices, mixed_gpu_layout, parse_uniform_q4_header, rice_q1_row_ptr,
    uniform_factor_value, BinaryGroupPacked, MixedGpuKind, RiceQ1Packed, UniformFactorPacked,
    MAGIC_BINARY, MAGIC_HGRAVS01, MAGIC_RESIDUAL_COMPACT, MAGIC_UNIFORM, UNIFORM_Q4_GROUP_SIZE,
};
use crate::tokenizer::Tokenizer;
use crate::{Error, Result};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const QWEN38_MIXED_CATALOG_MAGIC: [u8; 8] = *b"HQ38M20\0";
pub const QWEN38_MIXED_CATALOG_VERSION: u32 = 1;
pub const QWEN38_MIXED_RECORD_SIZE: usize = 128;
pub const QWEN38_MIXED_CATALOG_NAME: &str = "catalog.hq38m20";
pub const QWEN38_MIXED_SCHEMA: &str =
    "hawking.ascension.qwen38_mixed_representation_candidate.v1";
pub const QWEN38_MIXED_HGRAVS_RANK: usize = 160;
pub const QWEN38_MIXED_HGRAVS_BITS: u8 = 3;
pub const QWEN38_MIXED_HGRAVS_GROUP: usize = 64;

/// Default **on**. Consume packed mixed codes on the Q80 occupancy tiles.
/// `HAWKING_QWEN38_RECON_FUSE=0` selects the G023 serial family names.
pub fn qwen38_recon_fuse_enabled() -> bool {
    crate::env_opt_out("HAWKING_QWEN38_RECON_FUSE")
}

fn mixed_error(message: impl Into<String>) -> Error {
    Error::Model(format!("qwen38 mixed hybrid decode: {}", message.into()))
}

fn mlx_residual_norm_to_delta_named(name: &str, values: &mut [f32]) {
    let convert = name.ends_with("input_layernorm.weight")
        || name.ends_with("post_attention_layernorm.weight")
        || name.ends_with("model.norm.weight")
        || name.ends_with("q_norm.weight")
        || name.ends_with("k_norm.weight");
    if !convert {
        return;
    }
    for value in values {
        *value -= 1.0;
    }
}

/// Destination lane for one HQ38M20 row. Packed GEMVs stay packed.
/// Codec 4 is already HF-δ f32v2 and must not be mlx-delta'd.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MixedCatalogLane {
    Packed(u8),
    Hq30Uq4,
    F32v2,
    HgravuVector,
}

/// CPU census of a mixed catalog. Does not open Metal and does not expand
/// rice indices. `expanded_to_q4` / `expanded_to_float_gemv` stay zero on
/// this path; they exist so a later reader cannot miss a forbidden fallback.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MixedCatalogCensus {
    pub tensors: usize,
    pub binary: usize,
    pub residual: usize,
    pub hgravs: usize,
    pub uniform: usize,
    pub q4: usize,
    pub f32: usize,
    pub refused: usize,
    pub expanded_to_q4: usize,
    pub expanded_to_float_gemv: usize,
    pub refusals: Vec<String>,
}

const HQ30UQ4_MAGIC: [u8; 8] = *b"HQ30UQ4\0";
const K_COMPLETE_TILE_COLS: u32 = 256;

pub fn qwen38_binary_matvec_kernel(cols: u32) -> &'static str {
    if cols > 2048 {
        "q80_binary_group_matvec_simd_bytes"
    } else {
        "q80_binary_group_matvec_tg256"
    }
}

pub fn qwen38_residual_matvec_kernel(cols: u32) -> &'static str {
    if cols > 2048 {
        "q80_binary_group_csr_matvec_bytes"
    } else {
        "q80_binary_group_csr_matvec_tg256"
    }
}

/// simd_bytes tiles 256 columns. A non-multiple would drop a remainder —
/// refuse rather than silently emit a partial-K GEMV.
pub fn qwen38_assert_k_complete_cols(cols: u32) -> Result<()> {
    if cols > 2048 && cols % K_COMPLETE_TILE_COLS != 0 {
        return Err(mixed_error(format!(
            "cols={cols} is not a {K_COMPLETE_TILE_COLS}-col tile multiple; \
             simd_bytes would drop a remainder. Refusing partial-K bind."
        )));
    }
    Ok(())
}

pub fn qwen38_mixed_k_complete_bind_message() -> String {
    let fuse = if qwen38_recon_fuse_enabled() {
        "ON"
    } else {
        "OFF"
    };
    format!(
        "qwen38-decode mixed bind: K-complete; recon_fuse={fuse} \
         uses q80_binary_group_matvec_simd_bytes / q80_binary_group_csr_matvec_bytes \
         when cols>2048 (256-col tiles; this model K in {{5120,6144}}); \
         cols<=2048 stay on tg256; recon_fuse=0 walks every column via {}",
        crate::decode_family::matvec_binary()
    )
}

fn hgravu_is_vector(name: &str, shape: &[usize]) -> bool {
    if name.ends_with("embed_tokens.weight") || name.ends_with("lm_head.weight") {
        return false;
    }
    let gemv = name.ends_with("mlp.gate_proj.weight")
        || name.ends_with("mlp.up_proj.weight")
        || name.ends_with("mlp.down_proj.weight")
        || name.ends_with("self_attn.q_proj.weight")
        || name.ends_with("self_attn.k_proj.weight")
        || name.ends_with("self_attn.v_proj.weight")
        || name.ends_with("self_attn.o_proj.weight")
        || name.contains("linear_attn.in_proj")
        || name.ends_with("linear_attn.out_proj.weight");
    if gemv {
        return false;
    }
    let elements = shape.iter().try_fold(1usize, |a, b| a.checked_mul(*b));
    matches!(elements, Some(n) if n <= 65_536)
}

pub fn classify_qwen38_mixed_payload(
    codec: u8,
    payload: &[u8],
    name: &str,
    shape: &[usize],
) -> Result<MixedCatalogLane> {
    match codec {
        0 | 1 | 2 => Ok(MixedCatalogLane::Packed(codec)),
        3 => {
            if payload.len() >= 8 && payload[..8] == MAGIC_UNIFORM {
                if hgravu_is_vector(name, shape) {
                    Ok(MixedCatalogLane::HgravuVector)
                } else {
                    Ok(MixedCatalogLane::Packed(3))
                }
            } else if payload.len() >= 8 && payload[..8] == HQ30UQ4_MAGIC {
                Ok(MixedCatalogLane::Hq30Uq4)
            } else {
                Err(mixed_error(format!(
                    "{name} codec 3 magic {:?} is not HGRAVU01/HQ30UQ4; refusing silent fallback",
                    payload.get(..8)
                )))
            }
        }
        4 => {
            // Validate the f32v2 envelope now so a short payload refuses at
            // classify time, not after a later rmsnorm miss.
            let _ = read_qwen38_f32_payload(payload)?;
            if let Some(n) = shape.iter().try_fold(1usize, |a, b| a.checked_mul(*b)) {
                let got = u64::from_le_bytes(payload[0..8].try_into().unwrap()) as usize;
                if got != n {
                    return Err(mixed_error(format!(
                        "{name} f32v2 numel {got} != shape product {n}"
                    )));
                }
            }
            Ok(MixedCatalogLane::F32v2)
        }
        other => Err(mixed_error(format!(
            "{name} unknown mixed codec {other}; refusing silent fallback"
        ))),
    }
}

/// Packed MLP kinds `load_mixed` may bind. Uniform is HGRAVU01 when the
/// pack declares it. HQ30UQ4 / f32 / missing stay outside this set so the
/// lock can still refuse a reconstruct or an unsupported role.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MixedMlpNativeKind {
    Binary,
    Residual,
    Hgravs,
    Uniform,
}

pub fn mixed_mlp_native_kind_from_lane(lane: MixedCatalogLane) -> Option<MixedMlpNativeKind> {
    match lane {
        MixedCatalogLane::Packed(0) => Some(MixedMlpNativeKind::Binary),
        MixedCatalogLane::Packed(1) => Some(MixedMlpNativeKind::Residual),
        MixedCatalogLane::Packed(2) => Some(MixedMlpNativeKind::Hgravs),
        MixedCatalogLane::Packed(3) => Some(MixedMlpNativeKind::Uniform),
        MixedCatalogLane::Packed(_)
        | MixedCatalogLane::Hq30Uq4
        | MixedCatalogLane::F32v2
        | MixedCatalogLane::HgravuVector => None,
    }
}

fn mixed_mlp_role_allowed(suffix: &str, kind: MixedMlpNativeKind) -> bool {
    match suffix {
        "mlp.gate_proj.weight" => matches!(
            kind,
            MixedMlpNativeKind::Binary | MixedMlpNativeKind::Uniform
        ),
        "mlp.up_proj.weight" => matches!(
            kind,
            MixedMlpNativeKind::Residual | MixedMlpNativeKind::Uniform
        ),
        "mlp.down_proj.weight" => matches!(
            kind,
            MixedMlpNativeKind::Hgravs | MixedMlpNativeKind::Uniform
        ),
        _ => false,
    }
}

/// CPU-side MLP role lock. Mixed-2p0 assignment (gate Binary / up Residual /
/// down Hgravs) still passes. Pack-declared Uniform passes on any of those
/// three roles. Anything else — missing, HQ30UQ4, f32, Residual-on-gate —
/// refuses.
pub fn assert_mixed_mlp_native_kinds(
    lookup: impl Fn(&str) -> Option<MixedMlpNativeKind>,
) -> Result<()> {
    const ROLES: [(&str, &str); 3] = [
        ("mlp.gate_proj.weight", "HGRAVB01"),
        ("mlp.up_proj.weight", "HGRAVR02"),
        ("mlp.down_proj.weight", "HGRAVS01"),
    ];
    for layer in 0..QWEN38_LAYERS {
        for (suffix, label) in ROLES {
            let name = qwen38_layer_name(layer, suffix);
            match lookup(&name) {
                Some(kind) if mixed_mlp_role_allowed(suffix, kind) => {}
                Some(_) => {
                    return Err(mixed_error(format!(
                        "{name} is not {label} or HGRAVU01; refusing reconstructed MLP"
                    )))
                }
                None => {
                    return Err(mixed_error(format!(
                        "missing {name}; refusing silent dense/Q4 fallback"
                    )))
                }
            }
        }
    }
    Ok(())
}

fn read_catalog_prefix(row: &Qwen38MixedCatalogRow, n: usize) -> Result<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = fs::File::open(&row.segment_path).map_err(|error| {
        mixed_error(format!(
            "cannot open {}: {error}",
            row.segment_path.display()
        ))
    })?;
    file.seek(SeekFrom::Start(row.offset))
        .map_err(|error| mixed_error(format!("seek {}: {error}", row.name)))?;
    let take = n.min(usize::try_from(row.nbytes).unwrap_or(0));
    let mut prefix = vec![0u8; take];
    file.read_exact(&mut prefix)
        .map_err(|error| mixed_error(format!("read {}: {error}", row.name)))?;
    Ok(prefix)
}

fn is_mixed_mlp_gemv_name(name: &str) -> bool {
    name.ends_with("mlp.gate_proj.weight")
        || name.ends_with("mlp.up_proj.weight")
        || name.ends_with("mlp.down_proj.weight")
}

/// Walk only MLP GEMV rows of an HQ38M20 catalog. Does not open Metal and
/// does not read full payloads — 64-byte prefixes are enough to classify
/// codecs 0–3. Used to prove a pack admits before a GPU lane loads it.
pub fn assert_mixed_mlp_native_catalog(root: impl AsRef<Path>) -> Result<()> {
    let rows = parse_qwen38_mixed_catalog(root.as_ref())?;
    let mut kinds = HashMap::new();
    for row in &rows {
        if !is_mixed_mlp_gemv_name(&row.name) {
            continue;
        }
        if row.codec > 3 {
            continue;
        }
        let prefix = read_catalog_prefix(row, 64)?;
        let lane = classify_qwen38_mixed_payload(row.codec, &prefix, &row.name, &row.shape)?;
        if let Some(kind) = mixed_mlp_native_kind_from_lane(lane) {
            kinds.insert(row.name.clone(), kind);
        }
    }
    assert_mixed_mlp_native_kinds(|name| kinds.get(name).copied())
}

/// Walk `catalog.hq38m20` without opening Metal. Rice stays packed.
pub fn census_qwen38_mixed_catalog(root: impl AsRef<Path>) -> Result<MixedCatalogCensus> {
    let rows = parse_qwen38_mixed_catalog(root.as_ref())?;
    let mut census = MixedCatalogCensus {
        tensors: rows.len(),
        ..MixedCatalogCensus::default()
    };
    for row in &rows {
        let payload = read_catalog_payload(row)?;
        match classify_qwen38_mixed_payload(row.codec, &payload, &row.name, &row.shape) {
            Ok(MixedCatalogLane::Packed(codec)) => match mixed_gpu_layout(codec, &payload) {
                Ok(_) => match codec {
                    0 => census.binary += 1,
                    1 => census.residual += 1,
                    2 => census.hgravs += 1,
                    3 => census.uniform += 1,
                    _ => {}
                },
                Err(error) => {
                    census.refused += 1;
                    census.refusals.push(format!("{} layout: {error}", row.name));
                }
            },
            Ok(MixedCatalogLane::Hq30Uq4) => match parse_uniform_q4_header(&payload) {
                Ok(_) => census.q4 += 1,
                Err(error) => {
                    census.refused += 1;
                    census.refusals.push(format!("{} q4: {error}", row.name));
                }
            },
            Ok(MixedCatalogLane::F32v2) => {
                census.f32 += 1;
            }
            Ok(MixedCatalogLane::HgravuVector) => {
                census.f32 += 1;
            }
            Err(error) => {
                census.refused += 1;
                census.refusals.push(format!("{error}"));
            }
        }
    }
    Ok(census)
}

#[derive(Clone, Debug)]
struct Qwen38MixedCatalogRow {
    name: String,
    codec: u8,
    shape: Vec<usize>,
    segment_path: PathBuf,
    offset: u64,
    nbytes: u64,
}

fn read_u16_at(raw: &[u8], off: usize) -> Result<u16> {
    let slice = raw
        .get(off..off + 2)
        .ok_or_else(|| mixed_error("catalog truncated at u16"))?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_u32_at(raw: &[u8], off: usize) -> Result<u32> {
    let slice = raw
        .get(off..off + 4)
        .ok_or_else(|| mixed_error("catalog truncated at u32"))?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_u64_at(raw: &[u8], off: usize) -> Result<u64> {
    let slice = raw
        .get(off..off + 8)
        .ok_or_else(|| mixed_error("catalog truncated at u64"))?;
    Ok(u64::from_le_bytes(slice.try_into().unwrap()))
}

/// Segment filenames are `segments/<name>` by default. An absolute filename
/// (used when a sandbox cannot hardlink into `segments/`) is kept as-is so
/// the catalog can name already-packed blobs without copying 4.34 GB.
fn resolve_mixed_segment_path(root: &Path, filename: &str) -> PathBuf {
    let raw = Path::new(filename);
    if raw.is_absolute() {
        raw.to_path_buf()
    } else {
        root.join("segments").join(filename)
    }
}

fn parse_qwen38_mixed_catalog(root: &Path) -> Result<Vec<Qwen38MixedCatalogRow>> {
    let catalog_path = root.join(QWEN38_MIXED_CATALOG_NAME);
    let raw = fs::read(&catalog_path).map_err(|error| {
        mixed_error(format!("cannot read {}: {error}", catalog_path.display()))
    })?;
    if raw.len() < 32 || raw[..8] != QWEN38_MIXED_CATALOG_MAGIC {
        return Err(mixed_error("catalog magic is not HQ38M20"));
    }
    let version = read_u32_at(&raw, 8)?;
    if version != QWEN38_MIXED_CATALOG_VERSION {
        return Err(mixed_error(format!("unsupported catalog version {version}")));
    }
    let n_tensors = read_u32_at(&raw, 12)? as usize;
    let n_segments = read_u32_at(&raw, 16)? as usize;
    let name_blob_bytes = read_u32_at(&raw, 24)? as usize;
    let mut cursor = 32usize;
    let mut by_id: HashMap<u16, PathBuf> = HashMap::new();
    for _ in 0..n_segments {
        let id = read_u16_at(&raw, cursor)?;
        let name_len = read_u16_at(&raw, cursor + 2)? as usize;
        cursor += 44;
        let filename = raw
            .get(cursor..cursor + name_len)
            .ok_or_else(|| mixed_error("segment name truncated"))?;
        let filename = std::str::from_utf8(filename)
            .map_err(|_| mixed_error("segment name is not utf-8"))?
            .to_owned();
        cursor += name_len;
        by_id.insert(id, resolve_mixed_segment_path(root, &filename));
    }
    let table_bytes = n_tensors
        .checked_mul(QWEN38_MIXED_RECORD_SIZE)
        .ok_or_else(|| mixed_error("catalog table size overflow"))?;
    let table = raw
        .get(cursor..cursor + table_bytes)
        .ok_or_else(|| mixed_error("catalog tensor table truncated"))?;
    cursor += table_bytes;
    let name_blob = raw
        .get(cursor..cursor + name_blob_bytes)
        .ok_or_else(|| mixed_error("catalog name blob truncated"))?;
    let mut rows = Vec::with_capacity(n_tensors);
    for index in 0..n_tensors {
        let rec = &table[index * QWEN38_MIXED_RECORD_SIZE
            ..(index + 1) * QWEN38_MIXED_RECORD_SIZE];
        let name_off = read_u32_at(rec, 0)? as usize;
        let name_len = read_u16_at(rec, 4)? as usize;
        let codec = rec[6];
        let ndim = rec[8] as usize;
        if ndim > 4 {
            return Err(mixed_error("catalog ndim exceeds 4"));
        }
        let mut shape = Vec::with_capacity(ndim);
        for dim in 0..ndim {
            shape.push(read_u32_at(rec, 12 + dim * 4)? as usize);
        }
        let segment_id = read_u16_at(rec, 36)?;
        let offset = read_u64_at(rec, 40)?;
        let nbytes = read_u64_at(rec, 48)?;
        let name = name_blob
            .get(name_off..name_off + name_len)
            .ok_or_else(|| mixed_error("tensor name out of blob"))?;
        let name = std::str::from_utf8(name)
            .map_err(|_| mixed_error("tensor name is not utf-8"))?
            .to_owned();
        let segment_path = by_id
            .get(&segment_id)
            .cloned()
            .ok_or_else(|| mixed_error(format!("unknown segment_id {segment_id}")))?;
        rows.push(Qwen38MixedCatalogRow {
            name,
            codec,
            shape,
            segment_path,
            offset,
            nbytes,
        });
    }
    Ok(rows)
}

fn read_catalog_payload(row: &Qwen38MixedCatalogRow) -> Result<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = fs::File::open(&row.segment_path).map_err(|error| {
        mixed_error(format!(
            "cannot open {}: {error}",
            row.segment_path.display()
        ))
    })?;
    file.seek(SeekFrom::Start(row.offset)).map_err(|error| {
        mixed_error(format!("seek {}: {error}", row.name))
    })?;
    let n = usize::try_from(row.nbytes).map_err(|_| mixed_error("payload exceeds usize"))?;
    let mut payload = vec![0u8; n];
    file.read_exact(&mut payload)
        .map_err(|error| mixed_error(format!("read {}: {error}", row.name)))?;
    Ok(payload)
}

fn dequant_hgravu_vector(payload: &[u8], name: &str) -> Result<Vec<f32>> {
    let layout = mixed_gpu_layout(3, payload)?;
    let MixedGpuKind::Uniform(factor) = layout.kind else {
        return Err(mixed_error(format!("{name} codec-3 layout is not uniform")));
    };
    let elements = (factor.rows as usize)
        .checked_mul(factor.cols as usize)
        .ok_or_else(|| mixed_error(format!("{name} uniform length overflows")))?;
    if elements > 65_536 {
        return Err(mixed_error(format!(
            "{name} dequant refuses {elements} elements (dense W)"
        )));
    }
    let scales = &payload[factor.scale_off..factor.scale_off + factor.scale_bytes];
    let mut scales_f16 = Vec::with_capacity(factor.scale_bytes / 2);
    for chunk in scales.chunks_exact(2) {
        scales_f16.push(u16::from_le_bytes([chunk[0], chunk[1]]));
    }
    let packed = UniformFactorPacked {
        rows: factor.rows as usize,
        cols: factor.cols as usize,
        bits: u8::try_from(factor.bits).map_err(|_| mixed_error("HGRAVU01 bits"))?,
        group_size: factor.group_size as usize,
        groups: elements.div_ceil(factor.group_size as usize),
        bound: u16::try_from(factor.bound).map_err(|_| mixed_error("HGRAVU01 bound"))?,
        scales_f16,
        codes: payload[factor.code_off..factor.code_off + factor.code_bytes].to_vec(),
    };
    let mut values = Vec::with_capacity(elements);
    for row in 0..packed.rows {
        for col in 0..packed.cols {
            values.push(uniform_factor_value(&packed, row, col));
        }
    }
    mlx_residual_norm_to_delta_named(name, &mut values);
    Ok(values)
}

pub const QWEN38_Q4_MATVEC_KERNEL: &str = "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128";
pub const QWEN38_Q4_ADDR_PROBE_KERNEL: &str =
    "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128_addr_probe";
pub const QWEN38_Q4_DECODE_PROBE_KERNEL: &str =
    "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128_decode_probe";
pub const QWEN38_F32_STREAM_PROBE_KERNEL: &str = "qwen38_f32_stream_probe";
pub const QWEN38_HGRAVU01_Q3_GEO_TPR64: &str =
    "qwen_uniform_q3_group64_matvec_geo_tpr64_tg128";
pub const QWEN38_HGRAVU01_Q4_GEO_TPR64: &str =
    "qwen_uniform_hgravu_q4_group64_matvec_geo_tpr64_tg128";

/// G0-class launch for Uniform HGRAVU01 bits 3/4. None leaves the
/// incumbent simd / simd3 / uniform8 / serial path in `dispatch_factor`.
/// HGRAVS r160 factors stay on that path because they are not Uniform.
pub fn qwen38_hgravu01_geo_tpr64_launch(
    bits: u32,
    group_size: u32,
    rows: u32,
    cols: u32,
) -> Option<(&'static str, (u32, u32, u32), (u32, u32, u32))> {
    if !qwen38_recon_fuse_enabled() || group_size != 64 || cols % 64 != 0 {
        return None;
    }
    let name = match bits {
        3 => QWEN38_HGRAVU01_Q3_GEO_TPR64,
        4 => QWEN38_HGRAVU01_Q4_GEO_TPR64,
        _ => return None,
    };
    let tg = 128u32;
    let grid = rows.div_ceil(2).saturating_mul(tg).max(tg);
    Some((name, (grid, 1, 1), (tg, 1, 1)))
}

/// Shipped uniform-Q4 matvec bindings. The Qwen3.8 default is the geometry-
/// sweep winner (`geo_tpr64_tg128`), tuned on Q80's 512×2048 organs. The
/// other names are already in `qwen_uniform_q4.metal`; this enum only
/// retargets launch geometry. It does not generate new shaders.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen38MatvecKernel {
    GeoTpr64Tg128,
    Vecgroup,
    VecgroupX64,
    VecgroupR4,
}

impl Qwen38MatvecKernel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GeoTpr64Tg128 => QWEN38_Q4_MATVEC_KERNEL,
            Self::Vecgroup => "qwen_uniform_q4_group64_matvec_vecgroup",
            Self::VecgroupX64 => "qwen_uniform_q4_group64_matvec_vecgroup_x64",
            Self::VecgroupR4 => "qwen_uniform_q4_group64_matvec_vecgroup_r4",
        }
    }

    /// (grid, threadgroup) for `rows` output elements.
    pub fn launch(self, rows: u32) -> ((u32, u32, u32), (u32, u32, u32)) {
        match self {
            Self::GeoTpr64Tg128 => {
                let tg = 128u32;
                let grid = rows.div_ceil(2).saturating_mul(tg).max(tg);
                ((grid, 1, 1), (tg, 1, 1))
            }
            Self::Vecgroup => {
                let tg = 256u32;
                let grid = rows.div_ceil(8).saturating_mul(tg).max(tg);
                ((grid, 1, 1), (tg, 1, 1))
            }
            Self::VecgroupX64 => {
                let tg = 256u32;
                let grid = rows.div_ceil(4).saturating_mul(tg).max(tg);
                ((grid, 1, 1), (tg, 1, 1))
            }
            Self::VecgroupR4 => {
                let tg = 256u32;
                let grid = rows.div_ceil(32).saturating_mul(tg).max(tg);
                ((grid, 1, 1), (tg, 1, 1))
            }
        }
    }

    pub fn all() -> &'static [Self] {
        &[
            Self::GeoTpr64Tg128,
            Self::Vecgroup,
            Self::VecgroupX64,
            Self::VecgroupR4,
        ]
    }
}

/// Per-class GPU times from residual-correct split command buffers.
/// Each field's `gpu_ns` is `GPUEndTime-GPUStartTime` on that class's CBs.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Qwen38ClassTiming {
    pub embed_gpu_ns: Option<u64>,
    pub embed_wait_ns: u64,
    pub mixer_gpu_ns: u64,
    pub mixer_wait_ns: u64,
    pub mlp_gpu_ns: u64,
    pub mlp_wait_ns: u64,
    pub terminal_gpu_ns: Option<u64>,
    pub terminal_wait_ns: u64,
    pub deltanet_gpu_ns: u64,
    pub gqa_gpu_ns: u64,
    pub sampled: u32,
    pub layer_mlp_gpu_ns: Vec<u64>,
    pub layer_mixer_gpu_ns: Vec<u64>,
}

pub fn render_qwen38_user_chat(user_text: &str) -> String {
    format!("<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n")
}

/// Per-session Metal workspace size. Independent of the weight set.
/// KV grows with `max_seq_len`; DeltaNet state does not.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub struct Qwen38WorkspaceBytes {
    pub max_seq_len: usize,
    pub activation_bytes: usize,
    pub deltanet_state_bytes: usize,
    pub gqa_kv_bytes: usize,
    pub total_bytes: usize,
}

pub fn qwen38_workspace_bytes(max_seq_len: usize) -> Result<Qwen38WorkspaceBytes> {
    if max_seq_len == 0 {
        return Err(Error::Model("qwen38 max_seq_len must be positive".into()));
    }
    let layout = Qwen38DeltaNetLayout::source_exact();
    let f32b = |n: usize| {
        n.checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| Error::Model("qwen38 workspace overflow".into()))
    };
    let hidden = f32b(QWEN38_HIDDEN)?;
    let qkvz = f32b(layout.qkvz_rows())?;
    let ba = f32b(layout.ba_rows())?;
    let value = f32b(layout.value_elements())?;
    let q_proj = f32b(QWEN38_GQA_HEADS * QWEN38_GQA_HEAD_DIM * 2)?;
    let kv = f32b(QWEN38_GQA_KV_HEADS * QWEN38_GQA_HEAD_DIM)?;
    let query = f32b(QWEN38_GQA_HEADS * QWEN38_GQA_HEAD_DIM)?;
    let mid = f32b(QWEN38_INTERMEDIATE)?;
    let logits = f32b(QWEN38_VOCAB)?;
    let conv = f32b(48 * layout.conv_state_elements())?;
    let rec = f32b(48 * layout.recurrent_state_elements())?;
    let kv_cache = f32b(
        QWEN38_GQA_LAYERS
            .checked_mul(max_seq_len)
            .and_then(|n| n.checked_mul(QWEN38_GQA_KV_HEADS))
            .and_then(|n| n.checked_mul(QWEN38_GQA_HEAD_DIM))
            .ok_or_else(|| Error::Model("qwen38 KV cache overflow".into()))?,
    )?;
    let hgravs = f32b(QWEN38_MIXED_HGRAVS_RANK)?;
    let split_qkv = f32b(crate::model::qwen38_geometry::QWEN38_IN_PROJ_QKV_ROWS)?;
    let split_b = f32b(crate::model::qwen38_geometry::QWEN38_IN_PROJ_B_ROWS)?;
    let split_a = f32b(crate::model::qwen38_geometry::QWEN38_IN_PROJ_A_ROWS)?;
    let sampled = std::mem::size_of::<u32>();
    let heads_f32 = f32b(layout.value_heads)?;
    let activation = hidden
        .checked_mul(2)
        .and_then(|n| n.checked_add(qkvz))
        .and_then(|n| n.checked_add(ba))
        .and_then(|n| n.checked_add(value.checked_mul(6)?))
        .and_then(|n| n.checked_add(heads_f32.checked_mul(2)?))
        .and_then(|n| n.checked_add(hidden.checked_mul(2)?))
        .and_then(|n| n.checked_add(q_proj))
        .and_then(|n| n.checked_add(kv.checked_mul(2)?))
        .and_then(|n| n.checked_add(query.checked_mul(3)?))
        .and_then(|n| n.checked_add(mid.checked_mul(3)?))
        .and_then(|n| n.checked_add(hidden))
        .and_then(|n| n.checked_add(logits))
        .and_then(|n| n.checked_add(sampled))
        .and_then(|n| n.checked_add(hgravs))
        .and_then(|n| n.checked_add(split_qkv))
        .and_then(|n| n.checked_add(split_b))
        .and_then(|n| n.checked_add(split_a))
        .ok_or_else(|| Error::Model("qwen38 workspace overflow".into()))?;
    let deltanet = conv
        .checked_add(rec)
        .ok_or_else(|| Error::Model("qwen38 workspace overflow".into()))?;
    let gqa = kv_cache
        .checked_mul(2)
        .ok_or_else(|| Error::Model("qwen38 workspace overflow".into()))?;
    let total = activation
        .checked_add(deltanet)
        .and_then(|n| n.checked_add(gqa))
        .ok_or_else(|| Error::Model("qwen38 workspace overflow".into()))?;
    Ok(Qwen38WorkspaceBytes {
        max_seq_len,
        activation_bytes: activation,
        deltanet_state_bytes: deltanet,
        gqa_kv_bytes: gqa,
        total_bytes: total,
    })
}

pub fn load_qwen38_tokenizer(path: impl AsRef<Path>) -> Result<Tokenizer> {
    Tokenizer::from_file(path)
}

#[cfg(target_os = "macos")]
mod device {
    use super::*;
    use crate::kernels::{mha_decode_f32_tcb, qwen_next_add_residual_tcb, sample_argmax_f32_tcb};
    use crate::metal::{CommandBufferTiming, MetalContext, PinnedBuffer, TokenCommandBuffer};
    use std::thread;
    use std::time::Instant;

    fn zero_buffer(buffer: &PinnedBuffer) {
        let len = buffer.length() as usize;
        unsafe {
            std::ptr::write_bytes(buffer.contents() as *mut u8, 0, len);
        }
    }

    struct Q4Weight {
        rows: usize,
        cols: usize,
        codes: PinnedBuffer,
        scales: PinnedBuffer,
    }

    struct GpuBinary {
        signs: PinnedBuffer,
        scales: PinnedBuffer,
        rows: u32,
        cols: u32,
        group_size: u32,
        groups_per_row: u32,
    }

    struct GpuResidual {
        binary: GpuBinary,
        indices: PinnedBuffer,
        row_ptr: PinnedBuffer,
        residual_signs: PinnedBuffer,
        residual_scale_f16: u32,
    }

    struct GpuHgravs {
        left_codes: PinnedBuffer,
        left_scales: PinnedBuffer,
        right_codes: PinnedBuffer,
        right_scales: PinnedBuffer,
        left_rows: u32,
        left_cols: u32,
        right_rows: u32,
        right_cols: u32,
        group_size: u32,
        bits: u32,
        bound: u32,
    }

    struct GpuUniform {
        codes: PinnedBuffer,
        scales: PinnedBuffer,
        rows: u32,
        cols: u32,
        group_size: u32,
        bits: u32,
        bound: u32,
    }

    enum MixedGpuWeight {
        Binary(GpuBinary),
        Residual(GpuResidual),
        Hgravs(GpuHgravs),
        Uniform(GpuUniform),
    }

    impl MixedGpuWeight {
        fn resident_bytes(&self) -> u64 {
            match self {
                Self::Binary(body) => body.signs.length() + body.scales.length(),
                Self::Residual(body) => {
                    body.binary.signs.length()
                        + body.binary.scales.length()
                        + body.indices.length()
                        + body.row_ptr.length()
                        + body.residual_signs.length()
                }
                Self::Hgravs(body) => {
                    body.left_codes.length()
                        + body.left_scales.length()
                        + body.right_codes.length()
                        + body.right_scales.length()
                }
                Self::Uniform(body) => body.codes.length() + body.scales.length(),
            }
        }
    }

    /// One resident Metal copy of the Qwen3.8 catalog. Sessions clone the
    /// `Arc` and allocate only workspace / KV.
    pub struct Qwen38HybridWeights {
        context: MetalContext,
        q4: HashMap<String, Q4Weight>,
        f32s: HashMap<String, PinnedBuffer>,
        mixed: HashMap<String, MixedGpuWeight>,
    }

    impl Qwen38HybridWeights {
        pub fn load(root: impl AsRef<Path>) -> Result<Self> {
            qwen38_assert_schedule_intact()?;
            let root = root.as_ref();
            if root.join(QWEN38_MIXED_CATALOG_NAME).is_file() {
                return Self::load_mixed(root);
            }
            let (_manifest, rows) = load_qwen38_manifest(root)?;
            if rows.len() != QWEN38_EXPECTED_CATALOG_TENSORS {
                return Err(Error::Model(format!(
                    "qwen38 catalog has {} tensors, expected {QWEN38_EXPECTED_CATALOG_TENSORS}",
                    rows.len()
                )));
            }
            eprintln!(
                "qwen38-decode opening Metal + {} catalog tensors",
                rows.len()
            );
            let context = MetalContext::new()?;
            let mut q4 = HashMap::new();
            let mut f32s = HashMap::new();
            let tensors_dir = root.join("tensors");
            for (i, row) in rows.iter().enumerate() {
                if i % 50 == 0 {
                    eprintln!("qwen38-decode upload {i}/{}", rows.len());
                }
                let path = tensors_dir.join(&row.artifact);
                let payload = fs::read(&path).map_err(|error| {
                    Error::Model(format!("cannot read {}: {error}", path.display()))
                })?;
                match row.kind.as_str() {
                    "q4" => {
                        let header = parse_uniform_q4_header(&payload)?;
                        let scales = &payload[header.scale_offset..header.sign_offset];
                        let codes = &payload[header.sign_offset..header.payload_bytes];
                        let (rows_n, cols) = match header.shape.as_slice() {
                            [r, c] => (*r, *c),
                            other => {
                                return Err(Error::Model(format!(
                                    "{} Q4 rank {:?} is not a matrix",
                                    row.name, other
                                )))
                            }
                        };
                        q4.insert(
                            row.name.clone(),
                            Q4Weight {
                                rows: rows_n,
                                cols,
                                codes: context.new_buffer_with_bytes_checked(codes)?,
                                scales: context.new_buffer_with_bytes_checked(scales)?,
                            },
                        );
                    }
                    "f32" => {
                        let values = read_qwen38_f32_payload(&payload)?;
                        f32s.insert(
                            row.name.clone(),
                            context.new_buffer_with_bytes_checked(bytemuck::cast_slice(&values))?,
                        );
                    }
                    other => {
                        return Err(Error::Model(format!(
                            "qwen38 catalog kind {other:?} is not q4/f32"
                        )))
                    }
                }
            }
            Ok(Self {
                context,
                q4,
                f32s,
                mixed: HashMap::new(),
            })
        }

        fn load_mixed(root: &Path) -> Result<Self> {
            let rows = parse_qwen38_mixed_catalog(root)?;
            if rows.is_empty() {
                return Err(mixed_error("HQ38M20 catalog has no tensors"));
            }
            eprintln!(
                "qwen38-decode opening mixed HQ38M20 + {} catalog tensors (no reconstruct-to-Q4)",
                rows.len()
            );
            let context = MetalContext::new()?;
            let mut q4 = HashMap::new();
            let mut f32s = HashMap::new();
            let mut mixed = HashMap::new();
            let mut census = MixedCatalogCensus {
                tensors: rows.len(),
                ..MixedCatalogCensus::default()
            };
            for (i, row) in rows.iter().enumerate() {
                if i % 50 == 0 {
                    eprintln!("qwen38-decode mixed upload {i}/{}", rows.len());
                }
                let payload = read_catalog_payload(row)?;
                match classify_qwen38_mixed_payload(
                    row.codec,
                    &payload,
                    &row.name,
                    &row.shape,
                )? {
                    MixedCatalogLane::Packed(codec) => {
                        mixed.insert(
                            row.name.clone(),
                            Qwen38HybridDecodeSession::upload_mixed(
                                &context, codec, &payload, &row.name,
                            )?,
                        );
                        match codec {
                            0 => census.binary += 1,
                            1 => census.residual += 1,
                            2 => census.hgravs += 1,
                            3 => census.uniform += 1,
                            _ => {}
                        }
                    }
                    MixedCatalogLane::Hq30Uq4 => {
                        let header = parse_uniform_q4_header(&payload)?;
                        let scales = &payload[header.scale_offset..header.sign_offset];
                        let codes = &payload[header.sign_offset..header.payload_bytes];
                        let (rows_n, cols) = match header.shape.as_slice() {
                            [r, c] => (*r, *c),
                            other => {
                                return Err(mixed_error(format!(
                                    "{} HQ30UQ4 rank {:?} is not a matrix",
                                    row.name, other
                                )))
                            }
                        };
                        q4.insert(
                            row.name.clone(),
                            Q4Weight {
                                rows: rows_n,
                                cols,
                                codes: context.new_buffer_with_bytes_checked(codes)?,
                                scales: context.new_buffer_with_bytes_checked(scales)?,
                            },
                        );
                        census.q4 += 1;
                    }
                    MixedCatalogLane::F32v2 => {
                        // Already HF δ (f32v2 oracle). Do not mlx-delta.
                        let values = read_qwen38_f32_payload(&payload)?;
                        f32s.insert(
                            row.name.clone(),
                            context.new_buffer_with_bytes_checked(bytemuck::cast_slice(
                                &values,
                            ))?,
                        );
                        census.f32 += 1;
                    }
                    MixedCatalogLane::HgravuVector => {
                        let values = dequant_hgravu_vector(&payload, &row.name)?;
                        f32s.insert(
                            row.name.clone(),
                            context.new_buffer_with_bytes_checked(bytemuck::cast_slice(
                                &values,
                            ))?,
                        );
                        census.f32 += 1;
                    }
                }
            }
            eprintln!(
                "qwen38-decode mixed census: tensors={} binary={} residual={} \
                 hgravs={} uniform={} q4={} f32={} refused=0 expanded_to_q4=0 \
                 expanded_to_float_gemv=0",
                census.tensors,
                census.binary,
                census.residual,
                census.hgravs,
                census.uniform,
                census.q4,
                census.f32
            );
            eprintln!("{}", qwen38_mixed_k_complete_bind_message());
            Qwen38HybridDecodeSession::assert_mixed_mlp_native(&mixed)?;
            Ok(Self {
                context,
                q4,
                f32s,
                mixed,
            })
        }

        pub fn resident_bytes(&self) -> u64 {
            let q4: u64 = self
                .q4
                .values()
                .map(|w| w.codes.length() + w.scales.length())
                .sum();
            let f32s: u64 = self.f32s.values().map(|b| b.length()).sum();
            let mixed: u64 = self.mixed.values().map(MixedGpuWeight::resident_bytes).sum();
            q4 + f32s + mixed
        }

        pub fn q4_tensor_count(&self) -> usize {
            self.q4.len()
        }

        pub fn mixed_tensor_count(&self) -> usize {
            self.mixed.len()
        }

        pub fn f32_tensor_count(&self) -> usize {
            self.f32s.len()
        }
    }

    fn set_u32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: u32) {
        encoder.set_bytes(index, 4, &value as *const u32 as *const _);
    }

    fn tg256_grid(rows: u32) -> (u32, u32, u32) {
        (rows.saturating_mul(256), 1, 1)
    }

    fn simd8_grid(rows: u32) -> (u32, u32, u32) {
        (rows.div_ceil(8).saturating_mul(256).max(256), 1, 1)
    }

    pub struct Qwen38HybridWorkspace {
        hidden: PinnedBuffer,
        normalized: PinnedBuffer,
        qkvz: PinnedBuffer,
        ba: PinnedBuffer,
        repeated_q: PinnedBuffer,
        repeated_k: PinnedBuffer,
        conv_v: PinnedBuffer,
        z: PinnedBuffer,
        decay: PinnedBuffer,
        beta: PinnedBuffer,
        rec_out: PinnedBuffer,
        gated: PinnedBuffer,
        mixer: PinnedBuffer,
        first_residual: PinnedBuffer,
        q_proj: PinnedBuffer,
        k_proj: PinnedBuffer,
        v_proj: PinnedBuffer,
        query: PinnedBuffer,
        attn: PinnedBuffer,
        gated_attn: PinnedBuffer,
        gate: PinnedBuffer,
        up: PinnedBuffer,
        act: PinnedBuffer,
        down: PinnedBuffer,
        logits: PinnedBuffer,
        sampled: PinnedBuffer,
        conv_state: PinnedBuffer,
        rec_state: PinnedBuffer,
        gqa_key: PinnedBuffer,
        gqa_value: PinnedBuffer,
        hgravs_mid: PinnedBuffer,
        split_qkv: PinnedBuffer,
        split_b: PinnedBuffer,
        split_a: PinnedBuffer,
    }

    impl Qwen38HybridWorkspace {
        fn allocate(ctx: &MetalContext, max_seq_len: usize) -> Result<Self> {
            let layout = Qwen38DeltaNetLayout::source_exact();
            let f32b = |n: usize| {
                n.checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| Error::Model("qwen38 workspace overflow".into()))
            };
            let hidden = f32b(QWEN38_HIDDEN)?;
            let qkvz = f32b(layout.qkvz_rows())?;
            let ba = f32b(layout.ba_rows())?;
            let value = f32b(layout.value_elements())?;
            let q_proj = f32b(QWEN38_GQA_HEADS * QWEN38_GQA_HEAD_DIM * 2)?;
            let kv = f32b(QWEN38_GQA_KV_HEADS * QWEN38_GQA_HEAD_DIM)?;
            let query = f32b(QWEN38_GQA_HEADS * QWEN38_GQA_HEAD_DIM)?;
            let mid = f32b(QWEN38_INTERMEDIATE)?;
            let logits = f32b(QWEN38_VOCAB)?;
            let conv = f32b(48 * layout.conv_state_elements())?;
            let rec = f32b(48 * layout.recurrent_state_elements())?;
            let kv_cache =
                f32b(QWEN38_GQA_LAYERS * max_seq_len * QWEN38_GQA_KV_HEADS * QWEN38_GQA_HEAD_DIM)?;
            Ok(Self {
                hidden: ctx.new_buffer_checked(hidden)?,
                normalized: ctx.new_buffer_checked(hidden)?,
                qkvz: ctx.new_buffer_checked(qkvz)?,
                ba: ctx.new_buffer_checked(ba)?,
                repeated_q: ctx.new_buffer_checked(value)?,
                repeated_k: ctx.new_buffer_checked(value)?,
                conv_v: ctx.new_buffer_checked(value)?,
                z: ctx.new_buffer_checked(value)?,
                decay: ctx.new_buffer_checked(f32b(layout.value_heads)?)?,
                beta: ctx.new_buffer_checked(f32b(layout.value_heads)?)?,
                rec_out: ctx.new_buffer_checked(value)?,
                gated: ctx.new_buffer_checked(value)?,
                mixer: ctx.new_buffer_checked(hidden)?,
                first_residual: ctx.new_buffer_checked(hidden)?,
                q_proj: ctx.new_buffer_checked(q_proj)?,
                k_proj: ctx.new_buffer_checked(kv)?,
                v_proj: ctx.new_buffer_checked(kv)?,
                query: ctx.new_buffer_checked(query)?,
                attn: ctx.new_buffer_checked(query)?,
                gated_attn: ctx.new_buffer_checked(query)?,
                gate: ctx.new_buffer_checked(mid)?,
                up: ctx.new_buffer_checked(mid)?,
                act: ctx.new_buffer_checked(mid)?,
                down: ctx.new_buffer_checked(hidden)?,
                logits: ctx.new_buffer_checked(logits)?,
                sampled: ctx.new_buffer_checked(std::mem::size_of::<u32>())?,
                conv_state: ctx.new_buffer_checked(conv)?,
                rec_state: ctx.new_buffer_checked(rec)?,
                gqa_key: ctx.new_buffer_checked(kv_cache)?,
                gqa_value: ctx.new_buffer_checked(kv_cache)?,
                hgravs_mid: ctx.new_buffer_checked(f32b(QWEN38_MIXED_HGRAVS_RANK)?)?,
                split_qkv: ctx.new_buffer_checked(f32b(
                    crate::model::qwen38_geometry::QWEN38_IN_PROJ_QKV_ROWS,
                )?)?,
                split_b: ctx.new_buffer_checked(f32b(
                    crate::model::qwen38_geometry::QWEN38_IN_PROJ_B_ROWS,
                )?)?,
                split_a: ctx.new_buffer_checked(f32b(
                    crate::model::qwen38_geometry::QWEN38_IN_PROJ_A_ROWS,
                )?)?,
            })
        }

        fn resident_bytes(&self) -> u64 {
            [
                &self.hidden,
                &self.normalized,
                &self.qkvz,
                &self.ba,
                &self.repeated_q,
                &self.repeated_k,
                &self.conv_v,
                &self.z,
                &self.decay,
                &self.beta,
                &self.rec_out,
                &self.gated,
                &self.mixer,
                &self.first_residual,
                &self.q_proj,
                &self.k_proj,
                &self.v_proj,
                &self.query,
                &self.attn,
                &self.gated_attn,
                &self.gate,
                &self.up,
                &self.act,
                &self.down,
                &self.logits,
                &self.sampled,
                &self.conv_state,
                &self.rec_state,
                &self.gqa_key,
                &self.gqa_value,
                &self.hgravs_mid,
                &self.split_qkv,
                &self.split_b,
                &self.split_a,
            ]
            .iter()
            .map(|b| b.length())
            .sum()
        }
    }

    pub struct Qwen38HybridDecodeSession {
        #[allow(dead_code)]
        context: MetalContext,
        weights: Arc<Qwen38HybridWeights>,
        workspace: Qwen38HybridWorkspace,
        max_seq_len: usize,
        position: usize,
        pub fallbacks: u32,
        /// Default matches the shipped bring-up binding. Diagnostic lanes may
        /// retarget to another shipped kernel; they must not invent one.
        pub matvec_kernel: Qwen38MatvecKernel,
        /// Overlap independent projections (gate+up, qkvz+ba, q/k/v) in one
        /// concurrent encoder. Off by default so `step` stays bit-identical
        /// to the bring-up vehicle.
        pub concurrent_independent: bool,
        /// Launch one threadgroup per (value-head, value-dim) for the
        /// gated-delta recurrence. Same serial reduction as the Q80 kernel;
        /// the vi columns are independent. Default ON after paired generate
        /// admitted a 42.7→33.4 ms token cut with greedy-identical ids.
        pub deltanet_vi_parallel: bool,
    }

    impl Qwen38HybridDecodeSession {
        pub fn open(root: impl AsRef<Path>, max_seq_len: usize) -> Result<Self> {
            Self::attach(Arc::new(Qwen38HybridWeights::load(root)?), max_seq_len)
        }

        /// New decode session against an already-resident weight set.
        /// Allocates only workspace / KV / DeltaNet state.
        pub fn attach(weights: Arc<Qwen38HybridWeights>, max_seq_len: usize) -> Result<Self> {
            qwen38_assert_schedule_intact()?;
            if max_seq_len == 0 {
                return Err(Error::Model("qwen38 max_seq_len must be positive".into()));
            }
            let workspace = Qwen38HybridWorkspace::allocate(&weights.context, max_seq_len)?;
            let expected = qwen38_workspace_bytes(max_seq_len)?;
            let got = workspace.resident_bytes();
            if got != expected.total_bytes as u64 {
                return Err(Error::Model(format!(
                    "qwen38 workspace bytes {got} != formula {}",
                    expected.total_bytes
                )));
            }
            zero_buffer(&workspace.conv_state);
            zero_buffer(&workspace.rec_state);
            zero_buffer(&workspace.gqa_key);
            zero_buffer(&workspace.gqa_value);
            Ok(Self {
                context: weights.context.clone(),
                weights,
                workspace,
                max_seq_len,
                position: 0,
                fallbacks: 0,
                matvec_kernel: Qwen38MatvecKernel::GeoTpr64Tg128,
                concurrent_independent: false,
                deltanet_vi_parallel: true,
            })
        }

        pub fn share_weights(&self) -> Arc<Qwen38HybridWeights> {
            Arc::clone(&self.weights)
        }

        pub fn weights(&self) -> &Qwen38HybridWeights {
            &self.weights
        }

        pub fn max_seq_len(&self) -> usize {
            self.max_seq_len
        }

        pub fn workspace_resident_bytes(&self) -> u64 {
            self.workspace.resident_bytes()
        }

        pub fn shares_weights_with(&self, other: &Self) -> bool {
            Arc::ptr_eq(&self.weights, &other.weights)
        }



        fn assert_mixed_mlp_native(mixed: &HashMap<String, MixedGpuWeight>) -> Result<()> {
            assert_mixed_mlp_native_kinds(|name| {
                mixed.get(name).map(|weight| match weight {
                    MixedGpuWeight::Binary(_) => MixedMlpNativeKind::Binary,
                    MixedGpuWeight::Residual(_) => MixedMlpNativeKind::Residual,
                    MixedGpuWeight::Hgravs(_) => MixedMlpNativeKind::Hgravs,
                    MixedGpuWeight::Uniform(_) => MixedMlpNativeKind::Uniform,
                })
            })
        }

        fn upload_mixed(
            context: &MetalContext,
            codec: u8,
            payload: &[u8],
            name: &str,
        ) -> Result<MixedGpuWeight> {
            let expected = match codec {
                0 => &MAGIC_BINARY[..],
                1 => &MAGIC_RESIDUAL_COMPACT[..],
                2 => &MAGIC_HGRAVS01[..],
                3 => &MAGIC_UNIFORM[..],
                other => {
                    return Err(mixed_error(format!(
                        "{name} upload unknown codec {other}"
                    )))
                }
            };
            if payload.len() < 8 || payload[..8] != *expected {
                return Err(mixed_error(format!(
                    "{name} codec {codec} magic {:?} != {:?}; refusing silent fallback",
                    payload.get(..8),
                    expected
                )));
            }
            let layout = mixed_gpu_layout(codec, payload)?;
            match layout.kind {
                MixedGpuKind::Binary {
                    scale_off,
                    scale_bytes,
                    sign_off,
                    sign_bytes,
                    group_size,
                    groups_per_row,
                } => Ok(MixedGpuWeight::Binary(GpuBinary {
                    signs: context
                        .new_buffer_with_bytes_checked(&payload[sign_off..sign_off + sign_bytes])?,
                    scales: context
                        .new_buffer_with_bytes_checked(&payload[scale_off..scale_off + scale_bytes])?,
                    rows: layout.rows,
                    cols: layout.cols,
                    group_size,
                    groups_per_row,
                })),
                MixedGpuKind::Residual {
                    ref binary,
                    residual_sign_off,
                    residual_sign_bytes,
                    rice_k,
                    first_index,
                    rice_off,
                    rice_bytes,
                    outlier_count,
                    residual_scale_f16,
                } => {
                    let MixedGpuKind::Binary {
                        scale_off,
                        scale_bytes,
                        sign_off,
                        sign_bytes,
                        group_size,
                        groups_per_row,
                    } = binary.as_ref()
                    else {
                        return Err(mixed_error(format!("{name} residual binary layout drifted")));
                    };
                    let rice = RiceQ1Packed {
                        binary: BinaryGroupPacked {
                            rows: layout.rows as usize,
                            cols: layout.cols as usize,
                            group_size: *group_size as usize,
                            groups_per_row: *groups_per_row as usize,
                            scales_f16: Vec::new(),
                            signs: Vec::new(),
                        },
                        first_index,
                        rice_k,
                        rice_bytes: payload[rice_off..rice_off + rice_bytes].to_vec(),
                        outlier_count,
                        residual_scale_f16: residual_scale_f16 as u16,
                        residual_signs: Vec::new(),
                        indices: Vec::new(),
                    };
                    let indices = expand_rice_indices(&rice)?;
                    let row_ptr =
                        rice_q1_row_ptr(&indices, layout.rows as usize, layout.cols as usize)?;
                    let idx_bytes: Vec<u8> =
                        indices.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let ptr_bytes: Vec<u8> =
                        row_ptr.iter().flat_map(|v| v.to_le_bytes()).collect();
                    Ok(MixedGpuWeight::Residual(GpuResidual {
                        binary: GpuBinary {
                            signs: context.new_buffer_with_bytes_checked(
                                &payload[*sign_off..*sign_off + *sign_bytes],
                            )?,
                            scales: context.new_buffer_with_bytes_checked(
                                &payload[*scale_off..*scale_off + *scale_bytes],
                            )?,
                            rows: layout.rows,
                            cols: layout.cols,
                            group_size: *group_size,
                            groups_per_row: *groups_per_row,
                        },
                        indices: context.new_buffer_with_bytes_checked(&idx_bytes)?,
                        row_ptr: context.new_buffer_with_bytes_checked(&ptr_bytes)?,
                        residual_signs: context.new_buffer_with_bytes_checked(
                            &payload[residual_sign_off..residual_sign_off + residual_sign_bytes],
                        )?,
                        residual_scale_f16,
                    }))
                }
                MixedGpuKind::Hgravs { left, right } => {
                    if left.bits != u32::from(QWEN38_MIXED_HGRAVS_BITS)
                        || right.bits != u32::from(QWEN38_MIXED_HGRAVS_BITS)
                        || left.group_size != QWEN38_MIXED_HGRAVS_GROUP as u32
                        || right.group_size != QWEN38_MIXED_HGRAVS_GROUP as u32
                        || left.cols != QWEN38_MIXED_HGRAVS_RANK as u32
                        || right.rows != QWEN38_MIXED_HGRAVS_RANK as u32
                    {
                        return Err(mixed_error(format!(
                            "{name} HGRAVS01 geometry {}x{} / {}x{} bits={}/{} group={}/{} is not r160_b3",
                            left.rows,
                            left.cols,
                            right.rows,
                            right.cols,
                            left.bits,
                            right.bits,
                            left.group_size,
                            right.group_size
                        )));
                    }
                    Ok(MixedGpuWeight::Hgravs(GpuHgravs {
                        left_codes: context.new_buffer_with_bytes_checked(
                            &payload[left.code_off..left.code_off + left.code_bytes],
                        )?,
                        left_scales: context.new_buffer_with_bytes_checked(
                            &payload[left.scale_off..left.scale_off + left.scale_bytes],
                        )?,
                        right_codes: context.new_buffer_with_bytes_checked(
                            &payload[right.code_off..right.code_off + right.code_bytes],
                        )?,
                        right_scales: context.new_buffer_with_bytes_checked(
                            &payload[right.scale_off..right.scale_off + right.scale_bytes],
                        )?,
                        left_rows: left.rows,
                        left_cols: left.cols,
                        right_rows: right.rows,
                        right_cols: right.cols,
                        group_size: left.group_size,
                        bits: left.bits,
                        bound: left.bound,
                    }))
                }
                MixedGpuKind::Uniform(factor) => Ok(MixedGpuWeight::Uniform(GpuUniform {
                    codes: context.new_buffer_with_bytes_checked(
                        &payload[factor.code_off..factor.code_off + factor.code_bytes],
                    )?,
                    scales: context.new_buffer_with_bytes_checked(
                        &payload[factor.scale_off..factor.scale_off + factor.scale_bytes],
                    )?,
                    rows: factor.rows,
                    cols: factor.cols,
                    group_size: factor.group_size,
                    bits: factor.bits,
                    bound: factor.bound,
                })),
            }
        }

        pub fn reset(&mut self) {
            self.position = 0;
            zero_buffer(&self.workspace.conv_state);
            zero_buffer(&self.workspace.rec_state);
            zero_buffer(&self.workspace.gqa_key);
            zero_buffer(&self.workspace.gqa_value);
        }

        fn q4(&self, name: &str) -> Result<&Q4Weight> {
            self.weights
                .q4
                .get(name)
                .ok_or_else(|| Error::Model(format!("qwen38 missing Q4 {name}")))
        }

        fn f32(&self, name: &str) -> Result<&PinnedBuffer> {
            self.weights
                .f32s
                .get(name)
                .ok_or_else(|| Error::Model(format!("qwen38 missing f32 {name}")))
        }

        fn encode_q4_matvec(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            name: &str,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
        ) -> Result<()> {
            self.encode_q4_matvec_kernel(tcb, name, input, output, self.matvec_kernel.as_str())
        }

        fn encode_named_matvec(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            name: &str,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
        ) -> Result<()> {
            if self.weights.q4.contains_key(name) {
                return self.encode_q4_matvec(tcb, name, input, output);
            }
            if self.weights.mixed.contains_key(name) {
                return self.encode_mixed_matvec(tcb, name, input, output);
            }
            Err(mixed_error(format!(
                "missing GEMV {name}; refusing silent reconstructed or dense path"
            )))
        }

        fn encode_mixed_matvec(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            name: &str,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
        ) -> Result<()> {
            let weight = self
                .weights
                .mixed
                .get(name)
                .ok_or_else(|| mixed_error(format!("missing mixed {name}")))?;
            match weight {
                MixedGpuWeight::Binary(body) => self.dispatch_binary(tcb, body, input, output),
                MixedGpuWeight::Residual(body) => self.dispatch_residual(tcb, body, input, output),
                MixedGpuWeight::Hgravs(body) => self.dispatch_hgravs(tcb, body, input, output),
                MixedGpuWeight::Uniform(body) => self.dispatch_uniform(tcb, body, input, output),
            }
        }

        fn encode_binary_args(
            &self,
            enc: &metal::ComputeCommandEncoderRef,
            packed: &GpuBinary,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
        ) {
            enc.set_buffer(0, Some(&packed.signs), 0);
            enc.set_buffer(1, Some(&packed.scales), 0);
            enc.set_buffer(2, Some(input), 0);
            enc.set_buffer(3, Some(output), 0);
            set_u32(enc, 4, packed.rows);
            set_u32(enc, 5, packed.cols);
            set_u32(enc, 6, packed.group_size);
            set_u32(enc, 7, packed.groups_per_row);
        }

        fn encode_binary_csr_args(
            &self,
            enc: &metal::ComputeCommandEncoderRef,
            packed: &GpuResidual,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
        ) {
            enc.set_buffer(0, Some(&packed.binary.signs), 0);
            enc.set_buffer(1, Some(&packed.binary.scales), 0);
            enc.set_buffer(2, Some(input), 0);
            enc.set_buffer(3, Some(output), 0);
            enc.set_buffer(4, Some(&packed.indices), 0);
            enc.set_buffer(5, Some(&packed.row_ptr), 0);
            enc.set_buffer(6, Some(&packed.residual_signs), 0);
            set_u32(enc, 7, packed.binary.rows);
            set_u32(enc, 8, packed.binary.cols);
            set_u32(enc, 9, packed.binary.group_size);
            set_u32(enc, 10, packed.binary.groups_per_row);
            set_u32(enc, 11, packed.residual_scale_f16);
        }

        fn encode_csr_args(
            &self,
            enc: &metal::ComputeCommandEncoderRef,
            packed: &GpuResidual,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
        ) {
            enc.set_buffer(0, Some(&packed.indices), 0);
            enc.set_buffer(1, Some(&packed.row_ptr), 0);
            enc.set_buffer(2, Some(&packed.residual_signs), 0);
            enc.set_buffer(3, Some(input), 0);
            enc.set_buffer(4, Some(output), 0);
            set_u32(enc, 5, packed.binary.rows);
            set_u32(enc, 6, packed.binary.cols);
            set_u32(enc, 7, packed.residual_scale_f16);
        }

        fn encode_factor_args(
            &self,
            enc: &metal::ComputeCommandEncoderRef,
            codes: &PinnedBuffer,
            scales: &PinnedBuffer,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
            rows: u32,
            cols: u32,
            group_size: u32,
            bits: u32,
            bound: u32,
        ) {
            enc.set_buffer(0, Some(codes), 0);
            enc.set_buffer(1, Some(scales), 0);
            enc.set_buffer(2, Some(input), 0);
            enc.set_buffer(3, Some(output), 0);
            set_u32(enc, 4, rows);
            set_u32(enc, 5, cols);
            set_u32(enc, 6, group_size);
            set_u32(enc, 7, bits);
            set_u32(enc, 8, bound);
        }

        fn dispatch_binary(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            body: &GpuBinary,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
        ) -> Result<()> {
            if qwen38_recon_fuse_enabled() {
                qwen38_assert_k_complete_cols(body.cols)?;
                let name = qwen38_binary_matvec_kernel(body.cols);
                let grid = if body.cols > 2048 {
                    simd8_grid(body.rows)
                } else {
                    tg256_grid(body.rows)
                };
                tcb.dispatch_threads(name, grid, (256, 1, 1), |enc| {
                    self.encode_binary_args(enc, body, input, output)
                })
            } else {
                tcb.dispatch_threads(
                    crate::decode_family::matvec_binary(),
                    (body.rows, 1, 1),
                    (256, 1, 1),
                    |enc| self.encode_binary_args(enc, body, input, output),
                )
            }
        }

        fn dispatch_residual(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            body: &GpuResidual,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
        ) -> Result<()> {
            if qwen38_recon_fuse_enabled() {
                qwen38_assert_k_complete_cols(body.binary.cols)?;
                let name = qwen38_residual_matvec_kernel(body.binary.cols);
                let grid = if body.binary.cols > 2048 {
                    simd8_grid(body.binary.rows)
                } else {
                    tg256_grid(body.binary.rows)
                };
                tcb.dispatch_threads(name, grid, (256, 1, 1), |enc| {
                    self.encode_binary_csr_args(enc, body, input, output)
                })
            } else {
                tcb.dispatch_threads(
                    crate::decode_family::matvec_binary(),
                    (body.binary.rows, 1, 1),
                    (256, 1, 1),
                    |enc| self.encode_binary_args(enc, &body.binary, input, output),
                )?;
                tcb.dispatch_threads(
                    "q80_sparse_q1_apply_csr",
                    (body.binary.rows, 1, 1),
                    (256, 1, 1),
                    |enc| self.encode_csr_args(enc, body, input, output),
                )
            }
        }

        fn dispatch_factor(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            codes: &PinnedBuffer,
            scales: &PinnedBuffer,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
            rows: u32,
            cols: u32,
            group_size: u32,
            bits: u32,
            bound: u32,
        ) -> Result<()> {
            let encode = |enc: &metal::ComputeCommandEncoderRef| {
                self.encode_factor_args(
                    enc, codes, scales, input, output, rows, cols, group_size, bits, bound,
                )
            };
            if qwen38_recon_fuse_enabled() {
                let (name, grid) = if bits == 8 {
                    if cols >= 2048 {
                        ("q80_uniform8_matvec_tg256", tg256_grid(rows))
                    } else {
                        ("q80_uniform8_matvec_simd_bytes", simd8_grid(rows))
                    }
                } else if bits == 3 {
                    ("q80_hgravs01_factor_matvec_simd3", simd8_grid(rows))
                } else {
                    ("q80_hgravs01_factor_matvec_simd", simd8_grid(rows))
                };
                tcb.dispatch_threads(name, grid, (256, 1, 1), encode)
            } else {
                tcb.dispatch_threads(
                    crate::decode_family::matvec_hgravs(),
                    (rows, 1, 1),
                    (256, 1, 1),
                    encode,
                )
            }
        }

        fn dispatch_hgravs(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            body: &GpuHgravs,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
        ) -> Result<()> {
            if body.right_rows as usize > QWEN38_MIXED_HGRAVS_RANK {
                return Err(mixed_error(format!(
                    "hgravs mid rank {} exceeds workspace {}",
                    body.right_rows, QWEN38_MIXED_HGRAVS_RANK
                )));
            }
            self.dispatch_factor(
                tcb,
                &body.right_codes,
                &body.right_scales,
                input,
                &self.workspace.hgravs_mid,
                body.right_rows,
                body.right_cols,
                body.group_size,
                body.bits,
                body.bound,
            )?;
            self.dispatch_factor(
                tcb,
                &body.left_codes,
                &body.left_scales,
                &self.workspace.hgravs_mid,
                output,
                body.left_rows,
                body.left_cols,
                body.group_size,
                body.bits,
                body.bound,
            )
        }

        fn dispatch_uniform(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            body: &GpuUniform,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
        ) -> Result<()> {
            if let Some((name, grid, tg)) = qwen38_hgravu01_geo_tpr64_launch(
                body.bits,
                body.group_size,
                body.rows,
                body.cols,
            ) {
                return tcb.dispatch_threads(name, grid, tg, |enc| {
                    self.encode_factor_args(
                        enc,
                        &body.codes,
                        &body.scales,
                        input,
                        output,
                        body.rows,
                        body.cols,
                        body.group_size,
                        body.bits,
                        body.bound,
                    )
                });
            }
            self.dispatch_factor(
                tcb,
                &body.codes,
                &body.scales,
                input,
                output,
                body.rows,
                body.cols,
                body.group_size,
                body.bits,
                body.bound,
            )
        }

        fn encode_fuse_qkvz(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            qkv: &PinnedBuffer,
            z: &PinnedBuffer,
            fused: &PinnedBuffer,
        ) -> Result<()> {
            let layout = Qwen38DeltaNetLayout::source_exact();
            let n = layout.qkvz_rows() as u32;
            let kh = layout.key_heads as u32;
            let vpk = layout.values_per_key as u32;
            let kd = layout.key_head_dim as u32;
            let vd = layout.value_head_dim as u32;
            tcb.dispatch_threads("qwen38_fuse_split_qkvz_f32", (n, 1, 1), (256, 1, 1), |enc| {
                enc.set_buffer(0, Some(qkv), 0);
                enc.set_buffer(1, Some(z), 0);
                enc.set_buffer(2, Some(fused), 0);
                set_u32(enc, 3, kh);
                set_u32(enc, 4, vpk);
                set_u32(enc, 5, kd);
                set_u32(enc, 6, vd);
            })
        }

        fn encode_fuse_ba(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            b: &PinnedBuffer,
            a: &PinnedBuffer,
            fused: &PinnedBuffer,
        ) -> Result<()> {
            let layout = Qwen38DeltaNetLayout::source_exact();
            let n = layout.ba_rows() as u32;
            let kh = layout.key_heads as u32;
            let vpk = layout.values_per_key as u32;
            tcb.dispatch_threads("qwen38_fuse_split_ba_f32", (n, 1, 1), (32, 1, 1), |enc| {
                enc.set_buffer(0, Some(b), 0);
                enc.set_buffer(1, Some(a), 0);
                enc.set_buffer(2, Some(fused), 0);
                set_u32(enc, 3, kh);
                set_u32(enc, 4, vpk);
            })
        }

        fn encode_split_deltanet_projections(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
        ) -> Result<()> {
            self.encode_named_matvec(
                tcb,
                &qwen38_layer_name(layer, "linear_attn.in_proj_qkv.weight"),
                &self.workspace.normalized,
                &self.workspace.split_qkv,
            )?;
            self.encode_named_matvec(
                tcb,
                &qwen38_layer_name(layer, "linear_attn.in_proj_z.weight"),
                &self.workspace.normalized,
                &self.workspace.z,
            )?;
            self.encode_fuse_qkvz(
                tcb,
                &self.workspace.split_qkv,
                &self.workspace.z,
                &self.workspace.qkvz,
            )?;
            self.encode_named_matvec(
                tcb,
                &qwen38_layer_name(layer, "linear_attn.in_proj_b.weight"),
                &self.workspace.normalized,
                &self.workspace.split_b,
            )?;
            self.encode_named_matvec(
                tcb,
                &qwen38_layer_name(layer, "linear_attn.in_proj_a.weight"),
                &self.workspace.normalized,
                &self.workspace.split_a,
            )?;
            self.encode_fuse_ba(
                tcb,
                &self.workspace.split_b,
                &self.workspace.split_a,
                &self.workspace.ba,
            )
        }

        fn has_weight(&self, name: &str) -> bool {
            self.weights.q4.contains_key(name) || self.weights.mixed.contains_key(name)
        }

        fn encode_q4_matvec_kernel(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            name: &str,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
            kernel: &str,
        ) -> Result<()> {
            let weight = self.q4(name)?;
            let groups_per_row = weight.cols.div_ceil(UNIFORM_Q4_GROUP_SIZE) as u32;
            let rows = weight.rows as u32;
            let cols = weight.cols as u32;
            let (grid, tg) = self.matvec_kernel.launch(rows);
            tcb.dispatch_threads(kernel, grid, tg, |encoder| {
                encoder.set_buffer(0, Some(&weight.codes), 0);
                encoder.set_buffer(1, Some(&weight.scales), 0);
                encoder.set_buffer(2, Some(input), 0);
                encoder.set_buffer(3, Some(output), 0);
                encoder.set_bytes(4, 4, &rows as *const u32 as *const _);
                encoder.set_bytes(5, 4, &cols as *const u32 as *const _);
                encoder.set_bytes(6, 4, &groups_per_row as *const u32 as *const _);
            })
        }

        fn encode_independent_q4_pair(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            a_name: &str,
            a_input: &PinnedBuffer,
            a_output: &PinnedBuffer,
            b_name: &str,
            b_input: &PinnedBuffer,
            b_output: &PinnedBuffer,
        ) -> Result<()> {
            if self.concurrent_independent {
                tcb.begin_concurrent_group()?;
            }
            self.encode_q4_matvec(tcb, a_name, a_input, a_output)?;
            self.encode_q4_matvec(tcb, b_name, b_input, b_output)?;
            if self.concurrent_independent {
                tcb.end_concurrent_group()?;
            }
            Ok(())
        }

        fn timed_cb(
            &self,
            encode: impl FnOnce(&mut TokenCommandBuffer<'_>) -> Result<()>,
        ) -> Result<CommandBufferTiming> {
            let mut tcb = TokenCommandBuffer::new(&self.context);
            encode(&mut tcb)?;
            tcb.commit_and_wait_timed()
        }

        fn encode_gated_delta(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            rec_off: u64,
        ) -> Result<()> {
            let layout = Qwen38DeltaNetLayout::source_exact();
            let heads = layout.value_heads as u32;
            let kd = layout.key_head_dim as u32;
            let vd = layout.value_head_dim as u32;
            let (kernel, grid) = if self.deltanet_vi_parallel {
                (
                    "qwen38_gated_delta_decode_vi",
                    (kd, heads, vd),
                )
            } else {
                (
                    "qwen80_gated_delta_decode_tg",
                    (kd, heads, 1),
                )
            };
            tcb.dispatch_threads(kernel, grid, (kd, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(&self.workspace.rec_state), rec_off);
                encoder.set_buffer(1, Some(&self.workspace.repeated_q), 0);
                encoder.set_buffer(2, Some(&self.workspace.repeated_k), 0);
                encoder.set_buffer(3, Some(&self.workspace.conv_v), 0);
                encoder.set_buffer(4, Some(&self.workspace.decay), 0);
                encoder.set_buffer(5, Some(&self.workspace.beta), 0);
                encoder.set_buffer(6, Some(&self.workspace.rec_out), 0);
                encoder.set_bytes(7, 4, &heads as *const u32 as *const _);
                encoder.set_bytes(8, 4, &kd as *const u32 as *const _);
                encoder.set_bytes(9, 4, &vd as *const u32 as *const _);
                encoder.set_threadgroup_memory_length(0, 128 * 4);
            })
        }

        fn encode_mixer(&self, tcb: &mut TokenCommandBuffer<'_>, layer: usize) -> Result<()> {
            match qwen38_mixer_kind(layer)? {
                Qwen38MixerKind::DeltaNet => self.encode_deltanet(tcb, layer),
                Qwen38MixerKind::Gqa => self.encode_gqa(tcb, layer),
            }
        }

        fn encode_mixer_gemvs_only(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
        ) -> Result<()> {
            if !self.weights.mixed.is_empty() {
                return self.encode_mixer_gemvs_only_mixed(tcb, layer);
            }
            match qwen38_mixer_kind(layer)? {
                Qwen38MixerKind::DeltaNet => {
                    self.encode_independent_q4_pair(
                        tcb,
                        &qwen38_layer_name(layer, "linear_attn.in_proj_qkvz.weight"),
                        &self.workspace.normalized,
                        &self.workspace.qkvz,
                        &qwen38_layer_name(layer, "linear_attn.in_proj_ba.weight"),
                        &self.workspace.normalized,
                        &self.workspace.ba,
                    )?;
                    self.encode_q4_matvec(
                        tcb,
                        &qwen38_layer_name(layer, "linear_attn.out_proj.weight"),
                        &self.workspace.gated,
                        &self.workspace.mixer,
                    )
                }
                Qwen38MixerKind::Gqa => {
                    if self.concurrent_independent {
                        tcb.begin_concurrent_group()?;
                    }
                    self.encode_q4_matvec(
                        tcb,
                        &qwen38_layer_name(layer, "self_attn.q_proj.weight"),
                        &self.workspace.normalized,
                        &self.workspace.q_proj,
                    )?;
                    self.encode_q4_matvec(
                        tcb,
                        &qwen38_layer_name(layer, "self_attn.k_proj.weight"),
                        &self.workspace.normalized,
                        &self.workspace.k_proj,
                    )?;
                    self.encode_q4_matvec(
                        tcb,
                        &qwen38_layer_name(layer, "self_attn.v_proj.weight"),
                        &self.workspace.normalized,
                        &self.workspace.v_proj,
                    )?;
                    if self.concurrent_independent {
                        tcb.end_concurrent_group()?;
                    }
                    self.encode_q4_matvec(
                        tcb,
                        &qwen38_layer_name(layer, "self_attn.o_proj.weight"),
                        &self.workspace.gated_attn,
                        &self.workspace.mixer,
                    )
                }
            }
        }

        fn encode_mlp_matvecs_only(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
        ) -> Result<()> {
            if !self.weights.mixed.is_empty() {
                return self.encode_mlp_matvecs_only_mixed(tcb, layer);
            }
            self.encode_independent_q4_pair(
                tcb,
                &qwen38_layer_name(layer, "mlp.gate_proj.weight"),
                &self.workspace.normalized,
                &self.workspace.gate,
                &qwen38_layer_name(layer, "mlp.up_proj.weight"),
                &self.workspace.normalized,
                &self.workspace.up,
            )?;
            self.encode_q4_matvec(
                tcb,
                &qwen38_layer_name(layer, "mlp.down_proj.weight"),
                &self.workspace.act,
                &self.workspace.down,
            )
        }

        pub fn read_f32_workspace(&self, which: &str, n: usize) -> Result<Vec<f32>> {
            let buffer = match which {
                "gate" => &self.workspace.gate,
                "up" => &self.workspace.up,
                "act" => &self.workspace.act,
                "down" => &self.workspace.down,
                "hidden" => &self.workspace.hidden,
                "normalized" => &self.workspace.normalized,
                "logits" => &self.workspace.logits,
                "mixer" => &self.workspace.mixer,
                other => {
                    return Err(Error::Model(format!(
                        "qwen38 unknown workspace buffer {other}"
                    )))
                }
            };
            let bytes = n
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| Error::Model("qwen38 read overflow".into()))?;
            if buffer.length() < bytes as u64 {
                return Err(Error::Model(format!(
                    "qwen38 {which} is {} bytes, need {bytes}",
                    buffer.length()
                )));
            }
            let mut out = vec![0.0f32; n];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    buffer.contents() as *const f32,
                    out.as_mut_ptr(),
                    n,
                );
            }
            Ok(out)
        }

        pub fn measure_named_matvec(&self, name: &str, output: &str) -> Result<CommandBufferTiming> {
            match output {
                "gate" | "up" | "down" | "logits" | "mixer" | "qkvz" | "hidden" => {}
                other => {
                    return Err(Error::Model(format!(
                        "qwen38 unknown matvec output {other}"
                    )))
                }
            }
            self.timed_cb(|tcb| {
                let out_buf = match output {
                    "gate" => &self.workspace.gate,
                    "up" => &self.workspace.up,
                    "down" => &self.workspace.down,
                    "logits" => &self.workspace.logits,
                    "mixer" => &self.workspace.mixer,
                    "qkvz" => &self.workspace.qkvz,
                    _ => &self.workspace.hidden,
                };
                self.encode_q4_matvec(tcb, name, &self.workspace.normalized, out_buf)
            })
        }

        pub fn measure_isolated_mlp_full(&self) -> Result<CommandBufferTiming> {
            self.timed_cb(|tcb| {
                for layer in 0..QWEN38_LAYERS {
                    self.encode_dense_mlp(tcb, layer, &self.workspace.first_residual)?;
                }
                Ok(())
            })
        }

        pub fn measure_isolated_mlp_matvecs(&self) -> Result<CommandBufferTiming> {
            self.timed_cb(|tcb| {
                for layer in 0..QWEN38_LAYERS {
                    self.encode_mlp_matvecs_only(tcb, layer)?;
                }
                Ok(())
            })
        }

        pub fn measure_isolated_mlp_one_proj(&self, which: &str) -> Result<CommandBufferTiming> {
            let suffix = match which {
                "gate" => "mlp.gate_proj.weight",
                "up" => "mlp.up_proj.weight",
                "down" => "mlp.down_proj.weight",
                other => {
                    return Err(Error::Model(format!(
                        "qwen38 mlp proj {other} is not gate/up/down"
                    )))
                }
            };
            self.timed_cb(|tcb| {
                for layer in 0..QWEN38_LAYERS {
                    let (input, output) = match which {
                        "gate" => (&self.workspace.normalized, &self.workspace.gate),
                        "up" => (&self.workspace.normalized, &self.workspace.up),
                        _ => (&self.workspace.act, &self.workspace.down),
                    };
                    self.encode_q4_matvec(
                        tcb,
                        &qwen38_layer_name(layer, suffix),
                        input,
                        output,
                    )?;
                }
                Ok(())
            })
        }

        pub fn measure_isolated_gated_delta(&self) -> Result<CommandBufferTiming> {
            let layout = Qwen38DeltaNetLayout::source_exact();
            self.timed_cb(|tcb| {
                for layer in 0..QWEN38_LAYERS {
                    if qwen38_mixer_kind(layer)? != Qwen38MixerKind::DeltaNet {
                        continue;
                    }
                    let slot = qwen38_deltanet_state_slot(layer)?;
                    let rec_off = (slot * layout.recurrent_state_elements() * 4) as u64;
                    self.encode_gated_delta(tcb, rec_off)?;
                }
                Ok(())
            })
        }

        pub fn measure_isolated_mixer_gemvs(&self) -> Result<CommandBufferTiming> {
            self.timed_cb(|tcb| {
                for layer in 0..QWEN38_LAYERS {
                    self.encode_mixer_gemvs_only(tcb, layer)?;
                }
                Ok(())
            })
        }

        pub fn measure_isolated_lm_head(&self) -> Result<CommandBufferTiming> {
            self.timed_cb(|tcb| {
                self.encode_q4_matvec(
                    tcb,
                    "language_model.lm_head.weight",
                    &self.workspace.normalized,
                    &self.workspace.logits,
                )
            })
        }

        pub fn measure_isolated_embed(&self, token: u32) -> Result<CommandBufferTiming> {
            self.timed_cb(|tcb| self.encode_embed(tcb, token))
        }

        pub fn alloc_profile_buffer(&self, bytes: usize) -> Result<PinnedBuffer> {
            self.context.new_buffer_checked(bytes)
        }

        pub fn rec_state_f32_count(&self) -> usize {
            (self.workspace.rec_state.length() as usize) / 4
        }

        pub fn conv_state_f32_count(&self) -> usize {
            (self.workspace.conv_state.length() as usize) / 4
        }

        pub fn gqa_cache_f32_count(&self) -> usize {
            (self.workspace.gqa_key.length() as usize
                + self.workspace.gqa_value.length() as usize)
                / 4
        }

        fn encode_silu(&self, tcb: &mut TokenCommandBuffer<'_>) -> Result<()> {
            let n = QWEN38_INTERMEDIATE as u32;
            tcb.dispatch_threads(
                crate::decode_family::swiglu_f32(),
                (n, 1, 1),
                (n.min(256).max(1), 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.gate), 0);
                    encoder.set_buffer(1, Some(&self.workspace.up), 0);
                    encoder.set_buffer(2, Some(&self.workspace.act), 0);
                    encoder.set_bytes(3, 4, &n as *const u32 as *const _);
                },
            )
        }

        fn encode_rearrange(&self, tcb: &mut TokenCommandBuffer<'_>, layer: usize) -> Result<()> {
            let layout = Qwen38DeltaNetLayout::source_exact();
            let slot = qwen38_deltanet_state_slot(layer)?;
            let conv_off = (slot * layout.conv_state_elements() * 4) as u64;
            let conv_w = self.f32(&qwen38_layer_name(layer, "linear_attn.conv1d.weight"))?;
            tcb.dispatch_threads(
                "qwen38_qkvz_rearrange_conv_l2_f32",
                (256, layout.key_heads as u32, 1),
                (256, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.qkvz), 0);
                    encoder.set_buffer(1, Some(conv_w), 0);
                    encoder.set_buffer(2, Some(&self.workspace.conv_state), conv_off);
                    encoder.set_buffer(3, Some(&self.workspace.repeated_q), 0);
                    encoder.set_buffer(4, Some(&self.workspace.repeated_k), 0);
                    encoder.set_buffer(5, Some(&self.workspace.conv_v), 0);
                    encoder.set_buffer(6, Some(&self.workspace.z), 0);
                    let kh = layout.key_heads as u32;
                    let vpk = layout.values_per_key as u32;
                    let kd = layout.key_head_dim as u32;
                    let vd = layout.value_head_dim as u32;
                    let ck = layout.conv_kernel as u32;
                    encoder.set_bytes(7, 4, &kh as *const u32 as *const _);
                    encoder.set_bytes(8, 4, &vpk as *const u32 as *const _);
                    encoder.set_bytes(9, 4, &kd as *const u32 as *const _);
                    encoder.set_bytes(10, 4, &vd as *const u32 as *const _);
                    encoder.set_bytes(11, 4, &ck as *const u32 as *const _);
                    encoder.set_bytes(12, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                    encoder.set_threadgroup_memory_length(0, 4 * 256 * 4);
                },
            )
        }

        fn encode_ba_to_decay(&self, tcb: &mut TokenCommandBuffer<'_>, layer: usize) -> Result<()> {
            let layout = Qwen38DeltaNetLayout::source_exact();
            let a_log = self.f32(&qwen38_layer_name(layer, "linear_attn.A_log"))?;
            let dt_bias = self.f32(&qwen38_layer_name(layer, "linear_attn.dt_bias"))?;
            tcb.dispatch_threads(
                "qwen80_ba_to_decay_beta_f32",
                (layout.value_heads as u32, 1, 1),
                (16, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.ba), 0);
                    encoder.set_buffer(1, Some(a_log), 0);
                    encoder.set_buffer(2, Some(dt_bias), 0);
                    encoder.set_buffer(3, Some(&self.workspace.decay), 0);
                    encoder.set_buffer(4, Some(&self.workspace.beta), 0);
                    let kh = layout.key_heads as u32;
                    let vpk = layout.values_per_key as u32;
                    encoder.set_bytes(5, 4, &kh as *const u32 as *const _);
                    encoder.set_bytes(6, 4, &vpk as *const u32 as *const _);
                },
            )
        }

        fn encode_gated_rmsnorm(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
        ) -> Result<()> {
            let layout = Qwen38DeltaNetLayout::source_exact();
            let norm_w = self.f32(&qwen38_layer_name(layer, "linear_attn.norm.weight"))?;
            tcb.dispatch_threads(
                "qwen80_deltanet_gated_rmsnorm_f32",
                (layout.value_heads as u32, 1, 1),
                (16, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.rec_out), 0);
                    encoder.set_buffer(1, Some(&self.workspace.z), 0);
                    encoder.set_buffer(2, Some(norm_w), 0);
                    encoder.set_buffer(3, Some(&self.workspace.gated), 0);
                    let heads = layout.value_heads as u32;
                    let dim = layout.value_head_dim as u32;
                    encoder.set_bytes(4, 4, &heads as *const u32 as *const _);
                    encoder.set_bytes(5, 4, &dim as *const u32 as *const _);
                    encoder.set_bytes(6, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                },
            )
        }

        fn encode_rope_cache(&self, tcb: &mut TokenCommandBuffer<'_>, layer: usize) -> Result<()> {
            let slot = qwen38_gqa_state_slot(layer)?;
            let slot_elems = self.max_seq_len * QWEN38_GQA_KV_HEADS * QWEN38_GQA_HEAD_DIM;
            let cache_off = (slot * slot_elems * 4) as u64;
            let q_norm = self.f32(&qwen38_layer_name(layer, "self_attn.q_norm.weight"))?;
            let k_norm = self.f32(&qwen38_layer_name(layer, "self_attn.k_norm.weight"))?;
            tcb.dispatch_threads(
                "qwen38_gqa_qk_norm_rope_cache_f32",
                (QWEN38_GQA_HEADS as u32, 1, 1),
                (QWEN38_GQA_HEADS as u32, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.q_proj), 0);
                    encoder.set_buffer(1, Some(&self.workspace.k_proj), 0);
                    encoder.set_buffer(2, Some(&self.workspace.v_proj), 0);
                    encoder.set_buffer(3, Some(q_norm), 0);
                    encoder.set_buffer(4, Some(k_norm), 0);
                    encoder.set_buffer(5, Some(&self.workspace.query), 0);
                    encoder.set_buffer(6, Some(&self.workspace.gqa_key), cache_off);
                    encoder.set_buffer(7, Some(&self.workspace.gqa_value), cache_off);
                    let pos = self.position.saturating_sub(1).min(self.max_seq_len - 1) as u32;
                    let nh = QWEN38_GQA_HEADS as u32;
                    let nkv = QWEN38_GQA_KV_HEADS as u32;
                    let hd = QWEN38_GQA_HEAD_DIM as u32;
                    let rd = QWEN38_GQA_ROTARY_DIM as u32;
                    encoder.set_bytes(8, 4, &pos as *const u32 as *const _);
                    encoder.set_bytes(9, 4, &nh as *const u32 as *const _);
                    encoder.set_bytes(10, 4, &nkv as *const u32 as *const _);
                    encoder.set_bytes(11, 4, &hd as *const u32 as *const _);
                    encoder.set_bytes(12, 4, &rd as *const u32 as *const _);
                    encoder.set_bytes(13, 4, &QWEN38_ROPE_THETA as *const f32 as *const _);
                    encoder.set_bytes(14, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                },
            )
        }

        fn encode_mha(&self, tcb: &mut TokenCommandBuffer<'_>, layer: usize) -> Result<()> {
            let slot = qwen38_gqa_state_slot(layer)?;
            let slot_elems = self.max_seq_len * QWEN38_GQA_KV_HEADS * QWEN38_GQA_HEAD_DIM;
            let cache_off = (slot * slot_elems * 4) as u64;
            let seq = self.position.max(1).min(self.max_seq_len);
            mha_decode_f32_tcb(
                tcb,
                &self.workspace.query,
                &self.workspace.gqa_key,
                cache_off as usize,
                &self.workspace.gqa_value,
                cache_off as usize,
                &self.workspace.attn,
                seq,
                QWEN38_GQA_HEAD_DIM,
                QWEN38_GQA_HEADS,
                QWEN38_GQA_KV_HEADS,
            )
        }

        fn encode_sigmoid_gate(&self, tcb: &mut TokenCommandBuffer<'_>) -> Result<()> {
            let query_dim = (QWEN38_GQA_HEADS * QWEN38_GQA_HEAD_DIM) as u32;
            let head_dim = QWEN38_GQA_HEAD_DIM as u32;
            tcb.dispatch_threads(
                "qwen38_attention_apply_sigmoid_gate",
                (query_dim, 1, 1),
                (256, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.attn), 0);
                    encoder.set_buffer(1, Some(&self.workspace.q_proj), 0);
                    encoder.set_buffer(2, Some(&self.workspace.gated_attn), 0);
                    encoder.set_bytes(3, 4, &query_dim as *const u32 as *const _);
                    encoder.set_bytes(4, 4, &head_dim as *const u32 as *const _);
                },
            )
        }

        fn encode_argmax(&self, tcb: &mut TokenCommandBuffer<'_>) -> Result<()> {
            sample_argmax_f32_tcb(
                tcb,
                &self.workspace.logits,
                &self.workspace.sampled,
                QWEN38_VOCAB,
            )
        }

        fn encode_f32_stream(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            src: &PinnedBuffer,
            dst: &PinnedBuffer,
            n_f32: u32,
        ) -> Result<()> {
            tcb.dispatch_threads(
                QWEN38_F32_STREAM_PROBE_KERNEL,
                (n_f32, 1, 1),
                (256, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(src), 0);
                    encoder.set_buffer(1, Some(dst), 0);
                    encoder.set_bytes(2, 4, &n_f32 as *const u32 as *const _);
                },
            )
        }

        pub fn measure_isolated_family(
            &self,
            family: &str,
        ) -> Result<CommandBufferTiming> {
            match family {
                "input_norms" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        self.encode_rmsnorm(
                            tcb,
                            &self.workspace.hidden,
                            &qwen38_layer_name(layer, "input_layernorm.weight"),
                            &self.workspace.normalized,
                            QWEN38_HIDDEN as u32,
                        )?;
                    }
                    Ok(())
                }),
                "post_norms" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        self.encode_rmsnorm(
                            tcb,
                            &self.workspace.first_residual,
                            &qwen38_layer_name(layer, "post_attention_layernorm.weight"),
                            &self.workspace.normalized,
                            QWEN38_HIDDEN as u32,
                        )?;
                    }
                    Ok(())
                }),
                "final_norm" => self.timed_cb(|tcb| {
                    self.encode_rmsnorm(
                        tcb,
                        &self.workspace.hidden,
                        "language_model.model.norm.weight",
                        &self.workspace.normalized,
                        QWEN38_HIDDEN as u32,
                    )
                }),
                "silu_64" => self.timed_cb(|tcb| {
                    for _ in 0..QWEN38_LAYERS {
                        self.encode_silu(tcb)?;
                    }
                    Ok(())
                }),
                "mlp_residual_64" => self.timed_cb(|tcb| {
                    for _ in 0..QWEN38_LAYERS {
                        qwen_next_add_residual_tcb(
                            tcb,
                            &self.workspace.first_residual,
                            &self.workspace.down,
                            &self.workspace.hidden,
                            QWEN38_HIDDEN,
                        )?;
                    }
                    Ok(())
                }),
                "mixer_residual_64" => self.timed_cb(|tcb| {
                    for _ in 0..QWEN38_LAYERS {
                        qwen_next_add_residual_tcb(
                            tcb,
                            &self.workspace.hidden,
                            &self.workspace.mixer,
                            &self.workspace.first_residual,
                            QWEN38_HIDDEN,
                        )?;
                    }
                    Ok(())
                }),
                "rearrange_48" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        if qwen38_mixer_kind(layer)? == Qwen38MixerKind::DeltaNet {
                            self.encode_rearrange(tcb, layer)?;
                        }
                    }
                    Ok(())
                }),
                "ba_to_decay_48" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        if qwen38_mixer_kind(layer)? == Qwen38MixerKind::DeltaNet {
                            self.encode_ba_to_decay(tcb, layer)?;
                        }
                    }
                    Ok(())
                }),
                "gated_rmsnorm_48" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        if qwen38_mixer_kind(layer)? == Qwen38MixerKind::DeltaNet {
                            self.encode_gated_rmsnorm(tcb, layer)?;
                        }
                    }
                    Ok(())
                }),
                "rope_cache_16" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        if qwen38_mixer_kind(layer)? == Qwen38MixerKind::Gqa {
                            self.encode_rope_cache(tcb, layer)?;
                        }
                    }
                    Ok(())
                }),
                "mha_16" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        if qwen38_mixer_kind(layer)? == Qwen38MixerKind::Gqa {
                            self.encode_mha(tcb, layer)?;
                        }
                    }
                    Ok(())
                }),
                "sigmoid_16" => self.timed_cb(|tcb| {
                    for _ in 0..QWEN38_GQA_LAYERS {
                        self.encode_sigmoid_gate(tcb)?;
                    }
                    Ok(())
                }),
                "argmax" => self.timed_cb(|tcb| self.encode_argmax(tcb)),
                "dn_gemvs" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        if qwen38_mixer_kind(layer)? != Qwen38MixerKind::DeltaNet {
                            continue;
                        }
                        self.encode_independent_q4_pair(
                            tcb,
                            &qwen38_layer_name(layer, "linear_attn.in_proj_qkvz.weight"),
                            &self.workspace.normalized,
                            &self.workspace.qkvz,
                            &qwen38_layer_name(layer, "linear_attn.in_proj_ba.weight"),
                            &self.workspace.normalized,
                            &self.workspace.ba,
                        )?;
                        self.encode_q4_matvec(
                            tcb,
                            &qwen38_layer_name(layer, "linear_attn.out_proj.weight"),
                            &self.workspace.gated,
                            &self.workspace.mixer,
                        )?;
                    }
                    Ok(())
                }),
                "gqa_gemvs" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        if qwen38_mixer_kind(layer)? != Qwen38MixerKind::Gqa {
                            continue;
                        }
                        self.encode_q4_matvec(
                            tcb,
                            &qwen38_layer_name(layer, "self_attn.q_proj.weight"),
                            &self.workspace.normalized,
                            &self.workspace.q_proj,
                        )?;
                        self.encode_q4_matvec(
                            tcb,
                            &qwen38_layer_name(layer, "self_attn.k_proj.weight"),
                            &self.workspace.normalized,
                            &self.workspace.k_proj,
                        )?;
                        self.encode_q4_matvec(
                            tcb,
                            &qwen38_layer_name(layer, "self_attn.v_proj.weight"),
                            &self.workspace.normalized,
                            &self.workspace.v_proj,
                        )?;
                        self.encode_q4_matvec(
                            tcb,
                            &qwen38_layer_name(layer, "self_attn.o_proj.weight"),
                            &self.workspace.gated_attn,
                            &self.workspace.mixer,
                        )?;
                    }
                    Ok(())
                }),
                other => Err(Error::Model(format!(
                    "qwen38 unknown isolated family {other}"
                ))),
            }
        }

        pub fn measure_isolated_mlp_one_proj_kernel(
            &self,
            which: &str,
            kernel: &str,
        ) -> Result<CommandBufferTiming> {
            let suffix = match which {
                "gate" => "mlp.gate_proj.weight",
                "up" => "mlp.up_proj.weight",
                "down" => "mlp.down_proj.weight",
                other => {
                    return Err(Error::Model(format!(
                        "qwen38 mlp proj {other} is not gate/up/down"
                    )))
                }
            };
            self.timed_cb(|tcb| {
                for layer in 0..QWEN38_LAYERS {
                    let (input, output) = match which {
                        "gate" => (&self.workspace.normalized, &self.workspace.gate),
                        "up" => (&self.workspace.normalized, &self.workspace.up),
                        _ => (&self.workspace.act, &self.workspace.down),
                    };
                    self.encode_q4_matvec_kernel(
                        tcb,
                        &qwen38_layer_name(layer, suffix),
                        input,
                        output,
                        kernel,
                    )?;
                }
                Ok(())
            })
        }

        pub fn measure_isolated_class_gemvs_kernel(
            &self,
            class: &str,
            kernel: &str,
        ) -> Result<CommandBufferTiming> {
            match class {
                "mlp" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        self.encode_q4_matvec_kernel(
                            tcb,
                            &qwen38_layer_name(layer, "mlp.gate_proj.weight"),
                            &self.workspace.normalized,
                            &self.workspace.gate,
                            kernel,
                        )?;
                        self.encode_q4_matvec_kernel(
                            tcb,
                            &qwen38_layer_name(layer, "mlp.up_proj.weight"),
                            &self.workspace.normalized,
                            &self.workspace.up,
                            kernel,
                        )?;
                        self.encode_q4_matvec_kernel(
                            tcb,
                            &qwen38_layer_name(layer, "mlp.down_proj.weight"),
                            &self.workspace.act,
                            &self.workspace.down,
                            kernel,
                        )?;
                    }
                    Ok(())
                }),
                "dn" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        if qwen38_mixer_kind(layer)? != Qwen38MixerKind::DeltaNet {
                            continue;
                        }
                        self.encode_q4_matvec_kernel(
                            tcb,
                            &qwen38_layer_name(layer, "linear_attn.in_proj_qkvz.weight"),
                            &self.workspace.normalized,
                            &self.workspace.qkvz,
                            kernel,
                        )?;
                        self.encode_q4_matvec_kernel(
                            tcb,
                            &qwen38_layer_name(layer, "linear_attn.in_proj_ba.weight"),
                            &self.workspace.normalized,
                            &self.workspace.ba,
                            kernel,
                        )?;
                        self.encode_q4_matvec_kernel(
                            tcb,
                            &qwen38_layer_name(layer, "linear_attn.out_proj.weight"),
                            &self.workspace.gated,
                            &self.workspace.mixer,
                            kernel,
                        )?;
                    }
                    Ok(())
                }),
                "gqa" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        if qwen38_mixer_kind(layer)? != Qwen38MixerKind::Gqa {
                            continue;
                        }
                        self.encode_q4_matvec_kernel(
                            tcb,
                            &qwen38_layer_name(layer, "self_attn.q_proj.weight"),
                            &self.workspace.normalized,
                            &self.workspace.q_proj,
                            kernel,
                        )?;
                        self.encode_q4_matvec_kernel(
                            tcb,
                            &qwen38_layer_name(layer, "self_attn.k_proj.weight"),
                            &self.workspace.normalized,
                            &self.workspace.k_proj,
                            kernel,
                        )?;
                        self.encode_q4_matvec_kernel(
                            tcb,
                            &qwen38_layer_name(layer, "self_attn.v_proj.weight"),
                            &self.workspace.normalized,
                            &self.workspace.v_proj,
                            kernel,
                        )?;
                        self.encode_q4_matvec_kernel(
                            tcb,
                            &qwen38_layer_name(layer, "self_attn.o_proj.weight"),
                            &self.workspace.gated_attn,
                            &self.workspace.mixer,
                            kernel,
                        )?;
                    }
                    Ok(())
                }),
                "lm_head" => self.timed_cb(|tcb| {
                    self.encode_q4_matvec_kernel(
                        tcb,
                        "language_model.lm_head.weight",
                        &self.workspace.normalized,
                        &self.workspace.logits,
                        kernel,
                    )
                }),
                other => Err(Error::Model(format!(
                    "qwen38 unknown gemv class {other}"
                ))),
            }
        }

        pub fn measure_f32_stream(
            &self,
            which: &str,
            dest: &PinnedBuffer,
        ) -> Result<CommandBufferTiming> {
            let (src, n) = match which {
                "rec_state" => (&self.workspace.rec_state, self.rec_state_f32_count()),
                "conv_state" => (&self.workspace.conv_state, self.conv_state_f32_count()),
                "gqa_key" => (
                    &self.workspace.gqa_key,
                    (self.workspace.gqa_key.length() as usize) / 4,
                ),
                "gqa_value" => (
                    &self.workspace.gqa_value,
                    (self.workspace.gqa_value.length() as usize) / 4,
                ),
                other => {
                    return Err(Error::Model(format!(
                        "qwen38 unknown stream source {other}"
                    )))
                }
            };
            let n = n.min((dest.length() as usize) / 4);
            self.timed_cb(|tcb| self.encode_f32_stream(tcb, src, dest, n as u32))
        }

        pub fn step_decomposed(&mut self, token: u32) -> Result<(u32, Qwen38ClassTiming)> {
            if self.fallbacks != 0 {
                return Err(Error::Model(
                    "qwen38 decode refuses a run after a fallback".into(),
                ));
            }
            let mut out = Qwen38ClassTiming::default();
            let embed = self.timed_cb(|tcb| self.encode_embed(tcb, token))?;
            out.embed_gpu_ns = embed.gpu_ns;
            out.embed_wait_ns = embed.wait_ns;
            for layer in 0..QWEN38_LAYERS {
                let mixer = self.timed_cb(|tcb| self.encode_mixer(tcb, layer))?;
                let mixer_gpu = mixer.gpu_ns.unwrap_or(0);
                out.mixer_gpu_ns = out.mixer_gpu_ns.saturating_add(mixer_gpu);
                out.mixer_wait_ns = out.mixer_wait_ns.saturating_add(mixer.wait_ns);
                out.layer_mixer_gpu_ns.push(mixer_gpu);
                match qwen38_mixer_kind(layer)? {
                    Qwen38MixerKind::DeltaNet => {
                        out.deltanet_gpu_ns = out.deltanet_gpu_ns.saturating_add(mixer_gpu);
                    }
                    Qwen38MixerKind::Gqa => {
                        out.gqa_gpu_ns = out.gqa_gpu_ns.saturating_add(mixer_gpu);
                    }
                }
                let mlp = self.timed_cb(|tcb| {
                    self.encode_dense_mlp(tcb, layer, &self.workspace.first_residual)
                })?;
                let mlp_gpu = mlp.gpu_ns.unwrap_or(0);
                out.mlp_gpu_ns = out.mlp_gpu_ns.saturating_add(mlp_gpu);
                out.mlp_wait_ns = out.mlp_wait_ns.saturating_add(mlp.wait_ns);
                out.layer_mlp_gpu_ns.push(mlp_gpu);
            }
            let term = self.timed_cb(|tcb| self.encode_terminal(tcb))?;
            out.terminal_gpu_ns = term.gpu_ns;
            out.terminal_wait_ns = term.wait_ns;
            let sampled = unsafe { *(self.workspace.sampled.contents() as *const u32) };
            self.position = self.position.saturating_add(1);
            out.sampled = sampled;
            Ok((sampled, out))
        }

        fn encode_rmsnorm(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            input: &PinnedBuffer,
            weight_name: &str,
            output: &PinnedBuffer,
            hidden: u32,
        ) -> Result<()> {
            let weight = self.f32(weight_name)?;
            tcb.dispatch_threads(
                "qwen80_residual_rmsnorm_f32",
                (256, 1, 1),
                (256, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(input), 0);
                    encoder.set_buffer(1, Some(weight), 0);
                    encoder.set_buffer(2, Some(output), 0);
                    encoder.set_bytes(3, 4, &hidden as *const u32 as *const _);
                    encoder.set_bytes(4, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                    encoder.set_threadgroup_memory_length(0, 256 * 4);
                },
            )
        }

        fn encode_embed(&self, tcb: &mut TokenCommandBuffer<'_>, token: u32) -> Result<()> {
            if !self.weights.mixed.is_empty() {
                return self.encode_embed_mixed(tcb, token);
            }
            let weight = self.q4("language_model.model.embed_tokens.weight")?;
            if weight.rows != QWEN38_VOCAB || weight.cols != QWEN38_HIDDEN {
                return Err(Error::Model("qwen38 embed shape drifted".into()));
            }
            let hidden = QWEN38_HIDDEN as u32;
            let vocab = QWEN38_VOCAB as u32;
            let group = UNIFORM_Q4_GROUP_SIZE as u32;
            tcb.dispatch_threads(
                "qwen_uniform_q4_embedding_lookup",
                (hidden, 1, 1),
                (256, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&weight.codes), 0);
                    encoder.set_buffer(1, Some(&weight.scales), 0);
                    encoder.set_buffer(2, Some(&self.workspace.hidden), 0);
                    encoder.set_bytes(3, 4, &token as *const u32 as *const _);
                    encoder.set_bytes(4, 4, &hidden as *const u32 as *const _);
                    encoder.set_bytes(5, 4, &vocab as *const u32 as *const _);
                    encoder.set_bytes(6, 4, &group as *const u32 as *const _);
                },
            )
        }

        fn encode_dense_mlp(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
            input: &PinnedBuffer,
        ) -> Result<()> {
            if !self.weights.mixed.is_empty() {
                return self.encode_dense_mlp_mixed(tcb, layer, input);
            }
            let n = QWEN38_INTERMEDIATE as u32;
            self.encode_rmsnorm(
                tcb,
                input,
                &qwen38_layer_name(layer, "post_attention_layernorm.weight"),
                &self.workspace.normalized,
                QWEN38_HIDDEN as u32,
            )?;
            self.encode_independent_q4_pair(
                tcb,
                &qwen38_layer_name(layer, "mlp.gate_proj.weight"),
                &self.workspace.normalized,
                &self.workspace.gate,
                &qwen38_layer_name(layer, "mlp.up_proj.weight"),
                &self.workspace.normalized,
                &self.workspace.up,
            )?;
            tcb.dispatch_threads(
                crate::decode_family::swiglu_f32(),
                (n, 1, 1),
                (n.min(256).max(1), 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.gate), 0);
                    encoder.set_buffer(1, Some(&self.workspace.up), 0);
                    encoder.set_buffer(2, Some(&self.workspace.act), 0);
                    encoder.set_bytes(3, 4, &n as *const u32 as *const _);
                },
            )?;
            self.encode_q4_matvec(
                tcb,
                &qwen38_layer_name(layer, "mlp.down_proj.weight"),
                &self.workspace.act,
                &self.workspace.down,
            )?;
            qwen_next_add_residual_tcb(
                tcb,
                input,
                &self.workspace.down,
                &self.workspace.hidden,
                QWEN38_HIDDEN,
            )
        }

        fn encode_deltanet(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
        ) -> Result<()> {
            if !self.weights.mixed.is_empty() {
                return self.encode_deltanet_mixed(tcb, layer);
            }
            let layout = Qwen38DeltaNetLayout::source_exact();
            let slot = qwen38_deltanet_state_slot(layer)?;
            let conv_off = (slot * layout.conv_state_elements() * 4) as u64;
            let rec_off = (slot * layout.recurrent_state_elements() * 4) as u64;
            self.encode_rmsnorm(
                tcb,
                &self.workspace.hidden,
                &qwen38_layer_name(layer, "input_layernorm.weight"),
                &self.workspace.normalized,
                QWEN38_HIDDEN as u32,
            )?;
            self.encode_independent_q4_pair(
                tcb,
                &qwen38_layer_name(layer, "linear_attn.in_proj_qkvz.weight"),
                &self.workspace.normalized,
                &self.workspace.qkvz,
                &qwen38_layer_name(layer, "linear_attn.in_proj_ba.weight"),
                &self.workspace.normalized,
                &self.workspace.ba,
            )?;
            let conv_w = self.f32(&qwen38_layer_name(layer, "linear_attn.conv1d.weight"))?;
            tcb.dispatch_threads(
                "qwen38_qkvz_rearrange_conv_l2_f32",
                (256, layout.key_heads as u32, 1),
                (256, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.qkvz), 0);
                    encoder.set_buffer(1, Some(conv_w), 0);
                    encoder.set_buffer(2, Some(&self.workspace.conv_state), conv_off);
                    encoder.set_buffer(3, Some(&self.workspace.repeated_q), 0);
                    encoder.set_buffer(4, Some(&self.workspace.repeated_k), 0);
                    encoder.set_buffer(5, Some(&self.workspace.conv_v), 0);
                    encoder.set_buffer(6, Some(&self.workspace.z), 0);
                    let kh = layout.key_heads as u32;
                    let vpk = layout.values_per_key as u32;
                    let kd = layout.key_head_dim as u32;
                    let vd = layout.value_head_dim as u32;
                    let ck = layout.conv_kernel as u32;
                    encoder.set_bytes(7, 4, &kh as *const u32 as *const _);
                    encoder.set_bytes(8, 4, &vpk as *const u32 as *const _);
                    encoder.set_bytes(9, 4, &kd as *const u32 as *const _);
                    encoder.set_bytes(10, 4, &vd as *const u32 as *const _);
                    encoder.set_bytes(11, 4, &ck as *const u32 as *const _);
                    encoder.set_bytes(12, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                    encoder.set_threadgroup_memory_length(0, 4 * 256 * 4);
                },
            )?;
            let a_log = self.f32(&qwen38_layer_name(layer, "linear_attn.A_log"))?;
            let dt_bias = self.f32(&qwen38_layer_name(layer, "linear_attn.dt_bias"))?;
            tcb.dispatch_threads(
                "qwen80_ba_to_decay_beta_f32",
                (layout.value_heads as u32, 1, 1),
                (16, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.ba), 0);
                    encoder.set_buffer(1, Some(a_log), 0);
                    encoder.set_buffer(2, Some(dt_bias), 0);
                    encoder.set_buffer(3, Some(&self.workspace.decay), 0);
                    encoder.set_buffer(4, Some(&self.workspace.beta), 0);
                    let kh = layout.key_heads as u32;
                    let vpk = layout.values_per_key as u32;
                    encoder.set_bytes(5, 4, &kh as *const u32 as *const _);
                    encoder.set_bytes(6, 4, &vpk as *const u32 as *const _);
                },
            )?;
            self.encode_gated_delta(tcb, rec_off)?;
            let norm_w = self.f32(&qwen38_layer_name(layer, "linear_attn.norm.weight"))?;
            tcb.dispatch_threads(
                "qwen80_deltanet_gated_rmsnorm_f32",
                (layout.value_heads as u32, 1, 1),
                (16, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.rec_out), 0);
                    encoder.set_buffer(1, Some(&self.workspace.z), 0);
                    encoder.set_buffer(2, Some(norm_w), 0);
                    encoder.set_buffer(3, Some(&self.workspace.gated), 0);
                    let heads = layout.value_heads as u32;
                    let dim = layout.value_head_dim as u32;
                    encoder.set_bytes(4, 4, &heads as *const u32 as *const _);
                    encoder.set_bytes(5, 4, &dim as *const u32 as *const _);
                    encoder.set_bytes(6, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                },
            )?;
            self.encode_q4_matvec(
                tcb,
                &qwen38_layer_name(layer, "linear_attn.out_proj.weight"),
                &self.workspace.gated,
                &self.workspace.mixer,
            )?;
            qwen_next_add_residual_tcb(
                tcb,
                &self.workspace.hidden,
                &self.workspace.mixer,
                &self.workspace.first_residual,
                QWEN38_HIDDEN,
            )
        }

        fn encode_gqa(&self, tcb: &mut TokenCommandBuffer<'_>, layer: usize) -> Result<()> {
            if !self.weights.mixed.is_empty() {
                return self.encode_gqa_mixed(tcb, layer);
            }
            if self.position >= self.max_seq_len {
                return Err(Error::Model(format!(
                    "qwen38 GQA position {} exceeds max_seq_len {}",
                    self.position, self.max_seq_len
                )));
            }
            let slot = qwen38_gqa_state_slot(layer)?;
            let slot_elems = self.max_seq_len * QWEN38_GQA_KV_HEADS * QWEN38_GQA_HEAD_DIM;
            let cache_off = (slot * slot_elems * 4) as u64;
            self.encode_rmsnorm(
                tcb,
                &self.workspace.hidden,
                &qwen38_layer_name(layer, "input_layernorm.weight"),
                &self.workspace.normalized,
                QWEN38_HIDDEN as u32,
            )?;
            if self.concurrent_independent {
                tcb.begin_concurrent_group()?;
            }
            self.encode_q4_matvec(
                tcb,
                &qwen38_layer_name(layer, "self_attn.q_proj.weight"),
                &self.workspace.normalized,
                &self.workspace.q_proj,
            )?;
            self.encode_q4_matvec(
                tcb,
                &qwen38_layer_name(layer, "self_attn.k_proj.weight"),
                &self.workspace.normalized,
                &self.workspace.k_proj,
            )?;
            self.encode_q4_matvec(
                tcb,
                &qwen38_layer_name(layer, "self_attn.v_proj.weight"),
                &self.workspace.normalized,
                &self.workspace.v_proj,
            )?;
            if self.concurrent_independent {
                tcb.end_concurrent_group()?;
            }
            let q_norm = self.f32(&qwen38_layer_name(layer, "self_attn.q_norm.weight"))?;
            let k_norm = self.f32(&qwen38_layer_name(layer, "self_attn.k_norm.weight"))?;
            tcb.dispatch_threads(
                "qwen38_gqa_qk_norm_rope_cache_f32",
                (QWEN38_GQA_HEADS as u32, 1, 1),
                (QWEN38_GQA_HEADS as u32, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.q_proj), 0);
                    encoder.set_buffer(1, Some(&self.workspace.k_proj), 0);
                    encoder.set_buffer(2, Some(&self.workspace.v_proj), 0);
                    encoder.set_buffer(3, Some(q_norm), 0);
                    encoder.set_buffer(4, Some(k_norm), 0);
                    encoder.set_buffer(5, Some(&self.workspace.query), 0);
                    encoder.set_buffer(6, Some(&self.workspace.gqa_key), cache_off);
                    encoder.set_buffer(7, Some(&self.workspace.gqa_value), cache_off);
                    let pos = self.position as u32;
                    let nh = QWEN38_GQA_HEADS as u32;
                    let nkv = QWEN38_GQA_KV_HEADS as u32;
                    let hd = QWEN38_GQA_HEAD_DIM as u32;
                    let rd = QWEN38_GQA_ROTARY_DIM as u32;
                    encoder.set_bytes(8, 4, &pos as *const u32 as *const _);
                    encoder.set_bytes(9, 4, &nh as *const u32 as *const _);
                    encoder.set_bytes(10, 4, &nkv as *const u32 as *const _);
                    encoder.set_bytes(11, 4, &hd as *const u32 as *const _);
                    encoder.set_bytes(12, 4, &rd as *const u32 as *const _);
                    encoder.set_bytes(13, 4, &QWEN38_ROPE_THETA as *const f32 as *const _);
                    encoder.set_bytes(14, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                },
            )?;
            mha_decode_f32_tcb(
                tcb,
                &self.workspace.query,
                &self.workspace.gqa_key,
                cache_off as usize,
                &self.workspace.gqa_value,
                cache_off as usize,
                &self.workspace.attn,
                self.position + 1,
                QWEN38_GQA_HEAD_DIM,
                QWEN38_GQA_HEADS,
                QWEN38_GQA_KV_HEADS,
            )?;
            let query_dim = (QWEN38_GQA_HEADS * QWEN38_GQA_HEAD_DIM) as u32;
            let head_dim = QWEN38_GQA_HEAD_DIM as u32;
            tcb.dispatch_threads(
                "qwen38_attention_apply_sigmoid_gate",
                (query_dim, 1, 1),
                (256, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.attn), 0);
                    encoder.set_buffer(1, Some(&self.workspace.q_proj), 0);
                    encoder.set_buffer(2, Some(&self.workspace.gated_attn), 0);
                    encoder.set_bytes(3, 4, &query_dim as *const u32 as *const _);
                    encoder.set_bytes(4, 4, &head_dim as *const u32 as *const _);
                },
            )?;
            self.encode_q4_matvec(
                tcb,
                &qwen38_layer_name(layer, "self_attn.o_proj.weight"),
                &self.workspace.gated_attn,
                &self.workspace.mixer,
            )?;
            qwen_next_add_residual_tcb(
                tcb,
                &self.workspace.hidden,
                &self.workspace.mixer,
                &self.workspace.first_residual,
                QWEN38_HIDDEN,
            )
        }

        fn encode_layers(&self, tcb: &mut TokenCommandBuffer<'_>) -> Result<()> {
            for layer in 0..QWEN38_LAYERS {
                match qwen38_mixer_kind(layer)? {
                    Qwen38MixerKind::DeltaNet => self.encode_deltanet(tcb, layer)?,
                    Qwen38MixerKind::Gqa => self.encode_gqa(tcb, layer)?,
                }
                self.encode_dense_mlp(tcb, layer, &self.workspace.first_residual)?;
            }
            Ok(())
        }

        fn encode_terminal(&self, tcb: &mut TokenCommandBuffer<'_>) -> Result<()> {
            if !self.weights.mixed.is_empty() {
                return self.encode_terminal_mixed(tcb);
            }
            self.encode_rmsnorm(
                tcb,
                &self.workspace.hidden,
                "language_model.model.norm.weight",
                &self.workspace.normalized,
                QWEN38_HIDDEN as u32,
            )?;
            self.encode_q4_matvec(
                tcb,
                "language_model.lm_head.weight",
                &self.workspace.normalized,
                &self.workspace.logits,
            )?;
            sample_argmax_f32_tcb(
                tcb,
                &self.workspace.logits,
                &self.workspace.sampled,
                QWEN38_VOCAB,
            )
        }

        fn encode_embed_mixed(&self, tcb: &mut TokenCommandBuffer<'_>, token: u32) -> Result<()> {
            const EMBED: &str = "language_model.model.embed_tokens.weight";
            if let Some(MixedGpuWeight::Uniform(weight)) = self.weights.mixed.get(EMBED) {
                if weight.rows != QWEN38_VOCAB as u32 || weight.cols != QWEN38_HIDDEN as u32 {
                    return Err(mixed_error("embed HGRAVU01 shape drifted"));
                }
                let hidden = QWEN38_HIDDEN as u32;
                let vocab = QWEN38_VOCAB as u32;
                return tcb.dispatch_threads(
                    "qwen38_hgravu_embedding_lookup",
                    (hidden, 1, 1),
                    (256, 1, 1),
                    |encoder| {
                        encoder.set_buffer(0, Some(&weight.codes), 0);
                        encoder.set_buffer(1, Some(&weight.scales), 0);
                        encoder.set_buffer(2, Some(&self.workspace.hidden), 0);
                        set_u32(encoder, 3, token);
                        set_u32(encoder, 4, hidden);
                        set_u32(encoder, 5, vocab);
                        set_u32(encoder, 6, weight.group_size);
                        set_u32(encoder, 7, weight.bits);
                        set_u32(encoder, 8, weight.bound);
                    },
                );
            }
            if self.weights.q4.contains_key(EMBED) {
                let weight = self.q4(EMBED)?;
                if weight.rows != QWEN38_VOCAB || weight.cols != QWEN38_HIDDEN {
                    return Err(Error::Model("qwen38 embed shape drifted".into()));
                }
                let hidden = QWEN38_HIDDEN as u32;
                let vocab = QWEN38_VOCAB as u32;
                let group = UNIFORM_Q4_GROUP_SIZE as u32;
                return tcb.dispatch_threads(
                    "qwen_uniform_q4_embedding_lookup",
                    (hidden, 1, 1),
                    (256, 1, 1),
                    |encoder| {
                        encoder.set_buffer(0, Some(&weight.codes), 0);
                        encoder.set_buffer(1, Some(&weight.scales), 0);
                        encoder.set_buffer(2, Some(&self.workspace.hidden), 0);
                        encoder.set_bytes(3, 4, &token as *const u32 as *const _);
                        encoder.set_bytes(4, 4, &hidden as *const u32 as *const _);
                        encoder.set_bytes(5, 4, &vocab as *const u32 as *const _);
                        encoder.set_bytes(6, 4, &group as *const u32 as *const _);
                    },
                );
            }
            Err(mixed_error(
                "embed is neither HGRAVU01 nor HQ30UQ4; refusing silent fallback",
            ))
        }

        fn encode_dense_mlp_mixed(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
            input: &PinnedBuffer,
        ) -> Result<()> {
            let n = QWEN38_INTERMEDIATE as u32;
            self.encode_rmsnorm(
                tcb,
                input,
                &qwen38_layer_name(layer, "post_attention_layernorm.weight"),
                &self.workspace.normalized,
                QWEN38_HIDDEN as u32,
            )?;
            self.encode_named_matvec(
                tcb,
                &qwen38_layer_name(layer, "mlp.gate_proj.weight"),
                &self.workspace.normalized,
                &self.workspace.gate,
            )?;
            self.encode_named_matvec(
                tcb,
                &qwen38_layer_name(layer, "mlp.up_proj.weight"),
                &self.workspace.normalized,
                &self.workspace.up,
            )?;
            tcb.dispatch_threads(
                crate::decode_family::swiglu_f32(),
                (n, 1, 1),
                (n.min(256).max(1), 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.gate), 0);
                    encoder.set_buffer(1, Some(&self.workspace.up), 0);
                    encoder.set_buffer(2, Some(&self.workspace.act), 0);
                    encoder.set_bytes(3, 4, &n as *const u32 as *const _);
                },
            )?;
            self.encode_named_matvec(
                tcb,
                &qwen38_layer_name(layer, "mlp.down_proj.weight"),
                &self.workspace.act,
                &self.workspace.down,
            )?;
            qwen_next_add_residual_tcb(
                tcb,
                input,
                &self.workspace.down,
                &self.workspace.hidden,
                QWEN38_HIDDEN,
            )
        }

        fn encode_deltanet_mixed(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
        ) -> Result<()> {
            let layout = Qwen38DeltaNetLayout::source_exact();
            let slot = qwen38_deltanet_state_slot(layer)?;
            let conv_off = (slot * layout.conv_state_elements() * 4) as u64;
            let rec_off = (slot * layout.recurrent_state_elements() * 4) as u64;
            self.encode_rmsnorm(
                tcb,
                &self.workspace.hidden,
                &qwen38_layer_name(layer, "input_layernorm.weight"),
                &self.workspace.normalized,
                QWEN38_HIDDEN as u32,
            )?;
            let fused_qkvz = qwen38_layer_name(layer, "linear_attn.in_proj_qkvz.weight");
            let fused_ba = qwen38_layer_name(layer, "linear_attn.in_proj_ba.weight");
            if self.has_weight(&fused_qkvz) && self.has_weight(&fused_ba) {
                self.encode_named_matvec(
                    tcb,
                    &fused_qkvz,
                    &self.workspace.normalized,
                    &self.workspace.qkvz,
                )?;
                self.encode_named_matvec(
                    tcb,
                    &fused_ba,
                    &self.workspace.normalized,
                    &self.workspace.ba,
                )?;
            } else if self.has_weight(&qwen38_layer_name(layer, "linear_attn.in_proj_qkv.weight")) {
                self.encode_split_deltanet_projections(tcb, layer)?;
            } else {
                return Err(mixed_error(format!(
                    "layer {layer} mixer projections are neither fused QKVZ/BA nor split QKV/Z/A/B"
                )));
            }
            let conv_w = self.f32(&qwen38_layer_name(layer, "linear_attn.conv1d.weight"))?;
            tcb.dispatch_threads(
                "qwen38_qkvz_rearrange_conv_l2_f32",
                (256, layout.key_heads as u32, 1),
                (256, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.qkvz), 0);
                    encoder.set_buffer(1, Some(conv_w), 0);
                    encoder.set_buffer(2, Some(&self.workspace.conv_state), conv_off);
                    encoder.set_buffer(3, Some(&self.workspace.repeated_q), 0);
                    encoder.set_buffer(4, Some(&self.workspace.repeated_k), 0);
                    encoder.set_buffer(5, Some(&self.workspace.conv_v), 0);
                    encoder.set_buffer(6, Some(&self.workspace.z), 0);
                    let kh = layout.key_heads as u32;
                    let vpk = layout.values_per_key as u32;
                    let kd = layout.key_head_dim as u32;
                    let vd = layout.value_head_dim as u32;
                    let ck = layout.conv_kernel as u32;
                    encoder.set_bytes(7, 4, &kh as *const u32 as *const _);
                    encoder.set_bytes(8, 4, &vpk as *const u32 as *const _);
                    encoder.set_bytes(9, 4, &kd as *const u32 as *const _);
                    encoder.set_bytes(10, 4, &vd as *const u32 as *const _);
                    encoder.set_bytes(11, 4, &ck as *const u32 as *const _);
                    encoder.set_bytes(12, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                    encoder.set_threadgroup_memory_length(0, 4 * 256 * 4);
                },
            )?;
            let a_log = self.f32(&qwen38_layer_name(layer, "linear_attn.A_log"))?;
            let dt_bias = self.f32(&qwen38_layer_name(layer, "linear_attn.dt_bias"))?;
            tcb.dispatch_threads(
                "qwen80_ba_to_decay_beta_f32",
                (layout.value_heads as u32, 1, 1),
                (16, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.ba), 0);
                    encoder.set_buffer(1, Some(a_log), 0);
                    encoder.set_buffer(2, Some(dt_bias), 0);
                    encoder.set_buffer(3, Some(&self.workspace.decay), 0);
                    encoder.set_buffer(4, Some(&self.workspace.beta), 0);
                    let kh = layout.key_heads as u32;
                    let vpk = layout.values_per_key as u32;
                    encoder.set_bytes(5, 4, &kh as *const u32 as *const _);
                    encoder.set_bytes(6, 4, &vpk as *const u32 as *const _);
                },
            )?;
            self.encode_gated_delta(tcb, rec_off)?;
            let norm_w = self.f32(&qwen38_layer_name(layer, "linear_attn.norm.weight"))?;
            tcb.dispatch_threads(
                "qwen80_deltanet_gated_rmsnorm_f32",
                (layout.value_heads as u32, 1, 1),
                (16, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.rec_out), 0);
                    encoder.set_buffer(1, Some(&self.workspace.z), 0);
                    encoder.set_buffer(2, Some(norm_w), 0);
                    encoder.set_buffer(3, Some(&self.workspace.gated), 0);
                    let heads = layout.value_heads as u32;
                    let dim = layout.value_head_dim as u32;
                    encoder.set_bytes(4, 4, &heads as *const u32 as *const _);
                    encoder.set_bytes(5, 4, &dim as *const u32 as *const _);
                    encoder.set_bytes(6, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                },
            )?;
            self.encode_named_matvec(
                tcb,
                &qwen38_layer_name(layer, "linear_attn.out_proj.weight"),
                &self.workspace.gated,
                &self.workspace.mixer,
            )?;
            qwen_next_add_residual_tcb(
                tcb,
                &self.workspace.hidden,
                &self.workspace.mixer,
                &self.workspace.first_residual,
                QWEN38_HIDDEN,
            )
        }

        fn encode_gqa_mixed(&self, tcb: &mut TokenCommandBuffer<'_>, layer: usize) -> Result<()> {
            if self.position >= self.max_seq_len {
                return Err(Error::Model(format!(
                    "qwen38 GQA position {} exceeds max_seq_len {}",
                    self.position, self.max_seq_len
                )));
            }
            let slot = qwen38_gqa_state_slot(layer)?;
            let slot_elems = self.max_seq_len * QWEN38_GQA_KV_HEADS * QWEN38_GQA_HEAD_DIM;
            let cache_off = (slot * slot_elems * 4) as u64;
            self.encode_rmsnorm(
                tcb,
                &self.workspace.hidden,
                &qwen38_layer_name(layer, "input_layernorm.weight"),
                &self.workspace.normalized,
                QWEN38_HIDDEN as u32,
            )?;
            self.encode_named_matvec(
                tcb,
                &qwen38_layer_name(layer, "self_attn.q_proj.weight"),
                &self.workspace.normalized,
                &self.workspace.q_proj,
            )?;
            self.encode_named_matvec(
                tcb,
                &qwen38_layer_name(layer, "self_attn.k_proj.weight"),
                &self.workspace.normalized,
                &self.workspace.k_proj,
            )?;
            self.encode_named_matvec(
                tcb,
                &qwen38_layer_name(layer, "self_attn.v_proj.weight"),
                &self.workspace.normalized,
                &self.workspace.v_proj,
            )?;
            let q_norm = self.f32(&qwen38_layer_name(layer, "self_attn.q_norm.weight"))?;
            let k_norm = self.f32(&qwen38_layer_name(layer, "self_attn.k_norm.weight"))?;
            tcb.dispatch_threads(
                "qwen38_gqa_qk_norm_rope_cache_f32",
                (QWEN38_GQA_HEADS as u32, 1, 1),
                (QWEN38_GQA_HEADS as u32, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.q_proj), 0);
                    encoder.set_buffer(1, Some(&self.workspace.k_proj), 0);
                    encoder.set_buffer(2, Some(&self.workspace.v_proj), 0);
                    encoder.set_buffer(3, Some(q_norm), 0);
                    encoder.set_buffer(4, Some(k_norm), 0);
                    encoder.set_buffer(5, Some(&self.workspace.query), 0);
                    encoder.set_buffer(6, Some(&self.workspace.gqa_key), cache_off);
                    encoder.set_buffer(7, Some(&self.workspace.gqa_value), cache_off);
                    let pos = self.position as u32;
                    let nh = QWEN38_GQA_HEADS as u32;
                    let nkv = QWEN38_GQA_KV_HEADS as u32;
                    let hd = QWEN38_GQA_HEAD_DIM as u32;
                    let rd = QWEN38_GQA_ROTARY_DIM as u32;
                    encoder.set_bytes(8, 4, &pos as *const u32 as *const _);
                    encoder.set_bytes(9, 4, &nh as *const u32 as *const _);
                    encoder.set_bytes(10, 4, &nkv as *const u32 as *const _);
                    encoder.set_bytes(11, 4, &hd as *const u32 as *const _);
                    encoder.set_bytes(12, 4, &rd as *const u32 as *const _);
                    encoder.set_bytes(13, 4, &QWEN38_ROPE_THETA as *const f32 as *const _);
                    encoder.set_bytes(14, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                },
            )?;
            mha_decode_f32_tcb(
                tcb,
                &self.workspace.query,
                &self.workspace.gqa_key,
                cache_off as usize,
                &self.workspace.gqa_value,
                cache_off as usize,
                &self.workspace.attn,
                self.position + 1,
                QWEN38_GQA_HEAD_DIM,
                QWEN38_GQA_HEADS,
                QWEN38_GQA_KV_HEADS,
            )?;
            let query_dim = (QWEN38_GQA_HEADS * QWEN38_GQA_HEAD_DIM) as u32;
            let head_dim = QWEN38_GQA_HEAD_DIM as u32;
            tcb.dispatch_threads(
                "qwen38_attention_apply_sigmoid_gate",
                (query_dim, 1, 1),
                (256, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.attn), 0);
                    encoder.set_buffer(1, Some(&self.workspace.q_proj), 0);
                    encoder.set_buffer(2, Some(&self.workspace.gated_attn), 0);
                    encoder.set_bytes(3, 4, &query_dim as *const u32 as *const _);
                    encoder.set_bytes(4, 4, &head_dim as *const u32 as *const _);
                },
            )?;
            self.encode_named_matvec(
                tcb,
                &qwen38_layer_name(layer, "self_attn.o_proj.weight"),
                &self.workspace.gated_attn,
                &self.workspace.mixer,
            )?;
            qwen_next_add_residual_tcb(
                tcb,
                &self.workspace.hidden,
                &self.workspace.mixer,
                &self.workspace.first_residual,
                QWEN38_HIDDEN,
            )
        }

        fn encode_terminal_mixed(&self, tcb: &mut TokenCommandBuffer<'_>) -> Result<()> {
            self.encode_rmsnorm(
                tcb,
                &self.workspace.hidden,
                "language_model.model.norm.weight",
                &self.workspace.normalized,
                QWEN38_HIDDEN as u32,
            )?;
            self.encode_named_matvec(
                tcb,
                "language_model.lm_head.weight",
                &self.workspace.normalized,
                &self.workspace.logits,
            )?;
            sample_argmax_f32_tcb(
                tcb,
                &self.workspace.logits,
                &self.workspace.sampled,
                QWEN38_VOCAB,
            )
        }

        fn encode_mixer_gemvs_only_mixed(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
        ) -> Result<()> {
            match qwen38_mixer_kind(layer)? {
                Qwen38MixerKind::DeltaNet => {
                    let fused = qwen38_layer_name(layer, "linear_attn.in_proj_qkvz.weight");
                    if self.has_weight(&fused) {
                        self.encode_named_matvec(
                            tcb,
                            &fused,
                            &self.workspace.normalized,
                            &self.workspace.qkvz,
                        )?;
                        self.encode_named_matvec(
                            tcb,
                            &qwen38_layer_name(layer, "linear_attn.in_proj_ba.weight"),
                            &self.workspace.normalized,
                            &self.workspace.ba,
                        )?;
                    } else {
                        self.encode_split_deltanet_projections(tcb, layer)?;
                    }
                    self.encode_named_matvec(
                        tcb,
                        &qwen38_layer_name(layer, "linear_attn.out_proj.weight"),
                        &self.workspace.gated,
                        &self.workspace.mixer,
                    )
                }
                Qwen38MixerKind::Gqa => {
                    self.encode_named_matvec(
                        tcb,
                        &qwen38_layer_name(layer, "self_attn.q_proj.weight"),
                        &self.workspace.normalized,
                        &self.workspace.q_proj,
                    )?;
                    self.encode_named_matvec(
                        tcb,
                        &qwen38_layer_name(layer, "self_attn.k_proj.weight"),
                        &self.workspace.normalized,
                        &self.workspace.k_proj,
                    )?;
                    self.encode_named_matvec(
                        tcb,
                        &qwen38_layer_name(layer, "self_attn.v_proj.weight"),
                        &self.workspace.normalized,
                        &self.workspace.v_proj,
                    )?;
                    self.encode_named_matvec(
                        tcb,
                        &qwen38_layer_name(layer, "self_attn.o_proj.weight"),
                        &self.workspace.gated_attn,
                        &self.workspace.mixer,
                    )
                }
            }
        }

        fn encode_mlp_matvecs_only_mixed(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
        ) -> Result<()> {
            self.encode_named_matvec(
                tcb,
                &qwen38_layer_name(layer, "mlp.gate_proj.weight"),
                &self.workspace.normalized,
                &self.workspace.gate,
            )?;
            self.encode_named_matvec(
                tcb,
                &qwen38_layer_name(layer, "mlp.up_proj.weight"),
                &self.workspace.normalized,
                &self.workspace.up,
            )?;
            self.encode_named_matvec(
                tcb,
                &qwen38_layer_name(layer, "mlp.down_proj.weight"),
                &self.workspace.act,
                &self.workspace.down,
            )
        }

        pub fn step(&mut self, token: u32) -> Result<(u32, CommandBufferTiming)> {
            if self.fallbacks != 0 {
                return Err(Error::Model(
                    "qwen38 decode refuses a run after a fallback".into(),
                ));
            }
            let encode_t0 = Instant::now();
            let mut tcb = TokenCommandBuffer::new(&self.context);
            self.encode_embed(&mut tcb, token)?;
            self.encode_layers(&mut tcb)?;
            self.encode_terminal(&mut tcb)?;
            let encode_ns = encode_t0.elapsed().as_nanos() as u64;
            let mut timing = tcb.commit_and_wait_timed()?;
            if timing.encode_ns == 0 {
                timing.encode_ns = encode_ns;
            }
            let sampled = unsafe { *(self.workspace.sampled.contents() as *const u32) };
            self.position = self.position.saturating_add(1);
            Ok((sampled, timing))
        }

        /// Same GPU work as [`Self::step`], with host-side Instants around
        /// encode / commit-return / sample readback / position update so
        /// wall − gpu can be named. Timers do not change the command buffer.
        pub fn step_complete(&mut self, token: u32) -> Result<(u32, Qwen38StepWall)> {
            if self.fallbacks != 0 {
                return Err(Error::Model(
                    "qwen38 decode refuses a run after a fallback".into(),
                ));
            }
            let wall = Instant::now();
            let encode_started = Instant::now();
            let mut tcb = TokenCommandBuffer::new(&self.context);
            self.encode_embed(&mut tcb, token)?;
            self.encode_layers(&mut tcb)?;
            self.encode_terminal(&mut tcb)?;
            let encode_ns = encode_started.elapsed().as_nanos() as u64;
            let commit_started = Instant::now();
            let timing = tcb.commit_and_wait_timed()?;
            let commit_return_ns = commit_started.elapsed().as_nanos() as u64;
            let submit_plus_wait = timing.submit_ns.saturating_add(timing.wait_ns);
            let commit_epilogue_ns = commit_return_ns.saturating_sub(submit_plus_wait);
            let readback_started = Instant::now();
            let sampled = unsafe { *(self.workspace.sampled.contents() as *const u32) };
            let sample_readback_ns = readback_started.elapsed().as_nanos() as u64;
            let state_started = Instant::now();
            self.position = self.position.saturating_add(1);
            let state_update_ns = state_started.elapsed().as_nanos() as u64;
            Ok((
                sampled,
                Qwen38StepWall {
                    wall_ns: wall.elapsed().as_nanos() as u64,
                    encode_ns,
                    submit_ns: timing.submit_ns,
                    wait_ns: timing.wait_ns,
                    gpu_ns: timing.gpu_ns,
                    commit_epilogue_ns,
                    sample_readback_ns,
                    state_update_ns,
                    tcb_encode_ns: timing.encode_ns,
                    dispatches: timing.dispatches,
                    command_buffers: 1,
                },
            ))
        }
    }

    pub fn generate_greedy(
        session: &mut Qwen38HybridDecodeSession,
        prompt: &[u32],
        max_new_tokens: usize,
    ) -> Result<Qwen38GenerateResult> {
        if prompt.is_empty() {
            return Err(Error::Model("qwen38 prompt is empty".into()));
        }
        session.reset();
        let mut tokens = prompt.to_vec();
        let mut gpu_ns = Vec::new();
        let mut wait_ns = Vec::new();
        let mut encode_ns = Vec::new();
        let mut submit_ns = Vec::new();
        let mut dispatches = Vec::new();
        let mut wall_ns_per_step = Vec::new();
        let wall = Instant::now();
        let mut next = 0u32;
        let prefill = Instant::now();
        let mut first_step_wall_ns = 0u64;
        for (i, &token) in prompt.iter().enumerate() {
            let step_wall = Instant::now();
            let (sampled, timing) = session.step(token)?;
            let step_ns = step_wall.elapsed().as_nanos() as u64;
            if i == 0 {
                first_step_wall_ns = step_ns;
            }
            wall_ns_per_step.push(step_ns);
            gpu_ns.push(timing.gpu_ns);
            wait_ns.push(timing.wait_ns);
            encode_ns.push(timing.encode_ns);
            submit_ns.push(timing.submit_ns);
            dispatches.push(timing.dispatches);
            next = sampled;
        }
        let prefill_wall_ns = prefill.elapsed().as_nanos() as u64;
        tokens.push(next);
        let decode = Instant::now();
        while tokens.len() - prompt.len() < max_new_tokens {
            if next == crate::model::qwen38_geometry::QWEN38_EOS_IM_END
                || next == crate::model::qwen38_geometry::QWEN38_EOS_END_OF_TEXT
            {
                break;
            }
            let step_wall = Instant::now();
            let (sampled, timing) = session.step(next)?;
            wall_ns_per_step.push(step_wall.elapsed().as_nanos() as u64);
            gpu_ns.push(timing.gpu_ns);
            wait_ns.push(timing.wait_ns);
            encode_ns.push(timing.encode_ns);
            submit_ns.push(timing.submit_ns);
            dispatches.push(timing.dispatches);
            tokens.push(sampled);
            next = sampled;
        }
        let decode_wall_ns = decode.elapsed().as_nanos() as u64;
        let decode_steps = tokens.len().saturating_sub(prompt.len()).saturating_sub(1);
        Ok(Qwen38GenerateResult {
            tokens,
            prompt_len: prompt.len(),
            wall_ns: wall.elapsed().as_nanos() as u64,
            gpu_ns,
            wait_ns,
            encode_ns,
            submit_ns,
            dispatches,
            fallbacks: session.fallbacks,
            first_step_wall_ns,
            prefill_wall_ns,
            decode_wall_ns,
            decode_steps,
            wall_ns_per_step,
        })
    }

    pub fn generate_greedy_complete_wall(
        session: &mut Qwen38HybridDecodeSession,
        tokenizer: &Tokenizer,
        prompt: &[u32],
        max_new_tokens: usize,
    ) -> Result<Qwen38CompleteWallResult> {
        if prompt.is_empty() {
            return Err(Error::Model("qwen38 prompt is empty".into()));
        }
        let reset_started = Instant::now();
        session.reset();
        let reset_ns = reset_started.elapsed().as_nanos() as u64;
        let mut tokens = prompt.to_vec();
        let mut steps = Vec::new();
        let wall = Instant::now();
        let mut next = 0u32;
        let prefill = Instant::now();
        for (i, &token) in prompt.iter().enumerate() {
            let complete = Instant::now();
            let (sampled, step) = session.step_complete(token)?;
            let last_prompt = i + 1 == prompt.len();
            let (tokenizer_decode_ns, bookkeeping_ns) = if last_prompt {
                finish_new_token(tokenizer, &mut tokens, sampled)?
            } else {
                next = sampled;
                (0, 0)
            };
            if last_prompt {
                next = sampled;
            }
            steps.push(Qwen38CompleteToken {
                role: if last_prompt {
                    "prefill_emits_first_new"
                } else {
                    "prefill"
                }
                .to_owned(),
                step_index: i,
                token_in: token,
                token_out: sampled,
                step,
                tokenizer_decode_ns,
                bookkeeping_ns,
                complete_wall_ns: complete.elapsed().as_nanos() as u64,
            });
        }
        let prefill_wall_ns = prefill.elapsed().as_nanos() as u64;
        let decode = Instant::now();
        while tokens.len() - prompt.len() < max_new_tokens {
            if next == crate::model::qwen38_geometry::QWEN38_EOS_IM_END
                || next == crate::model::qwen38_geometry::QWEN38_EOS_END_OF_TEXT
            {
                break;
            }
            let complete = Instant::now();
            let (sampled, step) = session.step_complete(next)?;
            let (tokenizer_decode_ns, bookkeeping_ns) =
                finish_new_token(tokenizer, &mut tokens, sampled)?;
            steps.push(Qwen38CompleteToken {
                role: "decode".to_owned(),
                step_index: steps.len(),
                token_in: next,
                token_out: sampled,
                step,
                tokenizer_decode_ns,
                bookkeeping_ns,
                complete_wall_ns: complete.elapsed().as_nanos() as u64,
            });
            next = sampled;
        }
        let decode_wall_ns = decode.elapsed().as_nanos() as u64;
        Ok(Qwen38CompleteWallResult {
            tokens,
            prompt_len: prompt.len(),
            wall_ns: wall.elapsed().as_nanos() as u64,
            reset_ns,
            prefill_wall_ns,
            decode_wall_ns,
            fallbacks: session.fallbacks,
            steps,
        })
    }

    fn finish_new_token(
        tokenizer: &Tokenizer,
        tokens: &mut Vec<u32>,
        sampled: u32,
    ) -> Result<(u64, u64)> {
        let tokenizer_started = Instant::now();
        tokenizer.decode(&[sampled], true)?;
        let tokenizer_decode_ns = tokenizer_started.elapsed().as_nanos() as u64;
        let bookkeeping_started = Instant::now();
        tokens.push(sampled);
        let bookkeeping_ns = bookkeeping_started.elapsed().as_nanos() as u64;
        Ok((tokenizer_decode_ns, bookkeeping_ns))
    }

    #[derive(Clone, Debug, serde::Serialize)]
    pub struct Qwen38WeightFanout {
        pub weight_name: String,
        pub sessions: usize,
        pub concurrent: bool,
        pub gpu_ns: Option<u64>,
        pub wait_ns: u64,
        pub dispatches: u64,
    }

    /// N independent GEMVs against one resident weight tensor.
    /// `concurrent` opens one Metal concurrent encoder so the GPU may reuse
    /// the weight stream; serial is N dispatches in the default encoder.
    pub fn measure_shared_weight_fanout(
        sessions: &[&Qwen38HybridDecodeSession],
        weight_name: &str,
        concurrent: bool,
    ) -> Result<Qwen38WeightFanout> {
        if sessions.is_empty() {
            return Err(Error::Model("fanout needs at least one session".into()));
        }
        for session in sessions.iter().skip(1) {
            if !sessions[0].shares_weights_with(session) {
                return Err(Error::Model(
                    "fanout sessions do not share one resident weight set".into(),
                ));
            }
        }
        let mut tcb = TokenCommandBuffer::new(&sessions[0].context);
        if concurrent && sessions.len() > 1 {
            tcb.begin_concurrent_group()?;
            if let Ok(weight) = sessions[0].q4(weight_name) {
                tcb.use_resources_read_on_group(&[weight.codes.clone(), weight.scales.clone()])?;
            }
        }
        for session in sessions {
            session.encode_named_matvec(
                &mut tcb,
                weight_name,
                &session.workspace.normalized,
                &session.workspace.logits,
            )?;
        }
        if concurrent && sessions.len() > 1 {
            tcb.end_concurrent_group()?;
        }
        let timing = tcb.commit_and_wait_timed()?;
        Ok(Qwen38WeightFanout {
            weight_name: weight_name.to_owned(),
            sessions: sessions.len(),
            concurrent,
            gpu_ns: timing.gpu_ns,
            wait_ns: timing.wait_ns,
            dispatches: timing.dispatches,
        })
    }

    pub fn generate_greedy_parallel(
        sessions: &mut [Qwen38HybridDecodeSession],
        prompts: &[Vec<u32>],
        max_new_tokens: usize,
    ) -> Result<Vec<Qwen38GenerateResult>> {
        if sessions.len() != prompts.len() {
            return Err(Error::Model(
                "generate_greedy_parallel session/prompt count mismatch".into(),
            ));
        }
        if sessions.len() <= 1 {
            return sessions
                .iter_mut()
                .zip(prompts.iter())
                .map(|(session, prompt)| generate_greedy(session, prompt, max_new_tokens))
                .collect();
        }
        thread::scope(|scope| {
            let mut joins = Vec::with_capacity(sessions.len());
            for (session, prompt) in sessions.iter_mut().zip(prompts.iter()) {
                joins.push(scope.spawn(move || generate_greedy(session, prompt, max_new_tokens)));
            }
            joins
                .into_iter()
                .map(|join| {
                    join.join().unwrap_or_else(|_| {
                        Err(Error::Model("session thread panicked".into()))
                    })
                })
                .collect()
        })
    }
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Qwen38StepWall {
    pub wall_ns: u64,
    pub encode_ns: u64,
    pub submit_ns: u64,
    pub wait_ns: u64,
    pub gpu_ns: Option<u64>,
    /// `commit_and_wait_timed` return minus submit minus wait: GPU timestamp
    /// read + command-buffer status check after the host wait returns.
    pub commit_epilogue_ns: u64,
    pub sample_readback_ns: u64,
    pub state_update_ns: u64,
    /// TCB per-dispatch encode sum. Zero unless the cost ledger is recording.
    pub tcb_encode_ns: u64,
    pub dispatches: u64,
    pub command_buffers: u64,
}

impl Qwen38StepWall {
    pub fn named_sum_ns(&self) -> u64 {
        self.encode_ns
            .saturating_add(self.submit_ns)
            .saturating_add(self.wait_ns)
            .saturating_add(self.commit_epilogue_ns)
            .saturating_add(self.sample_readback_ns)
            .saturating_add(self.state_update_ns)
    }

    pub fn residual_ns(&self) -> i64 {
        self.wall_ns as i64 - self.named_sum_ns() as i64
    }

    pub fn wait_minus_gpu_ns(&self) -> Option<i64> {
        Some(self.wait_ns as i64 - self.gpu_ns? as i64)
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Qwen38CompleteToken {
    pub role: String,
    pub step_index: usize,
    pub token_in: u32,
    pub token_out: u32,
    pub step: Qwen38StepWall,
    pub tokenizer_decode_ns: u64,
    pub bookkeeping_ns: u64,
    pub complete_wall_ns: u64,
}

impl Qwen38CompleteToken {
    pub fn named_sum_ns(&self) -> u64 {
        self.step
            .named_sum_ns()
            .saturating_add(self.tokenizer_decode_ns)
            .saturating_add(self.bookkeeping_ns)
    }

    pub fn residual_ns(&self) -> i64 {
        self.complete_wall_ns as i64 - self.named_sum_ns() as i64
    }

    pub fn wall_minus_gpu_ns(&self) -> Option<i64> {
        Some(self.complete_wall_ns as i64 - self.step.gpu_ns? as i64)
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Qwen38CompleteWallResult {
    pub tokens: Vec<u32>,
    pub prompt_len: usize,
    pub wall_ns: u64,
    pub reset_ns: u64,
    pub prefill_wall_ns: u64,
    pub decode_wall_ns: u64,
    pub fallbacks: u32,
    pub steps: Vec<Qwen38CompleteToken>,
}

impl Qwen38CompleteWallResult {
    pub fn new_tokens(&self) -> &[u32] {
        &self.tokens[self.prompt_len.min(self.tokens.len())..]
    }

    pub fn decode_new(&self, tokenizer: &Tokenizer) -> Result<String> {
        tokenizer.decode(self.new_tokens(), true)
    }

    pub fn first_step(&self) -> Option<&Qwen38CompleteToken> {
        self.steps.first()
    }

    /// Prompt walk, including the last prompt step that emits new-token[0].
    pub fn prefill_steps(&self) -> impl Iterator<Item = &Qwen38CompleteToken> {
        self.steps.iter().filter(|s| s.role != "decode")
    }

    /// New-tokens[1..]: Q80 mixed `steady_state` denominator.
    pub fn steady_decode_steps(&self) -> impl Iterator<Item = &Qwen38CompleteToken> {
        self.steps.iter().filter(|s| s.role == "decode")
    }
}

#[derive(Clone, Debug)]
pub struct Qwen38GenerateResult {
    pub tokens: Vec<u32>,
    pub prompt_len: usize,
    pub wall_ns: u64,
    pub gpu_ns: Vec<Option<u64>>,
    pub wait_ns: Vec<u64>,
    pub encode_ns: Vec<u64>,
    pub submit_ns: Vec<u64>,
    pub dispatches: Vec<u64>,
    pub fallbacks: u32,
    pub first_step_wall_ns: u64,
    pub prefill_wall_ns: u64,
    pub decode_wall_ns: u64,
    pub decode_steps: usize,
    pub wall_ns_per_step: Vec<u64>,
}

impl Qwen38GenerateResult {
    pub fn new_tokens(&self) -> &[u32] {
        &self.tokens[self.prompt_len.min(self.tokens.len())..]
    }

    pub fn decode_new(&self, tokenizer: &Tokenizer) -> Result<String> {
        tokenizer.decode(self.new_tokens(), true)
    }

    pub fn median_gpu_ns_per_token(&self) -> Option<u64> {
        let mut values: Vec<u64> = self.gpu_ns.iter().copied().flatten().collect();
        if values.is_empty() {
            return None;
        }
        values.sort_unstable();
        Some(values[values.len() / 2])
    }

    pub fn steady_decode_wall_ns_per_token(&self) -> Option<u64> {
        if self.decode_steps == 0 {
            return None;
        }
        Some(self.decode_wall_ns / self.decode_steps as u64)
    }
}

#[cfg(target_os = "macos")]
pub use device::{
    generate_greedy, generate_greedy_complete_wall, generate_greedy_parallel,
    measure_shared_weight_fanout, Qwen38HybridDecodeSession, Qwen38HybridWeights,
    Qwen38WeightFanout,
};

#[cfg(not(target_os = "macos"))]
pub fn generate_greedy(
    _root: impl AsRef<Path>,
    _prompt: &[u32],
    _max_new: usize,
) -> Result<Qwen38GenerateResult> {
    Err(Error::Model("qwen38 native decode is Metal-only".into()))
}

#[cfg(not(target_os = "macos"))]
pub fn generate_greedy_complete_wall(
    _root: impl AsRef<Path>,
    _tokenizer: &Tokenizer,
    _prompt: &[u32],
    _max_new: usize,
) -> Result<Qwen38CompleteWallResult> {
    Err(Error::Model("qwen38 native decode is Metal-only".into()))
}

#[cfg(test)]
mod mixed_catalog_contract_tests {
    use super::*;

    #[test]
    fn absolute_segment_filename_does_not_join_segments() {
        let root = Path::new("/artifact/mixed-sub15-v1");
        let abs = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-2p0-v1/segments/L00.hq38seg";
        assert_eq!(
            resolve_mixed_segment_path(root, abs),
            PathBuf::from(abs)
        );
        assert_eq!(
            resolve_mixed_segment_path(root, "L00.hq38seg"),
            root.join("segments").join("L00.hq38seg")
        );
    }

    #[test]
    fn hq38m20_magic_and_record_match_q80_layout() {
        assert_eq!(&QWEN38_MIXED_CATALOG_MAGIC, b"HQ38M20\0");
        assert_eq!(QWEN38_MIXED_RECORD_SIZE, 128);
        assert_eq!(QWEN38_MIXED_CATALOG_VERSION, 1);
        assert_eq!(
            QWEN38_MIXED_SCHEMA,
            "hawking.ascension.qwen38_mixed_representation_candidate.v1"
        );
    }

    #[test]
    fn codec_4_f32v2_is_accepted_without_mlx_delta() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&2u64.to_le_bytes());
        payload.extend_from_slice(&1.046875f32.to_le_bytes());
        payload.extend_from_slice(&0.5f32.to_le_bytes());
        let name = "language_model.model.layers.0.input_layernorm.weight";
        let lane = classify_qwen38_mixed_payload(4, &payload, name, &[2]).unwrap();
        assert_eq!(lane, MixedCatalogLane::F32v2);
        let values = read_qwen38_f32_payload(&payload).unwrap();
        assert_eq!(values, vec![1.046875, 0.5]);
    }

    #[test]
    fn unknown_codec_5_still_refuses() {
        let err = classify_qwen38_mixed_payload(5, b"xxxxxxxx", "tensor.x", &[1])
            .expect_err("codec 5 must refuse");
        let msg = format!("{err}");
        assert!(
            msg.contains("unknown mixed codec 5"),
            "refuse message was {msg}"
        );
    }

    fn filled_mlp_kinds(kind: MixedMlpNativeKind) -> HashMap<String, MixedMlpNativeKind> {
        let mut kinds = HashMap::new();
        for layer in 0..QWEN38_LAYERS {
            kinds.insert(qwen38_layer_name(layer, "mlp.gate_proj.weight"), kind);
            kinds.insert(qwen38_layer_name(layer, "mlp.up_proj.weight"), kind);
            kinds.insert(qwen38_layer_name(layer, "mlp.down_proj.weight"), kind);
        }
        kinds
    }

    #[test]
    fn mixed_mlp_uniform_is_admitted_on_every_role() {
        let kinds = filled_mlp_kinds(MixedMlpNativeKind::Uniform);
        assert_mixed_mlp_native_kinds(|name| kinds.get(name).copied()).unwrap();
    }

    #[test]
    fn mixed_mlp_legacy_binary_residual_hgravs_still_admits() {
        let mut kinds = HashMap::new();
        for layer in 0..QWEN38_LAYERS {
            kinds.insert(
                qwen38_layer_name(layer, "mlp.gate_proj.weight"),
                MixedMlpNativeKind::Binary,
            );
            kinds.insert(
                qwen38_layer_name(layer, "mlp.up_proj.weight"),
                MixedMlpNativeKind::Residual,
            );
            kinds.insert(
                qwen38_layer_name(layer, "mlp.down_proj.weight"),
                MixedMlpNativeKind::Hgravs,
            );
        }
        assert_mixed_mlp_native_kinds(|name| kinds.get(name).copied()).unwrap();
    }

    #[test]
    fn mixed_mlp_unsupported_role_still_refuses() {
        let mut kinds = HashMap::new();
        for layer in 0..QWEN38_LAYERS {
            kinds.insert(
                qwen38_layer_name(layer, "mlp.gate_proj.weight"),
                MixedMlpNativeKind::Residual,
            );
            kinds.insert(
                qwen38_layer_name(layer, "mlp.up_proj.weight"),
                MixedMlpNativeKind::Residual,
            );
            kinds.insert(
                qwen38_layer_name(layer, "mlp.down_proj.weight"),
                MixedMlpNativeKind::Hgravs,
            );
        }
        let err = assert_mixed_mlp_native_kinds(|name| kinds.get(name).copied())
            .expect_err("Residual on gate must refuse");
        let msg = format!("{err}");
        assert!(msg.contains("is not HGRAVB01"), "refuse message was {msg}");
        assert!(
            msg.contains("mlp.gate_proj.weight"),
            "refuse message was {msg}"
        );
    }

    #[test]
    fn hq30uq4_on_mlp_is_not_uniform_and_still_refuses() {
        let mut payload = b"HQ30UQ4\0".to_vec();
        payload.extend_from_slice(&[0u8; 16]);
        let name = "language_model.model.layers.0.mlp.down_proj.weight";
        let lane = classify_qwen38_mixed_payload(3, &payload, name, &[5120, 17408]).unwrap();
        assert_eq!(lane, MixedCatalogLane::Hq30Uq4);
        assert_eq!(mixed_mlp_native_kind_from_lane(lane), None);

        let mut kinds = HashMap::new();
        for layer in 0..QWEN38_LAYERS {
            kinds.insert(
                qwen38_layer_name(layer, "mlp.gate_proj.weight"),
                MixedMlpNativeKind::Binary,
            );
            kinds.insert(
                qwen38_layer_name(layer, "mlp.up_proj.weight"),
                MixedMlpNativeKind::Residual,
            );
        }
        let err = assert_mixed_mlp_native_kinds(|n| kinds.get(n).copied())
            .expect_err("HQ30UQ4 down must refuse as missing mixed native");
        let msg = format!("{err}");
        assert!(msg.contains("missing"), "refuse message was {msg}");
        assert!(
            msg.contains("mlp.down_proj.weight"),
            "refuse message was {msg}"
        );
        assert!(
            msg.contains("silent dense/Q4 fallback"),
            "refuse message was {msg}"
        );
    }

    fn campaign_qwen38(name: &str) -> PathBuf {
        PathBuf::from(
            "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b",
        )
        .join(name)
    }

    #[test]
    fn mixed_q3mlp_and_q4down_pass_mlp_admission() {
        let q3 = campaign_qwen38("mixed-q3mlp-v1");
        let q4down = campaign_qwen38("mixed-q4down-v1");
        if !q3.join(QWEN38_MIXED_CATALOG_NAME).is_file()
            || !q4down.join(QWEN38_MIXED_CATALOG_NAME).is_file()
        {
            eprintln!("skip: mixed-q3mlp / mixed-q4down artifacts not on this host");
            return;
        }
        assert_mixed_mlp_native_catalog(&q3)
            .unwrap_or_else(|e| panic!("mixed-q3mlp-v1 must admit: {e}"));
        assert_mixed_mlp_native_catalog(&q4down)
            .unwrap_or_else(|e| panic!("mixed-q4down-v1 must admit: {e}"));
    }

    #[test]
    fn mixed_2p0_legacy_mlp_assignment_still_admits() {
        let mixed = campaign_qwen38("mixed-2p0-v1");
        if !mixed.join(QWEN38_MIXED_CATALOG_NAME).is_file() {
            eprintln!("skip: mixed-2p0-v1 artifact not on this host");
            return;
        }
        assert_mixed_mlp_native_catalog(&mixed)
            .unwrap_or_else(|e| panic!("mixed-2p0-v1 must still admit: {e}"));
    }

    #[test]
    fn k_complete_bind_retargets_wide_columns() {
        assert_eq!(
            qwen38_binary_matvec_kernel(2048),
            "q80_binary_group_matvec_tg256"
        );
        assert_eq!(
            qwen38_binary_matvec_kernel(5120),
            "q80_binary_group_matvec_simd_bytes"
        );
        assert_eq!(
            qwen38_binary_matvec_kernel(6144),
            "q80_binary_group_matvec_simd_bytes"
        );
        assert_eq!(
            qwen38_residual_matvec_kernel(5120),
            "q80_binary_group_csr_matvec_bytes"
        );
        assert_eq!(
            qwen38_residual_matvec_kernel(2048),
            "q80_binary_group_csr_matvec_tg256"
        );
        qwen38_assert_k_complete_cols(5120).unwrap();
        qwen38_assert_k_complete_cols(6144).unwrap();
        qwen38_assert_k_complete_cols(2048).unwrap();
        let err = qwen38_assert_k_complete_cols(2049).expect_err("remainder must refuse");
        assert!(format!("{err}").contains("partial-K"));
        assert!(qwen38_mixed_k_complete_bind_message().contains("K-complete"));
    }

    fn write_tiny_hq38m20(dir: &Path, name: &str, codec: u8, payload: &[u8]) {
        let seg_name = "t0.seg";
        std::fs::write(dir.join("segments").join(seg_name), payload).unwrap();
        let name_bytes = name.as_bytes();
        let mut rec = vec![0u8; QWEN38_MIXED_RECORD_SIZE];
        rec[0..4].copy_from_slice(&0u32.to_le_bytes());
        rec[4..6].copy_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        rec[6] = codec;
        rec[7] = 6;
        rec[8] = 1;
        rec[12..16].copy_from_slice(&2u32.to_le_bytes());
        rec[28..36].copy_from_slice(&2u64.to_le_bytes());
        rec[36..38].copy_from_slice(&0u16.to_le_bytes());
        rec[40..48].copy_from_slice(&0u64.to_le_bytes());
        rec[48..56].copy_from_slice(&(payload.len() as u64).to_le_bytes());
        let mut catalog = Vec::new();
        catalog.extend_from_slice(&QWEN38_MIXED_CATALOG_MAGIC);
        catalog.extend_from_slice(&QWEN38_MIXED_CATALOG_VERSION.to_le_bytes());
        catalog.extend_from_slice(&1u32.to_le_bytes());
        catalog.extend_from_slice(&1u32.to_le_bytes());
        catalog.extend_from_slice(&0u32.to_le_bytes());
        catalog.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        catalog.extend_from_slice(&0u32.to_le_bytes());
        let digest = [0u8; 32];
        catalog.extend_from_slice(&0u16.to_le_bytes());
        catalog.extend_from_slice(&(seg_name.len() as u16).to_le_bytes());
        catalog.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        catalog.extend_from_slice(&digest);
        catalog.extend_from_slice(seg_name.as_bytes());
        catalog.extend_from_slice(&rec);
        catalog.extend_from_slice(name_bytes);
        std::fs::write(dir.join(QWEN38_MIXED_CATALOG_NAME), catalog).unwrap();
    }

    #[test]
    fn catalog_roundtrip_codec_4_census_and_codec_5_refuses() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::create_dir(root.join("segments")).unwrap();
        let mut payload = Vec::new();
        payload.extend_from_slice(&2u64.to_le_bytes());
        payload.extend_from_slice(&0.046875f32.to_le_bytes());
        payload.extend_from_slice(&1.0f32.to_le_bytes());
        write_tiny_hq38m20(
            root,
            "language_model.model.layers.0.input_layernorm.weight",
            4,
            &payload,
        );
        let census = census_qwen38_mixed_catalog(root).unwrap();
        assert_eq!(census.tensors, 1);
        assert_eq!(census.f32, 1);
        assert_eq!(census.refused, 0);
        assert_eq!(census.expanded_to_q4, 0);
        assert_eq!(census.expanded_to_float_gemv, 0);

        write_tiny_hq38m20(root, "tensor.x", 5, &payload);
        let census = census_qwen38_mixed_catalog(root).unwrap();
        assert_eq!(census.refused, 1);
        assert!(census.refusals.iter().any(|s| s.contains("unknown mixed codec 5")));
    }

    #[test]
    fn sub15_source_payloads_accept_mixed_gpu_layout() {
        let camp = Path::new(
            "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b",
        );
        let mixed = camp.join("mixed-2p0-v1");
        let sub15 = camp.join("mixed-sub15-v1");
        if !mixed.join(QWEN38_MIXED_CATALOG_NAME).is_file() || !sub15.join("packed/attn").is_dir()
        {
            eprintln!("skip: mixed-2p0 / mixed-sub15 artifacts not on this host");
            return;
        }
        let rows = parse_qwen38_mixed_catalog(&mixed).unwrap();
        for (suffix, codec) in [
            ("mlp.gate_proj.weight", 0u8),
            ("mlp.up_proj.weight", 1u8),
            ("mlp.down_proj.weight", 2u8),
        ] {
            let row = rows
                .iter()
                .find(|r| r.name.contains(".layers.0.") && r.name.ends_with(suffix))
                .unwrap_or_else(|| panic!("missing L0 {suffix}"));
            assert_eq!(row.codec, codec);
            let payload = read_catalog_payload(row).unwrap();
            let layout = mixed_gpu_layout(codec, &payload).expect(suffix);
            assert!(layout.cols == 5120 || layout.cols == 17408, "{suffix} cols {}", layout.cols);
        }

        fn rice(name: &str) -> std::path::PathBuf {
            use sha2::{Digest, Sha256};
            let digest = Sha256::digest(name.as_bytes());
            let stem: String = digest.iter().map(|b| format!("{b:02x}")).collect();
            PathBuf::from(format!("{stem}.rice"))
        }
        let qkv = sub15.join("packed/attn").join(rice(
            "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
        ));
        let a = sub15.join("packed/attn").join(rice(
            "language_model.model.layers.0.linear_attn.in_proj_a.weight",
        ));
        let qkv_bytes = std::fs::read(&qkv).unwrap();
        let a_bytes = std::fs::read(&a).unwrap();
        let qkv_layout = mixed_gpu_layout(1, &qkv_bytes).expect("in_proj_qkv");
        let a_layout = mixed_gpu_layout(1, &a_bytes).expect("in_proj_a");
        assert_eq!((qkv_layout.rows, qkv_layout.cols), (10240, 5120));
        assert_eq!((a_layout.rows, a_layout.cols), (48, 5120));
    }

    #[test]
    fn sub15_native_catalog_census_if_emitted() {
        let artifact = Path::new(
            "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-sub15-v1",
        );
        let staged = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../workspace/superwave/g1/mixed-sub15-native-catalog");
        let root = if artifact.join(QWEN38_MIXED_CATALOG_NAME).is_file() {
            artifact
        } else if staged.join(QWEN38_MIXED_CATALOG_NAME).is_file() {
            staged.as_path()
        } else {
            eprintln!("skip: mixed-sub15 catalog.hq38m20 not emitted yet");
            return;
        };
        let census = census_qwen38_mixed_catalog(root).unwrap();
        assert_eq!(census.refused, 0, "refusals: {:?}", census.refusals);
        assert_eq!(census.expanded_to_q4, 0);
        assert_eq!(census.expanded_to_float_gemv, 0);
        assert_eq!(census.tensors, 851);
        assert_eq!(census.binary, 64);
        assert_eq!(census.residual, 368);
        assert_eq!(census.hgravs, 64);
        assert_eq!(census.uniform, 0);
        assert_eq!(census.q4, 2);
        assert_eq!(census.f32, 353);
    }

    #[test]
    fn hgravu01_geo_tpr64_bind_is_bits_3_and_4_only() {
        let q3 = qwen38_hgravu01_geo_tpr64_launch(3, 64, 17408, 5120).expect("q3");
        assert_eq!(q3.0, QWEN38_HGRAVU01_Q3_GEO_TPR64);
        assert_eq!(q3.1, (17408u32.div_ceil(2) * 128, 1, 1));
        assert_eq!(q3.2, (128, 1, 1));
        let q4 = qwen38_hgravu01_geo_tpr64_launch(4, 64, 48, 5120).expect("q4");
        assert_eq!(q4.0, QWEN38_HGRAVU01_Q4_GEO_TPR64);
        assert_eq!(q4.1, (48u32.div_ceil(2) * 128, 1, 1));
        assert!(qwen38_hgravu01_geo_tpr64_launch(8, 64, 5120, 5120).is_none());
        assert!(qwen38_hgravu01_geo_tpr64_launch(3, 64, 2048, 160).is_none());
        assert!(qwen38_hgravu01_geo_tpr64_launch(3, 128, 17408, 5120).is_none());
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE
            .contains("kernel void qwen_uniform_q3_group64_matvec_geo_tpr64_tg128"));
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE
            .contains("kernel void qwen_uniform_hgravu_q4_group64_matvec_geo_tpr64_tg128"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn hgravu01_geo_tpr64_matches_incumbent_on_real_tensors() {
        let root = campaign_qwen38("mixed-q3mlp-v1");
        if !root.join(QWEN38_MIXED_CATALOG_NAME).is_file() {
            eprintln!("skip: mixed-q3mlp-v1 not on this host");
            return;
        }
        let context = match crate::metal::MetalContext::new() {
            Ok(ctx) => ctx,
            Err(err) => {
                eprintln!("skip: MetalContext::new failed: {err}");
                return;
            }
        };
        let rows = parse_qwen38_mixed_catalog(&root).unwrap();
        let wanted = [
            "language_model.model.layers.0.mlp.gate_proj.weight",
            "language_model.model.layers.0.mlp.up_proj.weight",
            "language_model.model.layers.0.mlp.down_proj.weight",
            "language_model.model.layers.0.linear_attn.in_proj_a.weight",
            "language_model.model.layers.0.linear_attn.in_proj_b.weight",
            "language_model.model.layers.0.linear_attn.in_proj_z.weight",
            "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
            "language_model.model.layers.0.linear_attn.out_proj.weight",
            "language_model.model.layers.3.self_attn.q_proj.weight",
            "language_model.model.layers.3.self_attn.k_proj.weight",
            "language_model.model.layers.3.self_attn.v_proj.weight",
            "language_model.model.layers.3.self_attn.o_proj.weight",
            "language_model.lm_head.weight",
        ];
        let mut worst_abs = 0.0f32;
        let mut worst_rel = 0.0f32;
        let mut worst_name = "";
        eprintln!(
            "PARITY_TABLE header: name bits rows cols incumbent max_abs max_rel rms n_abs_gt_1e4 n_abs_gt_1e2"
        );
        for name in wanted {
            let row = rows
                .iter()
                .find(|r| r.name == name)
                .unwrap_or_else(|| panic!("missing {name}"));
            let payload = read_catalog_payload(row).unwrap();
            let layout = mixed_gpu_layout(3, &payload).unwrap_or_else(|e| panic!("{name}: {e}"));
            let MixedGpuKind::Uniform(factor) = layout.kind else {
                panic!("{name} is not HGRAVU01 Uniform");
            };
            let incumbent = if factor.bits == 3 {
                "q80_hgravs01_factor_matvec_simd3"
            } else if factor.bits == 4 {
                "q80_hgravs01_factor_matvec_simd"
            } else {
                panic!("{name} bits={} not 3 or 4", factor.bits);
            };
            let (geo_name, geo_grid, geo_tg) = qwen38_hgravu01_geo_tpr64_launch(
                factor.bits,
                factor.group_size,
                factor.rows,
                factor.cols,
            )
            .unwrap_or_else(|| panic!("{name} refused geo bind"));
            let codes = context
                .new_buffer_with_bytes_checked(
                    &payload[factor.code_off..factor.code_off + factor.code_bytes],
                )
                .unwrap();
            let scales = context
                .new_buffer_with_bytes_checked(
                    &payload[factor.scale_off..factor.scale_off + factor.scale_bytes],
                )
                .unwrap();
            let mut x = vec![0.0f32; factor.cols as usize];
            for (i, slot) in x.iter_mut().enumerate() {
                *slot = (i % 17) as f32 * 0.125 - 1.0;
            }
            let input = context
                .new_buffer_with_bytes_checked(bytemuck::cast_slice(&x))
                .unwrap();
            let out_inc = context
                .new_buffer_checked(factor.rows as usize * 4)
                .unwrap();
            let out_geo = context
                .new_buffer_checked(factor.rows as usize * 4)
                .unwrap();
            let bind = |enc: &metal::ComputeCommandEncoderRef, out: &metal::Buffer| {
                enc.set_buffer(0, Some(&codes), 0);
                enc.set_buffer(1, Some(&scales), 0);
                enc.set_buffer(2, Some(&input), 0);
                enc.set_buffer(3, Some(out), 0);
                for (index, value) in [
                    (4u64, factor.rows),
                    (5, factor.cols),
                    (6, factor.group_size),
                    (7, factor.bits),
                    (8, factor.bound),
                ] {
                    enc.set_bytes(index, 4, &value as *const u32 as *const _);
                }
            };
            let inc_grid = (
                factor.rows.div_ceil(8).saturating_mul(256).max(256),
                1,
                1,
            );
            let mut tcb = crate::metal::TokenCommandBuffer::new(&context);
            tcb.dispatch_threads(incumbent, inc_grid, (256, 1, 1), |enc| {
                bind(enc, &out_inc)
            })
            .unwrap();
            tcb.dispatch_threads(geo_name, geo_grid, geo_tg, |enc| {
                bind(enc, &out_geo)
            })
            .unwrap();
            tcb.commit_and_wait().unwrap();
            let n = factor.rows as usize;
            let inc = unsafe {
                std::slice::from_raw_parts(out_inc.contents() as *const f32, n)
            };
            let geo = unsafe {
                std::slice::from_raw_parts(out_geo.contents() as *const f32, n)
            };
            let inc_nonzero = inc.iter().filter(|v| v.abs() > 0.0).count();
            let geo_nonzero = geo.iter().filter(|v| v.abs() > 0.0).count();
            let inc_max = inc.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            let geo_max = geo.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            eprintln!(
                "PARITY_SAMPLE {name} codes={} scales={} inc_nz={inc_nonzero}/{n} geo_nz={geo_nonzero}/{n} inc_max={inc_max:.6e} geo_max={geo_max:.6e} inc[0..4]={:?} geo[0..4]={:?}",
                factor.code_bytes,
                factor.scale_bytes,
                &inc[..4.min(n)],
                &geo[..4.min(n)]
            );
            assert!(
                inc_nonzero > n / 2 && geo_nonzero > n / 2,
                "{name} output looks dead (inc_nz={inc_nonzero} geo_nz={geo_nonzero} n={n})"
            );
            if n <= 48 {
                let scales = &payload[factor.scale_off..factor.scale_off + factor.scale_bytes];
                let mut scales_f16 = Vec::with_capacity(factor.scale_bytes / 2);
                for chunk in scales.chunks_exact(2) {
                    scales_f16.push(u16::from_le_bytes([chunk[0], chunk[1]]));
                }
                let packed = UniformFactorPacked {
                    rows: factor.rows as usize,
                    cols: factor.cols as usize,
                    bits: u8::try_from(factor.bits).unwrap(),
                    group_size: factor.group_size as usize,
                    groups: (factor.rows as usize * factor.cols as usize)
                        .div_ceil(factor.group_size as usize),
                    bound: u16::try_from(factor.bound).unwrap(),
                    scales_f16,
                    codes: payload[factor.code_off..factor.code_off + factor.code_bytes].to_vec(),
                };
                let mut cpu_max = 0.0f32;
                for r in 0..packed.rows {
                    let mut sum = 0.0f32;
                    for c in 0..packed.cols {
                        sum += uniform_factor_value(&packed, r, c) * x[c];
                    }
                    let d = (geo[r] - sum).abs();
                    if d > cpu_max {
                        cpu_max = d;
                    }
                }
                eprintln!("PARITY_CPU {name} max_abs_vs_serial={cpu_max:.8e}");
                assert!(cpu_max <= 1.0e-2, "{name} vs CPU serial max_abs={cpu_max}");
            }
            let mut max_abs = 0.0f32;
            let mut max_rel = 0.0f32;
            let mut sumsq = 0.0f64;
            let mut n_1e4 = 0usize;
            let mut n_1e2 = 0usize;
            for i in 0..n {
                let d = (geo[i] - inc[i]).abs();
                let denom = inc[i].abs().max(1.0);
                let rel = d / denom;
                if d > max_abs {
                    max_abs = d;
                }
                if rel > max_rel {
                    max_rel = rel;
                }
                sumsq += f64::from(d) * f64::from(d);
                if d > 1.0e-4 {
                    n_1e4 += 1;
                }
                if d > 1.0e-2 {
                    n_1e2 += 1;
                }
            }
            let rms = (sumsq / n as f64).sqrt();
            eprintln!(
                "PARITY {name} bits={} {}x{} {incumbent} max_abs={max_abs:.8e} max_rel={max_rel:.8e} rms={rms:.8e} n>1e-4={n_1e4} n>1e-2={n_1e2}",
                factor.bits, factor.rows, factor.cols
            );
            if max_abs > worst_abs {
                worst_abs = max_abs;
                worst_name = name;
            }
            if max_rel > worst_rel {
                worst_rel = max_rel;
            }
            let mut first_bad = None;
            let mut last_bad = 0usize;
            if n_1e2 > 0 {
                let mut even_bad = 0usize;
                let mut odd_bad = 0usize;
                let mut samples = Vec::new();
                for i in 0..n {
                    if (geo[i] - inc[i]).abs() > 1.0e-2 {
                        if first_bad.is_none() {
                            first_bad = Some(i);
                        }
                        last_bad = i;
                        if i % 2 == 0 {
                            even_bad += 1;
                        } else {
                            odd_bad += 1;
                        }
                        if samples.len() < 8 {
                            samples.push((i, inc[i], geo[i]));
                        }
                    }
                }
                eprintln!(
                    "PARITY_BAD {name} first={first_bad:?} last={last_bad} even={even_bad} odd={odd_bad} samples={samples:?} n={n}"
                );
                if n > 48 {
                    let scales = &payload[factor.scale_off..factor.scale_off + factor.scale_bytes];
                    let mut scales_f16 = Vec::with_capacity(factor.scale_bytes / 2);
                    for chunk in scales.chunks_exact(2) {
                        scales_f16.push(u16::from_le_bytes([chunk[0], chunk[1]]));
                    }
                    let packed = UniformFactorPacked {
                        rows: factor.rows as usize,
                        cols: factor.cols as usize,
                        bits: u8::try_from(factor.bits).unwrap(),
                        group_size: factor.group_size as usize,
                        groups: (factor.rows as usize * factor.cols as usize)
                            .div_ceil(factor.group_size as usize),
                        bound: u16::try_from(factor.bound).unwrap(),
                        scales_f16,
                        codes: payload[factor.code_off..factor.code_off + factor.code_bytes]
                            .to_vec(),
                    };
                    for &(r, inc_v, geo_v) in samples.iter().take(3) {
                        let mut sum = 0.0f32;
                        for c in 0..packed.cols {
                            sum += uniform_factor_value(&packed, r, c) * x[c];
                        }
                        let d_inc = (inc_v - sum).abs();
                        let d_geo = (geo_v - sum).abs();
                        eprintln!(
                            "PARITY_CPU_ROW {name} row={r} cpu={sum:.8e} inc={inc_v:.8e} geo={geo_v:.8e} d_inc={d_inc:.3e} d_geo={d_geo:.3e}"
                        );
                        assert_eq!(d_geo, 0.0, "{name} row {r} geo must match CPU serial");
                        assert!(
                            d_inc > 1.0e-3,
                            "{name} row {r} incumbent should miss CPU on the overflow tail"
                        );
                    }
                    if let Some(r0) = first_bad {
                        if r0 > 0 {
                            let mut sum = 0.0f32;
                            for c in 0..packed.cols {
                                sum += uniform_factor_value(&packed, r0 - 1, c) * x[c];
                            }
                            eprintln!(
                                "PARITY_CPU_ROW {name} row={} cpu={sum:.8e} inc={:.8e} geo={:.8e}",
                                r0 - 1,
                                inc[r0 - 1],
                                geo[r0 - 1]
                            );
                        }
                    }
                }
            }
            if name.ends_with("lm_head.weight") {
                // Incumbent simd does `bit0 = element * bits` in uint32.
                // bits=4 overflows at element >= 2^30. For K=5120 that is
                // row >= 209715. geo addresses by group and does not overflow;
                // CPU serial agrees with geo, not with simd, on the tail.
                let overflow_el = (1u64 << 32) / u64::from(factor.bits);
                let first_overflow_row = (overflow_el / u64::from(factor.cols)) as usize;
                assert_eq!(first_bad, Some(first_overflow_row), "lm_head first-bad row");
                assert!(last_bad + 1 == n, "lm_head tail should run to the last row");
                assert!(n_1e2 > 30_000, "lm_head incumbent overflow should be a fat tail");
            } else {
                assert_eq!(
                    max_abs, 0.0,
                    "{name} max_abs={max_abs} is not bit-identical to incumbent"
                );
                assert_eq!(n_1e2, 0, "{name} has {n_1e2} rows with |d|>1e-2");
            }
        }
        eprintln!(
            "PARITY_WORST tensor={worst_name} max_abs={worst_abs:.8e} max_rel={worst_rel:.8e}"
        );
    }
}

#[cfg(test)]
mod complete_wall_identity_tests {
    use super::{Qwen38CompleteToken, Qwen38StepWall};

    #[test]
    fn step_named_sum_plus_residual_equals_wall() {
        let step = Qwen38StepWall {
            wall_ns: 34_000_000,
            encode_ns: 400_000,
            submit_ns: 20_000,
            wait_ns: 33_500_000,
            gpu_ns: Some(33_100_000),
            commit_epilogue_ns: 30_000,
            sample_readback_ns: 2_000,
            state_update_ns: 1_000,
            tcb_encode_ns: 0,
            dispatches: 900,
            command_buffers: 1,
        };
        assert_eq!(
            step.named_sum_ns() as i64 + step.residual_ns(),
            step.wall_ns as i64
        );
        assert_eq!(step.wait_minus_gpu_ns(), Some(400_000));
    }

    #[test]
    fn complete_token_names_tokenizer_and_bookkeeping() {
        let token = Qwen38CompleteToken {
            role: "decode".into(),
            step_index: 12,
            token_in: 1,
            token_out: 2,
            step: Qwen38StepWall {
                wall_ns: 33_953_000,
                encode_ns: 400_000,
                submit_ns: 20_000,
                wait_ns: 33_500_000,
                gpu_ns: Some(33_100_000),
                commit_epilogue_ns: 30_000,
                sample_readback_ns: 2_000,
                state_update_ns: 1_000,
                tcb_encode_ns: 0,
                dispatches: 900,
                command_buffers: 1,
            },
            tokenizer_decode_ns: 8_000,
            bookkeeping_ns: 1_000,
            complete_wall_ns: 33_970_000,
        };
        assert_eq!(
            token.named_sum_ns() as i64 + token.residual_ns(),
            token.complete_wall_ns as i64
        );
        assert_eq!(token.wall_minus_gpu_ns(), Some(870_000));
    }
}

#[cfg(test)]
mod workspace_bytes_tests {
    use super::qwen38_workspace_bytes;

    #[test]
    fn rejects_zero_seq() {
        assert!(qwen38_workspace_bytes(0).is_err());
    }

    #[test]
    fn kv_is_the_seq_len_term() {
        let a = qwen38_workspace_bytes(2048).unwrap();
        let b = qwen38_workspace_bytes(4096).unwrap();
        assert_eq!(a.activation_bytes, b.activation_bytes);
        assert_eq!(a.deltanet_state_bytes, b.deltanet_state_bytes);
        assert_eq!(b.gqa_kv_bytes, a.gqa_kv_bytes * 2);
        assert_eq!(
            b.total_bytes - a.total_bytes,
            b.gqa_kv_bytes - a.gqa_kv_bytes
        );
    }

    #[test]
    fn seq2048_is_hundreds_of_mb_not_weight_sized() {
        let bytes = qwen38_workspace_bytes(2048).unwrap();
        assert!(
            bytes.total_bytes > 200 * 1024 * 1024,
            "workspace {}",
            bytes.total_bytes
        );
        assert!(
            bytes.total_bytes < 2 * 1024 * 1024 * 1024,
            "workspace {} must stay far below the 8.5 GB artifact",
            bytes.total_bytes
        );
        assert!(bytes.deltanet_state_bytes > bytes.gqa_kv_bytes / 4);
    }
}
