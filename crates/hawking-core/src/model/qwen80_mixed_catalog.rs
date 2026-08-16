//! Compact mmap catalog for the Q80 mixed ≤1.5 packed artifact.
//!
//! Opens the small sealed JSON manifest plus `catalog.hq80m15`. Payload
//! bodies stay on disk until requested. This is not a Metal kernel, a
//! generation runtime, or a ≤1.5 coherence claim.

use crate::{Error, Result};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

pub const QWEN80_MIXED_CATALOG_MAGIC: [u8; 8] = *b"HQ80M15\0";
pub const QWEN80_MIXED_CATALOG_VERSION: u32 = 1;
pub const QWEN80_MIXED_RECORD_SIZE: usize = 128;
pub const QWEN80_MIXED_SCHEMA: &str =
    "hawking.ascension.qwen80_mixed_representation_candidate.v1";
pub const QWEN80_MIXED_MANIFEST_NAME: &str =
    "QWEN80_MIXED_1P5_V1_COMPLETE_BINARY_GRAVITY_CANDIDATE.json";
pub const QWEN80_MIXED_CATALOG_NAME: &str = "catalog.hq80m15";
pub const QWEN80_MIXED_EXPECTED_TENSOR_COUNT: usize = 74_391;

pub const CODEC_BINARY: u8 = 0;
pub const CODEC_RESIDUAL: u8 = 1;
pub const CODEC_HGRAVS01: u8 = 2;
pub const CODEC_UNIFORM8: u8 = 3;

pub const FLAG_SENSITIVE: u32 = 1 << 0;
pub const FLAG_GRAM_RANKDEF: u32 = 1 << 1;
pub const FLAG_WEIGHT_SPACE: u32 = 1 << 2;
pub const FLAG_ACTIVATION_WEIGHTED: u32 = 1 << 3;

const DEFAULT_ROOT_REL: &str =
    "workspace/campaign/records/ascension-sandbox/physical/qwen80/quality-candidates/mixed-1p5-v1";

fn mixed_error(message: impl Into<String>) -> Error {
    Error::Model(format!("qwen80 mixed catalog: {}", message.into()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn read_u16(raw: &[u8], off: usize) -> Result<u16> {
    let slice = raw
        .get(off..off + 2)
        .ok_or_else(|| mixed_error("catalog truncated at u16"))?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_u32(raw: &[u8], off: usize) -> Result<u32> {
    let slice = raw
        .get(off..off + 4)
        .ok_or_else(|| mixed_error("catalog truncated at u32"))?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_u64(raw: &[u8], off: usize) -> Result<u64> {
    let slice = raw
        .get(off..off + 8)
        .ok_or_else(|| mixed_error("catalog truncated at u64"))?;
    Ok(u64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}

#[derive(Clone, Debug)]
pub struct Qwen80MixedSegment {
    pub id: u16,
    pub filename: String,
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug)]
pub struct Qwen80MixedCatalogRow {
    pub tensor_name: String,
    pub codec: u8,
    pub organ: u8,
    pub shape: Vec<usize>,
    pub elements: u64,
    pub segment_id: u16,
    pub offset: u64,
    pub nbytes: u64,
    pub sha256: String,
    pub flags: u32,
    pub n_fit_rows: u32,
    pub achieved_rank: u16,
    pub codec_bpw: f32,
    pub segment_path: PathBuf,
}

#[derive(Debug)]
pub struct Qwen80MixedStreamingCatalog {
    pub root: PathBuf,
    pub manifest_path: PathBuf,
    pub manifest_seal_sha256: String,
    pub complete_physical_bpw: f64,
    pub tensor_payload_bytes: u64,
    pub catalog_sha256: String,
    rows: HashMap<String, Qwen80MixedCatalogRow>,
    segments: Vec<Qwen80MixedSegment>,
}

impl Qwen80MixedStreamingCatalog {
    pub fn default_root_hint() -> &'static str {
        DEFAULT_ROOT_REL
    }

    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = fs::canonicalize(root.as_ref()).map_err(|error| {
            mixed_error(format!(
                "cannot canonicalize {}: {error}",
                root.as_ref().display()
            ))
        })?;
        Self::open_manifest(root.join(QWEN80_MIXED_MANIFEST_NAME))
    }

    pub fn open_manifest(manifest_path: impl AsRef<Path>) -> Result<Self> {
        let manifest_path = fs::canonicalize(manifest_path.as_ref()).map_err(|error| {
            mixed_error(format!(
                "cannot canonicalize manifest {}: {error}",
                manifest_path.as_ref().display()
            ))
        })?;
        let root = manifest_path
            .parent()
            .ok_or_else(|| mixed_error("manifest has no parent"))?
            .to_path_buf();
        let raw = fs::read(&manifest_path).map_err(|error| {
            mixed_error(format!("cannot read {}: {error}", manifest_path.display()))
        })?;
        let document: Value = serde_json::from_slice(&raw)
            .map_err(|error| mixed_error(format!("manifest is not JSON: {error}")))?;
        let object = document
            .as_object()
            .ok_or_else(|| mixed_error("manifest root must be an object"))?;
        let schema = object
            .get("schema")
            .and_then(Value::as_str)
            .ok_or_else(|| mixed_error("manifest missing schema"))?;
        if schema != QWEN80_MIXED_SCHEMA {
            return Err(mixed_error(format!(
                "manifest schema {schema:?} is not {QWEN80_MIXED_SCHEMA}"
            )));
        }
        let manifest_seal_sha256 = object
            .get("seal_sha256")
            .and_then(Value::as_str)
            .ok_or_else(|| mixed_error("manifest missing seal_sha256"))?
            .to_owned();
        let ledger = object
            .get("complete_physical_bpw_ledger")
            .and_then(Value::as_object)
            .ok_or_else(|| mixed_error("manifest missing complete_physical_bpw_ledger"))?;
        let complete_physical_bpw = ledger
            .get("complete_physical_bpw")
            .and_then(Value::as_f64)
            .ok_or_else(|| mixed_error("manifest missing complete_physical_bpw"))?;
        let tensor_payload_bytes = ledger
            .get("tensor_payload_bytes")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let catalog_rel = object
            .get("catalog")
            .and_then(|c| c.get("path"))
            .and_then(Value::as_str)
            .ok_or_else(|| mixed_error("manifest missing catalog.path"))?;
        let declared_sha = object
            .get("catalog")
            .and_then(|c| c.get("sha256"))
            .and_then(Value::as_str)
            .ok_or_else(|| mixed_error("manifest missing catalog.sha256"))?;
        let catalog_path = PathBuf::from(catalog_rel);
        let catalog_path = if catalog_path.is_absolute() {
            catalog_path
        } else {
            root.join(catalog_path)
        };
        let catalog_bytes = fs::read(&catalog_path).map_err(|error| {
            mixed_error(format!(
                "cannot read catalog {}: {error}",
                catalog_path.display()
            ))
        })?;
        let catalog_sha256 = sha256_hex(&catalog_bytes);
        if catalog_sha256 != declared_sha {
            return Err(mixed_error(format!(
                "catalog sha256 {catalog_sha256} != manifest {declared_sha}"
            )));
        }
        let (rows, segments) = parse_catalog_bytes(&catalog_bytes, &root)?;
        Ok(Self {
            root,
            manifest_path,
            manifest_seal_sha256,
            complete_physical_bpw,
            tensor_payload_bytes,
            catalog_sha256,
            rows,
            segments,
        })
    }

    pub fn tensor_count(&self) -> usize {
        self.rows.len()
    }

    pub fn segments(&self) -> &[Qwen80MixedSegment] {
        &self.segments
    }

    pub fn require_row(&self, name: &str) -> Result<&Qwen80MixedCatalogRow> {
        self.rows
            .get(name)
            .ok_or_else(|| mixed_error(format!("missing tensor {name:?}")))
    }

    pub fn read_payload(&self, name: &str) -> Result<Vec<u8>> {
        let row = self.require_row(name)?;
        let file = fs::read(&row.segment_path).map_err(|error| {
            mixed_error(format!(
                "cannot read segment {} for {name:?}: {error}",
                row.segment_path.display()
            ))
        })?;
        let start = usize::try_from(row.offset)
            .map_err(|_| mixed_error(format!("{name:?} offset exceeds usize")))?;
        let end = start
            .checked_add(usize::try_from(row.nbytes).unwrap_or(usize::MAX))
            .ok_or_else(|| mixed_error(format!("{name:?} range overflow")))?;
        if end > file.len() {
            return Err(mixed_error(format!(
                "{name:?} range {start}..{end} exceeds segment {}",
                file.len()
            )));
        }
        let payload = file[start..end].to_vec();
        let observed = sha256_hex(&payload);
        if observed != row.sha256 {
            return Err(mixed_error(format!(
                "{name:?} payload sha256 {observed} != catalog {}",
                row.sha256
            )));
        }
        Ok(payload)
    }
}

fn parse_catalog_bytes(
    raw: &[u8],
    root: &Path,
) -> Result<(
    HashMap<String, Qwen80MixedCatalogRow>,
    Vec<Qwen80MixedSegment>,
)> {
    if raw.len() < 32 || raw[..8] != QWEN80_MIXED_CATALOG_MAGIC {
        return Err(mixed_error("catalog magic is not HQ80M15"));
    }
    let version = read_u32(raw, 8)?;
    if version != QWEN80_MIXED_CATALOG_VERSION {
        return Err(mixed_error(format!("unsupported catalog version {version}")));
    }
    let n_tensors = read_u32(raw, 12)? as usize;
    let n_segments = read_u32(raw, 16)? as usize;
    let name_blob_bytes = read_u32(raw, 24)? as usize;
    let mut cursor = 32usize;
    let mut segments = Vec::with_capacity(n_segments);
    let mut by_id: HashMap<u16, PathBuf> = HashMap::new();
    for _ in 0..n_segments {
        let id = read_u16(raw, cursor)?;
        let name_len = read_u16(raw, cursor + 2)? as usize;
        let bytes = read_u64(raw, cursor + 4)?;
        let digest = raw
            .get(cursor + 12..cursor + 44)
            .ok_or_else(|| mixed_error("segment digest truncated"))?;
        cursor += 44;
        let filename = raw
            .get(cursor..cursor + name_len)
            .ok_or_else(|| mixed_error("segment name truncated"))?;
        let filename = std::str::from_utf8(filename)
            .map_err(|_| mixed_error("segment name is not utf-8"))?
            .to_owned();
        cursor += name_len;
        let path = root.join("segments").join(&filename);
        by_id.insert(id, path.clone());
        segments.push(Qwen80MixedSegment {
            id,
            filename,
            path,
            bytes,
            sha256: hex_of(digest),
        });
    }
    let table_bytes = n_tensors
        .checked_mul(QWEN80_MIXED_RECORD_SIZE)
        .ok_or_else(|| mixed_error("catalog table size overflow"))?;
    let table = raw
        .get(cursor..cursor + table_bytes)
        .ok_or_else(|| mixed_error("catalog tensor table truncated"))?;
    cursor += table_bytes;
    let name_blob = raw
        .get(cursor..cursor + name_blob_bytes)
        .ok_or_else(|| mixed_error("catalog name blob truncated"))?;
    let mut rows = HashMap::with_capacity(n_tensors);
    for index in 0..n_tensors {
        let rec = &table[index * QWEN80_MIXED_RECORD_SIZE
            ..(index + 1) * QWEN80_MIXED_RECORD_SIZE];
        let name_off = read_u32(rec, 0)? as usize;
        let name_len = read_u16(rec, 4)? as usize;
        let codec = rec[6];
        let organ = rec[7];
        let ndim = rec[8] as usize;
        if ndim > 4 {
            return Err(mixed_error("catalog ndim exceeds 4"));
        }
        let mut shape = Vec::with_capacity(ndim);
        for dim in 0..ndim {
            shape.push(read_u32(rec, 12 + dim * 4)? as usize);
        }
        let elements = read_u64(rec, 28)?;
        let segment_id = read_u16(rec, 36)?;
        let achieved_rank = read_u16(rec, 38)?;
        let offset = read_u64(rec, 40)?;
        let nbytes = read_u64(rec, 48)?;
        let digest = rec
            .get(56..88)
            .ok_or_else(|| mixed_error("row digest truncated"))?;
        let flags = read_u32(rec, 88)?;
        let n_fit_rows = read_u32(rec, 92)?;
        let codec_bpw_bits = read_u32(rec, 96)?;
        let codec_bpw = f32::from_bits(codec_bpw_bits);
        let name = name_blob
            .get(name_off..name_off + name_len)
            .ok_or_else(|| mixed_error("tensor name out of blob"))?;
        let tensor_name = std::str::from_utf8(name)
            .map_err(|_| mixed_error("tensor name is not utf-8"))?
            .to_owned();
        let segment_path = by_id
            .get(&segment_id)
            .cloned()
            .ok_or_else(|| mixed_error(format!("unknown segment_id {segment_id}")))?;
        let row = Qwen80MixedCatalogRow {
            tensor_name: tensor_name.clone(),
            codec,
            organ,
            shape,
            elements,
            segment_id,
            offset,
            nbytes,
            sha256: hex_of(digest),
            flags,
            n_fit_rows,
            achieved_rank,
            codec_bpw,
            segment_path,
        };
        if rows.insert(tensor_name, row).is_some() {
            return Err(mixed_error("catalog contains a duplicate tensor_name"));
        }
    }
    Ok((rows, segments))
}

fn hex_of(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magic_and_record_size_match_the_python_writer() {
        assert_eq!(&QWEN80_MIXED_CATALOG_MAGIC, b"HQ80M15\0");
        assert_eq!(QWEN80_MIXED_RECORD_SIZE, 128);
        assert_eq!(CODEC_HGRAVS01, 2);
    }
}
