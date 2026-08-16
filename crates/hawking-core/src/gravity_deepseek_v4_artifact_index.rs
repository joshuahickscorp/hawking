//! Compact mmap-able DSV4F artifact index.
//!
//! Built once at admission time. A later process can reconstruct the tensor
//! map and the admission-trust identity table without parsing the 151 MB
//! manifest, the range journal, or the 13 MB JSON receipt.
//!
//! The file is a fixed header plus little-endian arrays. Load is a read-only
//! mmap and direct indexing / binary search — there is no JSON step.
//!
//! Validity is the same cheap identity used by admission trust: source-file
//! size + mtime_ns plus the manifest seal. A missing, truncated, corrupt, or
//! stale index is ignored and the caller takes today's JSON path.

use crate::gravity_deepseek_v4::{
    gravity, DeepSeekV4Segment, DeepSeekV4SourceIdentity, DeepSeekV4TensorMetadata,
};
use crate::gravity_deepseek_v4_admission_trust::{
    admission_receipt_cache_root, file_identity, DeepSeekV4AdmissionTrustIndex,
    DeepSeekV4ChunkFileIdentity,
};
use crate::Result;
use memmap2::MmapOptions;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Schema / format version stored in the header.
pub const ARTIFACT_INDEX_VERSION: u32 = 1;
/// Filename preferred beside the sealed artifact.
pub const ARTIFACT_INDEX_NAME: &str = ".hawking-artifact-index";
pub const ARTIFACT_INDEX_SCHEMA: &str = "hawking.gravity.deepseek_v4.artifact_index.v1";

const MAGIC: &[u8; 8] = b"DSV4IDX\x01";
const HEADER_BYTES: usize = 512;
const TENSOR_REC: usize = 80;
const SEGMENT_REC: usize = 64;
const CHUNK_REC: usize = 80;
const PAIR_REC: usize = 56;
const ASSET_REC: usize = 40;

const OFF_MAGIC: usize = 0;
const OFF_VERSION: usize = 8;
const OFF_HEADER_BYTES: usize = 12;
const OFF_MANIFEST_SEAL: usize = 24;
const OFF_CHUNK_DIGEST: usize = 56;
const OFF_INDEX_SEAL: usize = 88;
const OFF_MANIFEST_BYTES: usize = 120;
const OFF_MANIFEST_MTIME: usize = 128;
const OFF_RANGES_BYTES: usize = 136;
const OFF_RANGES_MTIME: usize = 144;
const OFF_JOURNAL_BYTES: usize = 152;
const OFF_JOURNAL_MTIME: usize = 160;
const OFF_TENSOR_COUNT: usize = 168;
const OFF_CHUNK_COUNT: usize = 172;
const OFF_SEGMENT_COUNT: usize = 176;
const OFF_STRING_COUNT: usize = 180;
const OFF_STRING_BLOB: usize = 184;
const OFF_PAIR_COUNT: usize = 188;
const OFF_ASSET_COUNT: usize = 192;
const OFF_STRINGS: usize = 200;
const OFF_STRING_OFFS: usize = 208;
const OFF_TENSORS: usize = 216;
const OFF_SEGMENTS: usize = 224;
const OFF_CHUNKS: usize = 232;
const OFF_PAIRS: usize = 240;
const OFF_SHAPES: usize = 248;
const OFF_ASSETS: usize = 256;
const OFF_TENSOR_BYTES: usize = 264;
const OFF_RESTART_SEAL: usize = 272;
const OFF_MANIFEST_FILE_SHA: usize = 304;
const OFF_TABLE_SHA: usize = 336;
const OFF_SEALED_AT: usize = 368;
const OFF_REPO_SID: usize = 376;
const OFF_REV_SID: usize = 380;
const OFF_VERIFIER_SID: usize = 384;

/// Result of writing one index.
#[derive(Debug, Clone)]
pub struct DeepSeekV4ArtifactIndexSeal {
    pub path: PathBuf,
    pub bytes: u64,
    pub wall_ms: u128,
    pub tensor_count: usize,
    pub chunk_count: usize,
    pub index_seal_sha256: String,
}

/// Why an on-disk index was not used.
#[derive(Debug, Clone)]
pub enum DeepSeekV4IndexLoad {
    Disabled,
    Missing,
    Rejected(String),
    Loaded(DeepSeekV4IndexContents),
}

/// Owned reconstruction of an index (structurally identical to the JSON path).
#[derive(Debug, Clone)]
pub struct DeepSeekV4IndexContents {
    pub path: PathBuf,
    pub source: DeepSeekV4SourceIdentity,
    pub manifest_seal_sha256: String,
    pub manifest_file_sha256: String,
    pub restart_seal_sha256: String,
    pub content_addressed_chunk_sha256: String,
    pub tensor_bytes: u64,
    pub tensors: BTreeMap<String, DeepSeekV4TensorMetadata>,
    pub chunks: BTreeMap<String, (String, u64)>,
    pub admission: DeepSeekV4AdmissionTrustIndex,
    pub source_metadata_sha256: BTreeMap<String, String>,
}

/// Inputs captured from an already-admitted reader.
pub struct IndexBuildInput<'a> {
    pub source_root: &'a Path,
    pub _reader_root: &'a Path,
    pub source: &'a DeepSeekV4SourceIdentity,
    pub manifest_seal_sha256: &'a str,
    pub manifest_file_sha256: &'a str,
    pub restart_seal_sha256: &'a str,
    pub content_addressed_chunk_sha256: &'a str,
    pub tensor_bytes: u64,
    pub tensors: &'a BTreeMap<String, DeepSeekV4TensorMetadata>,
    pub chunks: &'a BTreeMap<String, (String, u64)>,
    pub identities: &'a BTreeMap<String, DeepSeekV4ChunkFileIdentity>,
    pub source_metadata_sha256: &'a BTreeMap<String, String>,
    pub table_sha256: &'a str,
    pub sealed_at_unix_ms: u64,
    pub verifier_version: &'a str,
}

pub fn artifact_index_enabled() -> bool {
    match std::env::var("HAWKING_DSV4F_INDEX") {
        Ok(value) => {
            let v = value.trim().to_ascii_lowercase();
            !matches!(v.as_str(), "0" | "off" | "false" | "no" | "json")
        }
        Err(_) => true,
    }
}

pub fn artifact_index_path(root: impl AsRef<Path>) -> PathBuf {
    root.as_ref().join(ARTIFACT_INDEX_NAME)
}

pub fn artifact_index_cache_path(manifest_seal_sha256: &str) -> PathBuf {
    admission_receipt_cache_root().join(format!("{manifest_seal_sha256}.idx"))
}

fn explicit_index_path() -> Option<PathBuf> {
    std::env::var("HAWKING_DSV4F_INDEX_PATH")
        .ok()
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// Probe candidate index files for `source_root`. Never errors: a bad file
/// is a reject, not a hard fail.
pub fn load_artifact_index(source_root: impl AsRef<Path>) -> DeepSeekV4IndexLoad {
    if !artifact_index_enabled() {
        return DeepSeekV4IndexLoad::Disabled;
    }
    let source_root = source_root.as_ref();
    let Ok(expected) = source_identities(source_root) else {
        return DeepSeekV4IndexLoad::Missing;
    };
    let mut saw = false;
    let mut last_reject = None;
    for path in index_search_paths(source_root) {
        if !path.is_file() {
            continue;
        }
        saw = true;
        match load_index_file(&path, source_root, &expected) {
            Ok(contents) => return DeepSeekV4IndexLoad::Loaded(contents),
            Err(error) => last_reject = Some(format!("{error}")),
        }
    }
    if !saw {
        DeepSeekV4IndexLoad::Missing
    } else {
        DeepSeekV4IndexLoad::Rejected(
            last_reject
                .unwrap_or_else(|| "artifact index present but could not be verified".to_owned()),
        )
    }
}

fn index_search_paths(root: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(explicit) = explicit_index_path() {
        paths.push(explicit);
    }
    paths.push(artifact_index_path(root));
    if let Ok(ids) = source_identities(root) {
        let alias = artifact_index_by_manifest_path(&ids);
        paths.push(alias.clone());
        if let Some(seal) = peek_manifest_seal(&alias) {
            paths.push(artifact_index_cache_path(&seal));
        }
    }
    paths
}

fn peek_manifest_seal(path: &Path) -> Option<String> {
    let raw = fs::read(path).ok()?;
    if raw.len() < OFF_MANIFEST_SEAL + 32 {
        return None;
    }
    if &raw[OFF_MAGIC..OFF_MAGIC + 8] != MAGIC {
        return None;
    }
    Some(hex32(&raw[OFF_MANIFEST_SEAL..OFF_MANIFEST_SEAL + 32]))
}

fn artifact_index_by_manifest_path(ids: &SourceFileIds) -> PathBuf {
    admission_receipt_cache_root().join(format!(
        "by-manifest-{}-{}.idx",
        ids.manifest_bytes, ids.manifest_mtime_ns
    ))
}

struct SourceFileIds {
    manifest_bytes: u64,
    manifest_mtime_ns: i64,
    ranges_bytes: u64,
    ranges_mtime_ns: i64,
    journal_bytes: u64,
    journal_mtime_ns: i64,
}

fn source_identities(root: &Path) -> Result<SourceFileIds> {
    let manifest = file_identity(&root.join("manifest.json"), "full stream manifest")?;
    let ranges = file_identity(
        &root.join("stream-ranges.jsonl"),
        "full stream range journal",
    )?;
    let journal = file_identity(&root.join("stream-journal.json"), "full stream journal")?;
    Ok(SourceFileIds {
        manifest_bytes: manifest.bytes,
        manifest_mtime_ns: i128_to_i64(manifest.mtime_ns),
        ranges_bytes: ranges.bytes,
        ranges_mtime_ns: i128_to_i64(ranges.mtime_ns),
        journal_bytes: journal.bytes,
        journal_mtime_ns: i128_to_i64(journal.mtime_ns),
    })
}

fn i128_to_i64(v: i128) -> i64 {
    i64::try_from(v).unwrap_or(i64::MAX)
}

/// Build and atomically publish the index. Prefers `<source>/.hawking-artifact-index`,
/// then the admission cache (this host's artifact tree is TCC-locked).
pub fn write_artifact_index(input: IndexBuildInput<'_>) -> Result<DeepSeekV4ArtifactIndexSeal> {
    let wall = std::time::Instant::now();
    let ids = source_identities(input.source_root)?;
    let bytes = encode_index(&input, &ids)?;
    let seal = hex32(&bytes[OFF_INDEX_SEAL..OFF_INDEX_SEAL + 32]);
    let path = publish_index(input.source_root, input.manifest_seal_sha256, &bytes)?;
    Ok(DeepSeekV4ArtifactIndexSeal {
        path,
        bytes: bytes.len() as u64,
        wall_ms: wall.elapsed().as_millis(),
        tensor_count: input.tensors.len(),
        chunk_count: input.identities.len(),
        index_seal_sha256: seal,
    })
}

fn encode_index(input: &IndexBuildInput<'_>, ids: &SourceFileIds) -> Result<Vec<u8>> {
    let mut intern = Intern::default();
    let repo_sid = intern.get(&input.source.repository);
    let rev_sid = intern.get(&input.source.revision);
    let verifier_sid = intern.get(input.verifier_version);

    let mut chunk_order: Vec<&DeepSeekV4ChunkFileIdentity> = input.identities.values().collect();
    chunk_order.sort_by(|a, b| a.key.cmp(&b.key));
    let mut chunk_index_of: BTreeMap<&str, u32> = BTreeMap::new();
    for (i, chunk) in chunk_order.iter().enumerate() {
        intern.get(&chunk.key);
        chunk_index_of.insert(chunk.key.as_str(), i as u32);
    }

    let mut shapes: Vec<u64> = Vec::new();
    let mut tensor_recs: Vec<TensorRec> = Vec::with_capacity(input.tensors.len());
    let mut segment_recs: Vec<SegmentRec> = Vec::new();
    for (name, tensor) in input.tensors {
        let shape_start = shapes.len() as u32;
        shapes.extend_from_slice(&tensor.shape);
        let segment_start = segment_recs.len() as u32;
        for segment in &tensor.segments {
            let chunk_index = *chunk_index_of
                .get(segment.chunk_relpath.as_str())
                .ok_or_else(|| {
                    gravity(format!(
                        "artifact index: tensor {name} references unknown chunk {}",
                        segment.chunk_relpath
                    ))
                })?;
            segment_recs.push(SegmentRec {
                bytes: segment.bytes,
                chunk_index,
                source_file_start: segment.source_file_start,
                source_file_end: segment.source_file_end,
                tensor_start: segment.tensor_start,
                tensor_end: segment.tensor_end,
                row_start: segment.row_start,
                row_count: segment.row_count,
            });
        }
        let (layer, organ) = classify_tensor_name(name);
        tensor_recs.push(TensorRec {
            name_sid: intern.get(name),
            dtype_sid: intern.get(&tensor.dtype),
            shard_sid: intern.get(&tensor.source_shard),
            shape_start,
            shape_len: tensor.shape.len() as u32,
            segment_start,
            segment_count: tensor.segments.len() as u32,
            layer,
            organ,
            data_off0: tensor.data_offsets[0],
            data_off1: tensor.data_offsets[1],
            bytes: tensor.bytes,
            source_file_start: tensor.source_file_start,
            source_file_end: tensor.source_file_end,
        });
    }

    let mut pair_recs = Vec::new();
    for (weight_name, tensor) in input.tensors {
        let kind = match tensor.dtype.as_str() {
            "I8" => 0u32,
            "F8_E4M3" => 1u32,
            _ => continue,
        };
        let Some(stem) = weight_name.strip_suffix(".weight") else {
            continue;
        };
        let scale_name = format!("{stem}.scale");
        let Some(scale) = input.tensors.get(&scale_name) else {
            continue;
        };
        if scale.dtype != "F8_E8M0" || tensor.shape.len() != 2 || scale.shape.len() != 2 {
            continue;
        }
        let out_rows = tensor.shape[0];
        let packed_k = tensor.shape[1];
        let (logical_k, scale_rows, scale_cols) = match kind {
            0 => (
                packed_k.saturating_mul(2),
                out_rows,
                packed_k.saturating_mul(2) / 32,
            ),
            _ => (packed_k, out_rows / 128, packed_k / 128),
        };
        pair_recs.push(PairRec {
            weight_sid: intern.get(weight_name),
            scale_sid: intern.get(&scale_name),
            kind,
            out_rows,
            packed_k,
            logical_k,
            scale_rows,
            scale_cols,
        });
    }

    let mut asset_recs = Vec::new();
    for (key, sha) in input.source_metadata_sha256 {
        asset_recs.push(AssetRec {
            key_sid: intern.get(key),
            sha256: parse_sha256(sha)?,
        });
    }

    let string_count = intern.offs.len() as u32;
    intern.offs.push(intern.blob.len() as u32);
    let string_blob_bytes = intern.blob.len() as u32;

    let mut cursor = HEADER_BYTES as u64;
    let strings_off = cursor;
    cursor += intern.blob.len() as u64;
    let string_offs_off = align8(cursor);
    cursor = string_offs_off + (intern.offs.len() as u64) * 4;
    let shapes_off = align8(cursor);
    cursor = shapes_off + (shapes.len() as u64) * 8;
    let tensors_off = align8(cursor);
    cursor = tensors_off + (tensor_recs.len() as u64) * TENSOR_REC as u64;
    let segments_off = align8(cursor);
    cursor = segments_off + (segment_recs.len() as u64) * SEGMENT_REC as u64;
    let chunks_off = align8(cursor);
    cursor = chunks_off + (chunk_order.len() as u64) * CHUNK_REC as u64;
    let pairs_off = align8(cursor);
    cursor = pairs_off + (pair_recs.len() as u64) * PAIR_REC as u64;
    let assets_off = align8(cursor);
    cursor = assets_off + (asset_recs.len() as u64) * ASSET_REC as u64;
    let total = cursor as usize;

    let mut out = vec![0u8; total];
    write_bytes(&mut out, OFF_MAGIC, MAGIC);
    write_u32(&mut out, OFF_VERSION, ARTIFACT_INDEX_VERSION);
    write_u32(&mut out, OFF_HEADER_BYTES, HEADER_BYTES as u32);
    write_bytes(
        &mut out,
        OFF_MANIFEST_SEAL,
        &parse_sha256(input.manifest_seal_sha256)?,
    );
    write_bytes(
        &mut out,
        OFF_CHUNK_DIGEST,
        &parse_sha256(input.content_addressed_chunk_sha256)?,
    );
    write_u64(&mut out, OFF_MANIFEST_BYTES, ids.manifest_bytes);
    write_i64(&mut out, OFF_MANIFEST_MTIME, ids.manifest_mtime_ns);
    write_u64(&mut out, OFF_RANGES_BYTES, ids.ranges_bytes);
    write_i64(&mut out, OFF_RANGES_MTIME, ids.ranges_mtime_ns);
    write_u64(&mut out, OFF_JOURNAL_BYTES, ids.journal_bytes);
    write_i64(&mut out, OFF_JOURNAL_MTIME, ids.journal_mtime_ns);
    write_u32(&mut out, OFF_TENSOR_COUNT, tensor_recs.len() as u32);
    write_u32(&mut out, OFF_CHUNK_COUNT, chunk_order.len() as u32);
    write_u32(&mut out, OFF_SEGMENT_COUNT, segment_recs.len() as u32);
    write_u32(&mut out, OFF_STRING_COUNT, string_count);
    write_u32(&mut out, OFF_STRING_BLOB, string_blob_bytes);
    write_u32(&mut out, OFF_PAIR_COUNT, pair_recs.len() as u32);
    write_u32(&mut out, OFF_ASSET_COUNT, asset_recs.len() as u32);
    write_u64(&mut out, OFF_STRINGS, strings_off);
    write_u64(&mut out, OFF_STRING_OFFS, string_offs_off);
    write_u64(&mut out, OFF_TENSORS, tensors_off);
    write_u64(&mut out, OFF_SEGMENTS, segments_off);
    write_u64(&mut out, OFF_CHUNKS, chunks_off);
    write_u64(&mut out, OFF_PAIRS, pairs_off);
    write_u64(&mut out, OFF_SHAPES, shapes_off);
    write_u64(&mut out, OFF_ASSETS, assets_off);
    write_u64(&mut out, OFF_TENSOR_BYTES, input.tensor_bytes);
    write_bytes(
        &mut out,
        OFF_RESTART_SEAL,
        &parse_sha256(input.restart_seal_sha256)?,
    );
    write_bytes(
        &mut out,
        OFF_MANIFEST_FILE_SHA,
        &parse_sha256(input.manifest_file_sha256)?,
    );
    write_bytes(&mut out, OFF_TABLE_SHA, &parse_sha256(input.table_sha256)?);
    write_u64(&mut out, OFF_SEALED_AT, input.sealed_at_unix_ms);
    write_u32(&mut out, OFF_REPO_SID, repo_sid);
    write_u32(&mut out, OFF_REV_SID, rev_sid);
    write_u32(&mut out, OFF_VERIFIER_SID, verifier_sid);

    out[strings_off as usize..strings_off as usize + intern.blob.len()]
        .copy_from_slice(&intern.blob);
    for (i, off) in intern.offs.iter().enumerate() {
        write_u32(&mut out, string_offs_off as usize + i * 4, *off);
    }
    for (i, dim) in shapes.iter().enumerate() {
        write_u64(&mut out, shapes_off as usize + i * 8, *dim);
    }
    for (i, rec) in tensor_recs.iter().enumerate() {
        write_tensor(&mut out, tensors_off as usize + i * TENSOR_REC, rec);
    }
    for (i, rec) in segment_recs.iter().enumerate() {
        write_segment(&mut out, segments_off as usize + i * SEGMENT_REC, rec);
    }
    for (i, chunk) in chunk_order.iter().enumerate() {
        write_chunk(
            &mut out,
            chunks_off as usize + i * CHUNK_REC,
            intern.map[&chunk.key],
            chunk,
        )?;
    }
    for (i, rec) in pair_recs.iter().enumerate() {
        write_pair(&mut out, pairs_off as usize + i * PAIR_REC, rec);
    }
    for (i, rec) in asset_recs.iter().enumerate() {
        write_asset(&mut out, assets_off as usize + i * ASSET_REC, rec);
    }

    let seal = Sha256::digest(&out);
    out[OFF_INDEX_SEAL..OFF_INDEX_SEAL + 32].copy_from_slice(&seal);
    Ok(out)
}

fn load_index_file(
    path: &Path,
    source_root: &Path,
    expected: &SourceFileIds,
) -> Result<DeepSeekV4IndexContents> {
    let file = File::open(path).map_err(|error| {
        gravity(format!(
            "cannot open artifact index {}: {error}",
            path.display()
        ))
    })?;
    let meta = file.metadata()?;
    if meta.len() < HEADER_BYTES as u64 {
        return Err(gravity("artifact index is truncated"));
    }
    let mmap = unsafe {
        MmapOptions::new()
            .map(&file)
            .map_err(|error| gravity(format!("cannot mmap artifact index: {error}")))?
    };
    if mmap.len() < HEADER_BYTES {
        return Err(gravity("artifact index is truncated"));
    }
    if &mmap[OFF_MAGIC..OFF_MAGIC + 8] != MAGIC {
        return Err(gravity("artifact index magic mismatch"));
    }
    if read_u32(&mmap, OFF_VERSION)? != ARTIFACT_INDEX_VERSION {
        return Err(gravity("artifact index version is not v1"));
    }
    if read_u32(&mmap, OFF_HEADER_BYTES)? as usize != HEADER_BYTES {
        return Err(gravity("artifact index header size mismatch"));
    }
    let recorded_seal = mmap[OFF_INDEX_SEAL..OFF_INDEX_SEAL + 32].to_vec();
    let mut hasher = Sha256::new();
    hasher.update(&mmap[..OFF_INDEX_SEAL]);
    hasher.update([0u8; 32]);
    hasher.update(&mmap[OFF_INDEX_SEAL + 32..]);
    let observed = hasher.finalize();
    if recorded_seal.as_slice() != observed.as_slice() {
        return Err(gravity("artifact index seal mismatch"));
    }
    if read_u64(&mmap, OFF_MANIFEST_BYTES)? != expected.manifest_bytes
        || read_i64(&mmap, OFF_MANIFEST_MTIME)? != expected.manifest_mtime_ns
        || read_u64(&mmap, OFF_RANGES_BYTES)? != expected.ranges_bytes
        || read_i64(&mmap, OFF_RANGES_MTIME)? != expected.ranges_mtime_ns
        || read_u64(&mmap, OFF_JOURNAL_BYTES)? != expected.journal_bytes
        || read_i64(&mmap, OFF_JOURNAL_MTIME)? != expected.journal_mtime_ns
    {
        return Err(gravity(
            "artifact index source identity (size/mtime_ns) disagrees",
        ));
    }

    let string_count = read_u32(&mmap, OFF_STRING_COUNT)? as usize;
    let string_blob_bytes = read_u32(&mmap, OFF_STRING_BLOB)? as usize;
    let strings_off = read_u64(&mmap, OFF_STRINGS)? as usize;
    let string_offs_off = read_u64(&mmap, OFF_STRING_OFFS)? as usize;
    let tensors_off = read_u64(&mmap, OFF_TENSORS)? as usize;
    let segments_off = read_u64(&mmap, OFF_SEGMENTS)? as usize;
    let chunks_off = read_u64(&mmap, OFF_CHUNKS)? as usize;
    let assets_off = read_u64(&mmap, OFF_ASSETS)? as usize;
    let shapes_off = read_u64(&mmap, OFF_SHAPES)? as usize;
    let tensor_count = read_u32(&mmap, OFF_TENSOR_COUNT)? as usize;
    let chunk_count = read_u32(&mmap, OFF_CHUNK_COUNT)? as usize;
    let segment_count = read_u32(&mmap, OFF_SEGMENT_COUNT)? as usize;
    let asset_count = read_u32(&mmap, OFF_ASSET_COUNT)? as usize;

    need(&mmap, strings_off, string_blob_bytes)?;
    need(&mmap, string_offs_off, (string_count + 1) * 4)?;
    need(&mmap, tensors_off, tensor_count * TENSOR_REC)?;
    need(&mmap, segments_off, segment_count * SEGMENT_REC)?;
    need(&mmap, chunks_off, chunk_count * CHUNK_REC)?;
    need(&mmap, assets_off, asset_count * ASSET_REC)?;

    let intern = InternView {
        blob: &mmap[strings_off..strings_off + string_blob_bytes],
        offs: &mmap[string_offs_off..string_offs_off + (string_count + 1) * 4],
        count: string_count,
    };

    let mut chunk_relpaths = Vec::with_capacity(chunk_count);
    let mut chunks = BTreeMap::new();
    let mut identities = BTreeMap::new();
    for i in 0..chunk_count {
        let rec = read_chunk(&mmap, chunks_off + i * CHUNK_REC, &intern)?;
        chunk_relpaths.push(rec.relative.clone());
        chunks.insert(rec.relative.clone(), (rec.sha256.clone(), rec.bytes));
        identities.insert(
            rec.relative.clone(),
            DeepSeekV4ChunkFileIdentity {
                key: rec.relative,
                bytes: rec.bytes,
                mtime_ns: rec.mtime_ns,
                inode: rec.inode,
            },
        );
    }

    let mut tensors = BTreeMap::new();
    for i in 0..tensor_count {
        let base = tensors_off + i * TENSOR_REC;
        let name = intern.get(read_u32(&mmap, base)?)?.to_owned();
        let dtype = intern.get(read_u32(&mmap, base + 4)?)?.to_owned();
        let shard = intern.get(read_u32(&mmap, base + 8)?)?.to_owned();
        let shape_start = read_u32(&mmap, base + 12)? as usize;
        let shape_len = read_u32(&mmap, base + 16)? as usize;
        let segment_start = read_u32(&mmap, base + 20)? as usize;
        let segment_count_here = read_u32(&mmap, base + 24)? as usize;
        need(&mmap, shapes_off + shape_start * 8, shape_len * 8)?;
        let mut shape = Vec::with_capacity(shape_len);
        for s in 0..shape_len {
            shape.push(read_u64(&mmap, shapes_off + (shape_start + s) * 8)?);
        }
        if segment_start + segment_count_here > segment_count {
            return Err(gravity("artifact index tensor segment range escapes table"));
        }
        let mut segments = Vec::with_capacity(segment_count_here);
        for s in 0..segment_count_here {
            let seg = read_segment(&mmap, segments_off + (segment_start + s) * SEGMENT_REC)?;
            let rel = chunk_relpaths
                .get(seg.chunk_index as usize)
                .ok_or_else(|| gravity("artifact index segment chunk_index out of range"))?;
            let (sha, _) = chunks
                .get(rel)
                .ok_or_else(|| gravity("artifact index segment chunk missing"))?;
            segments.push(DeepSeekV4Segment {
                bytes: seg.bytes,
                chunk_relpath: rel.clone(),
                sha256: sha.clone(),
                source_file_start: seg.source_file_start,
                source_file_end: seg.source_file_end,
                tensor_start: seg.tensor_start,
                tensor_end: seg.tensor_end,
                row_start: seg.row_start,
                row_count: seg.row_count,
            });
        }
        tensors.insert(
            name.clone(),
            DeepSeekV4TensorMetadata {
                name,
                dtype,
                shape,
                data_offsets: [read_u64(&mmap, base + 32)?, read_u64(&mmap, base + 40)?],
                bytes: read_u64(&mmap, base + 48)?,
                source_file_start: read_u64(&mmap, base + 56)?,
                source_file_end: read_u64(&mmap, base + 64)?,
                source_shard: shard,
                segments,
            },
        );
    }

    let mut source_metadata_sha256 = BTreeMap::new();
    for i in 0..asset_count {
        let base = assets_off + i * ASSET_REC;
        let key = intern.get(read_u32(&mmap, base)?)?.to_owned();
        source_metadata_sha256.insert(key, hex32(&mmap[base + 8..base + 40]));
    }

    let manifest_seal = hex32(&mmap[OFF_MANIFEST_SEAL..OFF_MANIFEST_SEAL + 32]);
    let chunk_digest = hex32(&mmap[OFF_CHUNK_DIGEST..OFF_CHUNK_DIGEST + 32]);
    let table_sha256 = hex32(&mmap[OFF_TABLE_SHA..OFF_TABLE_SHA + 32]);
    let total_bytes = identities
        .values()
        .map(|c| c.bytes)
        .fold(0u64, |a, b| a.saturating_add(b));
    let admission = DeepSeekV4AdmissionTrustIndex {
        artifact_root: source_root.to_path_buf(),
        content_addressed_chunk_sha256: chunk_digest.clone(),
        manifest_seal_sha256: manifest_seal.clone(),
        chunk_count,
        total_bytes,
        chunks: identities,
        table_sha256,
        seal_sha256: hex32(&recorded_seal),
        sealed_at_unix_ms: read_u64(&mmap, OFF_SEALED_AT)?,
        verifier_version: intern.get(read_u32(&mmap, OFF_VERIFIER_SID)?)?.to_owned(),
    };

    Ok(DeepSeekV4IndexContents {
        path: path.to_path_buf(),
        source: DeepSeekV4SourceIdentity {
            repository: intern.get(read_u32(&mmap, OFF_REPO_SID)?)?.to_owned(),
            revision: intern.get(read_u32(&mmap, OFF_REV_SID)?)?.to_owned(),
        },
        manifest_seal_sha256: manifest_seal,
        manifest_file_sha256: hex32(&mmap[OFF_MANIFEST_FILE_SHA..OFF_MANIFEST_FILE_SHA + 32]),
        restart_seal_sha256: hex32(&mmap[OFF_RESTART_SEAL..OFF_RESTART_SEAL + 32]),
        content_addressed_chunk_sha256: chunk_digest,
        tensor_bytes: read_u64(&mmap, OFF_TENSOR_BYTES)?,
        tensors,
        chunks,
        admission,
        source_metadata_sha256,
    })
}

/// Compare two tensor maps field-for-field. Used by the correctness gate.
pub fn tensor_maps_structurally_equal(
    left: &BTreeMap<String, DeepSeekV4TensorMetadata>,
    right: &BTreeMap<String, DeepSeekV4TensorMetadata>,
) -> Result<()> {
    if left.len() != right.len() {
        return Err(gravity(format!(
            "tensor map size differs: {} vs {}",
            left.len(),
            right.len()
        )));
    }
    for ((lk, lv), (rk, rv)) in left.iter().zip(right.iter()) {
        if lk != rk {
            return Err(gravity(format!(
                "tensor key order differs: {lk:?} vs {rk:?}"
            )));
        }
        if lv != rv {
            return Err(gravity(format!("tensor {lk} differs between maps")));
        }
    }
    Ok(())
}

pub fn chunk_maps_structurally_equal(
    left: &BTreeMap<String, (String, u64)>,
    right: &BTreeMap<String, (String, u64)>,
) -> Result<()> {
    if left != right {
        return Err(gravity("chunk binding maps differ"));
    }
    Ok(())
}

fn classify_tensor_name(name: &str) -> (i16, u8) {
    let layer = name
        .strip_prefix("layers.")
        .and_then(|rest| rest.split_once('.'))
        .and_then(|(n, _)| n.parse::<i16>().ok())
        .unwrap_or(-1);
    let organ = if name.starts_with("embed") {
        1
    } else if name.starts_with("head") {
        2
    } else if name.contains(".attn.") {
        3
    } else if name.contains(".ffn.") {
        4
    } else if name.starts_with("hc_") {
        5
    } else {
        0
    };
    (layer, organ)
}

#[derive(Default)]
struct Intern {
    map: BTreeMap<String, u32>,
    blob: Vec<u8>,
    offs: Vec<u32>,
}

impl Intern {
    fn get(&mut self, s: &str) -> u32 {
        if let Some(id) = self.map.get(s) {
            return *id;
        }
        let id = self.offs.len() as u32;
        self.offs.push(self.blob.len() as u32);
        self.blob.extend_from_slice(s.as_bytes());
        self.map.insert(s.to_owned(), id);
        id
    }
}

struct InternView<'a> {
    blob: &'a [u8],
    offs: &'a [u8],
    count: usize,
}

impl InternView<'_> {
    fn get(&self, id: u32) -> Result<&str> {
        let i = id as usize;
        if i >= self.count {
            return Err(gravity("artifact index string id out of range"));
        }
        let start = read_u32_slice(self.offs, i * 4)? as usize;
        let end = read_u32_slice(self.offs, (i + 1) * 4)? as usize;
        if start > end || end > self.blob.len() {
            return Err(gravity("artifact index string range escapes blob"));
        }
        std::str::from_utf8(&self.blob[start..end])
            .map_err(|_| gravity("artifact index string is not UTF-8"))
    }
}

struct TensorRec {
    name_sid: u32,
    dtype_sid: u32,
    shard_sid: u32,
    shape_start: u32,
    shape_len: u32,
    segment_start: u32,
    segment_count: u32,
    layer: i16,
    organ: u8,
    data_off0: u64,
    data_off1: u64,
    bytes: u64,
    source_file_start: u64,
    source_file_end: u64,
}

struct SegmentRec {
    bytes: u64,
    chunk_index: u32,
    source_file_start: u64,
    source_file_end: u64,
    tensor_start: u64,
    tensor_end: u64,
    row_start: u64,
    row_count: u64,
}

struct PairRec {
    weight_sid: u32,
    scale_sid: u32,
    kind: u32,
    out_rows: u64,
    packed_k: u64,
    logical_k: u64,
    scale_rows: u64,
    scale_cols: u64,
}

struct AssetRec {
    key_sid: u32,
    sha256: [u8; 32],
}

struct LoadedChunk {
    relative: String,
    sha256: String,
    bytes: u64,
    mtime_ns: i128,
    inode: u64,
}

fn write_tensor(out: &mut [u8], at: usize, rec: &TensorRec) {
    write_u32(out, at, rec.name_sid);
    write_u32(out, at + 4, rec.dtype_sid);
    write_u32(out, at + 8, rec.shard_sid);
    write_u32(out, at + 12, rec.shape_start);
    write_u32(out, at + 16, rec.shape_len);
    write_u32(out, at + 20, rec.segment_start);
    write_u32(out, at + 24, rec.segment_count);
    write_i16(out, at + 28, rec.layer);
    out[at + 30] = rec.organ;
    write_u64(out, at + 32, rec.data_off0);
    write_u64(out, at + 40, rec.data_off1);
    write_u64(out, at + 48, rec.bytes);
    write_u64(out, at + 56, rec.source_file_start);
    write_u64(out, at + 64, rec.source_file_end);
}

fn write_segment(out: &mut [u8], at: usize, rec: &SegmentRec) {
    write_u64(out, at, rec.bytes);
    write_u32(out, at + 8, rec.chunk_index);
    write_u64(out, at + 16, rec.source_file_start);
    write_u64(out, at + 24, rec.source_file_end);
    write_u64(out, at + 32, rec.tensor_start);
    write_u64(out, at + 40, rec.tensor_end);
    write_u64(out, at + 48, rec.row_start);
    write_u64(out, at + 56, rec.row_count);
}

fn write_chunk(
    out: &mut [u8],
    at: usize,
    relative_sid: u32,
    chunk: &DeepSeekV4ChunkFileIdentity,
) -> Result<()> {
    write_u32(out, at, relative_sid);
    write_bytes(
        out,
        at + 8,
        &parse_sha256_from_rel_or_hex(&chunk.key, chunk)?,
    );
    write_u64(out, at + 40, chunk.bytes);
    let (lo, hi) = split_i128(chunk.mtime_ns);
    write_i64(out, at + 48, lo);
    write_i64(out, at + 56, hi);
    write_u64(out, at + 64, chunk.inode);
    Ok(())
}

fn parse_sha256_from_rel_or_hex(
    key: &str,
    chunk: &DeepSeekV4ChunkFileIdentity,
) -> Result<[u8; 32]> {
    // Chunk filename is the digest: chunks/<2 hex>/<64 hex>
    if let Some(name) = key.rsplit('/').next() {
        if name.len() == 64 {
            return parse_sha256(name);
        }
    }
    let _ = chunk;
    Err(gravity(format!(
        "artifact index chunk key is not a content-addressed digest path: {key}"
    )))
}

fn write_pair(out: &mut [u8], at: usize, rec: &PairRec) {
    write_u32(out, at, rec.weight_sid);
    write_u32(out, at + 4, rec.scale_sid);
    write_u32(out, at + 8, rec.kind);
    write_u64(out, at + 16, rec.out_rows);
    write_u64(out, at + 24, rec.packed_k);
    write_u64(out, at + 32, rec.logical_k);
    write_u64(out, at + 40, rec.scale_rows);
    write_u64(out, at + 48, rec.scale_cols);
}

fn write_asset(out: &mut [u8], at: usize, rec: &AssetRec) {
    write_u32(out, at, rec.key_sid);
    write_bytes(out, at + 8, &rec.sha256);
}

fn read_segment(buf: &[u8], at: usize) -> Result<SegmentRec> {
    Ok(SegmentRec {
        bytes: read_u64(buf, at)?,
        chunk_index: read_u32(buf, at + 8)?,
        source_file_start: read_u64(buf, at + 16)?,
        source_file_end: read_u64(buf, at + 24)?,
        tensor_start: read_u64(buf, at + 32)?,
        tensor_end: read_u64(buf, at + 40)?,
        row_start: read_u64(buf, at + 48)?,
        row_count: read_u64(buf, at + 56)?,
    })
}

fn read_chunk(buf: &[u8], at: usize, intern: &InternView<'_>) -> Result<LoadedChunk> {
    let relative = intern.get(read_u32(buf, at)?)?.to_owned();
    need(buf, at + 8, 32)?;
    Ok(LoadedChunk {
        relative,
        sha256: hex32(&buf[at + 8..at + 40]),
        bytes: read_u64(buf, at + 40)?,
        mtime_ns: join_i128(read_i64(buf, at + 48)?, read_i64(buf, at + 56)?),
        inode: read_u64(buf, at + 64)?,
    })
}

fn publish_index(source_root: &Path, manifest_seal: &str, bytes: &[u8]) -> Result<PathBuf> {
    let beside = artifact_index_path(source_root);
    match write_index_atomic(&beside, bytes) {
        Ok(path) => return Ok(path),
        Err(error) => {
            let message = format!("{error}");
            if !message.contains("Operation not permitted")
                && !message.contains("Read-only file system")
                && !message.contains("permission denied")
                && !message.contains("Permission denied")
            {
                return Err(error);
            }
        }
    }
    if let Some(explicit) = explicit_index_path() {
        if let Some(parent) = explicit.parent() {
            fs::create_dir_all(parent).map_err(|error| {
                gravity(format!(
                    "cannot create artifact index dir {}: {error}",
                    parent.display()
                ))
            })?;
        }
        return write_index_atomic(&explicit, bytes);
    }
    let cache = artifact_index_cache_path(manifest_seal);
    if let Some(parent) = cache.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            gravity(format!(
                "cannot create artifact index cache {}: {error}",
                parent.display()
            ))
        })?;
    }
    let published = write_index_atomic(&cache, bytes)?;
    if let Ok(ids) = source_identities(source_root) {
        let alias = artifact_index_by_manifest_path(&ids);
        let _ = fs::remove_file(&alias);
        if fs::hard_link(&published, &alias).is_err() {
            let _ = fs::copy(&published, &alias);
        }
    }
    Ok(published)
}

fn write_index_atomic(path: &Path, bytes: &[u8]) -> Result<PathBuf> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| gravity("artifact index path must have a UTF-8 file name"))?;
    let temporary = parent.join(format!(".{name}.{}.artifact-index.tmp", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| {
            gravity(format!(
                "cannot create artifact index temporary {}: {error}",
                temporary.display()
            ))
        })?;
    if let Err(error) = file.write_all(bytes).and_then(|_| file.sync_all()) {
        let _ = fs::remove_file(&temporary);
        return Err(gravity(format!(
            "cannot write artifact index temporary: {error}"
        )));
    }
    if let Err(error) = fs::rename(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(gravity(format!(
            "cannot publish artifact index {}: {error}",
            path.display()
        )));
    }
    if let Ok(dir) = File::open(parent) {
        let _ = dir.sync_all();
    }
    Ok(path.to_path_buf())
}

fn align8(n: u64) -> u64 {
    (n + 7) & !7
}

fn need(buf: &[u8], off: usize, len: usize) -> Result<()> {
    if off
        .checked_add(len)
        .map(|end| end <= buf.len())
        .unwrap_or(false)
    {
        Ok(())
    } else {
        Err(gravity("artifact index section escapes file"))
    }
}

fn write_bytes(out: &mut [u8], at: usize, bytes: &[u8]) {
    out[at..at + bytes.len()].copy_from_slice(bytes);
}

fn write_u32(out: &mut [u8], at: usize, v: u32) {
    out[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

fn write_u64(out: &mut [u8], at: usize, v: u64) {
    out[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

fn write_i64(out: &mut [u8], at: usize, v: i64) {
    out[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

fn write_i16(out: &mut [u8], at: usize, v: i16) {
    out[at..at + 2].copy_from_slice(&v.to_le_bytes());
}

fn read_u32(buf: &[u8], at: usize) -> Result<u32> {
    need(buf, at, 4)?;
    Ok(u32::from_le_bytes(buf[at..at + 4].try_into().unwrap()))
}

fn read_u32_slice(buf: &[u8], at: usize) -> Result<u32> {
    if at + 4 > buf.len() {
        return Err(gravity("artifact index u32 escapes slice"));
    }
    Ok(u32::from_le_bytes(buf[at..at + 4].try_into().unwrap()))
}

fn read_u64(buf: &[u8], at: usize) -> Result<u64> {
    need(buf, at, 8)?;
    Ok(u64::from_le_bytes(buf[at..at + 8].try_into().unwrap()))
}

fn read_i64(buf: &[u8], at: usize) -> Result<i64> {
    need(buf, at, 8)?;
    Ok(i64::from_le_bytes(buf[at..at + 8].try_into().unwrap()))
}

fn parse_sha256(hex: &str) -> Result<[u8; 32]> {
    if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(gravity(format!(
            "artifact index expected lowercase SHA-256, got {hex:?}"
        )));
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)
            .map_err(|_| gravity("artifact index SHA-256 hex is malformed"))?;
    }
    Ok(out)
}

fn hex32(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

fn split_i128(v: i128) -> (i64, i64) {
    (v as i64, (v >> 64) as i64)
}

fn join_i128(lo: i64, hi: i64) -> i128 {
    ((hi as i128) << 64) | ((lo as u64) as i128)
}
