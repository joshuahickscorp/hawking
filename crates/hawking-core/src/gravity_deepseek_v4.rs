//! Read-only admission for the sealed DeepSeek-V4-Flash full Gravity stream.
//!
//! The full V4 artifact is a content-addressed source stream, not an
//! executable Gravity model.  This module is deliberately below every engine,
//! serving, Metal, HCLI, and TPS surface.  It establishes the minimum safe
//! byte contract an eventual V4 adapter may consume:
//!
//! - the exact sealed full-stream manifest and pinned source identity;
//! - a non-symlink, regular-file content-addressed chunk tree;
//! - contiguous descriptor/segment/source-offset mappings;
//! - verified bounded range and explicitly bounded full-tensor reads; and
//! - exact native FP8 and packed-FP4 scale-pair geometry.
//!
//! No method here constructs an [`crate::Engine`], allocates Metal resources,
//! performs a model forward, or changes the public CLI admission policy.

use crate::{Error, Result};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Read;
use std::ops::Range;
use std::path::{Component, Path, PathBuf};

/// Immutable full-stream artifact schema produced by the Condense streamer.
pub const FULL_STREAM_SCHEMA: &str = "hawking.gravity.deepseek_v4.full_stream.v1";
/// The only status this reader admits.  In particular, this is not a runtime
/// readiness status.
pub const FULL_STREAM_STATUS: &str = "FULL_MODEL_STREAMED_SEALED_NOT_RUNTIME_READY";
/// The source repository bound into every admitted manifest and source window.
pub const PINNED_REPOSITORY: &str = "deepseek-ai/DeepSeek-V4-Flash";
/// The immutable source revision bound into every admitted manifest and source
/// window.
pub const PINNED_REVISION: &str = "60d8d70770c6776ff598c94bb586a859a38244f1";

const CONTENT_ADDRESSED_FORMAT: &str = "gravity.content_addressed.chunk_directory.v1";
const FULL_STREAM_KIND: &str = "deepseek_v4_full_43_layer_content_addressed_stream";
const EXPECTED_TENSOR_COUNT: usize = 69_187;
const EXPECTED_SOURCE_SHARDS: usize = 46;
const FP8_BLOCK: u64 = 128;
const FP4_LOGICAL_BLOCK: u64 = 32;
const READ_BUFFER_BYTES: usize = 1024 * 1024;

/// Pinned source identity exposed to a future adapter without opening a
/// source parent file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4SourceIdentity {
    pub repository: String,
    pub revision: String,
}

/// An admitted tensor descriptor.  `shape` is the source-native physical
/// safetensors shape: for packed FP4 weights the final dimension is packed-K,
/// while [`NativeScalePair::logical_k`] reports the decoded logical K.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4TensorMetadata {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub data_offsets: [u64; 2],
    pub bytes: u64,
    pub source_file_start: u64,
    pub source_file_end: u64,
    pub source_shard: String,
    pub segments: Vec<DeepSeekV4Segment>,
}

/// One source-contiguous, content-addressed segment of a tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4Segment {
    pub bytes: u64,
    pub chunk_relpath: String,
    pub sha256: String,
    pub source_file_start: u64,
    pub source_file_end: u64,
    pub tensor_start: u64,
    pub tensor_end: u64,
    pub row_start: u64,
    pub row_count: u64,
}

/// Native source representation selected by an admitted scale pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeScalePairKind {
    /// `I8 [out, packed_k]` containing two E2M1FN nibbles per byte, with one
    /// unsigned E8M0 scale per 32 logical K values.
    Fp4E2M1fnX2,
    /// `F8_E4M3 [out, logical_k]`, with an unsigned E8M0 scale per 128-by-128
    /// weight block.
    Fp8E4M3fn,
}

impl NativeScalePairKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fp4E2M1fnX2 => "native.fp4_e2m1fn_x2_e8m0",
            Self::Fp8E4M3fn => "native.fp8_e4m3fn_e8m0",
        }
    }
}

/// Geometry recorded for one native weight/scale pair.  It is derived from
/// and checked against the manifest rather than inferred by an adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeScalePairGeometry {
    pub kind: NativeScalePairKind,
    pub scale_name: String,
    pub out_rows: u64,
    pub packed_k: u64,
    pub logical_k: u64,
    pub scale_rows: u64,
    pub scale_cols: u64,
}

/// Borrowed lookup result for a validated source-native weight/scale pair.
#[derive(Debug, Clone, Copy)]
pub struct NativeScalePair<'a> {
    pub kind: NativeScalePairKind,
    pub weight: &'a DeepSeekV4TensorMetadata,
    pub scale: &'a DeepSeekV4TensorMetadata,
    pub out_rows: u64,
    pub packed_k: u64,
    pub logical_k: u64,
    pub scale_rows: u64,
    pub scale_cols: u64,
}

/// Exact content verified by a call to [`DeepSeekV4FullStreamReader::verify_all_chunks`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FullStreamChunkVerification {
    pub chunk_count: usize,
    pub bytes_verified: u64,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    schema: String,
    status: String,
    seal_sha256: String,
    artifact: ArtifactSection,
    source: SourceSection,
    full_model_scope: FullModelScope,
    architecture: ArchitectureSection,
    storage: StorageSection,
    runtime_adapter: RuntimeAdapter,
    restart_receipt: RestartReceiptBinding,
    tensors: BTreeMap<String, RawTensorMetadata>,
}

#[derive(Debug, Deserialize)]
struct ArtifactSection {
    kind: String,
    format: String,
    total_tensor_bytes: u64,
    source_index_total_size_bytes: u64,
    content_addressed_chunk_count: usize,
    content_addressed_chunk_sha256: String,
}

#[derive(Debug, Deserialize)]
struct SourceSection {
    repository: String,
    revision: String,
    metadata_assets: BTreeMap<String, MetadataAsset>,
    source_windows: Vec<SourceWindow>,
    source_shard_count: usize,
    source_parent_persisted: bool,
}

#[derive(Debug, Deserialize)]
struct MetadataAsset {
    path: String,
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Deserialize, Clone)]
struct SourceWindow {
    header_bytes: u64,
    streamed_full_file_sha256: String,
    source: SourceFile,
}

#[derive(Debug, Deserialize, Clone)]
struct SourceFile {
    repository: String,
    revision: String,
    commit_hash: String,
    shard: String,
    etag_sha256: String,
    xet_file_hash: String,
    file_size_bytes: u64,
}

#[derive(Debug, Deserialize)]
struct FullModelScope {
    full_width: bool,
    num_hidden_layers: usize,
    contains_all_routed_experts_per_layer: usize,
    contains_embedding_and_head: bool,
    tensor_count: usize,
    runtime_ready: bool,
}

#[derive(Debug, Deserialize)]
struct ArchitectureSection {
    model_type: String,
    layer_count: usize,
    tensor_count: usize,
}

#[derive(Debug, Deserialize)]
struct StorageSection {
    source_parent_retained: bool,
}

#[derive(Debug, Deserialize)]
struct RuntimeAdapter {
    id: Option<String>,
    registration: Option<String>,
    device: Option<String>,
    metal_dispatches: u64,
}

#[derive(Debug, Deserialize)]
struct RestartReceiptBinding {
    path: String,
    seal_sha256: String,
}

#[derive(Debug, Deserialize)]
struct RawTensorMetadata {
    name: String,
    dtype: String,
    shape: Vec<u64>,
    data_offsets: Vec<u64>,
    bytes: u64,
    source_file_start: u64,
    source_file_end: u64,
    source_shard: String,
    segments: Vec<RawSegment>,
}

#[derive(Debug, Deserialize)]
struct RawSegment {
    bytes: u64,
    chunk_relpath: String,
    sha256: String,
    source_file_start: u64,
    source_file_end: u64,
    tensor_start: u64,
    tensor_end: u64,
    row_start: u64,
    row_count: u64,
}

#[derive(Debug, Deserialize)]
struct SourceIndex {
    metadata: SourceIndexMetadata,
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct SourceIndexMetadata {
    total_size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ChunkBinding {
    relative: String,
    sha256: String,
    bytes: u64,
}

/// A successfully admitted full source stream.  It owns no decoded weights,
/// GPU buffers, or execution state.
#[derive(Debug)]
pub struct DeepSeekV4FullStreamReader {
    root: PathBuf,
    source: DeepSeekV4SourceIdentity,
    manifest_seal_sha256: String,
    manifest_file_sha256: String,
    restart_seal_sha256: String,
    tensor_bytes: u64,
    tensors: BTreeMap<String, DeepSeekV4TensorMetadata>,
    chunks: BTreeMap<String, ChunkBinding>,
    native_pairs: BTreeMap<String, NativeScalePairGeometry>,
    source_metadata_sha256: BTreeMap<String, String>,
}

impl DeepSeekV4FullStreamReader {
    /// Admit a complete, sealed V4 source stream without turning it into a
    /// runtime.  This validates all descriptor/chunk mappings and stats every
    /// declared content-addressed chunk; chunk SHA-256 bytes are checked on
    /// verified reads or by [`Self::verify_all_chunks`].
    pub fn admit(root: impl AsRef<Path>) -> Result<Self> {
        let root = canonical_non_symlink_directory(root.as_ref(), "DeepSeek-V4 full artifact")?;
        let manifest_path = checked_regular_path(&root, "manifest.json", "full stream manifest")?;
        let manifest_raw = read_regular_file(&manifest_path, "full stream manifest")?;
        let manifest_file_sha256 = sha256_hex(&manifest_raw);
        let manifest_value = parse_and_verify_sealed_json(&manifest_raw, "full stream manifest")?;
        let manifest: Manifest = serde_json::from_value(manifest_value).map_err(|error| {
            Error::Gravity(format!("DeepSeek-V4 full manifest schema decode: {error}"))
        })?;
        validate_manifest_identity(&manifest)?;

        let restart_path = checked_regular_path(
            &root,
            &manifest.restart_receipt.path,
            "full stream restart receipt",
        )?;
        if manifest.restart_receipt.path != "restart-receipt.json" {
            return Err(gravity("full stream restart receipt path is not canonical"));
        }
        let restart_raw = read_regular_file(&restart_path, "full stream restart receipt")?;
        let restart_value =
            parse_and_verify_sealed_json(&restart_raw, "full stream restart receipt")?;
        validate_restart_receipt(&restart_value, &manifest.restart_receipt.seal_sha256, &root)?;

        let source_metadata_sha256 = validate_metadata_assets(&root, &manifest.source)?;
        let index = load_and_verify_source_index(&root, &manifest.source)?;
        if index.metadata.total_size != manifest.artifact.source_index_total_size_bytes {
            return Err(gravity(
                "full stream source index total_size differs from manifest artifact bytes",
            ));
        }

        let source_windows = validate_source_windows(&manifest.source)?;
        let (tensors, chunks) = validate_tensors(&manifest, &index, &source_windows)?;
        validate_chunk_tree(&root, &chunks)?;
        let native_pairs = validate_native_scale_pairs(&tensors)?;

        Ok(Self {
            root,
            source: DeepSeekV4SourceIdentity {
                repository: manifest.source.repository,
                revision: manifest.source.revision,
            },
            manifest_seal_sha256: manifest.seal_sha256,
            manifest_file_sha256,
            restart_seal_sha256: manifest.restart_receipt.seal_sha256,
            tensor_bytes: manifest.artifact.total_tensor_bytes,
            tensors,
            chunks,
            native_pairs,
            source_metadata_sha256,
        })
    }

    /// Canonical non-symlink artifact root.
    pub fn artifact_root(&self) -> &Path {
        &self.root
    }

    /// Pinned source identity validated at admission.
    pub fn source_identity(&self) -> &DeepSeekV4SourceIdentity {
        &self.source
    }

    /// Content seal of the exact manifest admitted by this reader.
    pub fn manifest_seal_sha256(&self) -> &str {
        &self.manifest_seal_sha256
    }

    /// Hash of the physical canonical manifest file (including its seal field).
    pub fn manifest_file_sha256(&self) -> &str {
        &self.manifest_file_sha256
    }

    /// Content seal of the bound restart receipt.
    pub fn restart_seal_sha256(&self) -> &str {
        &self.restart_seal_sha256
    }

    /// Number of named source tensors in the admitted stream.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Sum of all source-native physical tensor bytes.
    pub fn tensor_bytes(&self) -> u64 {
        self.tensor_bytes
    }

    /// Number of unique content-addressed chunk files required by the stream.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Number of exact native FP4/FP8 weight-to-scale pairs admitted.
    pub fn native_scale_pair_count(&self) -> usize {
        self.native_pairs.len()
    }

    /// Number of admitted native pairs for one representation.
    pub fn native_scale_pair_count_for(&self, kind: NativeScalePairKind) -> usize {
        self.native_pairs
            .values()
            .filter(|geometry| geometry.kind == kind)
            .count()
    }

    /// SHA-256 of a metadata asset that was read and verified at admission.
    /// This is useful for binding a future native adapter to the pinned source
    /// model/kernel/convert grammar without loading an executable runtime.
    pub fn source_metadata_asset_sha256(&self, key: &str) -> Result<&str> {
        self.source_metadata_sha256
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| {
                gravity(format!(
                    "full stream has no admitted source metadata asset {key:?}"
                ))
            })
    }

    /// Number of source metadata assets that were byte/hash checked during
    /// admission.
    pub fn source_metadata_asset_count(&self) -> usize {
        self.source_metadata_sha256.len()
    }

    /// Return one pinned metadata asset after re-checking its regular-file
    /// path, byte length, and SHA-256 binding.  This remains a bounded
    /// read-only reader operation: callers must state an allocation ceiling
    /// and it does not decode a tensor or construct a runtime.
    ///
    /// Metadata files are authenticated at [`Self::admit`] already.  The
    /// second check here makes a later source-algorithm oracle resilient to a
    /// mutation between admission and use, just as verified tensor ranges are.
    pub fn read_verified_metadata_asset(
        &self,
        key: &str,
        max_output_bytes: usize,
    ) -> Result<Vec<u8>> {
        let expected_sha256 = self.source_metadata_asset_sha256(key)?;
        let metadata_root =
            canonical_non_symlink_directory(&self.root.join("metadata"), "full stream metadata")?;
        let path = checked_regular_path(&metadata_root, key, "source metadata asset")?;
        let file = open_checked_regular_file(&path, "source metadata asset")?;
        let bytes = usize::try_from(file.metadata()?.len())
            .map_err(|_| gravity(format!("metadata asset {key:?} exceeds host usize")))?;
        if bytes > max_output_bytes {
            return Err(gravity(format!(
                "metadata asset {key:?} has {bytes} bytes, exceeding explicit read bound {max_output_bytes}",
            )));
        }
        drop(file);
        let raw = read_regular_file(&path, "source metadata asset")?;
        if raw.len() != bytes || sha256_hex(&raw) != expected_sha256 {
            return Err(gravity(format!(
                "metadata asset {key:?} bytes/hash differs from its admitted binding",
            )));
        }
        Ok(raw)
    }

    /// Metadata only; this cannot read or execute a tensor.
    pub fn tensor_metadata(&self, name: &str) -> Result<&DeepSeekV4TensorMetadata> {
        self.tensors
            .get(name)
            .ok_or_else(|| gravity(format!("full stream has no tensor {name:?}")))
    }

    /// Return an already validated native scale-pair contract for `weight_name`.
    pub fn native_scale_pair(&self, weight_name: &str) -> Result<NativeScalePair<'_>> {
        let geometry = self.native_pairs.get(weight_name).ok_or_else(|| {
            gravity(format!(
                "full stream has no native scale pair for {weight_name:?}"
            ))
        })?;
        let weight = self.tensor_metadata(weight_name)?;
        let scale = self.tensor_metadata(&geometry.scale_name)?;
        Ok(NativeScalePair {
            kind: geometry.kind,
            weight,
            scale,
            out_rows: geometry.out_rows,
            packed_k: geometry.packed_k,
            logical_k: geometry.logical_k,
            scale_rows: geometry.scale_rows,
            scale_cols: geometry.scale_cols,
        })
    }

    /// Hash every segment that forms `name` without accumulating its bytes.
    pub fn verify_tensor(&self, name: &str) -> Result<()> {
        let tensor = self.tensor_metadata(name)?;
        for segment in &tensor.segments {
            self.verify_segment(segment)?;
        }
        Ok(())
    }

    /// Verify every unique physical chunk in the admitted artifact.  This is a
    /// full local byte scan and intentionally does not imply a forward/runtime
    /// claim.
    pub fn verify_all_chunks(&self) -> Result<FullStreamChunkVerification> {
        let mut bytes_verified = 0u64;
        for binding in self.chunks.values() {
            self.verify_chunk(binding)?;
            bytes_verified = bytes_verified
                .checked_add(binding.bytes)
                .ok_or_else(|| gravity("full stream verification byte count overflow"))?;
        }
        Ok(FullStreamChunkVerification {
            chunk_count: self.chunks.len(),
            bytes_verified,
        })
    }

    /// Return a verified source-native byte range.  Every source chunk touched
    /// by the range is SHA-256 checked in full before this function succeeds.
    /// `max_output_bytes` is mandatory so a caller cannot accidentally turn a
    /// bounded adapter read into an unbounded host allocation.
    pub fn read_verified_range(
        &self,
        name: &str,
        range: Range<u64>,
        max_output_bytes: usize,
    ) -> Result<Vec<u8>> {
        let tensor = self.tensor_metadata(name)?;
        if range.start >= range.end || range.end > tensor.bytes {
            return Err(gravity(format!(
                "{name}: invalid verified range {}..{} for {} bytes",
                range.start, range.end, tensor.bytes
            )));
        }
        let requested = range.end - range.start;
        let requested_usize = usize::try_from(requested)
            .map_err(|_| gravity(format!("{name}: requested range cannot fit host usize")))?;
        if requested_usize > max_output_bytes {
            return Err(gravity(format!(
                "{name}: requested {requested_usize} bytes exceeds explicit read bound {max_output_bytes}"
            )));
        }

        let mut out = Vec::with_capacity(requested_usize);
        let mut cursor = range.start;
        for segment in &tensor.segments {
            if segment.tensor_end <= range.start || segment.tensor_start >= range.end {
                continue;
            }
            let take_start = range.start.max(segment.tensor_start);
            let take_end = range.end.min(segment.tensor_end);
            self.read_verified_segment_slice(segment, take_start, take_end, &mut out)?;
            cursor = take_end;
        }
        if cursor != range.end || out.len() != requested_usize {
            return Err(gravity(format!(
                "{name}: verified segment range is not contiguous"
            )));
        }
        Ok(out)
    }

    /// Read an entire tensor after verifying all of its source chunks.  The
    /// caller must supply an explicit allocation ceiling; use
    /// [`Self::read_verified_range`] for small component windows.
    pub fn read_verified_full(&self, name: &str, max_output_bytes: usize) -> Result<Vec<u8>> {
        let bytes = self.tensor_metadata(name)?.bytes;
        self.read_verified_range(name, 0..bytes, max_output_bytes)
    }

    fn verify_segment(&self, segment: &DeepSeekV4Segment) -> Result<()> {
        let binding = self.chunks.get(&segment.chunk_relpath).ok_or_else(|| {
            gravity(format!(
                "segment binding missing for {}",
                segment.chunk_relpath
            ))
        })?;
        self.verify_chunk(binding)
    }

    fn verify_chunk(&self, binding: &ChunkBinding) -> Result<()> {
        let path = checked_regular_path(&self.root, &binding.relative, "content-addressed chunk")?;
        let file = open_checked_regular_file(&path, "content-addressed chunk")?;
        let observed_size = file.metadata()?.len();
        if observed_size != binding.bytes {
            return Err(gravity(format!(
                "chunk {} byte size {observed_size} differs from sealed {}",
                binding.relative, binding.bytes
            )));
        }
        let observed = sha256_reader(file)?;
        if observed != binding.sha256 {
            return Err(gravity(format!(
                "chunk {} sha256 differs from sealed segment digest",
                binding.relative
            )));
        }
        Ok(())
    }

    fn read_verified_segment_slice(
        &self,
        segment: &DeepSeekV4Segment,
        take_start: u64,
        take_end: u64,
        out: &mut Vec<u8>,
    ) -> Result<()> {
        if take_start < segment.tensor_start
            || take_end > segment.tensor_end
            || take_start >= take_end
        {
            return Err(gravity(
                "internal verified segment slice escaped its segment",
            ));
        }
        let binding = self.chunks.get(&segment.chunk_relpath).ok_or_else(|| {
            gravity(format!(
                "segment binding missing for {}",
                segment.chunk_relpath
            ))
        })?;
        let path = checked_regular_path(&self.root, &binding.relative, "content-addressed chunk")?;
        let mut file = open_checked_regular_file(&path, "content-addressed chunk")?;
        let observed_size = file.metadata()?.len();
        if observed_size != binding.bytes {
            return Err(gravity(format!(
                "chunk {} byte size differs from sealed segment bytes",
                binding.relative
            )));
        }

        let wanted_start = take_start - segment.tensor_start;
        let wanted_end = take_end - segment.tensor_start;
        let mut digest = Sha256::new();
        let mut buffer = vec![0u8; READ_BUFFER_BYTES];
        let mut offset = 0u64;
        loop {
            let got = file.read(&mut buffer)?;
            if got == 0 {
                break;
            }
            let end = offset.checked_add(got as u64).ok_or_else(|| {
                gravity("chunk byte offset overflow while reading verified segment")
            })?;
            digest.update(&buffer[..got]);
            let overlap_start = offset.max(wanted_start);
            let overlap_end = end.min(wanted_end);
            if overlap_start < overlap_end {
                let local_start = usize::try_from(overlap_start - offset)
                    .map_err(|_| gravity("chunk slice start exceeds usize"))?;
                let local_end = usize::try_from(overlap_end - offset)
                    .map_err(|_| gravity("chunk slice end exceeds usize"))?;
                out.extend_from_slice(&buffer[local_start..local_end]);
            }
            offset = end;
        }
        if offset != binding.bytes {
            return Err(gravity(format!(
                "chunk {} ended at {offset}, expected {} bytes",
                binding.relative, binding.bytes
            )));
        }
        let observed = format!("{:x}", digest.finalize());
        if observed != binding.sha256 {
            return Err(gravity(format!(
                "chunk {} sha256 differs from sealed segment digest",
                binding.relative
            )));
        }
        Ok(())
    }
}

fn validate_manifest_identity(manifest: &Manifest) -> Result<()> {
    if manifest.schema != FULL_STREAM_SCHEMA || manifest.status != FULL_STREAM_STATUS {
        return Err(gravity(
            "artifact is not the sealed DeepSeek-V4 full stream runtime-pending state",
        ));
    }
    if !is_sha256(&manifest.seal_sha256) {
        return Err(gravity(
            "full stream manifest seal is not lowercase SHA-256",
        ));
    }
    if manifest.source.repository != PINNED_REPOSITORY
        || manifest.source.revision != PINNED_REVISION
    {
        return Err(gravity(
            "full stream source identity is not the pinned DeepSeek-V4-Flash revision",
        ));
    }
    if manifest.source.source_parent_persisted || manifest.storage.source_parent_retained {
        return Err(gravity(
            "full stream violates the source-parent-evicted storage contract",
        ));
    }
    if manifest.artifact.kind != FULL_STREAM_KIND
        || manifest.artifact.format != CONTENT_ADDRESSED_FORMAT
    {
        return Err(gravity(
            "full stream has an unexpected artifact representation",
        ));
    }
    if manifest.artifact.total_tensor_bytes == 0
        || manifest.artifact.source_index_total_size_bytes != manifest.artifact.total_tensor_bytes
        || manifest.source.source_shard_count != EXPECTED_SOURCE_SHARDS
    {
        return Err(gravity(
            "full stream has invalid total byte or source shard accounting",
        ));
    }
    if manifest.full_model_scope.tensor_count != EXPECTED_TENSOR_COUNT
        || manifest.architecture.tensor_count != EXPECTED_TENSOR_COUNT
        || manifest.tensors.len() != EXPECTED_TENSOR_COUNT
        || manifest.full_model_scope.num_hidden_layers != 43
        || manifest.architecture.layer_count != 43
        || !manifest.full_model_scope.full_width
        || !manifest.full_model_scope.contains_embedding_and_head
        || manifest
            .full_model_scope
            .contains_all_routed_experts_per_layer
            != 256
        || manifest.full_model_scope.runtime_ready
        || manifest.architecture.model_type != "deepseek_v4"
    {
        return Err(gravity(
            "full stream architecture/scope is not the pinned 43-layer body",
        ));
    }
    if manifest.runtime_adapter.id.is_some()
        || manifest.runtime_adapter.registration.is_some()
        || manifest.runtime_adapter.device.is_some()
        || manifest.runtime_adapter.metal_dispatches != 0
    {
        return Err(gravity(
            "full stream reader refuses an artifact claiming a registered runtime",
        ));
    }
    if manifest.restart_receipt.path != "restart-receipt.json"
        || !is_sha256(&manifest.restart_receipt.seal_sha256)
    {
        return Err(gravity("full stream restart receipt binding is malformed"));
    }
    Ok(())
}

fn validate_restart_receipt(value: &Value, expected_seal: &str, root: &Path) -> Result<()> {
    let object = value
        .as_object()
        .ok_or_else(|| gravity("full stream restart receipt root is not an object"))?;
    let actual_seal = object
        .get("seal_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| gravity("full stream restart receipt has no seal"))?;
    if actual_seal != expected_seal {
        return Err(gravity(
            "full stream restart receipt seal differs from manifest binding",
        ));
    }
    if object.get("schema").and_then(Value::as_str)
        != Some("hawking.gravity.deepseek_v4.full_restart_receipt.v1")
        || object.get("status").and_then(Value::as_str) != Some("SEALED")
        || object
            .get("source_parent_retained")
            .and_then(Value::as_bool)
            != Some(false)
    {
        return Err(gravity(
            "full stream restart receipt has invalid identity/status",
        ));
    }
    let journal_sha = object
        .get("journal_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| gravity("full stream restart receipt lacks journal hash"))?;
    let ranges_sha = object
        .get("range_journal_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| gravity("full stream restart receipt lacks range journal hash"))?;
    if !is_sha256(journal_sha) || !is_sha256(ranges_sha) {
        return Err(gravity(
            "full stream restart receipt journal hash is malformed",
        ));
    }
    let journal = checked_regular_path(root, "stream-journal.json", "full stream journal")?;
    let ranges = checked_regular_path(root, "stream-ranges.jsonl", "full stream range journal")?;
    if sha256_hex(&read_regular_file(&journal, "full stream journal")?) != journal_sha
        || sha256_hex(&read_regular_file(&ranges, "full stream range journal")?) != ranges_sha
    {
        return Err(gravity(
            "full stream restart receipt journal binding differs",
        ));
    }
    Ok(())
}

fn load_and_verify_source_index(root: &Path, source: &SourceSection) -> Result<SourceIndex> {
    let asset = source
        .metadata_assets
        .get("model.safetensors.index.json")
        .ok_or_else(|| gravity("full stream lacks a source tensor index asset"))?;
    if asset.path != "model.safetensors.index.json" || !is_sha256(&asset.sha256) {
        return Err(gravity(
            "full stream source tensor index binding is malformed",
        ));
    }
    // The metadata root is handled separately so a symlink cannot redirect
    // the source index.
    let metadata_root =
        canonical_non_symlink_directory(&root.join("metadata"), "full stream metadata")?;
    let index_path = checked_regular_path(&metadata_root, &asset.path, "source tensor index")?;
    let raw = read_regular_file(&index_path, "source tensor index")?;
    if raw.len() as u64 != asset.bytes || sha256_hex(&raw) != asset.sha256 {
        return Err(gravity(
            "full stream source tensor index bytes/hash differs from manifest",
        ));
    }
    serde_json::from_slice(&raw)
        .map_err(|error| gravity(format!("full stream source tensor index JSON: {error}")))
}

fn validate_metadata_assets(
    root: &Path,
    source: &SourceSection,
) -> Result<BTreeMap<String, String>> {
    const REQUIRED: [&str; 8] = [
        "config.json",
        "inference/config.json",
        "inference/convert.py",
        "inference/kernel.py",
        "inference/model.py",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ];
    if source.metadata_assets.len() != REQUIRED.len()
        || REQUIRED
            .iter()
            .any(|key| !source.metadata_assets.contains_key(*key))
    {
        return Err(gravity(
            "full stream source metadata asset set is not the pinned V4 set",
        ));
    }
    let metadata_root =
        canonical_non_symlink_directory(&root.join("metadata"), "full stream metadata")?;
    let mut admitted = BTreeMap::new();
    for key in REQUIRED {
        let asset = source
            .metadata_assets
            .get(key)
            .expect("required metadata asset presence was checked above");
        if asset.path != key || !is_sha256(&asset.sha256) {
            return Err(gravity(format!(
                "full stream metadata asset binding is malformed for {key:?}"
            )));
        }
        let path = checked_regular_path(&metadata_root, &asset.path, "source metadata asset")?;
        let raw = read_regular_file(&path, "source metadata asset")?;
        if raw.len() as u64 != asset.bytes || sha256_hex(&raw) != asset.sha256 {
            return Err(gravity(format!(
                "full stream metadata asset bytes/hash differs for {key:?}"
            )));
        }
        admitted.insert(key.to_owned(), asset.sha256.clone());
    }
    Ok(admitted)
}

fn validate_source_windows(source: &SourceSection) -> Result<BTreeMap<String, SourceWindow>> {
    if source.source_windows.len() != EXPECTED_SOURCE_SHARDS {
        return Err(gravity(
            "full stream does not contain exactly 46 source windows",
        ));
    }
    let mut windows = BTreeMap::new();
    for window in &source.source_windows {
        let item = &window.source;
        if item.repository != PINNED_REPOSITORY
            || item.revision != PINNED_REVISION
            || item.commit_hash != PINNED_REVISION
            || item.file_size_bytes <= window.header_bytes
            || !is_sha256(&window.streamed_full_file_sha256)
            || !is_sha256(&item.etag_sha256)
            || !is_sha256(&item.xet_file_hash)
            || window.streamed_full_file_sha256 != item.etag_sha256
        {
            return Err(gravity(format!(
                "source window {:?} does not bind the pinned source identity",
                item.shard
            )));
        }
        if !is_expected_shard_name(&item.shard) {
            return Err(gravity(format!(
                "unexpected full stream source shard {:?}",
                item.shard
            )));
        }
        if windows.insert(item.shard.clone(), window.clone()).is_some() {
            return Err(gravity(format!("duplicate source window {:?}", item.shard)));
        }
    }
    if windows.len() != EXPECTED_SOURCE_SHARDS
        || (1..=EXPECTED_SOURCE_SHARDS)
            .any(|index| !windows.contains_key(&format!("model-{index:05}-of-00046.safetensors")))
    {
        return Err(gravity("full stream source window set is incomplete"));
    }
    Ok(windows)
}

fn validate_tensors(
    manifest: &Manifest,
    index: &SourceIndex,
    source_windows: &BTreeMap<String, SourceWindow>,
) -> Result<(
    BTreeMap<String, DeepSeekV4TensorMetadata>,
    BTreeMap<String, ChunkBinding>,
)> {
    if index.weight_map.len() != EXPECTED_TENSOR_COUNT
        || index.weight_map.len() != manifest.tensors.len()
    {
        return Err(gravity(
            "source tensor index does not contain the complete tensor set",
        ));
    }
    let mut tensors = BTreeMap::new();
    let mut chunks = BTreeMap::new();
    let mut total_bytes = 0u64;
    for (name, raw) in &manifest.tensors {
        let index_shard = index.weight_map.get(name).ok_or_else(|| {
            gravity(format!(
                "source tensor index lacks manifest tensor {name:?}"
            ))
        })?;
        if raw.name != *name || raw.source_shard != *index_shard {
            return Err(gravity(format!(
                "{name}: manifest tensor identity differs from source tensor index"
            )));
        }
        let window = source_windows.get(index_shard).ok_or_else(|| {
            gravity(format!(
                "{name}: source tensor index references an unsealed shard"
            ))
        })?;
        let metadata = validate_tensor_descriptor(name, raw, window)?;
        total_bytes = total_bytes
            .checked_add(metadata.bytes)
            .ok_or_else(|| gravity("full stream tensor byte sum overflow"))?;
        for segment in &metadata.segments {
            let expected = ChunkBinding {
                relative: segment.chunk_relpath.clone(),
                sha256: segment.sha256.clone(),
                bytes: segment.bytes,
            };
            if let Some(previous) = chunks.insert(expected.relative.clone(), expected.clone()) {
                if previous != expected {
                    return Err(gravity(format!(
                        "content-addressed chunk {} has conflicting manifest bindings",
                        segment.chunk_relpath
                    )));
                }
            }
        }
        tensors.insert(name.clone(), metadata);
    }
    if index
        .weight_map
        .keys()
        .any(|name| !tensors.contains_key(name))
    {
        return Err(gravity(
            "full stream manifest is missing source-index tensor entries",
        ));
    }
    if total_bytes != manifest.artifact.total_tensor_bytes
        || chunks.len() != manifest.artifact.content_addressed_chunk_count
    {
        return Err(gravity(
            "full stream tensor/chunk accounting differs from manifest",
        ));
    }
    let chunk_digests: Vec<String> = chunks.values().map(|chunk| chunk.sha256.clone()).collect();
    if sha256_hex(&canonical_json_array_strings(&chunk_digests))
        != manifest.artifact.content_addressed_chunk_sha256
    {
        return Err(gravity(
            "full stream content-addressed chunk digest list differs",
        ));
    }
    Ok((tensors, chunks))
}

fn validate_tensor_descriptor(
    name: &str,
    raw: &RawTensorMetadata,
    source_window: &SourceWindow,
) -> Result<DeepSeekV4TensorMetadata> {
    if raw.shape.is_empty() || raw.shape.iter().any(|&dimension| dimension == 0) {
        return Err(gravity(format!(
            "{name}: tensor shape must be non-empty and positive"
        )));
    }
    let item_width = dtype_bytes(&raw.dtype)?;
    let elements = raw.shape.iter().try_fold(1u64, |product, &dimension| {
        product
            .checked_mul(dimension)
            .ok_or_else(|| gravity(format!("{name}: tensor shape element count overflow")))
    })?;
    let expected_bytes = elements
        .checked_mul(item_width)
        .ok_or_else(|| gravity(format!("{name}: tensor byte count overflow")))?;
    if raw.bytes != expected_bytes || raw.data_offsets.len() != 2 {
        return Err(gravity(format!("{name}: invalid tensor byte geometry")));
    }
    let offsets = [raw.data_offsets[0], raw.data_offsets[1]];
    if offsets[0] >= offsets[1] || offsets[1] - offsets[0] != raw.bytes {
        return Err(gravity(format!("{name}: invalid source data offsets")));
    }
    let expected_source_start = source_window
        .header_bytes
        .checked_add(offsets[0])
        .ok_or_else(|| gravity(format!("{name}: source offset overflow")))?;
    let expected_source_end = source_window
        .header_bytes
        .checked_add(offsets[1])
        .ok_or_else(|| gravity(format!("{name}: source offset overflow")))?;
    if raw.source_file_start != expected_source_start
        || raw.source_file_end != expected_source_end
        || raw.source_file_end > source_window.source.file_size_bytes
    {
        return Err(gravity(format!(
            "{name}: descriptor source range does not bind its source window"
        )));
    }
    let row_count = if raw.shape.len() == 1 {
        1
    } else {
        raw.shape[0]
    };
    if raw.bytes % row_count != 0 || raw.segments.is_empty() {
        return Err(gravity(format!(
            "{name}: tensor does not have row-aligned segments"
        )));
    }
    let row_bytes = raw.bytes / row_count;
    let mut byte_cursor = 0u64;
    let mut row_cursor = 0u64;
    let mut segments = Vec::with_capacity(raw.segments.len());
    for segment in &raw.segments {
        validate_chunk_relative_path(&segment.chunk_relpath, &segment.sha256)?;
        if !is_sha256(&segment.sha256)
            || segment.bytes == 0
            || segment.tensor_start != byte_cursor
            || segment.tensor_end <= segment.tensor_start
            || segment.tensor_end - segment.tensor_start != segment.bytes
            || segment.source_file_end <= segment.source_file_start
            || segment.source_file_end - segment.source_file_start != segment.bytes
            || segment.source_file_start
                != raw
                    .source_file_start
                    .checked_add(segment.tensor_start)
                    .ok_or_else(|| gravity(format!("{name}: segment source offset overflow")))?
            || segment.source_file_end
                != raw
                    .source_file_start
                    .checked_add(segment.tensor_end)
                    .ok_or_else(|| gravity(format!("{name}: segment source offset overflow")))?
            || segment.row_start != row_cursor
            || segment.row_count == 0
            || segment.row_count > row_count - row_cursor
            || segment.bytes
                != row_bytes
                    .checked_mul(segment.row_count)
                    .ok_or_else(|| gravity(format!("{name}: segment row byte count overflow")))?
        {
            return Err(gravity(format!(
                "{name}: non-contiguous or malformed content-addressed segment"
            )));
        }
        byte_cursor = segment.tensor_end;
        row_cursor = row_cursor
            .checked_add(segment.row_count)
            .ok_or_else(|| gravity(format!("{name}: segment row count overflow")))?;
        segments.push(DeepSeekV4Segment {
            bytes: segment.bytes,
            chunk_relpath: segment.chunk_relpath.clone(),
            sha256: segment.sha256.clone(),
            source_file_start: segment.source_file_start,
            source_file_end: segment.source_file_end,
            tensor_start: segment.tensor_start,
            tensor_end: segment.tensor_end,
            row_start: segment.row_start,
            row_count: segment.row_count,
        });
    }
    if byte_cursor != raw.bytes || row_cursor != row_count {
        return Err(gravity(format!(
            "{name}: content-addressed segments do not cover the full tensor"
        )));
    }
    Ok(DeepSeekV4TensorMetadata {
        name: raw.name.clone(),
        dtype: raw.dtype.clone(),
        shape: raw.shape.clone(),
        data_offsets: offsets,
        bytes: raw.bytes,
        source_file_start: raw.source_file_start,
        source_file_end: raw.source_file_end,
        source_shard: raw.source_shard.clone(),
        segments,
    })
}

fn validate_native_scale_pairs(
    tensors: &BTreeMap<String, DeepSeekV4TensorMetadata>,
) -> Result<BTreeMap<String, NativeScalePairGeometry>> {
    let mut pairs = BTreeMap::new();
    let mut paired_scales = BTreeSet::new();
    for (name, weight) in tensors {
        let kind = match weight.dtype.as_str() {
            "I8" => NativeScalePairKind::Fp4E2M1fnX2,
            "F8_E4M3" => NativeScalePairKind::Fp8E4M3fn,
            _ => continue,
        };
        let stem = name.strip_suffix(".weight").ok_or_else(|| {
            gravity(format!(
                "{name}: native {}/weight name is not canonical",
                kind.as_str()
            ))
        })?;
        let scale_name = format!("{stem}.scale");
        let scale = tensors
            .get(&scale_name)
            .ok_or_else(|| gravity(format!("{name}: missing matching native E8M0 scale tensor")))?;
        if scale.dtype != "F8_E8M0" || weight.shape.len() != 2 || scale.shape.len() != 2 {
            return Err(gravity(format!(
                "{name}: native weight/scale pair is not rank-two E8M0"
            )));
        }
        let out_rows = weight.shape[0];
        let packed_k = weight.shape[1];
        let (logical_k, expected_scale_rows, expected_scale_cols) = match kind {
            NativeScalePairKind::Fp4E2M1fnX2 => {
                let logical_k = packed_k
                    .checked_mul(2)
                    .ok_or_else(|| gravity(format!("{name}: packed FP4 logical K overflow")))?;
                if logical_k % FP4_LOGICAL_BLOCK != 0 {
                    return Err(gravity(format!(
                        "{name}: FP4 logical K is not divisible by {FP4_LOGICAL_BLOCK}"
                    )));
                }
                (logical_k, out_rows, logical_k / FP4_LOGICAL_BLOCK)
            }
            NativeScalePairKind::Fp8E4M3fn => {
                if out_rows % FP8_BLOCK != 0 || packed_k % FP8_BLOCK != 0 {
                    return Err(gravity(format!(
                        "{name}: FP8 dimensions must be divisible by {FP8_BLOCK}"
                    )));
                }
                (packed_k, out_rows / FP8_BLOCK, packed_k / FP8_BLOCK)
            }
        };
        if scale.shape != [expected_scale_rows, expected_scale_cols]
            || scale.bytes
                != expected_scale_rows
                    .checked_mul(expected_scale_cols)
                    .ok_or_else(|| gravity(format!("{name}: native scale byte count overflow")))?
        {
            return Err(gravity(format!(
                "{name}: native scale shape is not exact for {}",
                kind.as_str()
            )));
        }
        if pairs
            .insert(
                name.clone(),
                NativeScalePairGeometry {
                    kind,
                    scale_name: scale_name.clone(),
                    out_rows,
                    packed_k,
                    logical_k,
                    scale_rows: expected_scale_rows,
                    scale_cols: expected_scale_cols,
                },
            )
            .is_some()
        {
            return Err(gravity(format!("duplicate native pair {name}")));
        }
        paired_scales.insert(scale_name);
    }
    for (name, scale) in tensors {
        if scale.dtype == "F8_E8M0" && !paired_scales.contains(name) {
            return Err(gravity(format!(
                "{name}: orphan E8M0 scale is not bound to a native weight"
            )));
        }
    }
    if pairs.is_empty() {
        return Err(gravity("full stream has no native FP4/FP8 scale pairs"));
    }
    Ok(pairs)
}

fn validate_chunk_tree(root: &Path, chunks: &BTreeMap<String, ChunkBinding>) -> Result<()> {
    let chunks_root =
        canonical_non_symlink_directory(&root.join("chunks"), "content-addressed chunk root")?;
    let mut discovered = BTreeSet::new();
    for prefix in fs::read_dir(&chunks_root)? {
        let prefix = prefix?;
        let prefix_path = prefix.path();
        let prefix_meta = fs::symlink_metadata(&prefix_path)?;
        if prefix_meta.file_type().is_symlink() || !prefix_meta.is_dir() {
            return Err(gravity(format!(
                "content-addressed chunk prefix must be a non-symlink directory: {}",
                prefix_path.display()
            )));
        }
        let prefix_name = prefix.file_name();
        let prefix_name = prefix_name.to_string_lossy();
        if prefix_name.len() != 2 || !prefix_name.bytes().all(is_lower_hex_byte) {
            return Err(gravity(format!(
                "content-addressed chunk prefix is not lowercase hex: {prefix_name}"
            )));
        }
        for entry in fs::read_dir(&prefix_path)? {
            let entry = entry?;
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path)?;
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                return Err(gravity(format!(
                    "content-addressed chunk must be a regular non-symlink file: {}",
                    path.display()
                )));
            }
            let file_name = entry.file_name();
            let file_name = file_name.to_string_lossy();
            if !is_sha256(&file_name) || !file_name.starts_with(prefix_name.as_ref()) {
                return Err(gravity(format!(
                    "content-addressed chunk name is not its lowercase SHA-256: {file_name}"
                )));
            }
            let relative = format!("chunks/{prefix_name}/{file_name}");
            let binding = chunks.get(&relative).ok_or_else(|| {
                gravity(format!("unreferenced content-addressed chunk {relative}"))
            })?;
            if binding.sha256 != file_name || binding.bytes != metadata.len() {
                return Err(gravity(format!(
                    "content-addressed chunk {relative} differs from sealed binding"
                )));
            }
            discovered.insert(relative);
        }
    }
    if discovered.len() != chunks.len() || chunks.keys().any(|path| !discovered.contains(path)) {
        return Err(gravity(
            "content-addressed chunk tree does not exactly match manifest mappings",
        ));
    }
    Ok(())
}

fn checked_regular_path(root: &Path, relative: &str, label: &str) -> Result<PathBuf> {
    let relative_path = Path::new(relative);
    if relative_path.is_absolute()
        || relative_path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(gravity(format!(
            "{label} path escapes artifact: {relative}"
        )));
    }
    let path = root.join(relative_path);
    let metadata = fs::symlink_metadata(&path).map_err(|error| {
        gravity(format!(
            "cannot inspect {label} {}: {error}",
            path.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(gravity(format!(
            "{label} must be a regular non-symlink file: {}",
            path.display()
        )));
    }
    let canonical = fs::canonicalize(&path).map_err(|error| {
        gravity(format!(
            "cannot canonicalize {label} {}: {error}",
            path.display()
        ))
    })?;
    if !canonical.starts_with(root) {
        return Err(gravity(format!(
            "{label} path escapes artifact: {relative}"
        )));
    }
    Ok(path)
}

fn canonical_non_symlink_directory(path: &Path, label: &str) -> Result<PathBuf> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        gravity(format!(
            "cannot inspect {label} {}: {error}",
            path.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(gravity(format!(
            "{label} must be a non-symlink directory: {}",
            path.display()
        )));
    }
    fs::canonicalize(path).map_err(|error| {
        gravity(format!(
            "cannot canonicalize {label} {}: {error}",
            path.display()
        ))
    })
}

fn open_checked_regular_file(path: &Path, label: &str) -> Result<File> {
    let before = fs::symlink_metadata(path).map_err(|error| {
        gravity(format!(
            "cannot inspect {label} {}: {error}",
            path.display()
        ))
    })?;
    if before.file_type().is_symlink() || !before.is_file() {
        return Err(gravity(format!(
            "{label} must be a regular non-symlink file: {}",
            path.display()
        )));
    }
    let file = File::open(path)?;
    let after = file.metadata()?;
    if !after.is_file() || after.len() != before.len() {
        return Err(gravity(format!(
            "{label} changed while opening: {}",
            path.display()
        )));
    }
    Ok(file)
}

fn read_regular_file(path: &Path, label: &str) -> Result<Vec<u8>> {
    let mut file = open_checked_regular_file(path, label)?;
    let bytes = file.metadata()?.len();
    let capacity = usize::try_from(bytes)
        .map_err(|_| gravity(format!("{label} is too large for host allocation")))?;
    let mut raw = Vec::with_capacity(capacity);
    file.read_to_end(&mut raw)?;
    if raw.len() as u64 != bytes {
        return Err(gravity(format!("{label} changed while being read")));
    }
    Ok(raw)
}

fn sha256_reader(mut reader: File) -> Result<String> {
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; READ_BUFFER_BYTES];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn parse_and_verify_sealed_json(raw: &[u8], label: &str) -> Result<Value> {
    let mut value: Value = serde_json::from_slice(raw)
        .map_err(|error| gravity(format!("{label} is not valid JSON: {error}")))?;
    let recorded = {
        let object = value
            .as_object_mut()
            .ok_or_else(|| gravity(format!("{label} root must be a JSON object")))?;
        object
            .remove("seal_sha256")
            .and_then(|item| item.as_str().map(str::to_owned))
            .ok_or_else(|| gravity(format!("{label} lacks a string seal_sha256")))?
    };
    if !is_sha256(&recorded) {
        return Err(gravity(format!(
            "{label} seal_sha256 is not lowercase SHA-256"
        )));
    }
    let observed = sha256_hex(&canonical_json(&value));
    if observed != recorded {
        return Err(gravity(format!(
            "{label} seal mismatch: recorded={recorded} observed={observed}"
        )));
    }
    value
        .as_object_mut()
        .expect("object was checked above")
        .insert("seal_sha256".to_owned(), Value::String(recorded));
    Ok(value)
}

/// Python's `json.dumps(sort_keys=True, separators=(",", ":"),
/// ensure_ascii=False)` layout used by the stream sealer.  This stays local
/// rather than widening the legacy Gravity container API.
fn canonical_json(value: &Value) -> Vec<u8> {
    let mut out = Vec::with_capacity(256);
    write_canonical_json(&mut out, value);
    out
}

fn write_canonical_json(out: &mut Vec<u8>, value: &Value) {
    match value {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(true) => out.extend_from_slice(b"true"),
        Value::Bool(false) => out.extend_from_slice(b"false"),
        Value::Number(number) => out.extend_from_slice(number.to_string().as_bytes()),
        Value::String(string) => out.extend_from_slice(
            serde_json::to_string(string)
                .expect("JSON string serialization is infallible")
                .as_bytes(),
        ),
        Value::Array(values) => {
            out.push(b'[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    out.push(b',');
                }
                write_canonical_json(out, value);
            }
            out.push(b']');
        }
        Value::Object(object) => {
            out.push(b'{');
            let mut keys: Vec<&String> = object.keys().collect();
            keys.sort();
            for (index, key) in keys.into_iter().enumerate() {
                if index != 0 {
                    out.push(b',');
                }
                out.extend_from_slice(
                    serde_json::to_string(key)
                        .expect("JSON string serialization is infallible")
                        .as_bytes(),
                );
                out.push(b':');
                write_canonical_json(out, &object[key]);
            }
            out.push(b'}');
        }
    }
}

fn canonical_json_array_strings(values: &[String]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len().saturating_mul(67).saturating_add(2));
    out.push(b'[');
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            out.push(b',');
        }
        out.extend_from_slice(
            serde_json::to_string(value)
                .expect("JSON string serialization is infallible")
                .as_bytes(),
        );
    }
    out.push(b']');
    out
}

fn validate_chunk_relative_path(relative: &str, sha256: &str) -> Result<()> {
    if !is_sha256(sha256) || relative != format!("chunks/{}/{}", &sha256[..2], sha256) {
        return Err(gravity(format!(
            "content-addressed segment path/digest is malformed: {relative}"
        )));
    }
    Ok(())
}

fn is_expected_shard_name(shard: &str) -> bool {
    (1..=EXPECTED_SOURCE_SHARDS)
        .map(|index| format!("model-{index:05}-of-00046.safetensors"))
        .any(|expected| shard == expected)
}

fn dtype_bytes(dtype: &str) -> Result<u64> {
    match dtype {
        "BF16" => Ok(2),
        "F32" => Ok(4),
        "I64" => Ok(8),
        "I8" | "F8_E4M3" | "F8_E8M0" => Ok(1),
        _ => Err(gravity(format!(
            "unsupported DeepSeek-V4 source dtype {dtype:?}"
        ))),
    }
}

fn is_lower_hex_byte(byte: u8) -> bool {
    byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(is_lower_hex_byte)
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn gravity(message: impl Into<String>) -> Error {
    Error::Gravity(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata(name: &str, dtype: &str, shape: &[u64], bytes: u64) -> DeepSeekV4TensorMetadata {
        DeepSeekV4TensorMetadata {
            name: name.to_owned(),
            dtype: dtype.to_owned(),
            shape: shape.to_vec(),
            data_offsets: [0, bytes],
            bytes,
            source_file_start: 0,
            source_file_end: bytes,
            source_shard: "model-00001-of-00046.safetensors".to_owned(),
            segments: Vec::new(),
        }
    }

    #[test]
    fn native_fp8_and_fp4_pair_geometry_is_exact() {
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "fp8.weight".to_owned(),
            metadata("fp8.weight", "F8_E4M3", &[1024, 4096], 1024 * 4096),
        );
        tensors.insert(
            "fp8.scale".to_owned(),
            metadata("fp8.scale", "F8_E8M0", &[8, 32], 8 * 32),
        );
        tensors.insert(
            "fp4.weight".to_owned(),
            metadata("fp4.weight", "I8", &[2048, 2048], 2048 * 2048),
        );
        tensors.insert(
            "fp4.scale".to_owned(),
            metadata("fp4.scale", "F8_E8M0", &[2048, 128], 2048 * 128),
        );

        let pairs = validate_native_scale_pairs(&tensors).expect("exact pair geometry");
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs["fp8.weight"].logical_k, 4096);
        assert_eq!(pairs["fp8.weight"].scale_rows, 8);
        assert_eq!(pairs["fp4.weight"].logical_k, 4096);
        assert_eq!(pairs["fp4.weight"].scale_cols, 128);
    }

    #[test]
    fn native_pair_rejects_wrong_scale_geometry_and_orphans() {
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "fp8.weight".to_owned(),
            metadata("fp8.weight", "F8_E4M3", &[128, 128], 128 * 128),
        );
        tensors.insert(
            "fp8.scale".to_owned(),
            metadata("fp8.scale", "F8_E8M0", &[1, 2], 2),
        );
        assert!(validate_native_scale_pairs(&tensors).is_err());

        tensors.insert(
            "fp8.scale".to_owned(),
            metadata("fp8.scale", "F8_E8M0", &[1, 1], 1),
        );
        tensors.insert(
            "orphan.scale".to_owned(),
            metadata("orphan.scale", "F8_E8M0", &[1, 1], 1),
        );
        assert!(validate_native_scale_pairs(&tensors).is_err());
    }

    #[test]
    fn canonical_json_sorts_keys_and_preserves_compact_layout() {
        let value: Value = serde_json::json!({"z": [true, null], "a": "é"});
        assert_eq!(
            canonical_json(&value),
            "{\"a\":\"é\",\"z\":[true,null]}".as_bytes().to_vec()
        );
    }

    #[test]
    fn content_addressed_paths_are_not_traversal_paths() {
        let digest = "a".repeat(64);
        assert!(validate_chunk_relative_path(&format!("chunks/aa/{digest}"), &digest).is_ok());
        assert!(validate_chunk_relative_path("chunks/aa/../bad", &digest).is_err());
    }
}
