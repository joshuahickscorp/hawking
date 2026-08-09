#![allow(dead_code)] // The static CLI deliberately does not execute the future reader surface.

//! CPU-only Qwen30 layer-streamed source-BF16 oracle reader/semantic contract.
//!
//! This target exists to close a very specific feasibility gap: a future
//! source oracle may stream verified BF16 tensor ranges from the sixteen
//! source safetensors shards, but it must never map, decode, or retain a full
//! shard (or the full model) just because a layer needs a weight row.
//!
//! The executable itself is deliberately static.  It opens no source payload,
//! model, GPU, Metal context, server, HCLI endpoint, or lease.  Its tests make
//! tiny synthetic safetensors fixtures and exercise the reader below.  A
//! future, separately leased source execution must bind the emitted contract
//! to the real source index/config/token trace and provide fresh evidence for
//! every field named in `future_exact_semantics_attestation`.
//!
//! Therefore this is neither a source comparison nor a coherence, HCLI, TG,
//! TPS, capability, or promotion result.

use half::bf16;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::path::{Component, Path, PathBuf};
use std::process;

const RESULT_SCHEMA: &str =
    "hawking.ascension.qwen30_layer_streamed_source_bf16_range_reader_semantic_contract.v1";
const RESULT_STATUS: &str =
    "PREPARED_QWEN30_LAYER_STREAMED_SOURCE_BF16_RANGE_READER_SEMANTIC_CONTRACT_NOT_EXECUTED";
const FUTURE_ATTESTATION_SCHEMA: &str =
    "hawking.ascension.qwen30_layer_streamed_source_bf16_exact_semantics_attestation.v1";
const FUTURE_ATTESTATION_STATUS: &str =
    "EARNED_QWEN30_LAYER_STREAMED_SOURCE_BF16_EXACT_SEMANTICS_ATTESTED";
const RANGE_MAP_SCHEMA: &str = "hawking.ascension.qwen30_source_bf16_range_map.v1";
const SOURCE_MODEL_ID: &str = "Qwen3-Coder-30B-A3B-Instruct";

const PREFIX_TOKEN_COUNT: usize = 369;
const FORCED_TOKEN_ID: u32 = 949;
const TRACE_FORWARD_COUNT: usize = PREFIX_TOKEN_COUNT + 1;
const LAYER_COUNT: usize = 48;
const SOURCE_TENSOR_COUNT: usize = 18_867;
const SHARD_COUNT: usize = 16;
const VOCAB_ROWS: usize = 151_936;
const HIDDEN_SIZE: usize = 2_048;
const ATTENTION_HEADS: usize = 32;
const KEY_VALUE_HEADS: usize = 4;
const HEAD_DIM: usize = 128;
const EXPERT_COUNT: usize = 128;
const TOP_K: usize = 8;
const MOE_INTERMEDIATE: usize = 768;

const BF16_BYTES: usize = std::mem::size_of::<u16>();
/// Payload reads, whole-shard checksum scans, and retained source bytes are
/// all bounded by this one fixed window.  The safetensors JSON header and the
/// independent source index are metadata, have independent caps, and are
/// dropped after structural validation.
const MAX_SOURCE_WINDOW_BYTES: usize = 1024 * 1024;
const MAX_RANGE_MAP_BYTES: usize = 32 * 1024 * 1024;
const MAX_SOURCE_INDEX_BYTES: usize = 32 * 1024 * 1024;
const MAX_SAFETENSORS_HEADER_BYTES: usize = 32 * 1024 * 1024;

#[derive(Debug)]
struct Args {
    out: Option<PathBuf>,
}

fn usage() -> &'static str {
    "usage: ascension_qwen30_streamed_bf16_oracle_preflight [--out NEW_ABSOLUTE_JSON]"
}

fn parse_args_from<I>(arguments: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut out = None;
    let mut values = arguments.into_iter();
    while let Some(flag) = values.next() {
        match flag.as_str() {
            "--out" => {
                let path = values
                    .next()
                    .ok_or_else(|| format!("missing value for --out; {}", usage()))?;
                if out.replace(PathBuf::from(path)).is_some() {
                    return Err(format!("--out was supplied more than once; {}", usage()));
                }
            }
            "--help" | "-h" => return Err(usage().into()),
            _ => return Err(format!("unsupported option {flag:?}; {}", usage())),
        }
    }
    if let Some(path) = &out {
        if !path.is_absolute() {
            return Err("--out must be absolute".into());
        }
        if !path.parent().is_some_and(Path::is_dir) {
            return Err("--out parent must already exist".into());
        }
    }
    Ok(Args { out })
}

fn parse_args() -> Result<Args, String> {
    parse_args_from(env::args().skip(1))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn checked_relative_path(value: &str, label: &str) -> Result<PathBuf, String> {
    let path = Path::new(value);
    if value.is_empty() || path.is_absolute() {
        return Err(format!("{label} must be a non-empty relative path"));
    }
    if path
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!(
            "{label} must not contain parent/current/root components"
        ));
    }
    Ok(path.to_path_buf())
}

fn regular_non_symlink_file(path: &Path, label: &str) -> Result<(), String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    Ok(())
}

fn regular_non_symlink_directory(path: &Path, label: &str) -> Result<(), String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?;
    if !metadata.file_type().is_dir() || metadata.file_type().is_symlink() {
        return Err(format!(
            "{label} must be a directory and must not be a symlink"
        ));
    }
    Ok(())
}

fn checked_u64_add(left: u64, right: u64, label: &str) -> Result<u64, String> {
    left.checked_add(right)
        .ok_or_else(|| format!("{label} overflows u64"))
}

fn checked_usize(value: u64, label: &str) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("{label} exceeds this platform"))
}

#[cfg(unix)]
fn positioned_read_exact(file: &File, offset: u64, destination: &mut [u8]) -> Result<(), String> {
    let mut read = 0usize;
    while read < destination.len() {
        let at = checked_u64_add(offset, read as u64, "positioned read offset")?;
        let amount = file
            .read_at(&mut destination[read..], at)
            .map_err(|error| format!("positioned read at byte {at} failed: {error}"))?;
        if amount == 0 {
            return Err(format!(
                "positioned read at byte {at} reached EOF before {} requested bytes",
                destination.len()
            ));
        }
        read = read
            .checked_add(amount)
            .ok_or_else(|| "positioned read byte count overflow".to_owned())?;
    }
    Ok(())
}

#[cfg(not(unix))]
compile_error!("the Qwen30 source range reader requires a positioned-read platform API");

/// The input index is a generated, sealed map rather than an invitation to
/// discover paths dynamically.  The independent original HF safetensors
/// index is validated against it before any shard range is used.
#[derive(Clone, Debug, Deserialize, Serialize)]
struct SourceIndexBinding {
    relative_path: String,
    sha256: String,
    format: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ShardRangeRecord {
    shard_id: String,
    relative_path: String,
    bytes: u64,
    sha256: String,
    safetensors_header_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct TensorRangeRecord {
    tensor_name: String,
    shard_id: String,
    dtype: String,
    shape: Vec<u64>,
    /// Absolute byte offset in the shard, including safetensors prefix/header.
    data_offset: u64,
    data_bytes: u64,
    raw_bf16_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct SourceRangeMap {
    schema: String,
    source_model_id: String,
    source_revision: String,
    source_tensor_count: usize,
    source_index: SourceIndexBinding,
    maximum_window_bytes: usize,
    shards: Vec<ShardRangeRecord>,
    tensors: Vec<TensorRangeRecord>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct ShardReadAttestation {
    shard_id: String,
    relative_path: String,
    bytes: u64,
    sha256: String,
    safetensors_header_sha256: String,
    checksum_verified_by_positioned_reads: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct TensorRangeReadAttestation {
    tensor_name: String,
    shard_id: String,
    first_element: u64,
    element_count: u64,
    absolute_data_offset: u64,
    raw_range_sha256: String,
    full_tensor_sha256_verified: bool,
    bf16_little_endian_row_major_order_verified: bool,
}

/// One reusable payload window.  It is intentionally the only long-lived
/// source-payload allocation in the reader.
#[derive(Debug)]
struct BoundedWindowCache {
    bytes: Vec<u8>,
    capacity: usize,
    maximum_observed_len: usize,
}

impl BoundedWindowCache {
    fn new(capacity: usize) -> Result<Self, String> {
        if capacity == 0 || capacity > MAX_SOURCE_WINDOW_BYTES {
            return Err(format!(
                "source window must be 1..={MAX_SOURCE_WINDOW_BYTES} bytes"
            ));
        }
        Ok(Self {
            bytes: Vec::with_capacity(capacity),
            capacity,
            maximum_observed_len: 0,
        })
    }

    fn load<'a>(&'a mut self, file: &File, offset: u64, length: usize) -> Result<&'a [u8], String> {
        if length == 0 || length > self.capacity {
            return Err(format!(
                "requested source window {length} exceeds bounded cache capacity {}",
                self.capacity
            ));
        }
        self.bytes.clear();
        self.bytes.resize(length, 0);
        positioned_read_exact(file, offset, &mut self.bytes)?;
        self.maximum_observed_len = self.maximum_observed_len.max(length);
        Ok(&self.bytes)
    }
}

fn bounded_metadata_bytes(path: &Path, label: &str, maximum: usize) -> Result<Vec<u8>, String> {
    regular_non_symlink_file(path, label)?;
    let length = fs::metadata(path)
        .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?
        .len();
    let length_usize = checked_usize(length, label)?;
    if length_usize == 0 || length_usize > maximum {
        return Err(format!(
            "{label} must be 1..={maximum} bytes of bounded metadata, observed {length_usize}"
        ));
    }
    let mut handle = File::open(path)
        .map_err(|error| format!("cannot open {label} {}: {error}", path.display()))?;
    let mut bytes = Vec::with_capacity(length_usize);
    handle
        .read_to_end(&mut bytes)
        .map_err(|error| format!("cannot read {label} {}: {error}", path.display()))?;
    if bytes.len() != length_usize {
        return Err(format!("{label} changed while it was read"));
    }
    Ok(bytes)
}

fn element_count(shape: &[u64], label: &str) -> Result<u64, String> {
    if shape.is_empty() || shape.iter().any(|dimension| *dimension == 0) {
        return Err(format!(
            "{label} shape must contain only positive dimensions"
        ));
    }
    shape.iter().try_fold(1u64, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| format!("{label} shape element count overflows u64"))
    })
}

fn validate_range_map(map: &SourceRangeMap) -> Result<(), String> {
    if map.schema != RANGE_MAP_SCHEMA {
        return Err("range map schema differs from the pinned contract".into());
    }
    if map.source_model_id.is_empty() || map.source_revision.is_empty() {
        return Err("range map source model/revision must be non-empty".into());
    }
    if map.source_tensor_count == 0 || map.source_tensor_count != map.tensors.len() {
        return Err("range map source tensor count must equal its tensor record count".into());
    }
    if map.maximum_window_bytes == 0 || map.maximum_window_bytes > MAX_SOURCE_WINDOW_BYTES {
        return Err(format!(
            "range map maximum_window_bytes must be 1..={MAX_SOURCE_WINDOW_BYTES}"
        ));
    }
    if map.source_index.format != "huggingface.safetensors.index.json"
        || !is_sha256(&map.source_index.sha256)
    {
        return Err("range map source index must be a SHA-256-bound HF safetensors index".into());
    }
    checked_relative_path(
        &map.source_index.relative_path,
        "range map source index path",
    )?;

    let mut shard_ids = BTreeSet::new();
    let mut shard_paths = BTreeSet::new();
    for shard in &map.shards {
        if shard.shard_id.is_empty()
            || !shard_ids.insert(shard.shard_id.as_str())
            || !shard_paths.insert(shard.relative_path.as_str())
        {
            return Err("range map shard IDs and paths must be non-empty and unique".into());
        }
        checked_relative_path(&shard.relative_path, "range map shard path")?;
        if shard.bytes == 0
            || !is_sha256(&shard.sha256)
            || !is_sha256(&shard.safetensors_header_sha256)
        {
            return Err(format!(
                "range map shard {} has invalid byte/checksum fields",
                shard.shard_id
            ));
        }
    }
    if map.shards.is_empty() {
        return Err("range map must name at least one shard".into());
    }

    let mut names = BTreeSet::new();
    let mut ranges_by_shard: BTreeMap<&str, Vec<(u64, u64, &str)>> = BTreeMap::new();
    for tensor in &map.tensors {
        if tensor.tensor_name.is_empty() || !names.insert(tensor.tensor_name.as_str()) {
            return Err("range map tensor names must be non-empty and unique".into());
        }
        if tensor.dtype != "BF16" || !is_sha256(&tensor.raw_bf16_sha256) {
            return Err(format!(
                "range map tensor {} must be BF16 with a valid raw checksum",
                tensor.tensor_name
            ));
        }
        if !shard_ids.contains(tensor.shard_id.as_str()) {
            return Err(format!(
                "range map tensor {} references an unknown shard",
                tensor.tensor_name
            ));
        }
        let elements = element_count(&tensor.shape, &tensor.tensor_name)?;
        let expected = elements
            .checked_mul(BF16_BYTES as u64)
            .ok_or_else(|| format!("{} BF16 byte count overflows u64", tensor.tensor_name))?;
        if tensor.data_bytes == 0 || tensor.data_bytes != expected || tensor.data_bytes % 2 != 0 {
            return Err(format!(
                "range map tensor {} BF16 shape/byte geometry disagrees",
                tensor.tensor_name
            ));
        }
        let end = checked_u64_add(tensor.data_offset, tensor.data_bytes, "tensor byte range")?;
        ranges_by_shard
            .entry(tensor.shard_id.as_str())
            .or_default()
            .push((tensor.data_offset, end, tensor.tensor_name.as_str()));
    }
    for ranges in ranges_by_shard.values_mut() {
        ranges.sort_by_key(|entry| entry.0);
        let mut previous_end = 0u64;
        for (start, end, name) in ranges {
            if *start < previous_end {
                return Err(format!(
                    "range map tensor {name} overlaps another tensor range"
                ));
            }
            previous_end = *end;
        }
    }
    Ok(())
}

fn json_object<'a>(
    value: &'a Value,
    label: &str,
) -> Result<&'a serde_json::Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be a JSON object"))
}

fn json_string<'a>(value: &'a Value, label: &str) -> Result<&'a str, String> {
    value
        .as_str()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{label} must be a non-empty string"))
}

fn json_u64(value: &Value, label: &str) -> Result<u64, String> {
    value
        .as_u64()
        .ok_or_else(|| format!("{label} must be a non-negative integer"))
}

/// A read-only verified range reader.  It retains file handles, a parsed range
/// map, and one bounded payload cache.  It intentionally has no API that can
/// hand a caller an entire shard or a decoded full source tensor.
#[derive(Debug)]
struct VerifiedBf16RangeReader {
    source_root: PathBuf,
    range_map_document_sha256: String,
    range_map: SourceRangeMap,
    files: BTreeMap<String, File>,
    shards: BTreeMap<String, ShardReadAttestation>,
    cache: BoundedWindowCache,
}

impl VerifiedBf16RangeReader {
    fn open(
        source_root: &Path,
        range_map_path: &Path,
        expected_range_map_sha256: &str,
    ) -> Result<Self, String> {
        regular_non_symlink_directory(source_root, "source root")?;
        if !range_map_path.is_absolute() {
            return Err("range map path must be absolute".into());
        }
        if !is_sha256(expected_range_map_sha256) {
            return Err("expected range-map checksum must be a SHA-256".into());
        }
        let range_map_bytes =
            bounded_metadata_bytes(range_map_path, "range map document", MAX_RANGE_MAP_BYTES)?;
        let range_map_document_sha256 = sha256_hex(&range_map_bytes);
        if range_map_document_sha256 != expected_range_map_sha256 {
            return Err("range map document checksum differs from its sealed binding".into());
        }
        let range_map: SourceRangeMap = serde_json::from_slice(&range_map_bytes)
            .map_err(|error| format!("range map document is invalid JSON: {error}"))?;
        validate_range_map(&range_map)?;

        let source_index_path = source_root.join(checked_relative_path(
            &range_map.source_index.relative_path,
            "source index path",
        )?);
        let source_index_bytes = bounded_metadata_bytes(
            &source_index_path,
            "source safetensors index",
            MAX_SOURCE_INDEX_BYTES,
        )?;
        if sha256_hex(&source_index_bytes) != range_map.source_index.sha256 {
            return Err("source safetensors index checksum differs from the range map".into());
        }
        let source_index: Value = serde_json::from_slice(&source_index_bytes)
            .map_err(|error| format!("source safetensors index is invalid JSON: {error}"))?;
        let weight_map = json_object(&source_index, "source safetensors index")?
            .get("weight_map")
            .ok_or_else(|| "source safetensors index lacks weight_map".to_owned())?;
        let weight_map = json_object(weight_map, "source safetensors index weight_map")?;
        if weight_map.len() != range_map.source_tensor_count {
            return Err(format!(
                "source safetensors index has {} tensor entries but range map binds {}",
                weight_map.len(),
                range_map.source_tensor_count
            ));
        }

        let mut cache = BoundedWindowCache::new(range_map.maximum_window_bytes)?;
        let mut files = BTreeMap::new();
        let mut attestations = BTreeMap::new();
        for shard in &range_map.shards {
            let path = source_root.join(checked_relative_path(
                &shard.relative_path,
                "source shard path",
            )?);
            regular_non_symlink_file(&path, "source shard")?;
            let metadata = fs::metadata(&path)
                .map_err(|error| format!("cannot stat source shard {}: {error}", path.display()))?;
            if metadata.len() != shard.bytes {
                return Err(format!(
                    "source shard {} byte length differs from its range map",
                    shard.shard_id
                ));
            }
            let file = File::open(&path)
                .map_err(|error| format!("cannot open source shard {}: {error}", path.display()))?;
            let checksum = streamed_file_sha256(&file, shard.bytes, &mut cache)?;
            if checksum != shard.sha256 {
                return Err(format!(
                    "source shard {} checksum differs from its range map",
                    shard.shard_id
                ));
            }
            validate_safetensors_header(&file, shard, &range_map.tensors, &mut cache)?;
            attestations.insert(
                shard.shard_id.clone(),
                ShardReadAttestation {
                    shard_id: shard.shard_id.clone(),
                    relative_path: shard.relative_path.clone(),
                    bytes: shard.bytes,
                    sha256: shard.sha256.clone(),
                    safetensors_header_sha256: shard.safetensors_header_sha256.clone(),
                    checksum_verified_by_positioned_reads: true,
                },
            );
            files.insert(shard.shard_id.clone(), file);
        }

        for tensor in &range_map.tensors {
            let expected_path = range_map
                .shards
                .iter()
                .find(|shard| shard.shard_id == tensor.shard_id)
                .ok_or_else(|| "validated tensor lost its shard record".to_owned())?
                .relative_path
                .as_str();
            let actual_path = weight_map
                .get(&tensor.tensor_name)
                .ok_or_else(|| format!("source index lacks tensor {}", tensor.tensor_name))?;
            if json_string(actual_path, "source index shard path")? != expected_path {
                return Err(format!(
                    "source index shard path for {} differs from the range map",
                    tensor.tensor_name
                ));
            }
        }
        for (tensor_name, shard_path) in weight_map {
            let tensor = range_map
                .tensors
                .iter()
                .find(|tensor| tensor.tensor_name == *tensor_name)
                .ok_or_else(|| {
                    format!(
                        "source index contains tensor {tensor_name} outside the sealed range map"
                    )
                })?;
            let expected_path = range_map
                .shards
                .iter()
                .find(|shard| shard.shard_id == tensor.shard_id)
                .ok_or_else(|| "validated tensor lost its shard record".to_owned())?
                .relative_path
                .as_str();
            if json_string(shard_path, "source index shard path")? != expected_path {
                return Err(format!(
                    "source index shard path for {tensor_name} differs from the range map"
                ));
            }
        }

        Ok(Self {
            source_root: source_root.to_path_buf(),
            range_map_document_sha256,
            range_map,
            files,
            shards: attestations,
            cache,
        })
    }

    fn source_root(&self) -> &Path {
        &self.source_root
    }

    fn range_map_document_sha256(&self) -> &str {
        &self.range_map_document_sha256
    }

    fn maximum_payload_cache_bytes(&self) -> usize {
        self.cache.capacity
    }

    fn maximum_payload_cache_observed_bytes(&self) -> usize {
        self.cache.maximum_observed_len
    }

    fn shard_attestations(&self) -> Vec<ShardReadAttestation> {
        self.shards.values().cloned().collect()
    }

    fn tensor(&self, name: &str) -> Result<TensorRangeRecord, String> {
        self.range_map
            .tensors
            .iter()
            .find(|tensor| tensor.tensor_name == name)
            .cloned()
            .ok_or_else(|| format!("range map has no tensor named {name}"))
    }

    /// Stream a contiguous row-major element range.  The callback observes
    /// original BF16 bits in increasing element index order.  The reader does
    /// not decode to an owning vector and does not allow a caller to request a
    /// range larger than the declared tensor geometry.
    fn stream_bf16_elements<F>(
        &mut self,
        tensor_name: &str,
        first_element: u64,
        element_count: u64,
        mut visit: F,
    ) -> Result<TensorRangeReadAttestation, String>
    where
        F: FnMut(u64, u16) -> Result<(), String>,
    {
        if element_count == 0 {
            return Err("BF16 range requests must contain at least one element".into());
        }
        let tensor = self.tensor(tensor_name)?;
        let total_elements = element_count_for_tensor(&tensor)?;
        let end_element = checked_u64_add(first_element, element_count, "BF16 element range")?;
        if end_element > total_elements {
            return Err(format!(
                "BF16 range [{first_element}, {end_element}) exceeds tensor {} element count {total_elements}",
                tensor.tensor_name
            ));
        }
        let byte_offset = first_element
            .checked_mul(BF16_BYTES as u64)
            .ok_or_else(|| "BF16 element offset overflows u64".to_owned())?;
        let byte_length = element_count
            .checked_mul(BF16_BYTES as u64)
            .ok_or_else(|| "BF16 element range byte count overflows u64".to_owned())?;
        let absolute_offset =
            checked_u64_add(tensor.data_offset, byte_offset, "BF16 absolute range")?;
        let requested_end = checked_u64_add(absolute_offset, byte_length, "BF16 absolute range")?;
        let tensor_end = checked_u64_add(tensor.data_offset, tensor.data_bytes, "tensor range")?;
        if requested_end > tensor_end {
            return Err("BF16 element range escapes its sealed tensor range".into());
        }
        let file = self
            .files
            .get(&tensor.shard_id)
            .ok_or_else(|| format!("reader has no open handle for shard {}", tensor.shard_id))?
            .try_clone()
            .map_err(|error| format!("cannot clone positioned source shard handle: {error}"))?;

        let mut raw_range_hasher = Sha256::new();
        let mut remaining = checked_usize(byte_length, "BF16 range byte count")?;
        let mut byte_cursor = absolute_offset;
        let mut index = first_element;
        while remaining > 0 {
            let window = remaining.min(self.cache.capacity);
            if window % BF16_BYTES != 0 {
                return Err("BF16 range window is not aligned to two-byte elements".into());
            }
            let bytes = self.cache.load(&file, byte_cursor, window)?;
            raw_range_hasher.update(bytes);
            for pair in bytes.chunks_exact(BF16_BYTES) {
                let bits = u16::from_le_bytes([pair[0], pair[1]]);
                if !bf16::from_bits(bits).is_finite() {
                    return Err(format!(
                        "source tensor {} has a non-finite BF16 element at row-major index {index}",
                        tensor.tensor_name
                    ));
                }
                visit(index, bits)?;
                index = checked_u64_add(index, 1, "BF16 row-major element index")?;
            }
            remaining -= window;
            byte_cursor = checked_u64_add(byte_cursor, window as u64, "BF16 range cursor")?;
        }
        if index != end_element {
            return Err("BF16 streaming did not visit the requested element count".into());
        }
        let raw_range_sha256 = format!("{:x}", raw_range_hasher.finalize());
        let full_tensor_sha256_verified = first_element == 0 && element_count == total_elements;
        if full_tensor_sha256_verified && raw_range_sha256 != tensor.raw_bf16_sha256 {
            return Err(format!(
                "source tensor {} raw BF16 checksum differs from its range map",
                tensor.tensor_name
            ));
        }
        Ok(TensorRangeReadAttestation {
            tensor_name: tensor.tensor_name,
            shard_id: tensor.shard_id,
            first_element,
            element_count,
            absolute_data_offset: absolute_offset,
            raw_range_sha256,
            full_tensor_sha256_verified,
            bf16_little_endian_row_major_order_verified: true,
        })
    }

    /// Translate a contiguous outer-dimension row interval to a verified
    /// row-major BF16 element interval.  This prevents callers from silently
    /// treating a transposed/tiled range as a source row.
    fn stream_bf16_rows<F>(
        &mut self,
        tensor_name: &str,
        first_row: u64,
        rows: u64,
        visit: F,
    ) -> Result<TensorRangeReadAttestation, String>
    where
        F: FnMut(u64, u16) -> Result<(), String>,
    {
        if rows == 0 {
            return Err("row ranges must contain at least one row".into());
        }
        let tensor = self.tensor(tensor_name)?;
        let row_count = *tensor
            .shape
            .first()
            .ok_or_else(|| "validated tensor has no outer row dimension".to_owned())?;
        let end_row = checked_u64_add(first_row, rows, "row range")?;
        if end_row > row_count {
            return Err(format!(
                "row range [{first_row}, {end_row}) exceeds tensor {} outer dimension {row_count}",
                tensor.tensor_name
            ));
        }
        let row_width = tensor.shape[1..]
            .iter()
            .try_fold(1u64, |count, dimension| {
                count
                    .checked_mul(*dimension)
                    .ok_or_else(|| "row width overflows u64".to_owned())
            })?;
        let first_element = first_row
            .checked_mul(row_width)
            .ok_or_else(|| "row range first element overflows u64".to_owned())?;
        let element_count = rows
            .checked_mul(row_width)
            .ok_or_else(|| "row range element count overflows u64".to_owned())?;
        self.stream_bf16_elements(tensor_name, first_element, element_count, visit)
    }
}

fn element_count_for_tensor(tensor: &TensorRangeRecord) -> Result<u64, String> {
    element_count(&tensor.shape, &tensor.tensor_name)
}

fn streamed_file_sha256(
    file: &File,
    bytes: u64,
    cache: &mut BoundedWindowCache,
) -> Result<String, String> {
    if bytes == 0 {
        return Err("source shard must not be empty".into());
    }
    let mut hasher = Sha256::new();
    let mut offset = 0u64;
    while offset < bytes {
        let remaining = bytes - offset;
        let length = checked_usize(
            remaining.min(cache.capacity as u64),
            "source checksum window",
        )?;
        let window = cache.load(file, offset, length)?;
        hasher.update(window);
        offset = checked_u64_add(offset, length as u64, "source checksum cursor")?;
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_safetensors_header(
    file: &File,
    shard: &ShardRangeRecord,
    tensors: &[TensorRangeRecord],
    cache: &mut BoundedWindowCache,
) -> Result<(), String> {
    let mut prefix = [0u8; 8];
    positioned_read_exact(file, 0, &mut prefix)?;
    let header_bytes = u64::from_le_bytes(prefix);
    let header_length = checked_usize(header_bytes, "safetensors header length")?;
    if header_length == 0 || header_length > MAX_SAFETENSORS_HEADER_BYTES {
        return Err(format!(
            "source shard {} safetensors header must be 1..={MAX_SAFETENSORS_HEADER_BYTES} bytes",
            shard.shard_id
        ));
    }
    let header_end = checked_u64_add(8, header_bytes, "safetensors header end")?;
    if header_end > shard.bytes {
        return Err(format!(
            "source shard {} safetensors header exceeds shard bytes",
            shard.shard_id
        ));
    }
    // Header metadata is bounded independently; this is never a payload cache.
    let mut header = vec![0u8; header_length];
    let mut remaining = header_length;
    let mut offset = 8u64;
    let mut copied = 0usize;
    while remaining > 0 {
        let length = remaining.min(cache.capacity);
        let window = cache.load(file, offset, length)?;
        header[copied..copied + length].copy_from_slice(window);
        copied += length;
        remaining -= length;
        offset = checked_u64_add(offset, length as u64, "safetensors header cursor")?;
    }
    if sha256_hex(&header) != shard.safetensors_header_sha256 {
        return Err(format!(
            "source shard {} safetensors header checksum differs from its range map",
            shard.shard_id
        ));
    }
    let document: Value = serde_json::from_slice(&header).map_err(|error| {
        format!(
            "source shard {} has invalid safetensors JSON: {error}",
            shard.shard_id
        )
    })?;
    let entries = json_object(&document, "safetensors header")?;
    for tensor in tensors
        .iter()
        .filter(|tensor| tensor.shard_id == shard.shard_id)
    {
        let entry = entries
            .get(&tensor.tensor_name)
            .ok_or_else(|| format!("safetensors header lacks tensor {}", tensor.tensor_name))?;
        let entry = json_object(entry, "safetensors tensor entry")?;
        if json_string(
            entry
                .get("dtype")
                .ok_or_else(|| format!("safetensors tensor {} lacks dtype", tensor.tensor_name))?,
            "safetensors tensor dtype",
        )? != "BF16"
        {
            return Err(format!(
                "safetensors tensor {} is not BF16",
                tensor.tensor_name
            ));
        }
        let header_shape = entry
            .get("shape")
            .ok_or_else(|| format!("safetensors tensor {} lacks shape", tensor.tensor_name))?
            .as_array()
            .ok_or_else(|| {
                format!(
                    "safetensors tensor {} shape must be an array",
                    tensor.tensor_name
                )
            })?;
        let header_shape = header_shape
            .iter()
            .map(|value| json_u64(value, "safetensors tensor shape dimension"))
            .collect::<Result<Vec<_>, _>>()?;
        if header_shape != tensor.shape {
            return Err(format!(
                "safetensors tensor {} shape differs from range map",
                tensor.tensor_name
            ));
        }
        let offsets = entry
            .get("data_offsets")
            .ok_or_else(|| {
                format!(
                    "safetensors tensor {} lacks data_offsets",
                    tensor.tensor_name
                )
            })?
            .as_array()
            .ok_or_else(|| {
                format!(
                    "safetensors tensor {} offsets must be an array",
                    tensor.tensor_name
                )
            })?;
        if offsets.len() != 2 {
            return Err(format!(
                "safetensors tensor {} needs two data offsets",
                tensor.tensor_name
            ));
        }
        let relative_start = json_u64(&offsets[0], "safetensors data start")?;
        let relative_end = json_u64(&offsets[1], "safetensors data end")?;
        if relative_end < relative_start {
            return Err(format!(
                "safetensors tensor {} has an inverted byte range",
                tensor.tensor_name
            ));
        }
        let header_data_start = checked_u64_add(8, header_bytes, "safetensors data start")?;
        let absolute_start = checked_u64_add(
            header_data_start,
            relative_start,
            "safetensors tensor offset",
        )?;
        if absolute_start != tensor.data_offset
            || relative_end - relative_start != tensor.data_bytes
        {
            return Err(format!(
                "safetensors tensor {} offset/length differs from range map",
                tensor.tensor_name
            ));
        }
        let absolute_end =
            checked_u64_add(absolute_start, tensor.data_bytes, "safetensors tensor end")?;
        if absolute_end > shard.bytes {
            return Err(format!(
                "safetensors tensor {} exceeds shard bytes",
                tensor.tensor_name
            ));
        }
    }
    Ok(())
}

/// The trace has one fixed 369-token source-template prefill and exactly one
/// forced token (`949`).  No sampling and no autoregressive candidate feedback
/// are represented: that is essential for source/control/candidate alignment.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum TracePhase {
    SourceTemplatePrefix,
    ForcedIdenticalContinuation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct TraceForward {
    forward_index: usize,
    token_position: usize,
    phase: TracePhase,
}

fn exact_trace() -> Vec<TraceForward> {
    let mut result = Vec::with_capacity(TRACE_FORWARD_COUNT);
    for position in 0..PREFIX_TOKEN_COUNT {
        result.push(TraceForward {
            forward_index: position,
            token_position: position,
            phase: TracePhase::SourceTemplatePrefix,
        });
    }
    result.push(TraceForward {
        forward_index: PREFIX_TOKEN_COUNT,
        token_position: PREFIX_TOKEN_COUNT,
        phase: TracePhase::ForcedIdenticalContinuation,
    });
    result
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum SourceOperator {
    TokenEmbeddingRow,
    InputRmsNorm,
    QueryAndKeyRmsNorm,
    QueryKeyValueProjectionSerialK,
    RopeThenCausalKvAppendAndRead,
    AttentionOutputProjectionSerialK,
    FirstResidualAdd,
    PostAttentionRmsNorm,
    RouterProjectionSerialK,
    Top8Selection,
    SelectedExpertGateUpSerialK,
    SourceSiluStoreBoundary,
    SelectedExpertDownSerialK,
    SourceOrderedRouteWeightedCombine,
    SecondResidualAdd,
    FinalRmsNorm,
    LmHeadSerialK,
    RetainFullF32EndpointLogits,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum AccumulationBoundary {
    RawBf16LittleEndianRowMajorRead,
    IncreasingKOrderRequired,
    SourceDefinedF32StoreBoundary,
    CausalKvAppendBeforeCausalRead,
    SourceDefinedResidualAddOrder,
    SourceDefinedRouteWeightedExpertCombineOrder,
    FullF32LogitRetention,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct TraceReplayInvocation {
    forward_index: usize,
    token_position: usize,
    phase: TracePhase,
    layer_index: Option<usize>,
    /// `Some(0..8)` only for the eight selected MoE bodies, in the source
    /// route-selection order.  The actual IDs/weights remain executor-owned
    /// source data and must be separately attested.
    route_slot: Option<usize>,
    operator: SourceOperator,
    boundaries: Vec<AccumulationBoundary>,
}

fn layer_operator_sequence() -> &'static [(SourceOperator, &'static [AccumulationBoundary])] {
    use AccumulationBoundary::*;
    use SourceOperator::*;
    &[
        (
            InputRmsNorm,
            &[
                RawBf16LittleEndianRowMajorRead,
                SourceDefinedF32StoreBoundary,
            ],
        ),
        (
            QueryAndKeyRmsNorm,
            &[
                RawBf16LittleEndianRowMajorRead,
                SourceDefinedF32StoreBoundary,
            ],
        ),
        (
            QueryKeyValueProjectionSerialK,
            &[RawBf16LittleEndianRowMajorRead, IncreasingKOrderRequired],
        ),
        (
            RopeThenCausalKvAppendAndRead,
            &[
                SourceDefinedF32StoreBoundary,
                CausalKvAppendBeforeCausalRead,
            ],
        ),
        (
            AttentionOutputProjectionSerialK,
            &[RawBf16LittleEndianRowMajorRead, IncreasingKOrderRequired],
        ),
        (FirstResidualAdd, &[SourceDefinedResidualAddOrder]),
        (
            PostAttentionRmsNorm,
            &[
                RawBf16LittleEndianRowMajorRead,
                SourceDefinedF32StoreBoundary,
            ],
        ),
        (
            RouterProjectionSerialK,
            &[RawBf16LittleEndianRowMajorRead, IncreasingKOrderRequired],
        ),
        (Top8Selection, &[SourceDefinedF32StoreBoundary]),
        (
            SourceOrderedRouteWeightedCombine,
            &[SourceDefinedRouteWeightedExpertCombineOrder],
        ),
        (SecondResidualAdd, &[SourceDefinedResidualAddOrder]),
    ]
}

fn selected_expert_operator_sequence(
) -> &'static [(SourceOperator, &'static [AccumulationBoundary])] {
    use AccumulationBoundary::*;
    use SourceOperator::*;
    &[
        (
            SelectedExpertGateUpSerialK,
            &[RawBf16LittleEndianRowMajorRead, IncreasingKOrderRequired],
        ),
        (SourceSiluStoreBoundary, &[SourceDefinedF32StoreBoundary]),
        (
            SelectedExpertDownSerialK,
            &[RawBf16LittleEndianRowMajorRead, IncreasingKOrderRequired],
        ),
    ]
}

/// A future executor supplies the actual source implementation behind this
/// interface.  This contract gives it only a verified reader and an immutable
/// event sequence; it cannot substitute an approximate runtime or reorder a
/// source accumulator without the attestation becoming false.
trait ExactTraceReplay {
    fn replay(&mut self, invocation: &TraceReplayInvocation) -> Result<(), String>;
}

fn replay_exact_trace<R: ExactTraceReplay>(replayer: &mut R) -> Result<(), String> {
    for forward in exact_trace() {
        replayer.replay(&TraceReplayInvocation {
            forward_index: forward.forward_index,
            token_position: forward.token_position,
            phase: forward.phase,
            layer_index: None,
            route_slot: None,
            operator: SourceOperator::TokenEmbeddingRow,
            boundaries: vec![AccumulationBoundary::RawBf16LittleEndianRowMajorRead],
        })?;
        for layer_index in 0..LAYER_COUNT {
            for (operator, boundaries) in layer_operator_sequence() {
                replayer.replay(&TraceReplayInvocation {
                    forward_index: forward.forward_index,
                    token_position: forward.token_position,
                    phase: forward.phase,
                    layer_index: Some(layer_index),
                    route_slot: None,
                    operator: *operator,
                    boundaries: boundaries.to_vec(),
                })?;
                if *operator == SourceOperator::Top8Selection {
                    for route_slot in 0..TOP_K {
                        for (selected_operator, selected_boundaries) in
                            selected_expert_operator_sequence()
                        {
                            replayer.replay(&TraceReplayInvocation {
                                forward_index: forward.forward_index,
                                token_position: forward.token_position,
                                phase: forward.phase,
                                layer_index: Some(layer_index),
                                route_slot: Some(route_slot),
                                operator: *selected_operator,
                                boundaries: selected_boundaries.to_vec(),
                            })?;
                        }
                    }
                }
            }
        }
        for (operator, boundaries) in [
            (
                SourceOperator::FinalRmsNorm,
                vec![
                    AccumulationBoundary::RawBf16LittleEndianRowMajorRead,
                    AccumulationBoundary::SourceDefinedF32StoreBoundary,
                ],
            ),
            (
                SourceOperator::LmHeadSerialK,
                vec![
                    AccumulationBoundary::RawBf16LittleEndianRowMajorRead,
                    AccumulationBoundary::IncreasingKOrderRequired,
                ],
            ),
            (
                SourceOperator::RetainFullF32EndpointLogits,
                vec![AccumulationBoundary::FullF32LogitRetention],
            ),
        ] {
            replayer.replay(&TraceReplayInvocation {
                forward_index: forward.forward_index,
                token_position: forward.token_position,
                phase: forward.phase,
                layer_index: None,
                route_slot: None,
                operator,
                boundaries,
            })?;
        }
    }
    Ok(())
}

/// The caller owns the arithmetic.  The range reader proves only that the
/// accumulator receives original BF16 bits in the declared row-major,
/// increasing-K order.  This makes it impossible to mistake a diagnostic
/// f64 fixture accumulator for the actual source arithmetic.
trait OrderedBf16Accumulator {
    fn begin_linear(&mut self, tensor: &TensorRangeRecord, expected_k: u64) -> Result<(), String>;
    fn push_bf16_pair(
        &mut self,
        k: u64,
        activation_bits: u16,
        weight_bits: u16,
    ) -> Result<(), String>;
    fn finish_linear(&mut self) -> Result<(), String>;
}

fn replay_one_bf16_row_dot<A: OrderedBf16Accumulator>(
    reader: &mut VerifiedBf16RangeReader,
    tensor_name: &str,
    activation_bits: &[u16],
    accumulator: &mut A,
) -> Result<TensorRangeReadAttestation, String> {
    let tensor = reader.tensor(tensor_name)?;
    let elements = element_count_for_tensor(&tensor)?;
    if elements != activation_bits.len() as u64 {
        return Err(format!(
            "source tensor {} has {elements} elements but activation row has {}",
            tensor.tensor_name,
            activation_bits.len()
        ));
    }
    if activation_bits
        .iter()
        .any(|bits| !bf16::from_bits(*bits).is_finite())
    {
        return Err("activation row contains a non-finite BF16 value".into());
    }
    accumulator.begin_linear(&tensor, elements)?;
    let mut next_k = 0usize;
    let result = reader.stream_bf16_elements(tensor_name, 0, elements, |k, weight_bits| {
        if k != next_k as u64 {
            return Err("source reader did not preserve increasing-K order".into());
        }
        let activation = activation_bits
            .get(next_k)
            .copied()
            .ok_or_else(|| "activation row ended before source tensor range".to_owned())?;
        accumulator.push_bf16_pair(k, activation, weight_bits)?;
        next_k += 1;
        Ok(())
    })?;
    if next_k != activation_bits.len() {
        return Err("source reader did not visit every BF16 dot-product term".into());
    }
    accumulator.finish_linear()?;
    Ok(result)
}

fn static_contract_document() -> Value {
    let per_pass_layer_traversals = TRACE_FORWARD_COUNT * LAYER_COUNT;
    json!({
        "schema": RESULT_SCHEMA,
        "status": RESULT_STATUS,
        "execution_boundary": {
            "source_tensor_payload_opened": false,
            "source_model_loaded": false,
            "whole_source_shard_mapped": false,
            "whole_source_shard_cached": false,
            "gpu_or_metal_invoked": false,
            "server_started": false,
            "hcli_invoked": false,
            "lease_requested": false,
            "source_control_candidate_comparison_performed": false,
            "coherence_claim_made": false,
            "tps_or_tg_claim_made": false
        },
        "bounded_range_reader": {
            "range_map_schema": RANGE_MAP_SCHEMA,
            "source_shard_format": "safetensors",
            "source_index_format": "huggingface.safetensors.index.json",
            "positioned_reads_required": true,
            "mmap_forbidden": true,
            "full_shard_return_api_absent": true,
            "full_tensor_decode_api_absent": true,
            "maximum_source_payload_window_bytes": MAX_SOURCE_WINDOW_BYTES,
            "maximum_range_map_metadata_bytes": MAX_RANGE_MAP_BYTES,
            "maximum_source_index_metadata_bytes": MAX_SOURCE_INDEX_BYTES,
            "maximum_safetensors_header_metadata_bytes": MAX_SAFETENSORS_HEADER_BYTES,
            "shard_checksum_scan_uses_the_same_bounded_window": true,
            "raw_bf16_little_endian_decode_required": true,
            "row_major_increasing_index_callbacks_required": true,
            "safetensors_header_offsets_shapes_and_bf16_dtype_must_match_range_map": true,
            "source_index_tensor_to_shard_mapping_must_match_range_map": true,
            "relative_non_symlink_source_paths_required": true
        },
        "exact_trace": {
            "source_template_token_count": PREFIX_TOKEN_COUNT,
            "forced_identical_continuation_token_id": FORCED_TOKEN_ID,
            "total_forwards_per_replay_arm": TRACE_FORWARD_COUNT,
            "layers": LAYER_COUNT,
            "layer_traversals_per_replay_arm": per_pass_layer_traversals,
            "two_arm_control_candidate_reference": {
                "total_forwards": TRACE_FORWARD_COUNT * 2,
                "layer_traversals": per_pass_layer_traversals * 2
            },
            "prefill_then_forced_cache_order": true,
            "sampling_or_autoregressive_feedback_forbidden": true,
            "operator_sequence_interface": [
                "token_embedding_row",
                "input_rmsnorm",
                "q_k_rmsnorm_then_qkv_serial_k",
                "rope_then_causal_kv_append_and_read",
                "attention_output_serial_k_then_first_residual",
                "post_attention_rmsnorm_then_router_top8",
                "selected_expert_gate_up_silu_down_one_body_at_a_time",
                "source_ordered_route_weighted_combine_then_second_residual",
                "final_rmsnorm_then_lm_head_serial_k",
                "retain_full_f32_logits_at_prefix_and_forced_endpoints"
            ],
            "accumulation_interface": {
                "raw_bf16_bits_passed_without_reordering": true,
                "linear_k_order_is_explicitly_checked": true,
                "all_eight_selected_expert_bodies_are_replayed_in_explicit_route_slot_order": true,
                "kv_append_before_causal_read_is_explicitly_required": true,
                "source_f32_store_boundaries_must_be_attested_by_future_executor": true,
                "source_residual_and_route_combine_order_must_be_attested_by_future_executor": true,
                "fixture_f64_accumulator_is_not_a_source_semantics_claim": true
            }
        },
        "future_exact_semantics_attestation": {
            "schema": FUTURE_ATTESTATION_SCHEMA,
            "status_only_after_a_real_separately_leased_execution": FUTURE_ATTESTATION_STATUS,
            "source_binding_required": {
                "source_config_sha256": "SHA256_OF_PINNED_QWEN30_SOURCE_CONFIG",
                "source_index_sha256": "SHA256_OF_PINNED_QWEN30_SOURCE_INDEX",
                "source_template_token_ids_u32le_sha256": "SHA256_OF_EXACT_369_TOKEN_LITERAL_HAWKING_TRACE",
                "source_weight_bytes": "EXACT_SOURCE_WEIGHT_BYTE_TOTAL",
                "source_tensor_count": SOURCE_TENSOR_COUNT,
                "source_shard_count": SHARD_COUNT,
                "source_model_id": SOURCE_MODEL_ID,
                "source_revision": "PINNED_SOURCE_REVISION"
            },
            "exact_trace_required": {
                "prefix_token_count": PREFIX_TOKEN_COUNT,
                "forced_token_id": FORCED_TOKEN_ID,
                "prefill_then_forced_cache_order": true,
                "sampling_or_autoregressive_feedback_forbidden": true
            },
            "exact_semantics_required": {
                "source_bf16_tensor_row_order_and_offsets_verified": true,
                "range_reader_never_maps_or_caches_a_complete_source_shard": true,
                "source_rmsnorm_rope_attention_router_topk_moe_operator_order_verified": true,
                "source_accumulation_and_expert_combine_order_verified": true,
                "all_48_layers_and_final_norm_head_verified": true,
                "full_f32_final_logits_at_prefix_and_forced_endpoints_verified": true
            },
            "fresh_execution_evidence_required": [
                "sealed source range-map checksum and all sixteen shard checksums",
                "per-tensor safetensors header/shape/offset/BF16 checks",
                "maximum observed source payload cache bytes <= declared bound",
                "fresh zero-swap observation before and after execution",
                "exclusive non-timed source-oracle lease",
                "two retained source full-F32 endpoint logit vectors with byte hashes",
                "aligned retained control and candidate vectors before three-way scoring"
            ]
        },
        "working_set_policy": {
            "row_tile_rows": 128,
            "max_simultaneous_expert_bodies": 1,
            "activation_element_bytes": 4,
            "kv_cache_element_bytes": 2,
            "attention_score_element_bytes": 4,
            "backend_allocator_reserve_bytes": 134217728,
            "minimum_unallocated_safety_margin_bytes": 1073741824
        },
        "source_geometry_reference": {
            "model_id": SOURCE_MODEL_ID,
            "layers": LAYER_COUNT,
            "hidden_size": HIDDEN_SIZE,
            "attention_heads": ATTENTION_HEADS,
            "key_value_heads": KEY_VALUE_HEADS,
            "head_dim": HEAD_DIM,
            "experts": EXPERT_COUNT,
            "top_k": TOP_K,
            "moe_intermediate": MOE_INTERMEDIATE,
            "vocab_rows": VOCAB_ROWS,
            "source_tensor_count": SOURCE_TENSOR_COUNT,
            "source_shard_count": SHARD_COUNT,
            "source_dtype": "BF16"
        },
        "claim_boundary": "Prepared reader and replay interfaces only; no real source tensor, source final logit, quality comparison, semantic coherence, runtime, HCLI, TPS, TG, or tournament result exists."
    })
}

fn write_new_json(path: &Path, value: &Value) -> Result<(), String> {
    if !path.is_absolute() {
        return Err("output path must be absolute".into());
    }
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("cannot serialize contract result: {error}"))?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create new output {}: {error}", path.display()))?;
    file.write_all(&bytes)
        .and_then(|_| file.write_all(b"\n"))
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("cannot durably write output {}: {error}", path.display()))
}

fn run(arguments: Args) -> Result<Value, String> {
    let document = static_contract_document();
    if let Some(path) = arguments.out {
        write_new_json(&path, &document)?;
    }
    Ok(document)
}

fn main() {
    match parse_args().and_then(run) {
        Ok(document) => match serde_json::to_string_pretty(&document) {
            Ok(text) => println!("{text}"),
            Err(error) => {
                eprintln!("cannot serialize result: {error}");
                process::exit(1);
            }
        },
        Err(error) => {
            eprintln!("{error}");
            process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| bf16::from_f32(*value).to_bits().to_le_bytes())
            .collect()
    }

    fn source_index_bytes(entries: &[(&str, &str)]) -> Vec<u8> {
        let weight_map = entries
            .iter()
            .map(|(name, shard)| ((*name).to_owned(), Value::String((*shard).to_owned())))
            .collect::<serde_json::Map<_, _>>();
        serde_json::to_vec(&json!({"metadata": {"total_size": 12}, "weight_map": weight_map}))
            .expect("serialize synthetic source index")
    }

    fn write_synthetic_fixture() -> (TempDir, PathBuf, String, Vec<u16>, Vec<u16>) {
        let directory = TempDir::new().expect("temp directory");
        let root = directory.path();
        let tensor_a = bf16_bytes(&[1.0, -2.0, 3.5, 0.25]);
        let tensor_b = bf16_bytes(&[4.0, 5.0]);
        let header = serde_json::to_vec(&json!({
            "tensor.a": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]},
            "tensor.b": {"dtype": "BF16", "shape": [2], "data_offsets": [8, 12]}
        }))
        .expect("serialize synthetic safetensors header");
        let mut shard = Vec::new();
        shard.extend_from_slice(&(header.len() as u64).to_le_bytes());
        shard.extend_from_slice(&header);
        shard.extend_from_slice(&tensor_a);
        shard.extend_from_slice(&tensor_b);
        let shard_path = root.join("model-00001-of-00001.safetensors");
        fs::write(&shard_path, &shard).expect("write synthetic shard");
        let index = source_index_bytes(&[
            ("tensor.a", "model-00001-of-00001.safetensors"),
            ("tensor.b", "model-00001-of-00001.safetensors"),
        ]);
        fs::write(root.join("model.safetensors.index.json"), &index)
            .expect("write synthetic index");
        let data_start = 8 + header.len() as u64;
        let range_map = SourceRangeMap {
            schema: RANGE_MAP_SCHEMA.into(),
            source_model_id: "synthetic-qwen30-fixture".into(),
            source_revision: "synthetic-only".into(),
            source_tensor_count: 2,
            source_index: SourceIndexBinding {
                relative_path: "model.safetensors.index.json".into(),
                sha256: sha256_hex(&index),
                format: "huggingface.safetensors.index.json".into(),
            },
            maximum_window_bytes: 4,
            shards: vec![ShardRangeRecord {
                shard_id: "shard-0".into(),
                relative_path: "model-00001-of-00001.safetensors".into(),
                bytes: shard.len() as u64,
                sha256: sha256_hex(&shard),
                safetensors_header_sha256: sha256_hex(&header),
            }],
            tensors: vec![
                TensorRangeRecord {
                    tensor_name: "tensor.a".into(),
                    shard_id: "shard-0".into(),
                    dtype: "BF16".into(),
                    shape: vec![2, 2],
                    data_offset: data_start,
                    data_bytes: tensor_a.len() as u64,
                    raw_bf16_sha256: sha256_hex(&tensor_a),
                },
                TensorRangeRecord {
                    tensor_name: "tensor.b".into(),
                    shard_id: "shard-0".into(),
                    dtype: "BF16".into(),
                    shape: vec![2],
                    data_offset: data_start + tensor_a.len() as u64,
                    data_bytes: tensor_b.len() as u64,
                    raw_bf16_sha256: sha256_hex(&tensor_b),
                },
            ],
        };
        let map_bytes = serde_json::to_vec_pretty(&range_map).expect("serialize range map");
        let map_path = root.join("range-map.json");
        fs::write(&map_path, &map_bytes).expect("write range map");
        let a_bits = tensor_a
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect();
        let b_bits = tensor_b
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect();
        (directory, map_path, sha256_hex(&map_bytes), a_bits, b_bits)
    }

    #[test]
    fn static_result_is_prepared_and_does_not_claim_execution() {
        let result = static_contract_document();
        assert_eq!(result["schema"], RESULT_SCHEMA);
        assert_eq!(result["status"], RESULT_STATUS);
        assert_eq!(
            result["execution_boundary"]["source_tensor_payload_opened"],
            false
        );
        assert_eq!(result["execution_boundary"]["gpu_or_metal_invoked"], false);
        assert_eq!(
            result["future_exact_semantics_attestation"]["schema"],
            FUTURE_ATTESTATION_SCHEMA
        );
        assert_eq!(
            result["future_exact_semantics_attestation"]["exact_semantics_required"]
                ["source_bf16_tensor_row_order_and_offsets_verified"],
            true
        );
    }

    #[test]
    fn positioned_reader_validates_index_header_shard_and_bf16_order_without_full_payloads() {
        let (directory, map_path, map_sha, expected_a, expected_b) = write_synthetic_fixture();
        let mut reader = VerifiedBf16RangeReader::open(directory.path(), &map_path, &map_sha)
            .expect("open verified synthetic reader");
        assert_eq!(reader.source_root(), directory.path());
        assert_eq!(reader.range_map_document_sha256(), map_sha);
        assert_eq!(reader.maximum_payload_cache_bytes(), 4);
        assert_eq!(reader.shard_attestations().len(), 1);

        let mut observed = Vec::new();
        let attestation = reader
            .stream_bf16_rows("tensor.a", 0, 2, |index, bits| {
                observed.push((index, bits));
                Ok(())
            })
            .expect("stream full tensor a");
        assert_eq!(
            observed,
            expected_a
                .iter()
                .enumerate()
                .map(|(index, bits)| (index as u64, *bits))
                .collect::<Vec<_>>()
        );
        assert!(attestation.full_tensor_sha256_verified);
        assert!(attestation.bf16_little_endian_row_major_order_verified);
        assert!(reader.maximum_payload_cache_observed_bytes() <= 4);

        let mut tail = Vec::new();
        let partial = reader
            .stream_bf16_elements("tensor.b", 1, 1, |index, bits| {
                tail.push((index, bits));
                Ok(())
            })
            .expect("stream one element of tensor b");
        assert_eq!(tail, vec![(1, expected_b[1])]);
        assert!(!partial.full_tensor_sha256_verified);
    }

    #[test]
    fn reader_rejects_bad_range_map_checksum_and_out_of_range_rows() {
        let (directory, map_path, _map_sha, _expected_a, _expected_b) = write_synthetic_fixture();
        let error = VerifiedBf16RangeReader::open(directory.path(), &map_path, &"0".repeat(64))
            .expect_err("bad range-map binding must refuse");
        assert!(error.contains("checksum"));

        let map_bytes = bounded_metadata_bytes(&map_path, "fixture range map", MAX_RANGE_MAP_BYTES)
            .expect("read fixture range map");
        let map_sha = sha256_hex(&map_bytes);
        let mut reader = VerifiedBf16RangeReader::open(directory.path(), &map_path, &map_sha)
            .expect("open good reader");
        let error = reader
            .stream_bf16_rows("tensor.a", 1, 2, |_index, _bits| Ok(()))
            .expect_err("out-of-range rows must refuse");
        assert!(error.contains("exceeds"));
    }

    #[test]
    fn reader_rejects_source_index_and_safetensors_header_drift() {
        let (directory, map_path, map_sha, _expected_a, _expected_b) = write_synthetic_fixture();
        fs::write(
            directory.path().join("model.safetensors.index.json"),
            b"{\"weight_map\":{}}",
        )
        .expect("mutate synthetic source index");
        let error = VerifiedBf16RangeReader::open(directory.path(), &map_path, &map_sha)
            .expect_err("index checksum drift must refuse");
        assert!(error.contains("index checksum"));

        let (directory, map_path, map_sha, _expected_a, _expected_b) = write_synthetic_fixture();
        let shard_path = directory.path().join("model-00001-of-00001.safetensors");
        let mut shard = fs::read(&shard_path).expect("read synthetic shard for corruption");
        shard[8] ^= 1;
        fs::write(&shard_path, shard).expect("corrupt synthetic safetensors header");
        let error = VerifiedBf16RangeReader::open(directory.path(), &map_path, &map_sha)
            .expect_err("shard/header drift must refuse");
        assert!(error.contains("checksum"));
    }

    #[derive(Default)]
    struct RecordingTraceReplay {
        invocations: Vec<TraceReplayInvocation>,
    }

    impl ExactTraceReplay for RecordingTraceReplay {
        fn replay(&mut self, invocation: &TraceReplayInvocation) -> Result<(), String> {
            self.invocations.push(invocation.clone());
            Ok(())
        }
    }

    #[test]
    fn exact_trace_has_369_prefix_forwards_then_forced_949_semantic_slot() {
        let trace = exact_trace();
        assert_eq!(trace.len(), 370);
        assert_eq!(trace[0].phase, TracePhase::SourceTemplatePrefix);
        assert_eq!(trace[368].phase, TracePhase::SourceTemplatePrefix);
        assert_eq!(trace[369].phase, TracePhase::ForcedIdenticalContinuation);
        assert_eq!(trace[369].token_position, 369);

        let mut replay = RecordingTraceReplay::default();
        replay_exact_trace(&mut replay).expect("replay static semantic event sequence");
        let expected_per_forward = 1
            + LAYER_COUNT
                * (layer_operator_sequence().len()
                    + TOP_K * selected_expert_operator_sequence().len())
            + 3;
        assert_eq!(
            replay.invocations.len(),
            TRACE_FORWARD_COUNT * expected_per_forward
        );
        let last = replay.invocations.last().expect("last event");
        assert_eq!(last.forward_index, 369);
        assert_eq!(last.phase, TracePhase::ForcedIdenticalContinuation);
        assert_eq!(last.operator, SourceOperator::RetainFullF32EndpointLogits);
        assert!(last
            .boundaries
            .contains(&AccumulationBoundary::FullF32LogitRetention));
        let selected_slots = replay
            .invocations
            .iter()
            .filter_map(|invocation| {
                (invocation.forward_index == 0
                    && invocation.layer_index == Some(0)
                    && invocation.operator == SourceOperator::SelectedExpertGateUpSerialK)
                    .then_some(invocation.route_slot)
            })
            .collect::<Vec<_>>();
        assert_eq!(selected_slots, (0..TOP_K).map(Some).collect::<Vec<_>>());
    }

    #[derive(Default)]
    struct FixtureAccumulator {
        expected_k: u64,
        next_k: u64,
        sum: f64,
        finished: bool,
    }

    impl OrderedBf16Accumulator for FixtureAccumulator {
        fn begin_linear(
            &mut self,
            _tensor: &TensorRangeRecord,
            expected_k: u64,
        ) -> Result<(), String> {
            self.expected_k = expected_k;
            self.next_k = 0;
            self.sum = 0.0;
            self.finished = false;
            Ok(())
        }

        fn push_bf16_pair(
            &mut self,
            k: u64,
            activation_bits: u16,
            weight_bits: u16,
        ) -> Result<(), String> {
            if k != self.next_k {
                return Err("fixture accumulator observed a reordered K term".into());
            }
            self.sum +=
                bf16::from_bits(activation_bits).to_f64() * bf16::from_bits(weight_bits).to_f64();
            self.next_k += 1;
            Ok(())
        }

        fn finish_linear(&mut self) -> Result<(), String> {
            if self.next_k != self.expected_k {
                return Err("fixture accumulator did not see all K terms".into());
            }
            self.finished = true;
            Ok(())
        }
    }

    #[test]
    fn accumulator_interface_receives_raw_bf16_pairs_in_increasing_k_order() {
        let (directory, map_path, map_sha, _expected_a, _expected_b) = write_synthetic_fixture();
        let mut reader = VerifiedBf16RangeReader::open(directory.path(), &map_path, &map_sha)
            .expect("open reader");
        let activation = [
            bf16::from_f32(2.0).to_bits(),
            bf16::from_f32(1.0).to_bits(),
            bf16::from_f32(-1.0).to_bits(),
            bf16::from_f32(4.0).to_bits(),
        ];
        let mut accumulator = FixtureAccumulator::default();
        let result =
            replay_one_bf16_row_dot(&mut reader, "tensor.a", &activation, &mut accumulator)
                .expect("replay synthetic raw BF16 dot row");
        assert!(result.full_tensor_sha256_verified);
        assert!(accumulator.finished);
        // 2*1 + 1*(-2) + (-1)*3.5 + 4*0.25 = -2.5.
        assert_eq!(accumulator.sum, -2.5);
    }

    #[test]
    fn cli_output_is_create_new_and_absolute_only() {
        assert!(parse_args_from(vec!["--out".into(), "relative.json".into()]).is_err());
        let directory = TempDir::new().expect("temp directory");
        let output = directory.path().join("contract.json");
        let result = run(Args {
            out: Some(output.clone()),
        })
        .expect("write create-new static result");
        assert_eq!(result["status"], RESULT_STATUS);
        assert!(output.is_file());
        assert!(run(Args { out: Some(output) }).is_err());
    }
}
