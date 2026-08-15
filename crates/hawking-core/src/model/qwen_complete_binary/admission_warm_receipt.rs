//! Warm payload-admission receipt for complete-binary catalogs.
//!
//! A cold admission still fully re-hashes every payload against the sealed
//! manifest. On success it writes a process-local receipt keyed by
//! `(manifest_seal, per-file size, mtime_ns, inode)`. device (st_dev) is
//! recorded for diagnostics but NOT part of the match key: it is a mount-time
//! artifact that a remount reassigns without changing the file.
//!
//! A later process may skip *recomputing* content SHA-256 only when every
//! payload file's identity metadata still matches the receipt. It never skips
//! checking that the files are unchanged. A single metadata mismatch forces a
//! full cold re-verify of the whole catalog.
//!
//! Proven on warm hit:
//!   - manifest seal matches the protected admission binding
//!   - source-chain seals still bind
//!   - every payload path exists as a regular file with the same size,
//!     mtime_ns, and inode recorded at the last cold verify (device excluded)
//!   - payload byte length equals the receipt size after read
//!   - direct header geometry still matches the manifest row
//!
//! Assumed on warm hit (not re-proven until cold rehash):
//!   - content bits are unchanged if identity metadata is unchanged
//!     (a metadata-preserving bitflip is outside this gate)

use super::{
    canonical_json, is_sha256, model_error, parse_complete_binary_header, read_regular_file,
    required_sha256, required_string, required_u64, sha256_hex, verify_sealed_document,
    CompleteBinaryHeader, CompleteBinaryTensor, SourceChain,
};
use crate::{Error, Result};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

pub const ADMISSION_WARM_RECEIPT_SCHEMA: &str =
    "hawking.ascension.complete_binary_payload_admission_receipt.v1";
pub const ADMISSION_WARM_RECEIPT_VERSION: u32 = 1;

/// Disable with `HAWKING_ADMISSION_WARM_RECEIPT=0`. Default: enabled.
pub fn warm_receipt_enabled() -> bool {
    match std::env::var("HAWKING_ADMISSION_WARM_RECEIPT") {
        Ok(v) if matches!(v.as_str(), "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF") => {
            false
        }
        _ => true,
    }
}

fn cache_root() -> PathBuf {
    if let Ok(dir) = std::env::var("HAWKING_ADMISSION_RECEIPT_DIR") {
        return PathBuf::from(dir);
    }
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        return PathBuf::from(xdg)
            .join("hawking")
            .join("complete_binary_admission_receipts");
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home)
            .join(".cache")
            .join("hawking")
            .join("complete_binary_admission_receipts");
    }
    PathBuf::from("/tmp/hawking-cache/complete_binary_admission_receipts")
}

pub fn receipt_path_for_seal(manifest_seal_sha256: &str) -> PathBuf {
    cache_root().join(format!("{manifest_seal_sha256}.json"))
}

#[derive(Clone, Debug)]
pub struct FileIdentity {
    pub size: u64,
    pub mtime_ns: i128,
    pub device: u64,
    pub inode: u64,
}

pub fn file_identity(path: &Path, label: &str) -> Result<FileIdentity> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| model_error(label, format!("cannot stat {}: {error}", path.display())))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(model_error(
            label,
            format!("{} must be a regular non-symlink file", path.display()),
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let mtime_ns =
            i128::from(metadata.mtime()) * 1_000_000_000i128 + i128::from(metadata.mtime_nsec());
        Ok(FileIdentity {
            size: metadata.len(),
            mtime_ns,
            device: metadata.dev(),
            inode: metadata.ino(),
        })
    }
    #[cfg(not(unix))]
    {
        let mtime_ns = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| {
                i128::from(d.as_secs()) * 1_000_000_000i128 + i128::from(d.subsec_nanos())
            })
            .unwrap_or(0);
        Ok(FileIdentity {
            size: metadata.len(),
            mtime_ns,
            device: 0,
            inode: 0,
        })
    }
}

fn identity_matches(observed: &FileIdentity, expected: &FileIdentity) -> bool {
    // device (st_dev) is deliberately NOT compared: it is a mount-time artifact,
    // so a reboot/remount reassigns it without touching the file and would
    // falsely invalidate the warm receipt, forcing a full ~10s cold rehash on
    // every post-reboot startup. (size, mtime_ns, inode) already pin the file on
    // a volume, and the cold verify's sealed content SHA is the ultimate
    // authority; device is still recorded in the receipt for diagnostics.
    observed.size == expected.size
        && observed.mtime_ns == expected.mtime_ns
        && observed.inode == expected.inode
}

/// Layout marker stored in the warm receipt.
///
/// - `hq30g1b1`: direct complete-binary payload; receipt may carry a parsed header.
/// - `hgravs01`: activation-weighted SVD factor payload; warm path re-parses geometry
///   from bytes + manifest and never assumes header equality from the receipt alone.
/// - `identity_only`: identity+sha catalog entry without a stored direct header
///   (used by mixed HGRAVS catalogs where layout is re-proven on every load).
pub const LAYOUT_KIND_HQ30G1B1: &str = "hq30g1b1";
pub const LAYOUT_KIND_HGRAVS01: &str = "hgravs01";
pub const LAYOUT_KIND_IDENTITY_ONLY: &str = "identity_only";

#[derive(Clone, Debug)]
pub struct ReceiptEntry {
    pub tensor_name: String,
    pub artifact_path: PathBuf,
    pub artifact_sha256: String,
    pub identity: FileIdentity,
    /// Present for direct complete-binary warm loads that compare header equality.
    /// Absent for mixed / HGRAVS catalogs that re-parse layout on every start.
    pub header: Option<CompleteBinaryHeader>,
    pub layout_kind: String,
    pub source_shard: String,
    pub source_shard_sha256: String,
    pub source_dtype: String,
}

/// One catalog row for building a warm receipt without depending on a specific
/// tensor struct type (complete-binary or activation-weighted).
#[derive(Clone, Debug)]
pub struct ReceiptEntrySpec {
    pub tensor_name: String,
    pub artifact_path: PathBuf,
    pub artifact_sha256: String,
    pub source_shard: String,
    pub source_shard_sha256: String,
    pub source_dtype: String,
    pub header: Option<CompleteBinaryHeader>,
    pub layout_kind: String,
}

#[derive(Clone, Debug)]
pub struct WarmAdmissionReceipt {
    pub manifest_seal_sha256: String,
    pub manifest_path: PathBuf,
    pub catalog_tensor_count: usize,
    pub entries: BTreeMap<String, ReceiptEntry>,
    pub sealed_at_unix_ms: u64,
}

fn header_to_json(header: &CompleteBinaryHeader) -> Value {
    json!({
        "version": header.version,
        "group_size": header.group_size,
        "shape": header.shape,
        "elements": header.elements,
        "groups": header.groups,
        "scale_offset": header.scale_offset,
        "sign_offset": header.sign_offset,
        "payload_bytes": header.payload_bytes,
    })
}

fn header_from_json(value: &Value, label: &str) -> Result<CompleteBinaryHeader> {
    let object = value
        .as_object()
        .ok_or_else(|| model_error(label, "header must be an object"))?;
    let shape = object
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| model_error(label, "header.shape must be an array"))?
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| model_error(label, "header.shape entry must be usize"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(CompleteBinaryHeader {
        version: required_u64(object, "version", label)? as u32,
        group_size: required_u64(object, "group_size", label)? as usize,
        shape,
        elements: required_u64(object, "elements", label)? as usize,
        groups: required_u64(object, "groups", label)? as usize,
        scale_offset: required_u64(object, "scale_offset", label)? as usize,
        sign_offset: required_u64(object, "sign_offset", label)? as usize,
        payload_bytes: required_u64(object, "payload_bytes", label)? as usize,
    })
}

fn required_i128_field(object: &Map<String, Value>, key: &str, label: &str) -> Result<i128> {
    object
        .get(key)
        .and_then(|value| {
            value
                .as_i64()
                .map(i128::from)
                .or_else(|| value.as_u64().map(i128::from))
                .or_else(|| {
                    value.as_str().and_then(|s| s.parse::<i128>().ok())
                })
        })
        .ok_or_else(|| model_error(label, format!("missing signed integer field {key:?}")))
}

fn receipt_document_unsigned(receipt: &WarmAdmissionReceipt) -> Value {
    let mut entries = Vec::with_capacity(receipt.entries.len());
    for (name, entry) in &receipt.entries {
        entries.push(json!({
            "tensor_name": name,
            "artifact_path": entry.artifact_path.to_string_lossy(),
            "artifact_sha256": entry.artifact_sha256,
            "size": entry.identity.size,
            // String form keeps i128 mtime exact under JSON numbers.
            "mtime_ns": entry.identity.mtime_ns.to_string(),
            "device": entry.identity.device,
            "inode": entry.identity.inode,
            "source_shard": entry.source_shard,
            "source_shard_sha256": entry.source_shard_sha256,
            "source_dtype": entry.source_dtype,
            "layout_kind": entry.layout_kind,
            "header": entry.header.as_ref().map(header_to_json).unwrap_or(Value::Null),
        }));
    }
    json!({
        "schema": ADMISSION_WARM_RECEIPT_SCHEMA,
        "version": ADMISSION_WARM_RECEIPT_VERSION,
        "manifest_seal_sha256": receipt.manifest_seal_sha256,
        "manifest_path": receipt.manifest_path.to_string_lossy(),
        "catalog_tensor_count": receipt.catalog_tensor_count,
        "sealed_at_unix_ms": receipt.sealed_at_unix_ms,
        "entries": entries,
    })
}

pub fn build_receipt_from_admitted(
    manifest_path: &Path,
    manifest_seal_sha256: &str,
    tensors: &BTreeMap<String, CompleteBinaryTensor>,
) -> Result<WarmAdmissionReceipt> {
    let specs: Vec<ReceiptEntrySpec> = tensors
        .iter()
        .map(|(name, tensor)| ReceiptEntrySpec {
            tensor_name: name.clone(),
            artifact_path: tensor.artifact_path.clone(),
            artifact_sha256: tensor.artifact_sha256.clone(),
            source_shard: tensor.source_shard.clone(),
            source_shard_sha256: tensor.source_shard_sha256.clone(),
            source_dtype: tensor.source_dtype.clone(),
            header: Some(tensor.header.clone()),
            layout_kind: LAYOUT_KIND_HQ30G1B1.to_owned(),
        })
        .collect();
    build_receipt_from_specs(manifest_path, manifest_seal_sha256, &specs)
}

/// Build a warm receipt from generic per-tensor identity specs.
/// Used by both complete-binary and activation-weighted (HGRAVS) admission paths
/// so they share one caching design.
pub fn build_receipt_from_specs(
    manifest_path: &Path,
    manifest_seal_sha256: &str,
    specs: &[ReceiptEntrySpec],
) -> Result<WarmAdmissionReceipt> {
    let mut entries = BTreeMap::new();
    for spec in specs {
        let identity = file_identity(&spec.artifact_path, "warm admission receipt seal")?;
        entries.insert(
            spec.tensor_name.clone(),
            ReceiptEntry {
                tensor_name: spec.tensor_name.clone(),
                artifact_path: spec.artifact_path.clone(),
                artifact_sha256: spec.artifact_sha256.clone(),
                identity,
                header: spec.header.clone(),
                layout_kind: if spec.layout_kind.is_empty() {
                    LAYOUT_KIND_IDENTITY_ONLY.to_owned()
                } else {
                    spec.layout_kind.clone()
                },
                source_shard: spec.source_shard.clone(),
                source_shard_sha256: spec.source_shard_sha256.clone(),
                source_dtype: spec.source_dtype.clone(),
            },
        );
    }
    let sealed_at_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    Ok(WarmAdmissionReceipt {
        manifest_seal_sha256: manifest_seal_sha256.to_owned(),
        manifest_path: manifest_path.to_path_buf(),
        catalog_tensor_count: specs.len(),
        entries,
        sealed_at_unix_ms,
    })
}

pub fn write_receipt(receipt: &WarmAdmissionReceipt) -> Result<PathBuf> {
    if !is_sha256(&receipt.manifest_seal_sha256) {
        return Err(model_error(
            "warm admission receipt",
            "manifest seal must be lowercase sha256",
        ));
    }
    let path = receipt_path_for_seal(&receipt.manifest_seal_sha256);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            model_error(
                "warm admission receipt",
                format!("cannot create receipt dir {}: {e}", parent.display()),
            )
        })?;
    }
    let mut unsigned = receipt_document_unsigned(receipt);
    let seal = sha256_hex(&canonical_json(&unsigned)?);
    if let Some(obj) = unsigned.as_object_mut() {
        obj.insert("seal_sha256".into(), Value::String(seal));
    }
    let bytes = serde_json::to_vec_pretty(&unsigned)
        .map_err(|e| model_error("warm admission receipt", e.to_string()))?;
    let tmp = path.with_extension("json.tmp");
    {
        let mut file = fs::File::create(&tmp).map_err(|e| {
            model_error(
                "warm admission receipt",
                format!("cannot create {}: {e}", tmp.display()),
            )
        })?;
        file.write_all(&bytes).map_err(|e| {
            model_error(
                "warm admission receipt",
                format!("cannot write {}: {e}", tmp.display()),
            )
        })?;
        file.sync_all().ok();
    }
    fs::rename(&tmp, &path).map_err(|e| {
        model_error(
            "warm admission receipt",
            format!("cannot publish {}: {e}", path.display()),
        )
    })?;
    Ok(path)
}

pub fn load_receipt(manifest_seal_sha256: &str) -> Result<Option<WarmAdmissionReceipt>> {
    let path = receipt_path_for_seal(manifest_seal_sha256);
    if !path.is_file() {
        return Ok(None);
    }
    let raw = fs::read(&path).map_err(|e| {
        model_error(
            "warm admission receipt",
            format!("cannot read {}: {e}", path.display()),
        )
    })?;
    let value: Value = serde_json::from_slice(&raw).map_err(|e| {
        model_error(
            "warm admission receipt",
            format!("cannot parse {}: {e}", path.display()),
        )
    })?;
    let object = value
        .as_object()
        .ok_or_else(|| model_error("warm admission receipt", "root must be an object"))?;
    if object.get("schema").and_then(Value::as_str) != Some(ADMISSION_WARM_RECEIPT_SCHEMA) {
        return Ok(None);
    }
    if object.get("version").and_then(Value::as_u64) != Some(u64::from(ADMISSION_WARM_RECEIPT_VERSION))
    {
        return Ok(None);
    }
    // Corrupt / rewritten receipt: ignore and force cold re-verify.
    if verify_sealed_document(&value, "warm admission receipt").is_err() {
        return Ok(None);
    }
    let manifest_seal = required_sha256(object, "manifest_seal_sha256", "warm admission receipt")?;
    if manifest_seal != manifest_seal_sha256 {
        return Ok(None);
    }
    let manifest_path = PathBuf::from(required_string(
        object,
        "manifest_path",
        "warm admission receipt",
    )?);
    let catalog_tensor_count =
        required_u64(object, "catalog_tensor_count", "warm admission receipt")? as usize;
    let sealed_at_unix_ms =
        required_u64(object, "sealed_at_unix_ms", "warm admission receipt").unwrap_or(0);
    let entries_val = object
        .get("entries")
        .and_then(Value::as_array)
        .ok_or_else(|| model_error("warm admission receipt", "entries must be an array"))?;
    let mut entries = BTreeMap::new();
    for entry in entries_val {
        let row = entry
            .as_object()
            .ok_or_else(|| model_error("warm admission receipt", "entry must be an object"))?;
        let tensor_name = required_string(row, "tensor_name", "warm admission receipt")?.to_owned();
        let artifact_path =
            PathBuf::from(required_string(row, "artifact_path", "warm admission receipt")?);
        let artifact_sha256 =
            required_sha256(row, "artifact_sha256", "warm admission receipt")?.to_owned();
        let identity = FileIdentity {
            size: required_u64(row, "size", "warm admission receipt")?,
            mtime_ns: required_i128_field(row, "mtime_ns", "warm admission receipt")?,
            device: required_u64(row, "device", "warm admission receipt")?,
            inode: required_u64(row, "inode", "warm admission receipt")?,
        };
        let header = match row.get("header") {
            None | Some(Value::Null) => None,
            Some(value) => Some(header_from_json(value, "warm admission receipt header")?),
        };
        let layout_kind = row
            .get("layout_kind")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| {
                if header.is_some() {
                    LAYOUT_KIND_HQ30G1B1.to_owned()
                } else {
                    LAYOUT_KIND_IDENTITY_ONLY.to_owned()
                }
            });
        let source_shard =
            required_string(row, "source_shard", "warm admission receipt")?.to_owned();
        let source_shard_sha256 =
            required_sha256(row, "source_shard_sha256", "warm admission receipt")?.to_owned();
        let source_dtype =
            required_string(row, "source_dtype", "warm admission receipt")?.to_owned();
        entries.insert(
            tensor_name.clone(),
            ReceiptEntry {
                tensor_name,
                artifact_path,
                artifact_sha256,
                identity,
                header,
                layout_kind,
                source_shard,
                source_shard_sha256,
                source_dtype,
            },
        );
    }
    if entries.len() != catalog_tensor_count {
        return Ok(None);
    }
    Ok(Some(WarmAdmissionReceipt {
        manifest_seal_sha256: manifest_seal,
        manifest_path,
        catalog_tensor_count,
        entries,
        sealed_at_unix_ms,
    }))
}

/// Re-check identity metadata for every receipt entry without reading content.
/// Returns Ok(true) only when every file still matches.
pub fn receipt_identities_still_match(receipt: &WarmAdmissionReceipt) -> Result<bool> {
    for entry in receipt.entries.values() {
        let observed = match file_identity(&entry.artifact_path, "warm admission identity check") {
            Ok(id) => id,
            Err(_) => return Ok(false),
        };
        if !identity_matches(&observed, &entry.identity) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Load payload bytes under a warm hit: identity already checked, skip content hash.
/// Shared by complete-binary and HGRAVS paths. Does not parse layout geometry.
pub fn load_payload_bytes_warm_skip_hash(entry: &ReceiptEntry) -> Result<Arc<[u8]>> {
    // Re-stat immediately before read so a race still fails closed to mismatch.
    let observed = file_identity(&entry.artifact_path, "warm admission payload load")?;
    if !identity_matches(&observed, &entry.identity) {
        return Err(Error::Model(format!(
            "warm admission identity race for {:?}: file changed after receipt match",
            entry.tensor_name
        )));
    }
    let payload = read_regular_file(&entry.artifact_path, "warm admission payload load")?;
    if u64::try_from(payload.len()).ok() != Some(entry.identity.size) {
        return Err(Error::Model(format!(
            "warm admission payload {:?} length {} != receipt size {}",
            entry.tensor_name,
            payload.len(),
            entry.identity.size
        )));
    }
    Ok(Arc::from(payload))
}

/// Load one direct complete-binary payload under a warm hit: identity already
/// checked, skip content hash, re-parse and compare stored header.
pub fn load_payload_warm_skip_hash(
    entry: &ReceiptEntry,
) -> Result<(CompleteBinaryTensor, Arc<[u8]>)> {
    let expected_header = entry.header.as_ref().ok_or_else(|| {
        Error::Model(format!(
            "warm admission direct load for {:?} missing stored header (layout_kind={})",
            entry.tensor_name, entry.layout_kind
        ))
    })?;
    let payload = load_payload_bytes_warm_skip_hash(entry)?;
    let header = parse_complete_binary_header(payload.as_ref())?;
    if &header != expected_header {
        return Err(Error::Model(format!(
            "warm admission payload {:?} header diverged from receipt without identity change",
            entry.tensor_name
        )));
    }
    Ok((
        CompleteBinaryTensor {
            tensor_name: entry.tensor_name.clone(),
            source_shard: entry.source_shard.clone(),
            source_shard_sha256: entry.source_shard_sha256.clone(),
            source_dtype: entry.source_dtype.clone(),
            artifact_path: entry.artifact_path.clone(),
            artifact_sha256: entry.artifact_sha256.clone(),
            header,
        },
        payload,
    ))
}

/// Confirm a warm receipt still names the same catalog as the live manifest rows.
pub fn receipt_covers_manifest_rows(
    receipt: &WarmAdmissionReceipt,
    rows: &[Value],
    _root: &Path,
    source: &SourceChain,
) -> Result<bool> {
    if rows.len() != receipt.catalog_tensor_count || rows.len() != receipt.entries.len() {
        return Ok(false);
    }
    if receipt.entries.keys().ne(source.weight_map.keys()) {
        return Ok(false);
    }
    for value in rows {
        let row = match value.as_object() {
            Some(r) => r,
            None => return Ok(false),
        };
        let name = match required_string(row, "tensor_name", "warm admission catalog check") {
            Ok(n) => n,
            Err(_) => return Ok(false),
        };
        let entry = match receipt.entries.get(name) {
            Some(e) => e,
            None => return Ok(false),
        };
        let expected_sha =
            match required_sha256(row, "artifact_sha256", "warm admission catalog check") {
                Ok(s) => s,
                Err(_) => return Ok(false),
            };
        if entry.artifact_sha256 != expected_sha {
            return Ok(false);
        }
        // Deterministic filename binding: tensors/<sha256(name)>.hq30g
        let filename = format!("{}.hq30g", sha256_hex(name.as_bytes()));
        if entry.artifact_path.file_name().and_then(|s| s.to_str()) != Some(filename.as_str()) {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::{identity_matches, FileIdentity};

    fn id(size: u64, mtime_ns: i128, device: u64, inode: u64) -> FileIdentity {
        FileIdentity { size, mtime_ns, device, inode }
    }

    #[test]
    fn device_shift_alone_still_matches() {
        // A remount reassigns st_dev without touching the file: same file,
        // must remain a warm hit (else every reboot pays a full cold rehash).
        let base = id(4_000_000_000, 1_786_064_136_014_875_545, 16_777_234, 197_011_751);
        let remounted = id(4_000_000_000, 1_786_064_136_014_875_545, 16_777_233, 197_011_751);
        assert!(identity_matches(&remounted, &base));
    }

    #[test]
    fn content_change_signals_still_invalidate() {
        let base = id(4_000_000_000, 1_786_064_136_014_875_545, 16_777_234, 197_011_751);
        // A rewrite bumps mtime and/or size; a swap changes inode. Each must
        // still force a cold re-verify.
        assert!(!identity_matches(&id(4_000_000_001, base.mtime_ns, base.device, base.inode), &base));
        assert!(!identity_matches(&id(base.size, base.mtime_ns + 1, base.device, base.inode), &base));
        assert!(!identity_matches(&id(base.size, base.mtime_ns, base.device, base.inode + 1), &base));
    }
}
