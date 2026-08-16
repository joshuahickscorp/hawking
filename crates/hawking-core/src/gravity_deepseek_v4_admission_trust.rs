//! Sealed admission-time trust for the DeepSeek-V4-Flash content-addressed stream.
//!
//! A chunk's SHA-256 **is** its filename. The sealed artifact is immutable
//! and was verified end-to-end at restoration; re-deriving every digest on
//! the token critical path re-proves a receipt that already exists.
//!
//! This module writes that receipt once (full SHA-256, parallel) and lets a
//! later reader skip hashing a chunk whose cheap identity (byte length,
//! mtime_ns, inode) still matches. It never skips hashing when the receipt
//! is missing, its own digest fails, or any cheap invariant disagrees.
//!
//! Threat model (honest, not overclaimed):
//!
//! Covers:
//! - replace, truncate, or delete of a chunk (size / mtime / inode change)
//! - a rewritten receipt whose table digest or document seal no longer match
//! - a stale receipt whose artifact-level digest, chunk count, or total
//!   bytes disagree with the admitted manifest
//!
//! Does not cover:
//! - an in-place bit flip that preserves size, mtime_ns, **and** inode
//!   (bit-rot that does not update timestamps; a write that restores mtime)
//! - a forged receipt written by a process that can already write
//!   `.hawking-admission.json` (the receipt is tamper-evident, not secretly
//!   signed)
//!
//! `HAWKING_DSV4F_VERIFY=full` closes the residual window by hashing every
//! chunk exactly as before this path existed.

use crate::gravity_deepseek_v4::{
    canonical_json, canonical_non_symlink_directory, checked_regular_path, gravity, is_sha256,
    map_chunk_readonly, sha256_hex,
};
use crate::{Error, Result};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Schema bound into every admission-trust receipt.
pub const ADMISSION_TRUST_SCHEMA: &str = "hawking.gravity.deepseek_v4.admission_trust.v1";
/// Receipt filename stored beside the sealed artifact.
pub const ADMISSION_TRUST_RECEIPT_NAME: &str = ".hawking-admission.json";

/// How the reader authenticates a chunk before returning its bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekV4VerifyMode {
    /// Today's behaviour: SHA-256 every first-touch chunk.
    Full,
    /// Skip SHA-256 when a valid admission receipt's cheap invariants hold.
    Admission,
}

impl DeepSeekV4VerifyMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Admission => "admission",
        }
    }

    pub fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "full" => Ok(Self::Full),
            "admission" => Ok(Self::Admission),
            other => Err(gravity(format!(
                "HAWKING_DSV4F_VERIFY={other:?} is not 'full' or 'admission'"
            ))),
        }
    }

    /// Default is `admission`. A missing receipt still hashes (never skip
    /// without a valid seal). Unknown values are a hard error, not a silent
    /// fallback.
    pub fn from_env() -> Result<Self> {
        match std::env::var("HAWKING_DSV4F_VERIFY") {
            Err(std::env::VarError::NotPresent) => Ok(Self::Admission),
            Err(error) => Err(gravity(format!("HAWKING_DSV4F_VERIFY: {error}"))),
            Ok(value) if value.is_empty() => Ok(Self::Admission),
            Ok(value) => Self::parse(&value),
        }
    }
}

/// Cheap identity recorded for one content-addressed chunk file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4ChunkFileIdentity {
    pub key: String,
    pub bytes: u64,
    pub mtime_ns: i128,
    pub inode: u64,
}

/// In-memory index of a loaded, seal-checked admission receipt.
#[derive(Debug, Clone)]
pub struct DeepSeekV4AdmissionTrustIndex {
    pub artifact_root: PathBuf,
    pub content_addressed_chunk_sha256: String,
    pub manifest_seal_sha256: String,
    pub chunk_count: usize,
    pub total_bytes: u64,
    pub chunks: BTreeMap<String, DeepSeekV4ChunkFileIdentity>,
    pub table_sha256: String,
    pub seal_sha256: String,
    pub sealed_at_unix_ms: u64,
    pub verifier_version: String,
}

/// Why a receipt on disk was not used.
#[derive(Debug, Clone)]
pub enum DeepSeekV4AdmissionLoad {
    Missing,
    Rejected(String),
    Loaded(DeepSeekV4AdmissionTrustIndex),
}

/// Result of a full parallel admission pass that writes the receipt.
#[derive(Debug, Clone)]
pub struct DeepSeekV4AdmissionTrustSeal {
    pub path: PathBuf,
    pub wall_ms: u128,
    pub hash_wall_ms: u128,
    pub chunk_count: usize,
    pub total_bytes: u64,
    pub bytes_hashed: u64,
    pub threads: usize,
    pub table_sha256: String,
    pub seal_sha256: String,
    pub content_addressed_chunk_sha256: String,
    pub manifest_seal_sha256: String,
    pub verifier_version: String,
    pub identities: BTreeMap<String, DeepSeekV4ChunkFileIdentity>,
    pub index_path: Option<PathBuf>,
    pub index_bytes: Option<u64>,
    pub index_wall_ms: Option<u128>,
}

/// One chunk the sealer must hash and identify.
#[derive(Debug, Clone)]
pub struct DeepSeekV4AdmissionChunkSpec {
    pub relative: String,
    pub sha256: String,
    pub bytes: u64,
}

pub fn admission_receipt_path(root: impl AsRef<Path>) -> PathBuf {
    root.as_ref().join(ADMISSION_TRUST_RECEIPT_NAME)
}

/// Directory for receipts when the artifact itself is not writable
/// (this host's Downloads tree is TCC-locked).
pub fn admission_receipt_cache_root() -> PathBuf {
    if let Ok(dir) = std::env::var("HAWKING_DSV4F_ADMISSION_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        return PathBuf::from(xdg).join("hawking").join("dsv4f-admission");
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home)
            .join(".cache")
            .join("hawking")
            .join("dsv4f-admission");
    }
    PathBuf::from("/tmp/hawking-cache/dsv4f-admission")
}

pub fn admission_receipt_cache_path(manifest_seal_sha256: &str) -> PathBuf {
    admission_receipt_cache_root().join(format!("{manifest_seal_sha256}.json"))
}

fn explicit_receipt_path() -> Option<PathBuf> {
    std::env::var("HAWKING_DSV4F_ADMISSION_RECEIPT")
        .ok()
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn receipt_search_paths(root: impl AsRef<Path>, manifest_seal: &str) -> Vec<PathBuf> {
    let mut paths = vec![admission_receipt_path(root)];
    if let Some(explicit) = explicit_receipt_path() {
        paths.push(explicit);
    }
    paths.push(admission_receipt_cache_path(manifest_seal));
    paths
}

/// Worker count for the one-time admission hash. Override with
/// `HAWKING_DSV4F_ADMIT_THREADS`.
pub fn admission_hash_threads() -> usize {
    if let Ok(value) = std::env::var("HAWKING_DSV4F_ADMIT_THREADS") {
        if let Ok(n) = value.parse::<usize>() {
            return n.max(1);
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

pub fn file_identity(path: &Path, label: &str) -> Result<DeepSeekV4ChunkFileIdentity> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| gravity(format!("cannot stat {label} {}: {error}", path.display())))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(gravity(format!(
            "{label} must be a regular non-symlink file: {}",
            path.display()
        )));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let mtime_ns =
            i128::from(metadata.mtime()) * 1_000_000_000i128 + i128::from(metadata.mtime_nsec());
        Ok(DeepSeekV4ChunkFileIdentity {
            key: String::new(),
            bytes: metadata.len(),
            mtime_ns,
            inode: metadata.ino(),
        })
    }
    #[cfg(not(unix))]
    {
        let mtime_ns = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| i128::from(d.as_secs()) * 1_000_000_000i128 + i128::from(d.subsec_nanos()))
            .unwrap_or(0);
        Ok(DeepSeekV4ChunkFileIdentity {
            key: String::new(),
            bytes: metadata.len(),
            mtime_ns,
            inode: 0,
        })
    }
}

pub fn identity_matches(
    observed: &DeepSeekV4ChunkFileIdentity,
    expected: &DeepSeekV4ChunkFileIdentity,
) -> bool {
    // device (st_dev) is deliberately not compared: a remount reassigns it
    // without touching the file. (size, mtime_ns, inode) pin the file on a
    // volume; SHA-256 remains the authority when any of those disagree.
    observed.bytes == expected.bytes
        && observed.mtime_ns == expected.mtime_ns
        && observed.inode == expected.inode
}

fn threat_model_json() -> Value {
    json!({
        "covers": [
            "chunk replace, truncate, or delete (size, mtime_ns, or inode change)",
            "rewritten receipt whose table_sha256 or seal_sha256 no longer match",
            "stale receipt whose content_addressed_chunk_sha256, chunk_count, or total_bytes disagree with the admitted manifest",
        ],
        "does_not_cover": [
            "in-place bit flip that preserves size, mtime_ns, and inode",
            "forged receipt written by a process with write access to .hawking-admission.json",
        ],
        "closes_residual_with": "HAWKING_DSV4F_VERIFY=full",
    })
}

fn table_json(chunks: &BTreeMap<String, DeepSeekV4ChunkFileIdentity>) -> Value {
    Value::Array(
        chunks
            .values()
            .map(|chunk| {
                json!({
                    "key": chunk.key,
                    "bytes": chunk.bytes,
                    "mtime_ns": chunk.mtime_ns.to_string(),
                    "inode": chunk.inode,
                })
            })
            .collect(),
    )
}

fn unsigned_receipt_document(
    artifact_root: &Path,
    manifest_seal_sha256: &str,
    content_addressed_chunk_sha256: &str,
    chunks: &BTreeMap<String, DeepSeekV4ChunkFileIdentity>,
    table_sha256: &str,
    sealed_at_unix_ms: u64,
    verifier_version: &str,
) -> Value {
    json!({
        "schema": ADMISSION_TRUST_SCHEMA,
        "artifact_root": artifact_root.to_string_lossy(),
        "manifest_seal_sha256": manifest_seal_sha256,
        "content_addressed_chunk_sha256": content_addressed_chunk_sha256,
        "chunk_count": chunks.len(),
        "total_bytes": chunks.values().map(|c| c.bytes).fold(0u64, |a, b| a.saturating_add(b)),
        "chunks": table_json(chunks),
        "table_sha256": table_sha256,
        "sealed_at_unix_ms": sealed_at_unix_ms,
        "verifier_version": verifier_version,
        "threat_model": threat_model_json(),
    })
}

fn parse_i128_field(value: &Value, label: &str) -> Result<i128> {
    value
        .as_i64()
        .map(i128::from)
        .or_else(|| value.as_u64().map(i128::from))
        .or_else(|| value.as_str().and_then(|s| s.parse::<i128>().ok()))
        .ok_or_else(|| gravity(format!("{label} must be an integer or integer string")))
}

fn parse_chunk_row(value: &Value) -> Result<DeepSeekV4ChunkFileIdentity> {
    let object = value
        .as_object()
        .ok_or_else(|| gravity("admission receipt chunk row must be an object"))?;
    let key = object
        .get("key")
        .and_then(Value::as_str)
        .ok_or_else(|| gravity("admission receipt chunk row lacks key"))?
        .to_owned();
    let bytes = object
        .get("bytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| gravity(format!("admission receipt chunk {key} lacks bytes")))?;
    let mtime_ns = parse_i128_field(
        object
            .get("mtime_ns")
            .ok_or_else(|| gravity(format!("admission receipt chunk {key} lacks mtime_ns")))?,
        "mtime_ns",
    )?;
    let inode = object
        .get("inode")
        .and_then(Value::as_u64)
        .ok_or_else(|| gravity(format!("admission receipt chunk {key} lacks inode")))?;
    Ok(DeepSeekV4ChunkFileIdentity {
        key,
        bytes,
        mtime_ns,
        inode,
    })
}

/// Load and seal-check a receipt beside `root`. Binding mismatches (wrong
/// manifest seal, wrong artifact digest, wrong counts) reject the whole
/// receipt so a stale file cannot authorize skip-hash.
pub fn load_admission_receipt(
    root: impl AsRef<Path>,
    expected_manifest_seal: &str,
    expected_chunk_digest: &str,
    expected_chunk_count: usize,
    expected_total_bytes: u64,
) -> DeepSeekV4AdmissionLoad {
    let mut saw_file = false;
    let mut last_reject = None;
    for path in receipt_search_paths(root, expected_manifest_seal) {
        if !path.is_file() {
            continue;
        }
        saw_file = true;
        match load_admission_receipt_file(
            &path,
            expected_manifest_seal,
            expected_chunk_digest,
            expected_chunk_count,
            expected_total_bytes,
        ) {
            Ok(index) => return DeepSeekV4AdmissionLoad::Loaded(index),
            Err(error) => last_reject = Some(format!("{error}")),
        }
    }
    if !saw_file {
        DeepSeekV4AdmissionLoad::Missing
    } else {
        DeepSeekV4AdmissionLoad::Rejected(
            last_reject.unwrap_or_else(|| {
                "admission receipt present but could not be verified".to_owned()
            }),
        )
    }
}

fn load_admission_receipt_file(
    path: &Path,
    expected_manifest_seal: &str,
    expected_chunk_digest: &str,
    expected_chunk_count: usize,
    expected_total_bytes: u64,
) -> Result<DeepSeekV4AdmissionTrustIndex> {
    let raw = fs::read(path).map_err(|error| {
        gravity(format!(
            "cannot read admission receipt {}: {error}",
            path.display()
        ))
    })?;
    let mut value: Value = serde_json::from_slice(&raw)
        .map_err(|error| gravity(format!("admission receipt is not valid JSON: {error}")))?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| gravity("admission receipt root must be a JSON object"))?;
    let recorded_seal = object
        .remove("seal_sha256")
        .and_then(|item| item.as_str().map(str::to_owned))
        .ok_or_else(|| gravity("admission receipt lacks seal_sha256"))?;
    if !is_sha256(&recorded_seal) {
        return Err(gravity(
            "admission receipt seal_sha256 is not lowercase SHA-256",
        ));
    }
    let observed_seal = sha256_hex(&canonical_json(&value));
    if observed_seal != recorded_seal {
        return Err(gravity(format!(
            "admission receipt seal mismatch: recorded={recorded_seal} observed={observed_seal}"
        )));
    }
    let object = value
        .as_object()
        .ok_or_else(|| gravity("admission receipt root must be a JSON object"))?;
    if object.get("schema").and_then(Value::as_str) != Some(ADMISSION_TRUST_SCHEMA) {
        return Err(gravity(
            "admission receipt schema is not admission_trust.v1",
        ));
    }
    let manifest_seal = object
        .get("manifest_seal_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| gravity("admission receipt lacks manifest_seal_sha256"))?;
    if manifest_seal != expected_manifest_seal {
        return Err(gravity(
            "admission receipt manifest_seal_sha256 differs from the admitted stream",
        ));
    }
    let chunk_digest = object
        .get("content_addressed_chunk_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| gravity("admission receipt lacks content_addressed_chunk_sha256"))?;
    if chunk_digest != expected_chunk_digest {
        return Err(gravity(
            "admission receipt content_addressed_chunk_sha256 differs from the admitted stream",
        ));
    }
    let chunk_count = object
        .get("chunk_count")
        .and_then(Value::as_u64)
        .ok_or_else(|| gravity("admission receipt lacks chunk_count"))?
        as usize;
    if chunk_count != expected_chunk_count {
        return Err(gravity(
            "admission receipt chunk_count differs from the admitted stream",
        ));
    }
    let total_bytes = object
        .get("total_bytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| gravity("admission receipt lacks total_bytes"))?;
    if total_bytes != expected_total_bytes {
        return Err(gravity(
            "admission receipt total_bytes differs from the admitted stream",
        ));
    }
    let artifact_root = PathBuf::from(
        object
            .get("artifact_root")
            .and_then(Value::as_str)
            .ok_or_else(|| gravity("admission receipt lacks artifact_root"))?,
    );
    let table_sha256 = object
        .get("table_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| gravity("admission receipt lacks table_sha256"))?
        .to_owned();
    if !is_sha256(&table_sha256) {
        return Err(gravity(
            "admission receipt table_sha256 is not lowercase SHA-256",
        ));
    }
    let sealed_at_unix_ms = object
        .get("sealed_at_unix_ms")
        .and_then(Value::as_u64)
        .ok_or_else(|| gravity("admission receipt lacks sealed_at_unix_ms"))?;
    let verifier_version = object
        .get("verifier_version")
        .and_then(Value::as_str)
        .ok_or_else(|| gravity("admission receipt lacks verifier_version"))?
        .to_owned();
    let rows = object
        .get("chunks")
        .and_then(Value::as_array)
        .ok_or_else(|| gravity("admission receipt chunks must be an array"))?;
    let mut chunks = BTreeMap::new();
    for row in rows {
        let identity = parse_chunk_row(row)?;
        if chunks.insert(identity.key.clone(), identity).is_some() {
            return Err(gravity("admission receipt has a duplicate chunk key"));
        }
    }
    if chunks.len() != chunk_count {
        return Err(gravity(
            "admission receipt chunk array length differs from chunk_count",
        ));
    }
    let observed_table = sha256_hex(&canonical_json(&table_json(&chunks)));
    if observed_table != table_sha256 {
        return Err(gravity(format!(
            "admission receipt table digest mismatch: recorded={table_sha256} observed={observed_table}"
        )));
    }
    let summed: u64 = chunks
        .values()
        .map(|c| c.bytes)
        .fold(0u64, |a, b| a.saturating_add(b));
    if summed != total_bytes {
        return Err(gravity(
            "admission receipt total_bytes disagrees with the chunk table",
        ));
    }
    Ok(DeepSeekV4AdmissionTrustIndex {
        artifact_root,
        content_addressed_chunk_sha256: chunk_digest.to_owned(),
        manifest_seal_sha256: manifest_seal.to_owned(),
        chunk_count,
        total_bytes,
        chunks,
        table_sha256,
        seal_sha256: recorded_seal,
        sealed_at_unix_ms,
        verifier_version,
    })
}

/// Hash every chunk under `receipt_root` (never skip), record cheap
/// identities, and write `<receipt_root>/.hawking-admission.json`.
pub fn seal_admission_trust_at(
    receipt_root: impl AsRef<Path>,
    manifest_seal_sha256: &str,
    content_addressed_chunk_sha256: &str,
    specs: &[DeepSeekV4AdmissionChunkSpec],
    verifier_version: &str,
) -> Result<DeepSeekV4AdmissionTrustSeal> {
    let wall = Instant::now();
    let receipt_root =
        canonical_non_symlink_directory(receipt_root.as_ref(), "admission receipt root")?;
    if specs.is_empty() {
        return Err(gravity("admission seal requires at least one chunk"));
    }
    let threads = admission_hash_threads().min(specs.len()).max(1);
    let hash_wall = Instant::now();
    let identities = hash_and_identify_chunks(&receipt_root, specs, threads)?;
    let hash_wall_ms = hash_wall.elapsed().as_millis();
    let total_bytes = identities.values().try_fold(0u64, |acc, chunk| {
        acc.checked_add(chunk.bytes)
            .ok_or_else(|| gravity("admission seal total byte count overflow"))
    })?;
    let table_sha256 = sha256_hex(&canonical_json(&table_json(&identities)));
    let sealed_at_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let mut unsigned = unsigned_receipt_document(
        &receipt_root,
        manifest_seal_sha256,
        content_addressed_chunk_sha256,
        &identities,
        &table_sha256,
        sealed_at_unix_ms,
        verifier_version,
    );
    let seal_sha256 = sha256_hex(&canonical_json(&unsigned));
    unsigned
        .as_object_mut()
        .expect("unsigned receipt is an object")
        .insert("seal_sha256".to_owned(), Value::String(seal_sha256.clone()));
    let path = publish_receipt(&receipt_root, manifest_seal_sha256, &unsigned)?;
    Ok(DeepSeekV4AdmissionTrustSeal {
        path,
        wall_ms: wall.elapsed().as_millis(),
        hash_wall_ms,
        chunk_count: identities.len(),
        total_bytes,
        bytes_hashed: total_bytes,
        threads,
        table_sha256,
        seal_sha256,
        content_addressed_chunk_sha256: content_addressed_chunk_sha256.to_owned(),
        manifest_seal_sha256: manifest_seal_sha256.to_owned(),
        verifier_version: verifier_version.to_owned(),
        identities,
        index_path: None,
        index_bytes: None,
        index_wall_ms: None,
    })
}

fn hash_and_identify_chunks(
    receipt_root: &Path,
    specs: &[DeepSeekV4AdmissionChunkSpec],
    threads: usize,
) -> Result<BTreeMap<String, DeepSeekV4ChunkFileIdentity>> {
    let errors: Mutex<Vec<Error>> = Mutex::new(Vec::new());
    let rows: Mutex<Vec<DeepSeekV4ChunkFileIdentity>> = Mutex::new(Vec::with_capacity(specs.len()));
    let chunk_size = (specs.len() + threads - 1) / threads;
    std::thread::scope(|scope| {
        let errors = &errors;
        let rows = &rows;
        for work in specs.chunks(chunk_size.max(1)) {
            scope.spawn(move || {
                for spec in work {
                    if errors.lock().expect("admission hash error mutex").len() > 0 {
                        break;
                    }
                    match hash_and_identify_one(receipt_root, spec) {
                        Ok(identity) => rows
                            .lock()
                            .expect("admission identity mutex")
                            .push(identity),
                        Err(error) => errors
                            .lock()
                            .expect("admission hash error mutex")
                            .push(error),
                    }
                }
            });
        }
    });
    let errors = errors.into_inner().expect("admission hash error mutex");
    if let Some(error) = errors.into_iter().next() {
        return Err(error);
    }
    let mut identities = BTreeMap::new();
    for identity in rows.into_inner().expect("admission identity mutex") {
        if identities.insert(identity.key.clone(), identity).is_some() {
            return Err(gravity("admission seal saw a duplicate chunk key"));
        }
    }
    if identities.len() != specs.len() {
        return Err(gravity(
            "admission seal did not identify every declared chunk",
        ));
    }
    Ok(identities)
}

fn hash_and_identify_one(
    receipt_root: &Path,
    spec: &DeepSeekV4AdmissionChunkSpec,
) -> Result<DeepSeekV4ChunkFileIdentity> {
    let path = checked_regular_path(receipt_root, &spec.relative, "content-addressed chunk")?;
    let mmap = map_chunk_readonly(&path, spec.bytes, &spec.relative)?;
    let observed = sha256_hex(&mmap);
    if observed != spec.sha256 {
        return Err(gravity(format!(
            "chunk {} sha256 differs from sealed segment digest",
            spec.relative
        )));
    }
    drop(mmap);
    let mut identity = file_identity(&path, "content-addressed chunk")?;
    if identity.bytes != spec.bytes {
        return Err(gravity(format!(
            "chunk {} byte size {} differs from sealed {}",
            spec.relative, identity.bytes, spec.bytes
        )));
    }
    identity.key = spec.relative.clone();
    Ok(identity)
}

fn publish_receipt(root: &Path, manifest_seal: &str, receipt: &Value) -> Result<PathBuf> {
    let beside = admission_receipt_path(root);
    match write_receipt_atomic(&beside, receipt) {
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
    if let Some(explicit) = explicit_receipt_path() {
        if let Some(parent) = explicit.parent() {
            fs::create_dir_all(parent).map_err(|error| {
                gravity(format!(
                    "cannot create admission receipt dir {}: {error}",
                    parent.display()
                ))
            })?;
        }
        return write_receipt_atomic(&explicit, receipt);
    }
    let cache = admission_receipt_cache_path(manifest_seal);
    if let Some(parent) = cache.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            gravity(format!(
                "cannot create admission receipt cache {}: {error}",
                parent.display()
            ))
        })?;
    }
    write_receipt_atomic(&cache, receipt)
}

fn write_receipt_atomic(path: &Path, receipt: &Value) -> Result<PathBuf> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| gravity("admission receipt path must have a UTF-8 file name"))?;
    let temporary = parent.join(format!(
        ".{name}.{}.admission-trust.tmp",
        std::process::id()
    ));
    let bytes = serde_json::to_vec_pretty(receipt)
        .map_err(|error| gravity(format!("cannot encode admission receipt: {error}")))?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| {
            gravity(format!(
                "cannot create admission receipt temporary {}: {error}",
                temporary.display()
            ))
        })?;
    if let Err(error) = file
        .write_all(&bytes)
        .and_then(|_| file.write_all(b"\n"))
        .and_then(|_| file.sync_all())
    {
        let _ = fs::remove_file(&temporary);
        return Err(gravity(format!(
            "cannot write admission receipt temporary: {error}"
        )));
    }
    if let Err(error) = fs::rename(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(gravity(format!(
            "cannot publish admission receipt {}: {error}",
            path.display()
        )));
    }
    if let Ok(dir) = File::open(parent) {
        let _ = dir.sync_all();
    }
    Ok(path.to_path_buf())
}

/// Resolve the file that will be mapped for `relative`.
///
/// Prefer the local path when its cheap identity matches the receipt.
/// Otherwise, if the receipt was sealed against a different durable root
/// (the clone-view case used when `stream-ranges.jsonl` was appended), map
/// that durable file when *its* identity still matches. The caller hashes
/// the local file when neither identity matches.
pub fn resolve_trusted_chunk_path(
    local_root: &Path,
    relative: &str,
    index: &DeepSeekV4AdmissionTrustIndex,
) -> Result<(PathBuf, Option<DeepSeekV4ChunkFileIdentity>, bool)> {
    let local = checked_regular_path(local_root, relative, "content-addressed chunk")?;
    let expected = match index.chunks.get(relative) {
        Some(expected) => expected,
        None => return Ok((local, None, false)),
    };
    let mut local_id = file_identity(&local, "content-addressed chunk")?;
    local_id.key = relative.to_owned();
    if identity_matches(&local_id, expected) {
        return Ok((local, Some(local_id), true));
    }
    if index.artifact_root != local_root {
        if let Ok(sealed_root) =
            canonical_non_symlink_directory(&index.artifact_root, "admission receipt artifact_root")
        {
            if sealed_root != local_root {
                if let Ok(sealed) =
                    checked_regular_path(&sealed_root, relative, "content-addressed chunk")
                {
                    if let Ok(mut sealed_id) = file_identity(&sealed, "content-addressed chunk") {
                        sealed_id.key = relative.to_owned();
                        if identity_matches(&sealed_id, expected) {
                            return Ok((sealed, Some(sealed_id), true));
                        }
                    }
                }
            }
        }
    }
    Ok((local, Some(local_id), false))
}

/// Required JSON fields used only by tests that rewrite a receipt body.
#[allow(dead_code)]
pub fn receipt_object_fields_for_tests(raw: &[u8]) -> Result<Map<String, Value>> {
    let value: Value = serde_json::from_slice(raw)
        .map_err(|error| gravity(format!("test receipt JSON: {error}")))?;
    value
        .as_object()
        .cloned()
        .ok_or_else(|| gravity("test receipt root must be an object"))
}
