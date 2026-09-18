//! Strict, non-executing admission for an immutable Hawking NR package.
//!
//! This is deliberately below `Engine`, Metal, Hawking, and serving.  It binds a
//! versioned descriptor to a closed declared package tree plus complete
//! payload/build/provenance bytes and denies source-checkpoint and
//! external-inference-framework dependencies, but it does **not** establish
//! that the declared execution contract is true.  A later native executor must
//! bind this exact identity and independently earn every correctness,
//! capability, and performance gate.
//!
//! The filesystem checks are admission-time local checks, not an OS-level
//! immutable lease, a source-independence proof, or a semantic verifier for a
//! provenance receipt.  A future executor must revalidate a stable artifact
//! store and use schema-specific provenance/runtime checks before trusting the
//! identity for execution.

use crate::{Error, Result};
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::{self, File, Metadata};
use std::io::{BufReader, Read};
use std::path::{Component, Path, PathBuf};
use std::{fmt, io};

/// The one descriptor layout admitted by this deliberately non-executing
/// tranche.  A schema change must add a new explicit reader rather than
/// weakening this one to accept an ambiguous document.
pub const ADMITTED_NR_DESCRIPTOR_SCHEMA: &str = "hawking.admitted_nr_descriptor.v1";
pub const ADMITTED_NR_DESCRIPTOR_VERSION: u64 = 1;
pub const ADMITTED_NR_DESCRIPTOR_SEAL_FORMAT: &str = "sha256_canonical_json_v1";

const MAX_DESCRIPTOR_BYTES: u64 = 1024 * 1024;
const MAX_PROVENANCE_RECEIPT_BYTES: u64 = 8 * 1024 * 1024;

const REQUIRED_EXECUTION_CONTRACT: &[&str] = &[
    "tokenization",
    "templates",
    "prefill",
    "live_routing",
    "required_model_operations",
    "recurrent_or_kv_state",
    "sampling",
    "incremental_output",
    "cancellation",
    "concurrency_isolation",
    "unload",
    "recovery",
];

const FORBIDDEN_RUNTIME_PATH_MARKERS: &[&str] = &[
    ".safetensors",
    "model.safetensors",
    "source_bf16",
    "source-bf16",
    "source_checkpoint",
    "source-checkpoint",
    "transformers",
    "llama.cpp",
    "ollama",
    "mlx",
    ".py",
];

fn nr_error(detail: impl Into<String>) -> Error {
    Error::Model(format!("admitted NR: {}", detail.into()))
}

/// One full-file hash binding in an admitted NR package.  The verifier retains
/// identity only; it does not retain or expose the payload bytes for execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NrFileBinding {
    pub id: String,
    pub relative_path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
}

/// Byte-bound reference to a separately sealed provenance receipt.
///
/// The receipt's seal value and format are checked for identity and the entire
/// receipt file is SHA-256 bound here.  Semantic verification remains with the
/// receipt schema's independent verifier; this generic package reader never
/// pretends to interpret every historical receipt format itself.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NrProvenanceReceiptBinding {
    pub relative_path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
    pub seal_format: String,
    pub seal_sha256: String,
}

/// Immutable identity produced only by successful descriptor admission.
///
/// Its fields stay private so a future executor cannot manufacture an
/// apparently admitted identity through a struct literal.  It must accept the
/// opaque [`AdmittedNrArtifact`] (or re-admit exact bytes) instead.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdmittedNrIdentity {
    artifact_id: String,
    artifact_root: PathBuf,
    descriptor_path: PathBuf,
    /// Exact descriptor bytes observed at admission. The separately declared
    /// seal binds canonical semantics; this digest also records the precise
    /// serialized descriptor that was reviewed.
    descriptor_sha256: String,
    descriptor_seal_sha256: String,
    build: NrFileBinding,
    components: Vec<NrFileBinding>,
    source_provenance_receipts: Vec<NrProvenanceReceiptBinding>,
    declared_execution_contract: Vec<String>,
}

impl AdmittedNrIdentity {
    pub fn artifact_id(&self) -> &str {
        &self.artifact_id
    }

    pub fn artifact_root(&self) -> &Path {
        &self.artifact_root
    }

    pub fn descriptor_path(&self) -> &Path {
        &self.descriptor_path
    }

    pub fn descriptor_sha256(&self) -> &str {
        &self.descriptor_sha256
    }

    pub fn descriptor_seal_sha256(&self) -> &str {
        &self.descriptor_seal_sha256
    }

    pub fn build(&self) -> &NrFileBinding {
        &self.build
    }

    pub fn components(&self) -> &[NrFileBinding] {
        &self.components
    }

    pub fn source_provenance_receipts(&self) -> &[NrProvenanceReceiptBinding] {
        &self.source_provenance_receipts
    }

    pub fn declared_execution_contract(&self) -> &[String] {
        &self.declared_execution_contract
    }
}

/// Admission deliberately stops before any executor is bound.  Qualification
/// needs a future, separately evidence-bound type rather than a variant that a
/// caller can construct today.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmittedNrReadiness {
    ArtifactVerifiedNoExecutor,
}

/// A verified package identity with no model, tokenizer, device, subprocess,
/// or serving capability attached.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdmittedNrArtifact {
    identity: AdmittedNrIdentity,
}

impl AdmittedNrArtifact {
    pub fn identity(&self) -> &AdmittedNrIdentity {
        &self.identity
    }

    pub fn readiness(&self) -> AdmittedNrReadiness {
        AdmittedNrReadiness::ArtifactVerifiedNoExecutor
    }

    pub fn can_execute(&self) -> bool {
        false
    }

    /// Always refuses in this tranche.  Keeping the refusal explicit prevents
    /// a verified descriptor from being mistaken for a runnable worker.
    pub fn refuse_execution(&self) -> Result<()> {
        Err(nr_error(
            "artifact identity is verified, but no native executor is bound or qualified",
        ))
    }
}

#[derive(Debug)]
struct NoDuplicateJson(Value);

impl<'de> Deserialize<'de> for NoDuplicateJson {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(NoDuplicateJsonVisitor)
    }
}

struct NoDuplicateJsonVisitor;

impl<'de> Visitor<'de> for NoDuplicateJsonVisitor {
    type Value = NoDuplicateJson;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value without duplicate object keys")
    }

    fn visit_bool<E>(self, value: bool) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(NoDuplicateJson(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(NoDuplicateJson(Value::Number(value.into())))
    }

    fn visit_u64<E>(self, value: u64) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(NoDuplicateJson(Value::Number(value.into())))
    }

    fn visit_f64<E>(self, value: f64) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        serde_json::Number::from_f64(value)
            .map(|number| NoDuplicateJson(Value::Number(number)))
            .ok_or_else(|| E::custom("JSON number must be finite"))
    }

    fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(NoDuplicateJson(Value::String(value.to_owned())))
    }

    fn visit_string<E>(self, value: String) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(NoDuplicateJson(Value::String(value)))
    }

    fn visit_none<E>(self) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(NoDuplicateJson(Value::Null))
    }

    fn visit_unit<E>(self) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(NoDuplicateJson(Value::Null))
    }

    fn visit_some<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        NoDuplicateJson::deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(NoDuplicateJson(value)) = sequence.next_element()? {
            values.push(value);
        }
        Ok(NoDuplicateJson(Value::Array(values)))
    }

    fn visit_map<A>(self, mut map_access: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = Map::new();
        while let Some(key) = map_access.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(de::Error::custom(format!("duplicate JSON key {key:?}")));
            }
            let NoDuplicateJson(value) = map_access.next_value()?;
            values.insert(key, value);
        }
        Ok(NoDuplicateJson(Value::Object(values)))
    }
}

fn parse_json_no_duplicate_keys(raw: &[u8], label: &str) -> Result<Value> {
    let mut deserializer = serde_json::Deserializer::from_slice(raw);
    let NoDuplicateJson(value) = NoDuplicateJson::deserialize(&mut deserializer)
        .map_err(|error| nr_error(format!("{label} is invalid JSON: {error}")))?;
    deserializer
        .end()
        .map_err(|error| nr_error(format!("{label} has trailing JSON data: {error}")))?;
    Ok(value)
}

fn canonical_json(value: &Value) -> Result<Vec<u8>> {
    fn write(value: &Value, out: &mut Vec<u8>) -> Result<()> {
        match value {
            Value::Null => out.extend_from_slice(b"null"),
            Value::Bool(true) => out.extend_from_slice(b"true"),
            Value::Bool(false) => out.extend_from_slice(b"false"),
            Value::Number(number) => out.extend_from_slice(number.to_string().as_bytes()),
            Value::String(string) => out.extend_from_slice(
                serde_json::to_string(string)
                    .map_err(|error| nr_error(format!("cannot render canonical string: {error}")))?
                    .as_bytes(),
            ),
            Value::Array(values) => {
                out.push(b'[');
                for (index, child) in values.iter().enumerate() {
                    if index != 0 {
                        out.push(b',');
                    }
                    write(child, out)?;
                }
                out.push(b']');
            }
            Value::Object(values) => {
                out.push(b'{');
                let mut keys: Vec<&String> = values.keys().collect();
                keys.sort_unstable();
                for (index, key) in keys.into_iter().enumerate() {
                    if index != 0 {
                        out.push(b',');
                    }
                    out.extend_from_slice(
                        serde_json::to_string(key)
                            .map_err(|error| {
                                nr_error(format!("cannot render canonical key: {error}"))
                            })?
                            .as_bytes(),
                    );
                    out.push(b':');
                    write(&values[key], out)?;
                }
                out.push(b'}');
            }
        }
        Ok(())
    }

    let mut output = Vec::new();
    write(value, &mut output)?;
    Ok(output)
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_reader(mut reader: impl Read) -> io::Result<(u64, String)> {
    let mut digest = Sha256::new();
    let mut total = 0u64;
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        total = total
            .checked_add(count as u64)
            .ok_or_else(|| io::Error::other("file byte count overflow"))?;
        digest.update(&buffer[..count]);
    }
    Ok((total, format!("{:x}", digest.finalize())))
}

fn require_object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| nr_error(format!("{label} must be a JSON object")))
}

fn require_only_keys(object: &Map<String, Value>, allowed: &[&str], label: &str) -> Result<()> {
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(nr_error(format!("{label} has unsupported field {key:?}")));
        }
    }
    Ok(())
}

fn require_nonempty_exact_string(value: Option<&Value>, label: &str) -> Result<String> {
    let value = value
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| nr_error(format!("{label} must be a non-empty string")))?;
    if value.trim() != value {
        return Err(nr_error(format!(
            "{label} must not contain leading or trailing whitespace"
        )));
    }
    if value.chars().any(char::is_control) {
        return Err(nr_error(format!(
            "{label} must not contain control characters"
        )));
    }
    Ok(value.to_owned())
}

fn require_string(object: &Map<String, Value>, key: &str, label: &str) -> Result<String> {
    require_nonempty_exact_string(object.get(key), &format!("{label}.{key}"))
}

fn require_false(object: &Map<String, Value>, key: &str, label: &str) -> Result<()> {
    match object.get(key) {
        Some(Value::Bool(false)) => Ok(()),
        _ => Err(nr_error(format!("{label}.{key} must be false"))),
    }
}

fn require_u64(object: &Map<String, Value>, key: &str, label: &str) -> Result<u64> {
    object
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| nr_error(format!("{label}.{key} must be an unsigned integer")))
}

fn require_sha256(value: &str, label: &str) -> Result<String> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(nr_error(format!(
            "{label} must be a 64-character SHA-256 hex digest"
        )));
    }
    let lower = value.to_ascii_lowercase();
    if lower != value {
        return Err(nr_error(format!("{label} must use lowercase hexadecimal")));
    }
    Ok(lower)
}

fn relative_path(value: &str, label: &str) -> Result<PathBuf> {
    let path = Path::new(value);
    if value.trim().is_empty()
        || path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir
                    | Component::RootDir
                    | Component::Prefix(_)
                    | Component::CurDir
            )
        })
    {
        return Err(nr_error(format!(
            "{label} must be a non-empty contained relative path"
        )));
    }
    let normalized = PathBuf::from(path);
    if normalized.components().count() == 0 {
        return Err(nr_error(format!("{label} must name a file")));
    }
    Ok(normalized)
}

fn is_immutable(metadata: &Metadata) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        metadata.permissions().mode() & 0o222 == 0
    }
    #[cfg(not(unix))]
    {
        metadata.permissions().readonly()
    }
}

fn regular_immutable_file(path: &Path, label: &str) -> Result<Metadata> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        nr_error(format!(
            "cannot inspect {label} {}: {error}",
            path.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(nr_error(format!(
            "{label} must be a regular non-symlink file: {}",
            path.display()
        )));
    }
    if !is_immutable(&metadata) {
        return Err(nr_error(format!(
            "{label} must not be write-enabled at admission: {}",
            path.display()
        )));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        // A writable file outside the package can otherwise share this inode
        // through a hard link while every declared path looks immutable.  An
        // admission identity must not rely on an external alias remaining
        // benign, so this target intentionally admits only singly linked
        // payload, build, provenance, and descriptor files.
        if metadata.nlink() != 1 {
            return Err(nr_error(format!(
                "{label} must not be hard-linked at admission: {}",
                path.display()
            )));
        }
        Ok(metadata)
    }
    #[cfg(not(unix))]
    {
        // The native worker target is POSIX/macOS.  Without a stable file-ID
        // and link-count primitive here, accepting a package would weaken the
        // closed-file identity guarantee rather than merely reduce coverage.
        let _ = metadata;
        Err(nr_error(
            "admitted NR package identity requires Unix hard-link inspection",
        ))
    }
}

fn immutable_directory(path: &Path, label: &str) -> Result<()> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        nr_error(format!(
            "cannot inspect {label} {}: {error}",
            path.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(nr_error(format!(
            "{label} must be a directory, not a symlink: {}",
            path.display()
        )));
    }
    if !is_immutable(&metadata) {
        return Err(nr_error(format!(
            "{label} must not be write-enabled at admission: {}",
            path.display()
        )));
    }
    Ok(())
}

fn canonical_descriptor_path(path: &Path) -> Result<PathBuf> {
    regular_immutable_file(path, "descriptor")?;
    let canonical = fs::canonicalize(path).map_err(|error| {
        nr_error(format!(
            "cannot canonicalize descriptor {}: {error}",
            path.display()
        ))
    })?;
    let root = canonical
        .parent()
        .ok_or_else(|| nr_error("descriptor has no artifact-root parent"))?;
    immutable_directory(root, "artifact root")?;
    Ok(canonical)
}

fn checked_relative_file(root: &Path, relative: &Path, label: &str) -> Result<PathBuf> {
    immutable_directory(root, "artifact root")?;
    let mut current = root.to_path_buf();
    let components: Vec<_> = relative.components().collect();
    for (index, component) in components.iter().enumerate() {
        let Component::Normal(name) = component else {
            return Err(nr_error(format!("{label} path is not contained")));
        };
        current.push(name);
        if index + 1 == components.len() {
            regular_immutable_file(&current, label)?;
        } else {
            immutable_directory(&current, &format!("{label} parent"))?;
        }
    }
    let canonical = fs::canonicalize(&current).map_err(|error| {
        nr_error(format!(
            "cannot canonicalize {label} {}: {error}",
            current.display()
        ))
    })?;
    if !canonical.starts_with(root) {
        return Err(nr_error(format!(
            "{label} path escapes admitted artifact root"
        )));
    }
    Ok(canonical)
}

fn read_immutable_limited(path: &Path, maximum: u64, label: &str) -> Result<Vec<u8>> {
    let before = regular_immutable_file(path, label)?;
    if before.len() > maximum {
        return Err(nr_error(format!(
            "{label} is {} bytes, above its bounded admission limit of {maximum}",
            before.len()
        )));
    }
    let file = File::open(path)
        .map_err(|error| nr_error(format!("cannot open {label} {}: {error}", path.display())))?;
    let mut reader = BufReader::new(file);
    let mut bytes = Vec::with_capacity(before.len() as usize);
    reader
        .read_to_end(&mut bytes)
        .map_err(|error| nr_error(format!("cannot read {label} {}: {error}", path.display())))?;
    let after = regular_immutable_file(path, label)?;
    if before.len() != after.len() || bytes.len() as u64 != before.len() {
        return Err(nr_error(format!("{label} changed while being read")));
    }
    Ok(bytes)
}

fn verify_hashed_file(root: &Path, binding: &NrFileBinding, label: &str) -> Result<PathBuf> {
    let path = checked_relative_file(root, &binding.relative_path, label)?;
    let before = regular_immutable_file(&path, label)?;
    if before.len() != binding.bytes {
        return Err(nr_error(format!(
            "{label} byte count differs from descriptor: expected {}, observed {}",
            binding.bytes,
            before.len()
        )));
    }
    let file = File::open(&path)
        .map_err(|error| nr_error(format!("cannot open {label} {}: {error}", path.display())))?;
    let (observed_bytes, observed_sha256) = sha256_reader(BufReader::new(file))
        .map_err(|error| nr_error(format!("cannot hash {label} {}: {error}", path.display())))?;
    let after = regular_immutable_file(&path, label)?;
    if before.len() != after.len() || observed_bytes != before.len() {
        return Err(nr_error(format!("{label} changed while being hashed")));
    }
    if observed_sha256 != binding.sha256 {
        return Err(nr_error(format!(
            "{label} SHA-256 differs from descriptor: expected {}, observed {observed_sha256}",
            binding.sha256
        )));
    }
    Ok(path)
}

fn verify_closed_package_tree(
    artifact_root: &Path,
    descriptor_path: &Path,
    declared_paths: &BTreeSet<PathBuf>,
) -> Result<()> {
    let descriptor_relative = descriptor_path
        .strip_prefix(artifact_root)
        .map_err(|error| {
            nr_error(format!(
                "descriptor {} is not beneath artifact root {}: {error}",
                descriptor_path.display(),
                artifact_root.display()
            ))
        })?;
    if declared_paths.contains(descriptor_relative) {
        return Err(nr_error(
            "descriptor must not also be declared as a component, build, or provenance receipt",
        ));
    }
    let mut expected = declared_paths.clone();
    expected.insert(descriptor_relative.to_path_buf());

    fn walk(artifact_root: &Path, directory: &Path, expected: &BTreeSet<PathBuf>) -> Result<()> {
        immutable_directory(directory, "artifact package directory")?;
        let entries = fs::read_dir(directory).map_err(|error| {
            nr_error(format!(
                "cannot enumerate admitted artifact directory {}: {error}",
                directory.display()
            ))
        })?;
        for entry in entries {
            let entry = entry.map_err(|error| {
                nr_error(format!(
                    "cannot enumerate admitted artifact directory {}: {error}",
                    directory.display()
                ))
            })?;
            let path = entry.path();
            let relative = path.strip_prefix(artifact_root).map_err(|error| {
                nr_error(format!(
                    "artifact entry {} escaped root {}: {error}",
                    path.display(),
                    artifact_root.display()
                ))
            })?;
            let file_type = entry.file_type().map_err(|error| {
                nr_error(format!(
                    "cannot inspect artifact entry {}: {error}",
                    path.display()
                ))
            })?;
            if file_type.is_symlink() {
                return Err(nr_error(format!(
                    "artifact package contains a symlink entry: {}",
                    relative.display()
                )));
            }
            if file_type.is_dir() {
                if !expected
                    .iter()
                    .any(|declared| declared.starts_with(relative))
                {
                    return Err(nr_error(format!(
                        "artifact package contains an undeclared directory: {}",
                        relative.display()
                    )));
                }
                walk(artifact_root, &path, expected)?;
            } else if file_type.is_file() {
                regular_immutable_file(&path, "artifact package file")?;
                if !expected.contains(relative) {
                    return Err(nr_error(format!(
                        "artifact package contains an undeclared file: {}",
                        relative.display()
                    )));
                }
            } else {
                return Err(nr_error(format!(
                    "artifact package contains an unsupported entry: {}",
                    relative.display()
                )));
            }
        }
        Ok(())
    }

    walk(artifact_root, artifact_root, &expected)
}

fn reject_forbidden_runtime_path(value: &str, label: &str) -> Result<()> {
    let lower = value.to_ascii_lowercase();
    if let Some(marker) = FORBIDDEN_RUNTIME_PATH_MARKERS
        .iter()
        .find(|marker| lower.contains(**marker))
    {
        return Err(nr_error(format!(
            "{label} names forbidden source/external-inference runtime marker {marker:?}"
        )));
    }
    Ok(())
}

fn parse_file_binding(value: &Value, label: &str, require_id: bool) -> Result<NrFileBinding> {
    let object = require_object(value, label)?;
    let allowed = if require_id {
        &["id", "path", "bytes", "sha256"][..]
    } else {
        &["path", "bytes", "sha256"][..]
    };
    require_only_keys(object, allowed, label)?;
    let id = if require_id {
        let id = require_string(object, "id", label)?;
        if !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        {
            return Err(nr_error(format!("{label}.id has unsupported characters")));
        }
        reject_forbidden_runtime_path(&id, &format!("{label}.id"))?;
        id
    } else {
        "native_build".to_owned()
    };
    let path_text = require_string(object, "path", label)?;
    reject_forbidden_runtime_path(&path_text, &format!("{label}.path"))?;
    let relative_path = relative_path(&path_text, &format!("{label}.path"))?;
    let bytes = require_u64(object, "bytes", label)?;
    if bytes == 0 {
        return Err(nr_error(format!("{label}.bytes must be positive")));
    }
    let sha256 = require_sha256(
        &require_string(object, "sha256", label)?,
        &format!("{label}.sha256"),
    )?;
    Ok(NrFileBinding {
        id,
        relative_path,
        bytes,
        sha256,
    })
}

fn parse_provenance_binding(value: &Value, label: &str) -> Result<NrProvenanceReceiptBinding> {
    let object = require_object(value, label)?;
    require_only_keys(
        object,
        &["path", "bytes", "sha256", "seal_format", "seal_sha256"],
        label,
    )?;
    let path_text = require_string(object, "path", label)?;
    let relative_path = relative_path(&path_text, &format!("{label}.path"))?;
    let bytes = require_u64(object, "bytes", label)?;
    if bytes == 0 || bytes > MAX_PROVENANCE_RECEIPT_BYTES {
        return Err(nr_error(format!(
            "{label}.bytes must be within 1..={MAX_PROVENANCE_RECEIPT_BYTES}"
        )));
    }
    let sha256 = require_sha256(
        &require_string(object, "sha256", label)?,
        &format!("{label}.sha256"),
    )?;
    let seal_format = require_string(object, "seal_format", label)?;
    let seal_sha256 = require_sha256(
        &require_string(object, "seal_sha256", label)?,
        &format!("{label}.seal_sha256"),
    )?;
    Ok(NrProvenanceReceiptBinding {
        relative_path,
        bytes,
        sha256,
        seal_format,
        seal_sha256,
    })
}

fn parse_execution_contract(value: &Value) -> Result<Vec<String>> {
    let values = value
        .as_array()
        .ok_or_else(|| nr_error("execution_contract must be an array"))?;
    let mut seen = BTreeSet::new();
    let mut parsed = Vec::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        let item =
            require_nonempty_exact_string(Some(value), &format!("execution_contract[{index}]"))?;
        if !seen.insert(item.clone()) {
            return Err(nr_error(format!("execution_contract duplicates {item:?}")));
        }
        parsed.push(item);
    }
    for required in REQUIRED_EXECUTION_CONTRACT {
        if !seen.contains(*required) {
            return Err(nr_error(format!(
                "execution_contract omits required declared capability {required:?}"
            )));
        }
    }
    Ok(parsed)
}

fn verify_runtime_dependency_policy(value: &Value) -> Result<()> {
    // This validates only the descriptor's declared policy.  It is not an
    // import/link/launch or clean-staging verifier, so it cannot by itself
    // establish the runtime-independence gate required for promotion.
    let object = require_object(value, "runtime_dependency_policy")?;
    require_only_keys(
        object,
        &[
            "requires_source_checkpoint",
            "allows_network_provider_fallback",
            "requires_python_inference_executable",
            "inference_frameworks",
        ],
        "runtime_dependency_policy",
    )?;
    require_false(
        object,
        "requires_source_checkpoint",
        "runtime_dependency_policy",
    )?;
    require_false(
        object,
        "allows_network_provider_fallback",
        "runtime_dependency_policy",
    )?;
    require_false(
        object,
        "requires_python_inference_executable",
        "runtime_dependency_policy",
    )?;
    let frameworks = object
        .get("inference_frameworks")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            nr_error("runtime_dependency_policy.inference_frameworks must be an array")
        })?;
    if !frameworks.is_empty() {
        return Err(nr_error(
            "runtime_dependency_policy.inference_frameworks must be empty for an admitted native NR",
        ));
    }
    Ok(())
}

fn verify_descriptor_seal(object: &Map<String, Value>) -> Result<String> {
    let seal_format = require_string(object, "seal_format", "descriptor")?;
    if seal_format != ADMITTED_NR_DESCRIPTOR_SEAL_FORMAT {
        return Err(nr_error(format!(
            "descriptor.seal_format must be {ADMITTED_NR_DESCRIPTOR_SEAL_FORMAT:?}"
        )));
    }
    let recorded = require_sha256(
        &require_string(object, "seal_sha256", "descriptor")?,
        "descriptor.seal_sha256",
    )?;
    let mut unsigned = object.clone();
    unsigned.remove("seal_sha256");
    let expected = sha256_hex(&canonical_json(&Value::Object(unsigned))?);
    if recorded != expected {
        return Err(nr_error(format!(
            "descriptor seal mismatch: recorded {recorded}, expected {expected}"
        )));
    }
    Ok(recorded)
}

fn verify_provenance_receipt(root: &Path, binding: &NrProvenanceReceiptBinding) -> Result<()> {
    let file_binding = NrFileBinding {
        id: "source_provenance_receipt".to_owned(),
        relative_path: binding.relative_path.clone(),
        bytes: binding.bytes,
        sha256: binding.sha256.clone(),
    };
    let path = verify_hashed_file(root, &file_binding, "source provenance receipt")?;
    let raw = read_immutable_limited(
        &path,
        MAX_PROVENANCE_RECEIPT_BYTES,
        "source provenance receipt",
    )?;
    if raw.len() as u64 != binding.bytes || sha256_hex(&raw) != binding.sha256 {
        return Err(nr_error(
            "source provenance receipt changed between hash verification and read",
        ));
    }
    let value = parse_json_no_duplicate_keys(&raw, "source provenance receipt")?;
    let object = require_object(&value, "source provenance receipt")?;
    require_string(object, "schema", "source provenance receipt")?;
    if require_string(object, "seal_format", "source provenance receipt")? != binding.seal_format {
        return Err(nr_error(
            "source provenance receipt seal format differs from descriptor",
        ));
    }
    if require_sha256(
        &require_string(object, "seal_sha256", "source provenance receipt")?,
        "source provenance receipt.seal_sha256",
    )? != binding.seal_sha256
    {
        return Err(nr_error(
            "source provenance receipt seal differs from descriptor",
        ));
    }
    Ok(())
}

/// Admit a descriptor rooted beside its immutable package files.
///
/// Admission fully hashes every declared payload and build, validates each
/// provenance receipt reference, and returns only an in-memory identity.  It
/// never opens a source checkpoint, creates an engine, contacts a provider,
/// starts a child process, or initializes Metal.
pub fn admit_nr_descriptor(descriptor_path: impl AsRef<Path>) -> Result<AdmittedNrArtifact> {
    let descriptor_path = canonical_descriptor_path(descriptor_path.as_ref())?;
    let artifact_root = descriptor_path
        .parent()
        .ok_or_else(|| nr_error("descriptor has no artifact-root parent"))?
        .to_path_buf();
    let raw = read_immutable_limited(&descriptor_path, MAX_DESCRIPTOR_BYTES, "descriptor")?;
    let descriptor_sha256 = sha256_hex(&raw);
    let document = parse_json_no_duplicate_keys(&raw, "descriptor")?;
    let object = require_object(&document, "descriptor")?;
    require_only_keys(
        object,
        &[
            "schema",
            "format_version",
            "artifact_id",
            "components",
            "build",
            "source_provenance_receipts",
            "execution_contract",
            "runtime_dependency_policy",
            "seal_format",
            "seal_sha256",
        ],
        "descriptor",
    )?;
    if require_string(object, "schema", "descriptor")? != ADMITTED_NR_DESCRIPTOR_SCHEMA {
        return Err(nr_error(format!(
            "descriptor.schema must be {ADMITTED_NR_DESCRIPTOR_SCHEMA:?}"
        )));
    }
    if require_u64(object, "format_version", "descriptor")? != ADMITTED_NR_DESCRIPTOR_VERSION {
        return Err(nr_error(format!(
            "descriptor.format_version must be {ADMITTED_NR_DESCRIPTOR_VERSION}"
        )));
    }
    let descriptor_seal_sha256 = verify_descriptor_seal(object)?;
    let artifact_id = require_string(object, "artifact_id", "descriptor")?;
    let execution_contract = parse_execution_contract(
        object
            .get("execution_contract")
            .ok_or_else(|| nr_error("descriptor.execution_contract is required"))?,
    )?;
    verify_runtime_dependency_policy(
        object
            .get("runtime_dependency_policy")
            .ok_or_else(|| nr_error("descriptor.runtime_dependency_policy is required"))?,
    )?;

    let component_values = object
        .get("components")
        .and_then(Value::as_array)
        .filter(|values| !values.is_empty())
        .ok_or_else(|| nr_error("descriptor.components must be a non-empty array"))?;
    let mut component_ids = BTreeSet::new();
    let mut occupied_paths = BTreeSet::new();
    let mut components = Vec::with_capacity(component_values.len());
    for (index, value) in component_values.iter().enumerate() {
        let binding = parse_file_binding(value, &format!("descriptor.components[{index}]"), true)?;
        if !component_ids.insert(binding.id.clone()) {
            return Err(nr_error(format!(
                "descriptor.components duplicates id {:?}",
                binding.id
            )));
        }
        if !occupied_paths.insert(binding.relative_path.clone()) {
            return Err(nr_error(format!(
                "descriptor.components duplicates path {}",
                binding.relative_path.display()
            )));
        }
        components.push(binding);
    }

    let build = parse_file_binding(
        object
            .get("build")
            .ok_or_else(|| nr_error("descriptor.build is required"))?,
        "descriptor.build",
        false,
    )?;
    if !occupied_paths.insert(build.relative_path.clone()) {
        return Err(nr_error(
            "descriptor.build path overlaps a declared component",
        ));
    }

    let receipt_values = object
        .get("source_provenance_receipts")
        .and_then(Value::as_array)
        .filter(|values| !values.is_empty())
        .ok_or_else(|| {
            nr_error("descriptor.source_provenance_receipts must be a non-empty array")
        })?;
    let mut source_provenance_receipts = Vec::with_capacity(receipt_values.len());
    for (index, value) in receipt_values.iter().enumerate() {
        let binding = parse_provenance_binding(
            value,
            &format!("descriptor.source_provenance_receipts[{index}]"),
        )?;
        if !occupied_paths.insert(binding.relative_path.clone()) {
            return Err(nr_error(
                "source provenance receipt path overlaps a component or build",
            ));
        }
        source_provenance_receipts.push(binding);
    }

    verify_closed_package_tree(&artifact_root, &descriptor_path, &occupied_paths)?;

    for component in &components {
        verify_hashed_file(&artifact_root, component, "declared NR component")?;
    }
    verify_hashed_file(&artifact_root, &build, "declared native build")?;
    for receipt in &source_provenance_receipts {
        verify_provenance_receipt(&artifact_root, receipt)?;
    }

    Ok(AdmittedNrArtifact {
        identity: AdmittedNrIdentity {
            artifact_id,
            artifact_root,
            descriptor_path,
            descriptor_sha256,
            descriptor_seal_sha256,
            build,
            components,
            source_provenance_receipts,
            declared_execution_contract: execution_contract,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    struct Fixture {
        _temp: TempDir,
        root: PathBuf,
        descriptor: PathBuf,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            // The fixture deliberately freezes its package root so normal
            // tempfile cleanup must make that synthetic directory writable
            // again. This is test-only and never part of admission.
            if let Ok(metadata) = fs::metadata(&self.root) {
                let mut permissions = metadata.permissions();
                permissions.set_readonly(false);
                let _ = fs::set_permissions(&self.root, permissions);
            }
        }
    }

    fn freeze(path: &Path) {
        let mut permissions = fs::metadata(path).unwrap().permissions();
        permissions.set_readonly(true);
        fs::set_permissions(path, permissions).unwrap();
    }

    fn thaw(path: &Path) {
        let mut permissions = fs::metadata(path).unwrap().permissions();
        permissions.set_readonly(false);
        fs::set_permissions(path, permissions).unwrap();
    }

    fn binding(path: &Path, id: Option<&str>) -> Value {
        let bytes = fs::read(path).unwrap();
        let mut value = serde_json::Map::new();
        if let Some(id) = id {
            value.insert("id".into(), Value::String(id.into()));
        }
        value.insert(
            "path".into(),
            Value::String(path.file_name().unwrap().to_string_lossy().into_owned()),
        );
        value.insert("bytes".into(), json!(bytes.len() as u64));
        value.insert("sha256".into(), Value::String(sha256_hex(&bytes)));
        Value::Object(value)
    }

    fn execution_contract() -> Value {
        Value::Array(
            REQUIRED_EXECUTION_CONTRACT
                .iter()
                .map(|item| Value::String((*item).to_owned()))
                .collect(),
        )
    }

    fn seal_descriptor(mut document: Value) -> Value {
        let object = document.as_object_mut().unwrap();
        object.insert(
            "seal_format".into(),
            Value::String(ADMITTED_NR_DESCRIPTOR_SEAL_FORMAT.into()),
        );
        let seal = sha256_hex(&canonical_json(&Value::Object(object.clone())).unwrap());
        object.insert("seal_sha256".into(), Value::String(seal));
        document
    }

    fn fixture_with(mutate: impl FnOnce(&mut Value), freeze_weight: bool) -> Fixture {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().to_path_buf();
        let weights = root.join("weights.nr");
        let tokenizer = root.join("tokenizer.bin");
        let build = root.join("worker.bin");
        let receipt = root.join("provenance.json");
        fs::write(&weights, b"immutable packed NR weights").unwrap();
        fs::write(&tokenizer, b"native tokenizer bytes").unwrap();
        fs::write(&build, b"native worker build").unwrap();
        let receipt_seal = "b".repeat(64);
        fs::write(
            &receipt,
            serde_json::to_vec(&json!({
                "schema": "fixture.provenance.v1",
                "seal_format": "fixture_sha256_v1",
                "seal_sha256": receipt_seal,
            }))
            .unwrap(),
        )
        .unwrap();

        let receipt_bytes = fs::read(&receipt).unwrap();
        let mut document = json!({
            "schema": ADMITTED_NR_DESCRIPTOR_SCHEMA,
            "format_version": ADMITTED_NR_DESCRIPTOR_VERSION,
            "artifact_id": "fixture-nr",
            "components": [binding(&weights, Some("weights")), binding(&tokenizer, Some("tokenizer"))],
            "build": binding(&build, None),
            "source_provenance_receipts": [{
                "path": "provenance.json",
                "bytes": receipt_bytes.len() as u64,
                "sha256": sha256_hex(&receipt_bytes),
                "seal_format": "fixture_sha256_v1",
                "seal_sha256": "b".repeat(64),
            }],
            "execution_contract": execution_contract(),
            "runtime_dependency_policy": {
                "requires_source_checkpoint": false,
                "allows_network_provider_fallback": false,
                "requires_python_inference_executable": false,
                "inference_frameworks": [],
            },
        });
        mutate(&mut document);
        let descriptor = root.join("nr-descriptor.json");
        fs::write(
            &descriptor,
            serde_json::to_vec(&seal_descriptor(document)).unwrap(),
        )
        .unwrap();
        if freeze_weight {
            freeze(&weights);
        }
        freeze(&tokenizer);
        freeze(&build);
        freeze(&receipt);
        freeze(&descriptor);
        freeze(&root);
        Fixture {
            _temp: temp,
            root,
            descriptor,
        }
    }

    fn fixture() -> Fixture {
        fixture_with(|_| {}, true)
    }

    #[test]
    fn admits_an_immutable_identity_but_never_an_executor() {
        let fixture = fixture();
        let admitted = admit_nr_descriptor(&fixture.descriptor).unwrap();
        assert_eq!(
            admitted.readiness(),
            AdmittedNrReadiness::ArtifactVerifiedNoExecutor
        );
        assert!(!admitted.can_execute());
        assert_eq!(admitted.identity().components().len(), 2);
        assert_eq!(
            admitted.identity().descriptor_sha256(),
            sha256_hex(&fs::read(&fixture.descriptor).unwrap())
        );
        assert!(admitted
            .refuse_execution()
            .unwrap_err()
            .to_string()
            .contains("no native executor"));
    }

    #[test]
    fn rejects_duplicate_descriptor_keys() {
        let fixture = fixture();
        thaw(&fixture.descriptor);
        fs::write(
            &fixture.descriptor,
            format!(
                r#"{{"schema":"{ADMITTED_NR_DESCRIPTOR_SCHEMA}","schema":"{ADMITTED_NR_DESCRIPTOR_SCHEMA}"}}"#
            ),
        )
        .unwrap();
        freeze(&fixture.descriptor);
        assert!(admit_nr_descriptor(&fixture.descriptor)
            .unwrap_err()
            .to_string()
            .contains("duplicate JSON key"));
    }

    #[test]
    fn rejects_a_stale_descriptor_seal() {
        let fixture = fixture();
        thaw(&fixture.descriptor);
        let raw = fs::read_to_string(&fixture.descriptor).unwrap();
        fs::write(
            &fixture.descriptor,
            raw.replace("fixture-nr", "stale-fixture-nr"),
        )
        .unwrap();
        freeze(&fixture.descriptor);
        assert!(admit_nr_descriptor(&fixture.descriptor)
            .unwrap_err()
            .to_string()
            .contains("descriptor seal mismatch"));
    }

    #[test]
    fn rejects_path_escape_and_incomplete_hashes_before_payload_use() {
        let escaped = fixture_with(
            |document| {
                document["components"][0]["path"] = json!("../outside.nr");
            },
            true,
        );
        assert!(admit_nr_descriptor(&escaped.descriptor)
            .unwrap_err()
            .to_string()
            .contains("contained relative path"));

        let incomplete = fixture_with(
            |document| {
                document["components"][0]["sha256"] = json!("");
            },
            true,
        );
        assert!(admit_nr_descriptor(&incomplete.descriptor)
            .unwrap_err()
            .to_string()
            .contains("components[0].sha256"));
    }

    #[test]
    fn rejects_missing_contract_and_external_inference_dependencies() {
        let missing_contract = fixture_with(
            |document| {
                document["execution_contract"] = json!(["tokenization"]);
            },
            true,
        );
        assert!(admit_nr_descriptor(&missing_contract.descriptor)
            .unwrap_err()
            .to_string()
            .contains("omits required"));

        let external_framework = fixture_with(
            |document| {
                document["runtime_dependency_policy"]["inference_frameworks"] = json!(["MLX"]);
            },
            true,
        );
        assert!(admit_nr_descriptor(&external_framework.descriptor)
            .unwrap_err()
            .to_string()
            .contains("must be empty"));
    }

    #[test]
    fn rejects_a_write_enabled_payload() {
        let fixture = fixture_with(|_| {}, false);
        assert!(admit_nr_descriptor(&fixture.descriptor)
            .unwrap_err()
            .to_string()
            .contains("must not be write-enabled"));
    }

    #[test]
    fn rejects_an_undeclared_package_sidecar() {
        let fixture = fixture();
        thaw(&fixture.root);
        let sidecar = fixture.root.join("undeclared-runtime-sidecar");
        fs::write(&sidecar, b"not an admitted dependency").unwrap();
        freeze(&sidecar);
        freeze(&fixture.root);
        assert!(admit_nr_descriptor(&fixture.descriptor)
            .unwrap_err()
            .to_string()
            .contains("undeclared file"));
    }

    #[test]
    fn rejects_descriptor_strings_with_surrounding_whitespace() {
        let fixture = fixture_with(
            |document| {
                document["artifact_id"] = json!(" fixture-nr ");
            },
            true,
        );
        assert!(admit_nr_descriptor(&fixture.descriptor)
            .unwrap_err()
            .to_string()
            .contains("leading or trailing whitespace"));
    }

    #[test]
    fn rejects_descriptor_strings_with_embedded_control_characters() {
        let fixture = fixture_with(
            |document| {
                document["artifact_id"] = json!("fixture\nnr");
            },
            true,
        );
        assert!(admit_nr_descriptor(&fixture.descriptor)
            .unwrap_err()
            .to_string()
            .contains("control characters"));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_a_symlinked_payload() {
        use std::os::unix::fs::symlink;

        let fixture = fixture();
        thaw(&fixture.root);
        let weights = fixture.root.join("weights.nr");
        fs::remove_file(&weights).unwrap();
        symlink(fixture.root.join("tokenizer.bin"), &weights).unwrap();
        freeze(&fixture.root);
        assert!(admit_nr_descriptor(&fixture.descriptor)
            .unwrap_err()
            .to_string()
            .contains("symlink"));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_hardlinked_admission_files() {
        // A read-only source file may still receive an external POSIX hard
        // link: write permission is required on the destination directory,
        // not the source.  Exercise every central admission file class.
        for filename in [
            "nr-descriptor.json",
            "weights.nr",
            "worker.bin",
            "provenance.json",
        ] {
            let fixture = fixture();
            let outside = tempfile::tempdir().unwrap();
            fs::hard_link(
                fixture.root.join(filename),
                outside.path().join("external-alias"),
            )
            .unwrap();

            let error = admit_nr_descriptor(&fixture.descriptor)
                .unwrap_err()
                .to_string();
            assert!(error.contains("hard-linked"), "{filename}: {error}");
        }
    }
}
