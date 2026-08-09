#![allow(dead_code)] // This executable intentionally exposes its fixture primitives to focused tests.

//! Fixture-only Qwen30 single-layer streamed-BF16 operator primitive.
//!
//! This program is deliberately a *small, non-source* proving ground for the
//! future layer-streamed source oracle.  It accepts only three metadata JSON
//! contracts already produced by the Q30 workstream:
//!
//! * the source-operator / scalar-order semantics document;
//! * the metadata-only range-map authority; and
//! * the guarded outer-controller / executor contract.
//!
//! It then constructs an entirely synthetic BF16 shard, reads each synthetic
//! row by explicit byte offset with one bounded buffer, and evaluates one
//! representative sparse-MoE layer in the declared order:
//!
//! `RMSNorm -> router -> softmax/top-8/normalise -> gate+up -> SiLU*up ->
//! down -> route-weight -> ordered index-add -> residual`.
//!
//! The fixture has no source-root, safetensors, model, GPU, server, HCLI,
//! lease, benchmark, or generation option.  In particular, it is not source
//! inference, source-equivalence, quality, coherence, HCLI, TPS, TG,
//! capability, promotion, or tournament evidence.  Its only numerical bytes
//! are generated in-process for a tiny synthetic shard whose filename cannot
//! be supplied by the caller.

use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};

type Result<T> = std::result::Result<T, String>;

const RESULT_SCHEMA: &str = "hawking.ascension.qwen30_streamed_bf16_single_layer_fixture.v1";
const RESULT_STATUS: &str =
    "PREPARED_QWEN30_STREAMED_BF16_SINGLE_LAYER_FIXTURE_ONLY_NOT_SOURCE_EXECUTION";
const SEMANTICS_SCHEMA: &str =
    "hawking.ascension.qwen30_layer_streamed_source_operator_accumulation_semantics_attester.v1";
const SEMANTICS_STATUS: &str =
    "PREPARED_QWEN30_LAYER_STREAMED_SOURCE_OPERATOR_ACCUMULATION_SEMANTICS_NOT_EXECUTED";
const RANGE_AUTHORITY_SCHEMA: &str =
    "hawking.ascension.qwen30_streamed_oracle_metadata_only_range_map_authority.v1";
const RANGE_AUTHORITY_STATUS: &str =
    "PREPARED_QWEN30_STREAMED_ORACLE_SOURCE_RANGE_MAP_AUTHORITY_NOT_EXECUTED";
const OUTER_EXECUTOR_SCHEMA: &str =
    "hawking.ascension.qwen30_guarded_streamed_source_oracle_outer_controller.v1";
const OUTER_EXECUTOR_STATUS: &str =
    "BLOCKED_QWEN30_GUARDED_STREAMED_SOURCE_ORACLE_OUTER_NO_EXECUTOR_NOT_EXECUTED";

/// This is the same one-window upper bound carried by the Q30 range authority
/// and outer controller.  The test fixture is much smaller, but its reader
/// enforces the physical bound rather than merely reporting it.
const MAX_WINDOW_BYTES: usize = 1024 * 1024;
const MAX_CONTRACT_BYTES: u64 = 32 * 1024 * 1024;
const SYNTHETIC_HIDDEN: usize = 4;
const SYNTHETIC_EXPERTS: usize = 8;
const SYNTHETIC_TOP_K: usize = 8;
const SYNTHETIC_INTERMEDIATE: usize = 3;
const RMS_EPSILON: f32 = 1.0e-6;

static SYNTHETIC_FILE_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
struct Args {
    semantics_contract: PathBuf,
    range_authority: PathBuf,
    executor_contract: PathBuf,
    out: PathBuf,
}

fn usage() -> &'static str {
    "usage: ascension_qwen30_streamed_bf16_layer_fixture --semantics-contract ABSOLUTE_SEMANTICS_JSON --range-authority ABSOLUTE_RANGE_AUTHORITY_JSON --executor-contract ABSOLUTE_OUTER_EXECUTOR_JSON --out NEW_ABSOLUTE_RECEIPT_JSON"
}

fn parse_args_from<I>(arguments: I) -> Result<Args>
where
    I: IntoIterator<Item = String>,
{
    let mut semantics_contract = None;
    let mut range_authority = None;
    let mut executor_contract = None;
    let mut out = None;
    let mut values = arguments.into_iter();
    while let Some(flag) = values.next() {
        let slot = match flag.as_str() {
            "--semantics-contract" => &mut semantics_contract,
            "--range-authority" => &mut range_authority,
            "--executor-contract" => &mut executor_contract,
            "--out" => &mut out,
            "--help" | "-h" => return Err(usage().to_owned()),
            _ => return Err(format!("unsupported option {flag:?}; {}", usage())),
        };
        let value = values
            .next()
            .ok_or_else(|| format!("missing value for {flag}; {}", usage()))?;
        if slot.replace(PathBuf::from(value)).is_some() {
            return Err(format!("{flag} was supplied more than once; {}", usage()));
        }
    }
    let semantics_contract = semantics_contract
        .ok_or_else(|| format!("--semantics-contract is required; {}", usage()))?;
    let range_authority =
        range_authority.ok_or_else(|| format!("--range-authority is required; {}", usage()))?;
    let executor_contract =
        executor_contract.ok_or_else(|| format!("--executor-contract is required; {}", usage()))?;
    let out = out.ok_or_else(|| format!("--out is required; {}", usage()))?;
    for (path, label) in [
        (&semantics_contract, "--semantics-contract"),
        (&range_authority, "--range-authority"),
        (&executor_contract, "--executor-contract"),
        (&out, "--out"),
    ] {
        if !path.is_absolute() {
            return Err(format!("{label} must be absolute"));
        }
    }
    if !out.parent().is_some_and(Path::is_dir) {
        return Err("--out parent must already exist".into());
    }
    Ok(Args {
        semantics_contract,
        range_authority,
        executor_contract,
        out,
    })
}

fn parse_args() -> Result<Args> {
    parse_args_from(env::args().skip(1))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a serde_json::Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be a JSON object"))
}

fn required<'a>(
    value: &'a serde_json::Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<&'a Value> {
    value
        .get(key)
        .ok_or_else(|| format!("{label} lacks required field {key:?}"))
}

fn required_string<'a>(
    value: &'a serde_json::Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<&'a str> {
    required(value, key, label)?
        .as_str()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{label}.{key} must be a non-empty string"))
}

fn required_bool(value: &serde_json::Map<String, Value>, key: &str, label: &str) -> Result<bool> {
    required(value, key, label)?
        .as_bool()
        .ok_or_else(|| format!("{label}.{key} must be a boolean"))
}

#[derive(Clone, Debug)]
struct LoadedDocument {
    path: PathBuf,
    bytes: Vec<u8>,
    value: Value,
    sha256: String,
}

/// Metadata-only reader: it accepts regular `.json` documents only.  It has
/// no generic payload-reader API, so a model shard cannot be accidentally fed
/// into this fixture primitive.
fn read_metadata_document(path: &Path, label: &str) -> Result<LoadedDocument> {
    if !path.is_absolute() || path.extension().and_then(|item| item.to_str()) != Some("json") {
        return Err(format!(
            "{label} must be an absolute .json metadata document"
        ));
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    if metadata.len() == 0 || metadata.len() > MAX_CONTRACT_BYTES {
        return Err(format!(
            "{label} must contain 1..={MAX_CONTRACT_BYTES} metadata bytes, observed {}",
            metadata.len()
        ));
    }
    let mut bytes = vec![0u8; usize::try_from(metadata.len()).map_err(|_| "contract too large")?];
    let mut file = File::open(path)
        .map_err(|error| format!("cannot open {label} {}: {error}", path.display()))?;
    file.read_exact(&mut bytes)
        .map_err(|error| format!("cannot read {label} {}: {error}", path.display()))?;
    let restat = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot restat {label} {}: {error}", path.display()))?;
    if restat.file_type().is_symlink()
        || !restat.file_type().is_file()
        || restat.len() != metadata.len()
    {
        return Err(format!("{label} changed while it was read"));
    }
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("{label} is not valid JSON: {error}"))?;
    Ok(LoadedDocument {
        path: path.to_path_buf(),
        sha256: sha256_hex(&bytes),
        bytes,
        value,
    })
}

#[derive(Debug, Serialize)]
struct ContractBinding {
    path: String,
    raw_document_sha256: String,
    schema: String,
    status: String,
}

#[derive(Debug, Serialize)]
struct ExternalContractBindings {
    semantics: ContractBinding,
    range_authority: ContractBinding,
    executor_outer_controller: ContractBinding,
    maximum_source_reader_cached_bytes: usize,
    maximum_source_reader_cached_windows: usize,
    source_payload_opened_by_this_fixture: bool,
    actual_streamed_executor_present: bool,
}

fn binding(
    document: &LoadedDocument,
    root: &serde_json::Map<String, Value>,
    label: &str,
) -> Result<ContractBinding> {
    Ok(ContractBinding {
        path: document.path.display().to_string(),
        raw_document_sha256: document.sha256.clone(),
        schema: required_string(root, "schema", label)?.to_owned(),
        status: required_string(root, "status", label)?.to_owned(),
    })
}

/// Bind the fixture to the existing semantics/range/executor documents by
/// their *actual bytes*, rather than duplicating their intended values.  The
/// current outer executor is deliberately blocked; accepting any "earned"
/// executor status here would blur the fixture/source boundary.
fn validate_external_contracts(
    semantics: &LoadedDocument,
    range_authority: &LoadedDocument,
    executor: &LoadedDocument,
) -> Result<ExternalContractBindings> {
    let semantics_root = object(&semantics.value, "semantics contract")?;
    if required_string(semantics_root, "schema", "semantics contract")? != SEMANTICS_SCHEMA
        || required_string(semantics_root, "status", "semantics contract")? != SEMANTICS_STATUS
    {
        return Err(
            "semantics contract schema/status is not the prepared Q30 scalar-order contract".into(),
        );
    }
    let semantics_requirements = object(
        required(
            semantics_root,
            "exact_scalar_and_order_requirements",
            "semantics contract",
        )?,
        "semantics contract.exact_scalar_and_order_requirements",
    )?;
    for field in [
        "rmsnorm",
        "router_topk_and_weights",
        "selected_expert_gate_up_swiglu_down",
        "residuals",
    ] {
        let requirement = object(
            required(
                semantics_requirements,
                field,
                "semantics scalar/order requirements",
            )?,
            &format!("semantics scalar/order requirements.{field}"),
        )?;
        if !requirement.contains_key("required_sequence") {
            return Err(format!(
                "semantics scalar/order requirements.{field} lacks required_sequence"
            ));
        }
    }

    let range_outer = object(&range_authority.value, "range authority wrapper")?;
    let range_root = object(
        required(range_outer, "authority", "range authority wrapper")?,
        "range authority wrapper.authority",
    )?;
    if required_string(range_root, "schema", "range authority")? != RANGE_AUTHORITY_SCHEMA
        || required_string(range_root, "status", "range authority")? != RANGE_AUTHORITY_STATUS
    {
        return Err(
            "range authority schema/status is not the prepared Q30 metadata-only authority".into(),
        );
    }
    if !required(range_root, "tensors", "range authority")?.is_array() {
        return Err("range authority.tensors must be an array".into());
    }

    let executor_root = object(&executor.value, "outer executor contract")?;
    if required_string(executor_root, "schema", "outer executor contract")? != OUTER_EXECUTOR_SCHEMA
        || required_string(executor_root, "status", "outer executor contract")?
            != OUTER_EXECUTOR_STATUS
    {
        return Err(
            "executor contract must remain the current blocked no-executor outer controller".into(),
        );
    }
    let claim_boundary = object(
        required(executor_root, "claim_boundary", "outer executor contract")?,
        "outer executor contract.claim_boundary",
    )?;
    if !required_bool(
        claim_boundary,
        "metadata_only_preflight",
        "outer executor contract.claim_boundary",
    )? || !required_bool(
        claim_boundary,
        "does_not_open_source_tensor_payloads_or_load_a_source_model",
        "outer executor contract.claim_boundary",
    )? {
        return Err(
            "executor contract must retain its metadata-only/no-source-payload boundary".into(),
        );
    }
    let future = object(
        required(
            executor_root,
            "future_source_launch_contract",
            "outer executor contract",
        )?,
        "outer executor contract.future_source_launch_contract",
    )?;
    let max_bytes = required(
        future,
        "maximum_source_reader_cached_bytes",
        "outer executor future",
    )?
    .as_u64()
    .ok_or_else(|| {
        "outer executor future maximum_source_reader_cached_bytes must be u64".to_owned()
    })?;
    let max_windows = required(
        future,
        "maximum_source_reader_cached_windows",
        "outer executor future",
    )?
    .as_u64()
    .ok_or_else(|| {
        "outer executor future maximum_source_reader_cached_windows must be u64".to_owned()
    })?;
    if usize::try_from(max_bytes).ok() != Some(MAX_WINDOW_BYTES) || max_windows != 1 {
        return Err(format!(
            "outer executor must require exactly one {MAX_WINDOW_BYTES}-byte source window"
        ));
    }
    if required_bool(
        future,
        "actual_streamed_executor_present",
        "outer executor future",
    )? {
        return Err(
            "fixture primitive refuses an outer contract that claims an actual source executor"
                .into(),
        );
    }

    let outer_semantics = object(
        required(
            executor_root,
            "metadata_only_operator_semantics",
            "outer executor contract",
        )?,
        "outer executor contract.metadata_only_operator_semantics",
    )?;
    if required_string(
        outer_semantics,
        "sha256",
        "outer executor contract.metadata_only_operator_semantics",
    )? != semantics.sha256
    {
        return Err(
            "outer executor semantics SHA-256 does not bind the supplied semantics document".into(),
        );
    }
    let outer_range = object(
        required(
            executor_root,
            "metadata_only_range_authority",
            "outer executor contract",
        )?,
        "outer executor contract.metadata_only_range_authority",
    )?;
    if required_string(
        outer_range,
        "sha256",
        "outer executor contract.metadata_only_range_authority",
    )? != range_authority.sha256
    {
        return Err(
            "outer executor range SHA-256 does not bind the supplied range authority".into(),
        );
    }

    Ok(ExternalContractBindings {
        semantics: binding(semantics, semantics_root, "semantics contract")?,
        range_authority: binding(range_authority, range_root, "range authority")?,
        executor_outer_controller: binding(executor, executor_root, "outer executor contract")?,
        maximum_source_reader_cached_bytes: MAX_WINDOW_BYTES,
        maximum_source_reader_cached_windows: 1,
        source_payload_opened_by_this_fixture: false,
        actual_streamed_executor_present: false,
    })
}

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    // IEEE round-to-nearest-even when discarding the low 16 bits.
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn bf16_round_f32(value: f32) -> f32 {
    bf16_bits_to_f32(f32_to_bf16_bits(value))
}

fn bf16_round_f64(value: f64) -> f64 {
    f64::from(bf16_round_f32(value as f32))
}

fn f32_digest(values: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<u32>());
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    sha256_hex(&bytes)
}

fn f64_digest(values: &[f64]) -> String {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<u64>());
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    sha256_hex(&bytes)
}

#[derive(Clone, Debug, Serialize)]
struct RangeRecord {
    name: String,
    offset_bytes: u64,
    length_bytes: usize,
    elements: usize,
    sha256: String,
}

#[derive(Default)]
struct FixtureShardBuilder {
    bytes: Vec<u8>,
    records: BTreeMap<String, RangeRecord>,
}

impl FixtureShardBuilder {
    fn add_bf16_vector(&mut self, name: impl Into<String>, values: &[f32]) -> Result<()> {
        let name = name.into();
        if name.is_empty() || self.records.contains_key(&name) || values.is_empty() {
            return Err("fixture ranges require unique non-empty names and finite values".into());
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err("fixture BF16 values must be finite".into());
        }
        let offset_bytes =
            u64::try_from(self.bytes.len()).map_err(|_| "fixture offset overflow")?;
        let mut encoded = Vec::with_capacity(values.len() * 2);
        for value in values {
            encoded.extend_from_slice(&f32_to_bf16_bits(*value).to_le_bytes());
        }
        if encoded.len() > MAX_WINDOW_BYTES {
            return Err(format!(
                "fixture record {name:?} exceeds the one-window {MAX_WINDOW_BYTES}-byte bound"
            ));
        }
        let record = RangeRecord {
            name: name.clone(),
            offset_bytes,
            length_bytes: encoded.len(),
            elements: values.len(),
            sha256: sha256_hex(&encoded),
        };
        self.bytes.extend_from_slice(&encoded);
        self.records.insert(name, record);
        Ok(())
    }

    fn maximum_declared_range_bytes(&self) -> usize {
        self.records
            .values()
            .map(|record| record.length_bytes)
            .max()
            .unwrap_or(0)
    }

    fn materialize(self) -> Result<PositionedBf16Reader> {
        let fixture = SyntheticFixtureFile::create(&self.bytes)?;
        PositionedBf16Reader::new(fixture, self.records, self.bytes.len())
    }
}

struct SyntheticFixtureFile {
    path: PathBuf,
}

impl SyntheticFixtureFile {
    fn create(bytes: &[u8]) -> Result<Self> {
        for _ in 0..64 {
            let nonce = SYNTHETIC_FILE_NONCE.fetch_add(1, Ordering::Relaxed);
            let path = env::temp_dir().join(format!(
                "hawking-q30-streamed-bf16-fixture-{}-{nonce}.bin",
                process::id()
            ));
            let opened = OpenOptions::new().write(true).create_new(true).open(&path);
            let mut file = match opened {
                Ok(file) => file,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(format!(
                        "cannot create private synthetic BF16 fixture {}: {error}",
                        path.display()
                    ))
                }
            };
            if let Err(error) = file.write_all(bytes).and_then(|_| file.sync_all()) {
                let _ = fs::remove_file(&path);
                return Err(format!(
                    "cannot write synthetic BF16 fixture {}: {error}",
                    path.display()
                ));
            }
            return Ok(Self { path });
        }
        Err("could not reserve a unique synthetic fixture filename".into())
    }
}

impl Drop for SyntheticFixtureFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[derive(Clone, Debug, Serialize)]
struct ReaderStats {
    positioned_read_calls: usize,
    total_source_like_bytes_read: usize,
    maximum_live_raw_bf16_window_bytes: usize,
    maximum_declared_range_bytes: usize,
    simultaneous_raw_bf16_windows: usize,
    all_reads_verified_by_range_sha256: bool,
    read_order: Vec<String>,
}

struct PositionedBf16Reader {
    /// Holds the private synthetic file's cleanup token.  This is never a
    /// source path and its name is generated internally.
    _fixture: SyntheticFixtureFile,
    file: File,
    file_length: usize,
    records: BTreeMap<String, RangeRecord>,
    active_window_bytes: usize,
    stats: ReaderStats,
}

impl PositionedBf16Reader {
    fn new(
        fixture: SyntheticFixtureFile,
        records: BTreeMap<String, RangeRecord>,
        file_length: usize,
    ) -> Result<Self> {
        let file = File::open(&fixture.path).map_err(|error| {
            format!(
                "cannot reopen internally-generated synthetic fixture {}: {error}",
                fixture.path.display()
            )
        })?;
        let maximum_declared_range_bytes = records
            .values()
            .map(|record| record.length_bytes)
            .max()
            .unwrap_or(0);
        if maximum_declared_range_bytes > MAX_WINDOW_BYTES {
            return Err("synthetic range map exceeds the one-window bound".into());
        }
        Ok(Self {
            _fixture: fixture,
            file,
            file_length,
            records,
            active_window_bytes: 0,
            stats: ReaderStats {
                positioned_read_calls: 0,
                total_source_like_bytes_read: 0,
                maximum_live_raw_bf16_window_bytes: 0,
                maximum_declared_range_bytes,
                simultaneous_raw_bf16_windows: 0,
                all_reads_verified_by_range_sha256: true,
                read_order: Vec::new(),
            },
        })
    }

    fn with_verified_window<T>(
        &mut self,
        name: &str,
        operation: impl FnOnce(&[u8]) -> Result<T>,
    ) -> Result<T> {
        if self.active_window_bytes != 0 {
            return Err("reader attempted to retain more than one raw BF16 window".into());
        }
        let record = self
            .records
            .get(name)
            .cloned()
            .ok_or_else(|| format!("synthetic range map lacks {name:?}"))?;
        if record.length_bytes == 0 || record.length_bytes > MAX_WINDOW_BYTES {
            return Err(format!(
                "range {} is not within the 1..={MAX_WINDOW_BYTES} byte bound",
                record.name
            ));
        }
        let start = usize::try_from(record.offset_bytes).map_err(|_| "range offset overflow")?;
        let end = start
            .checked_add(record.length_bytes)
            .ok_or_else(|| "range end overflow".to_owned())?;
        if end > self.file_length {
            return Err(format!("range {} escapes the synthetic shard", record.name));
        }
        self.file
            .seek(SeekFrom::Start(record.offset_bytes))
            .map_err(|error| format!("cannot seek synthetic range {}: {error}", record.name))?;
        let mut window = vec![0u8; record.length_bytes];
        self.file.read_exact(&mut window).map_err(|error| {
            format!(
                "cannot positioned-read synthetic range {}: {error}",
                record.name
            )
        })?;
        if sha256_hex(&window) != record.sha256 {
            self.stats.all_reads_verified_by_range_sha256 = false;
            return Err(format!("synthetic range {} checksum drifted", record.name));
        }
        self.active_window_bytes = window.len();
        self.stats.positioned_read_calls += 1;
        self.stats.total_source_like_bytes_read += window.len();
        self.stats.maximum_live_raw_bf16_window_bytes = self
            .stats
            .maximum_live_raw_bf16_window_bytes
            .max(window.len());
        self.stats.simultaneous_raw_bf16_windows = self.stats.simultaneous_raw_bf16_windows.max(1);
        self.stats.read_order.push(record.name);
        let result = operation(&window);
        self.active_window_bytes = 0;
        result
    }

    fn decode_bf16_vector(&mut self, name: &str, expected_elements: usize) -> Result<Vec<f32>> {
        self.with_verified_window(name, |window| {
            if window.len() != expected_elements * 2 {
                return Err(format!(
                    "range {name:?} must contain exactly {} BF16 bytes, observed {}",
                    expected_elements * 2,
                    window.len()
                ));
            }
            let mut decoded = Vec::with_capacity(expected_elements);
            for chunk in window.chunks_exact(2) {
                decoded.push(bf16_bits_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])));
            }
            if decoded.iter().any(|value| !value.is_finite()) {
                return Err(format!("range {name:?} decoded a non-finite BF16 value"));
            }
            Ok(decoded)
        })
    }

    fn stats(&self) -> ReaderStats {
        self.stats.clone()
    }
}

fn dot_f32(left: &[f32], right: &[f32]) -> Result<f32> {
    if left.len() != right.len() {
        return Err("F32 scalar dot operands have different lengths".into());
    }
    let mut accumulator = 0.0f32;
    for index in 0..left.len() {
        accumulator = accumulator + left[index] * right[index];
    }
    Ok(accumulator)
}

fn dot_f64(left: &[f64], right: &[f64]) -> Result<f64> {
    if left.len() != right.len() {
        return Err("F64 scalar dot operands have different lengths".into());
    }
    let mut accumulator = 0.0f64;
    for index in 0..left.len() {
        accumulator += left[index] * right[index];
    }
    Ok(accumulator)
}

fn rmsnorm_f32(input: &[f32], weight: &[f32]) -> Result<Vec<f32>> {
    if input.len() != SYNTHETIC_HIDDEN || weight.len() != SYNTHETIC_HIDDEN {
        return Err("synthetic RMSNorm must use the declared hidden width".into());
    }
    let mut sum_squares = 0.0f32;
    for value in input {
        sum_squares = sum_squares + value * value;
    }
    let scale = (sum_squares / input.len() as f32 + RMS_EPSILON)
        .sqrt()
        .recip();
    Ok(input
        .iter()
        .zip(weight)
        .map(|(hidden, norm_weight)| bf16_round_f32(hidden * scale * norm_weight))
        .collect())
}

fn rmsnorm_f64(input: &[f64], weight: &[f64]) -> Result<Vec<f64>> {
    if input.len() != SYNTHETIC_HIDDEN || weight.len() != SYNTHETIC_HIDDEN {
        return Err("synthetic RMSNorm must use the declared hidden width".into());
    }
    let mut sum_squares = 0.0f64;
    for value in input {
        sum_squares += value * value;
    }
    let scale = (sum_squares / input.len() as f64 + f64::from(RMS_EPSILON))
        .sqrt()
        .recip();
    Ok(input
        .iter()
        .zip(weight)
        .map(|(hidden, norm_weight)| bf16_round_f64(hidden * scale * norm_weight))
        .collect())
}

fn softmax_f32(logits: &[f32]) -> Result<Vec<f32>> {
    let max = logits
        .iter()
        .copied()
        .reduce(f32::max)
        .ok_or_else(|| "router logits cannot be empty".to_owned())?;
    let mut denominator = 0.0f32;
    let mut exponents = Vec::with_capacity(logits.len());
    for logit in logits {
        let exponent = (*logit - max).exp();
        denominator += exponent;
        exponents.push(exponent);
    }
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err("F32 router softmax denominator is invalid".into());
    }
    Ok(exponents
        .into_iter()
        .map(|value| bf16_round_f32(value / denominator))
        .collect())
}

fn softmax_f64(logits: &[f64]) -> Result<Vec<f64>> {
    let max = logits
        .iter()
        .copied()
        .reduce(f64::max)
        .ok_or_else(|| "router logits cannot be empty".to_owned())?;
    let mut denominator = 0.0f64;
    let mut exponents = Vec::with_capacity(logits.len());
    for logit in logits {
        let exponent = (*logit - max).exp();
        denominator += exponent;
        exponents.push(exponent);
    }
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err("F64 router softmax denominator is invalid".into());
    }
    Ok(exponents
        .into_iter()
        .map(|value| bf16_round_f64(value / denominator))
        .collect())
}

fn selected_routes_f32(probabilities: &[f32]) -> Result<Vec<(usize, f32)>> {
    if probabilities.len() != SYNTHETIC_EXPERTS {
        return Err("synthetic router must score all eight fixture experts".into());
    }
    let mut indexed = probabilities
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<_>>();
    indexed.sort_by(|(left_id, left), (right_id, right)| {
        right.total_cmp(left).then_with(|| left_id.cmp(right_id))
    });
    indexed.truncate(SYNTHETIC_TOP_K);
    let mut sum = 0.0f32;
    for (_, probability) in &indexed {
        sum += *probability;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err("synthetic selected F32 router weights are invalid".into());
    }
    Ok(indexed
        .into_iter()
        .map(|(expert, probability)| (expert, bf16_round_f32(probability / sum)))
        .collect())
}

fn selected_routes_f64(probabilities: &[f64]) -> Result<Vec<(usize, f64)>> {
    if probabilities.len() != SYNTHETIC_EXPERTS {
        return Err("synthetic router must score all eight fixture experts".into());
    }
    let mut indexed = probabilities
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<_>>();
    indexed.sort_by(|(left_id, left), (right_id, right)| {
        right.total_cmp(left).then_with(|| left_id.cmp(right_id))
    });
    indexed.truncate(SYNTHETIC_TOP_K);
    let mut sum = 0.0f64;
    for (_, probability) in &indexed {
        sum += *probability;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err("synthetic selected F64 router weights are invalid".into());
    }
    Ok(indexed
        .into_iter()
        .map(|(expert, probability)| (expert, bf16_round_f64(probability / sum)))
        .collect())
}

fn expert_row_name(expert: usize, role: &str, row: usize) -> String {
    format!("fixture.layer17.expert{expert}.{role}.row{row}")
}

fn matrix_vector_f32(
    reader: &mut PositionedBf16Reader,
    expert: usize,
    role: &str,
    rows: usize,
    columns: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    if input.len() != columns {
        return Err("synthetic F32 matvec input width drifted".into());
    }
    let mut output = Vec::with_capacity(rows);
    for row in 0..rows {
        let weight = reader.decode_bf16_vector(&expert_row_name(expert, role, row), columns)?;
        output.push(bf16_round_f32(dot_f32(&weight, input)?));
    }
    Ok(output)
}

fn matrix_vector_f64(
    reader: &mut PositionedBf16Reader,
    expert: usize,
    role: &str,
    rows: usize,
    columns: usize,
    input: &[f64],
) -> Result<Vec<f64>> {
    if input.len() != columns {
        return Err("synthetic F64 matvec input width drifted".into());
    }
    let mut output = Vec::with_capacity(rows);
    for row in 0..rows {
        let weight = reader
            .decode_bf16_vector(&expert_row_name(expert, role, row), columns)?
            .into_iter()
            .map(f64::from)
            .collect::<Vec<_>>();
        output.push(bf16_round_f64(dot_f64(&weight, input)?));
    }
    Ok(output)
}

#[derive(Clone, Debug, Serialize)]
struct F32BoundaryOutputs {
    rmsnorm_bf16_store: Vec<f32>,
    router_logits_bf16_store: Vec<f32>,
    router_probabilities_bf16_store: Vec<f32>,
    selected_expert_ids_in_router_order: Vec<usize>,
    normalized_selected_weights_bf16_store: Vec<f32>,
    ordered_index_add_combined_bf16_store: Vec<f32>,
    residual_bf16_store: Vec<f32>,
    residual_f32_bits_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct F64BoundaryOutputs {
    rmsnorm_bf16_store: Vec<f64>,
    router_logits_bf16_store: Vec<f64>,
    router_probabilities_bf16_store: Vec<f64>,
    selected_expert_ids_in_router_order: Vec<usize>,
    normalized_selected_weights_bf16_store: Vec<f64>,
    ordered_index_add_combined_bf16_store: Vec<f64>,
    residual_bf16_store: Vec<f64>,
    residual_f64_bits_sha256: String,
}

fn execute_f32_layer(
    reader: &mut PositionedBf16Reader,
) -> Result<(F32BoundaryOutputs, Vec<String>)> {
    let mut events = vec!["input_residual_bf16_decode".to_owned()];
    let input = reader.decode_bf16_vector("fixture.layer17.input_residual", SYNTHETIC_HIDDEN)?;
    let norm_weight =
        reader.decode_bf16_vector("fixture.layer17.rmsnorm_weight", SYNTHETIC_HIDDEN)?;
    let normalized = rmsnorm_f32(&input, &norm_weight)?;
    events.push("rmsnorm_f32_accumulate_then_bf16_store".to_owned());

    let mut router_logits = Vec::with_capacity(SYNTHETIC_EXPERTS);
    for expert in 0..SYNTHETIC_EXPERTS {
        let row = reader.decode_bf16_vector(
            &format!("fixture.layer17.router.row{expert}"),
            SYNTHETIC_HIDDEN,
        )?;
        router_logits.push(bf16_round_f32(dot_f32(&row, &normalized)?));
    }
    events.push("router_all_8_logits_f32_increasing_k_then_bf16_store".to_owned());
    let probabilities = softmax_f32(&router_logits)?;
    let routes = selected_routes_f32(&probabilities)?;
    events.push("router_softmax_top8_normalize_selected_weights".to_owned());

    let mut combined = vec![0.0f32; SYNTHETIC_HIDDEN];
    for (slot, (expert, weight)) in routes.iter().copied().enumerate() {
        let gate = matrix_vector_f32(
            reader,
            expert,
            "gate",
            SYNTHETIC_INTERMEDIATE,
            SYNTHETIC_HIDDEN,
            &normalized,
        )?;
        let up = matrix_vector_f32(
            reader,
            expert,
            "up",
            SYNTHETIC_INTERMEDIATE,
            SYNTHETIC_HIDDEN,
            &normalized,
        )?;
        let activated = gate
            .iter()
            .zip(&up)
            .map(|(gate, up)| bf16_round_f32((gate / (1.0 + (-gate).exp())) * up))
            .collect::<Vec<_>>();
        let down = matrix_vector_f32(
            reader,
            expert,
            "down",
            SYNTHETIC_HIDDEN,
            SYNTHETIC_INTERMEDIATE,
            &activated,
        )?;
        for hidden in 0..SYNTHETIC_HIDDEN {
            let weighted = bf16_round_f32(down[hidden] * weight);
            combined[hidden] = bf16_round_f32(combined[hidden] + weighted);
        }
        events.push(format!(
            "route_slot_{slot}_expert_{expert}:gate_up_same_current_state_swiglu_down_weight_index_add"
        ));
    }
    let residual = input
        .iter()
        .zip(&combined)
        .map(|(input, moe)| bf16_round_f32(input + moe))
        .collect::<Vec<_>>();
    events.push("residual_input_plus_ordered_index_add_bf16_store".to_owned());
    Ok((
        F32BoundaryOutputs {
            rmsnorm_bf16_store: normalized,
            router_logits_bf16_store: router_logits,
            router_probabilities_bf16_store: probabilities,
            selected_expert_ids_in_router_order: routes.iter().map(|(expert, _)| *expert).collect(),
            normalized_selected_weights_bf16_store: routes
                .iter()
                .map(|(_, weight)| *weight)
                .collect(),
            ordered_index_add_combined_bf16_store: combined,
            residual_f32_bits_sha256: f32_digest(&residual),
            residual_bf16_store: residual,
        },
        events,
    ))
}

fn execute_f64_layer(
    reader: &mut PositionedBf16Reader,
) -> Result<(F64BoundaryOutputs, Vec<String>)> {
    let mut events = vec!["input_residual_bf16_decode".to_owned()];
    let input = reader
        .decode_bf16_vector("fixture.layer17.input_residual", SYNTHETIC_HIDDEN)?
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    let norm_weight = reader
        .decode_bf16_vector("fixture.layer17.rmsnorm_weight", SYNTHETIC_HIDDEN)?
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    let normalized = rmsnorm_f64(&input, &norm_weight)?;
    events.push("rmsnorm_f64_accumulate_then_same_bf16_store".to_owned());

    let mut router_logits = Vec::with_capacity(SYNTHETIC_EXPERTS);
    for expert in 0..SYNTHETIC_EXPERTS {
        let row = reader
            .decode_bf16_vector(
                &format!("fixture.layer17.router.row{expert}"),
                SYNTHETIC_HIDDEN,
            )?
            .into_iter()
            .map(f64::from)
            .collect::<Vec<_>>();
        router_logits.push(bf16_round_f64(dot_f64(&row, &normalized)?));
    }
    events.push("router_all_8_logits_f64_increasing_k_then_same_bf16_store".to_owned());
    let probabilities = softmax_f64(&router_logits)?;
    let routes = selected_routes_f64(&probabilities)?;
    events.push("router_softmax_top8_normalize_selected_weights".to_owned());

    let mut combined = vec![0.0f64; SYNTHETIC_HIDDEN];
    for (slot, (expert, weight)) in routes.iter().copied().enumerate() {
        let gate = matrix_vector_f64(
            reader,
            expert,
            "gate",
            SYNTHETIC_INTERMEDIATE,
            SYNTHETIC_HIDDEN,
            &normalized,
        )?;
        let up = matrix_vector_f64(
            reader,
            expert,
            "up",
            SYNTHETIC_INTERMEDIATE,
            SYNTHETIC_HIDDEN,
            &normalized,
        )?;
        let activated = gate
            .iter()
            .zip(&up)
            .map(|(gate, up)| bf16_round_f64((gate / (1.0 + (-gate).exp())) * up))
            .collect::<Vec<_>>();
        let down = matrix_vector_f64(
            reader,
            expert,
            "down",
            SYNTHETIC_HIDDEN,
            SYNTHETIC_INTERMEDIATE,
            &activated,
        )?;
        for hidden in 0..SYNTHETIC_HIDDEN {
            let weighted = bf16_round_f64(down[hidden] * weight);
            combined[hidden] = bf16_round_f64(combined[hidden] + weighted);
        }
        events.push(format!(
            "route_slot_{slot}_expert_{expert}:gate_up_same_current_state_swiglu_down_weight_index_add"
        ));
    }
    let residual = input
        .iter()
        .zip(&combined)
        .map(|(input, moe)| bf16_round_f64(input + moe))
        .collect::<Vec<_>>();
    events.push("residual_input_plus_ordered_index_add_bf16_store".to_owned());
    Ok((
        F64BoundaryOutputs {
            rmsnorm_bf16_store: normalized,
            router_logits_bf16_store: router_logits,
            router_probabilities_bf16_store: probabilities,
            selected_expert_ids_in_router_order: routes.iter().map(|(expert, _)| *expert).collect(),
            normalized_selected_weights_bf16_store: routes
                .iter()
                .map(|(_, weight)| *weight)
                .collect(),
            ordered_index_add_combined_bf16_store: combined,
            residual_f64_bits_sha256: f64_digest(&residual),
            residual_bf16_store: residual,
        },
        events,
    ))
}

fn add_matrix(
    builder: &mut FixtureShardBuilder,
    expert: usize,
    role: &str,
    rows: usize,
    columns: usize,
    generator: impl Fn(usize, usize) -> f32,
) -> Result<()> {
    for row in 0..rows {
        let values = (0..columns)
            .map(|column| generator(row, column))
            .collect::<Vec<_>>();
        builder.add_bf16_vector(expert_row_name(expert, role, row), &values)?;
    }
    Ok(())
}

/// Build a deliberately tiny and non-model-shaped fixture.  Its values are
/// synthetic formulas, not a slice, quantisation, or transformation of any
/// Q30 source tensor.
fn synthetic_fixture_builder() -> Result<FixtureShardBuilder> {
    let mut builder = FixtureShardBuilder::default();
    builder.add_bf16_vector(
        "fixture.layer17.input_residual",
        &[0.75, -1.125, 0.3125, 1.5],
    )?;
    builder.add_bf16_vector(
        "fixture.layer17.rmsnorm_weight",
        &[0.875, 1.0625, 0.9375, 1.125],
    )?;
    for expert in 0..SYNTHETIC_EXPERTS {
        let row = (0..SYNTHETIC_HIDDEN)
            .map(|hidden| 0.03125 * (expert as f32 + 1.0) + 0.0078125 * (hidden as f32 + 1.0))
            .collect::<Vec<_>>();
        builder.add_bf16_vector(format!("fixture.layer17.router.row{expert}"), &row)?;
        add_matrix(
            &mut builder,
            expert,
            "gate",
            SYNTHETIC_INTERMEDIATE,
            SYNTHETIC_HIDDEN,
            |row, column| {
                let polarity = if (expert + row + column) % 2 == 0 {
                    1.0
                } else {
                    -1.0
                };
                polarity
                    * (0.015625 * (expert as f32 + 1.0) + 0.0078125 * (row + column + 1) as f32)
            },
        )?;
        add_matrix(
            &mut builder,
            expert,
            "up",
            SYNTHETIC_INTERMEDIATE,
            SYNTHETIC_HIDDEN,
            |row, column| {
                0.01171875 * (expert as f32 + 1.0) + 0.00390625 * (row * 2 + column + 1) as f32
            },
        )?;
        add_matrix(
            &mut builder,
            expert,
            "down",
            SYNTHETIC_HIDDEN,
            SYNTHETIC_INTERMEDIATE,
            |row, column| {
                let polarity = if (expert + row) % 2 == 0 { 1.0 } else { -1.0 };
                polarity
                    * (0.01953125 * (column as f32 + 1.0) + 0.001953125 * (expert + row + 1) as f32)
            },
        )?;
    }
    Ok(builder)
}

#[derive(Debug, Serialize)]
struct FixtureRun {
    fixture_geometry: FixtureGeometry,
    synthetic_f32_boundary_policy: Vec<&'static str>,
    f32: F32BoundaryOutputs,
    f64: F64BoundaryOutputs,
    f32_reader: ReaderStats,
    f64_reader: ReaderStats,
    f32_operator_events: Vec<String>,
    f64_operator_events: Vec<String>,
}

#[derive(Debug, Serialize)]
struct FixtureGeometry {
    representative_layer_index: usize,
    hidden_size: usize,
    router_experts: usize,
    top_k: usize,
    expert_intermediate: usize,
}

fn run_fixture_layer() -> Result<FixtureRun> {
    let builder = synthetic_fixture_builder()?;
    let maximum_declared_range_bytes = builder.maximum_declared_range_bytes();
    if maximum_declared_range_bytes == 0 || maximum_declared_range_bytes > MAX_WINDOW_BYTES {
        return Err("fixture range declaration does not fit the one-window bound".into());
    }
    // Rebuild from the same deterministic formula for the independent F32 and
    // F64 passes.  They do not share an open reader, raw window, or state.
    let mut f32_reader = builder.materialize()?;
    let (f32, f32_operator_events) = execute_f32_layer(&mut f32_reader)?;
    let f32_stats = f32_reader.stats();
    drop(f32_reader);
    let mut f64_reader = synthetic_fixture_builder()?.materialize()?;
    let (f64, f64_operator_events) = execute_f64_layer(&mut f64_reader)?;
    let f64_stats = f64_reader.stats();
    if f32.selected_expert_ids_in_router_order != f64.selected_expert_ids_in_router_order {
        return Err(
            "synthetic F32/F64 router ordering drifted; fixture is not stable enough".into(),
        );
    }
    for stats in [&f32_stats, &f64_stats] {
        if stats.maximum_live_raw_bf16_window_bytes > MAX_WINDOW_BYTES
            || stats.maximum_declared_range_bytes > MAX_WINDOW_BYTES
            || stats.simultaneous_raw_bf16_windows != 1
            || !stats.all_reads_verified_by_range_sha256
        {
            return Err("fixture reader violated the one verified <=1 MiB window policy".into());
        }
    }
    Ok(FixtureRun {
        fixture_geometry: FixtureGeometry {
            representative_layer_index: 17,
            hidden_size: SYNTHETIC_HIDDEN,
            router_experts: SYNTHETIC_EXPERTS,
            top_k: SYNTHETIC_TOP_K,
            expert_intermediate: SYNTHETIC_INTERMEDIATE,
        },
        synthetic_f32_boundary_policy: vec![
            "all synthetic weights and input are decoded from BF16 by explicit little-endian positioned reads",
            "RMSNorm and every scalar dot accumulate in increasing k order before a synthetic BF16 store boundary",
            "router softmax/top-8 weights, SwiGLU, route weighting, each ordered index-add, and final residual are explicitly rounded to synthetic BF16 stores",
            "the F64 pass uses the same decoded values and BF16 store boundaries only as a deterministic diagnostic reference, not as source equivalence evidence",
        ],
        f32,
        f64,
        f32_reader: f32_stats,
        f64_reader: f64_stats,
        f32_operator_events,
        f64_operator_events,
    })
}

fn result_document(bindings: ExternalContractBindings, run: FixtureRun) -> Value {
    json!({
        "schema": RESULT_SCHEMA,
        "status": RESULT_STATUS,
        "fixture_only": true,
        "claim_boundary": {
            "synthetic_range_map_and_weights_only": true,
            "source_tensor_payload_opened": false,
            "source_model_instantiated": false,
            "gpu_or_metal_invoked": false,
            "server_or_hcli_invoked": false,
            "lease_requested_or_issued": false,
            "tps_tg_or_tournament_claim_made": false,
            "not_source_execution_or_source_equivalence": true
        },
        "external_contract_bindings": bindings,
        "fixture_run": run,
        "result_interpretation": "Prepared synthetic single-layer BF16 positioned-read primitive. It proves only its own bounded fixture reader and deterministic scalar traces. A future separately leased Q30 source executor must independently prove source payload admission, row offsets, all-layer semantics, cache/attention behavior, real source boundary dtypes, and final-logit parity."
    })
}

fn write_new_json(path: &Path, value: &Value) -> Result<()> {
    if !path.is_absolute() || !path.parent().is_some_and(Path::is_dir) || path.exists() {
        return Err(format!(
            "refusing to overwrite or relativize fixture receipt {}",
            path.display()
        ));
    }
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("cannot serialize fixture receipt: {error}"))?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create fixture receipt {}: {error}", path.display()))?;
    file.write_all(&bytes)
        .and_then(|_| file.write_all(b"\n"))
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("cannot persist fixture receipt {}: {error}", path.display()))?;
    Ok(())
}

fn main() {
    let result = (|| -> Result<()> {
        let args = parse_args()?;
        let semantics = read_metadata_document(&args.semantics_contract, "--semantics-contract")?;
        let range_authority = read_metadata_document(&args.range_authority, "--range-authority")?;
        let executor = read_metadata_document(&args.executor_contract, "--executor-contract")?;
        let bindings = validate_external_contracts(&semantics, &range_authority, &executor)?;
        let run = run_fixture_layer()?;
        write_new_json(&args.out, &result_document(bindings, run))
    })();
    if let Err(error) = result {
        eprintln!("Q30 streamed-BF16 single-layer fixture refused: {error}");
        process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn loaded(value: Value, name: &str) -> LoadedDocument {
        let bytes = serde_json::to_vec(&value).expect("serialize fixture contract");
        LoadedDocument {
            path: PathBuf::from(format!("/synthetic/{name}.json")),
            sha256: sha256_hex(&bytes),
            bytes,
            value,
        }
    }

    fn external_contracts() -> (LoadedDocument, LoadedDocument, LoadedDocument) {
        let semantics = loaded(
            json!({
                "schema": SEMANTICS_SCHEMA,
                "status": SEMANTICS_STATUS,
                "exact_scalar_and_order_requirements": {
                    "rmsnorm": {"required_sequence": ["f32_accumulate"]},
                    "router_topk_and_weights": {"required_sequence": ["softmax_then_topk"]},
                    "selected_expert_gate_up_swiglu_down": {"required_sequence": ["gate_up_swiglu_down_index_add"]},
                    "residuals": {"required_sequence": ["second_residual"]}
                }
            }),
            "semantics",
        );
        let range = loaded(
            json!({
                "authority": {
                    "schema": RANGE_AUTHORITY_SCHEMA,
                    "status": RANGE_AUTHORITY_STATUS,
                    "tensors": []
                }
            }),
            "range",
        );
        let executor = loaded(
            json!({
                "schema": OUTER_EXECUTOR_SCHEMA,
                "status": OUTER_EXECUTOR_STATUS,
                "claim_boundary": {
                    "metadata_only_preflight": true,
                    "does_not_open_source_tensor_payloads_or_load_a_source_model": true
                },
                "metadata_only_operator_semantics": {"sha256": semantics.sha256},
                "metadata_only_range_authority": {"sha256": range.sha256},
                "future_source_launch_contract": {
                    "maximum_source_reader_cached_bytes": MAX_WINDOW_BYTES,
                    "maximum_source_reader_cached_windows": 1,
                    "actual_streamed_executor_present": false
                }
            }),
            "executor",
        );
        (semantics, range, executor)
    }

    #[test]
    fn bf16_decode_and_rounding_are_exact_for_known_words() {
        assert_eq!(bf16_bits_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_bits_to_f32(0xbf80), -1.0);
        assert_eq!(f32_to_bf16_bits(1.0), 0x3f80);
        assert_eq!(bf16_round_f32(1.00390625), 1.0);
    }

    #[test]
    fn external_contracts_are_byte_bound_and_currently_blocked() {
        let (semantics, range, executor) = external_contracts();
        let bindings = validate_external_contracts(&semantics, &range, &executor)
            .expect("synthetic documents model the external prepared contracts");
        assert_eq!(
            bindings.maximum_source_reader_cached_bytes,
            MAX_WINDOW_BYTES
        );
        assert_eq!(bindings.maximum_source_reader_cached_windows, 1);
        assert!(!bindings.actual_streamed_executor_present);

        let mut drifted_value = executor.value.clone();
        drifted_value["future_source_launch_contract"]["maximum_source_reader_cached_bytes"] =
            json!(MAX_WINDOW_BYTES + 2);
        let drifted = loaded(drifted_value, "executor-drifted");
        assert!(validate_external_contracts(&semantics, &range, &drifted).is_err());
    }

    #[test]
    fn fixture_execution_is_deterministic_and_never_exceeds_one_mib() {
        let first = run_fixture_layer().expect("first synthetic pass");
        let second = run_fixture_layer().expect("second synthetic pass");
        assert_eq!(
            first.f32.residual_f32_bits_sha256,
            second.f32.residual_f32_bits_sha256
        );
        assert_eq!(
            first.f64.residual_f64_bits_sha256,
            second.f64.residual_f64_bits_sha256
        );
        assert_eq!(
            first.f32.selected_expert_ids_in_router_order,
            first.f64.selected_expert_ids_in_router_order
        );
        assert_eq!(
            first.f32.selected_expert_ids_in_router_order.len(),
            SYNTHETIC_TOP_K
        );
        for stats in [&first.f32_reader, &first.f64_reader] {
            assert!(stats.positioned_read_calls > SYNTHETIC_EXPERTS);
            assert!(stats.maximum_live_raw_bf16_window_bytes <= MAX_WINDOW_BYTES);
            assert!(stats.maximum_declared_range_bytes <= MAX_WINDOW_BYTES);
            assert_eq!(stats.simultaneous_raw_bf16_windows, 1);
            assert!(stats.all_reads_verified_by_range_sha256);
        }
        assert!(
            first
                .f32_operator_events
                .iter()
                .any(|event| event
                    .contains("gate_up_same_current_state_swiglu_down_weight_index_add"))
        );
    }

    #[test]
    fn reader_refuses_a_range_larger_than_the_one_window_cap() {
        let mut builder = FixtureShardBuilder::default();
        let values = vec![0.5f32; MAX_WINDOW_BYTES / 2 + 1];
        let error = builder
            .add_bf16_vector("oversized.synthetic.row", &values)
            .expect_err("a >1MiB BF16 row must be rejected at range-map construction");
        assert!(error.contains("one-window"));
    }

    #[test]
    fn cli_refuses_to_replay_or_write_relative_receipts() {
        let root = TempDir::new().expect("temporary fixture root");
        let receipt = root.path().join("fixture-receipt.json");
        let value = json!({"fixture_only": true});
        write_new_json(&receipt, &value).expect("create-new fixture receipt");
        assert!(write_new_json(&receipt, &value).is_err());
        assert!(write_new_json(Path::new("relative.json"), &value).is_err());
    }
}
