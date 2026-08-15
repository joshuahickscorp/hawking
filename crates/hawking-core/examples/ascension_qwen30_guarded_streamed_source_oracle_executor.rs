#![allow(dead_code)] // The fixture-only executor exposes testable readiness interfaces.

//! Fixture-only readiness executor for the guarded Qwen30 streamed-source oracle.
//!
//! This binary deliberately has no real-source execution surface.  It does
//! not accept a source root, safetensors path, model command, GPU setting,
//! server endpoint, HCLI request, or benchmark option.  Its two modes are:
//!
//! * `--mode preflight`: validate a fresh, sealed **fixture-only** outer lease
//!   and write an immutable readiness receipt;
//! * `--mode fixture-only`: exercise a small in-memory range-map/weight
//!   fixture, write six small F32LE fixture payloads, and prove receipt-last
//!   source -> eviction -> native sequencing.
//!
//! The exact source-BF16 teacher remains blocked behind its own future lease,
//! memory evidence, and separately reviewed implementation.  In particular,
//! this program makes no source-quality, coherence, HCLI, TPS, TG, capability,
//! serving, promotion, or tournament claim.

use serde::Serialize;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process;

const PRELIGHT_SCHEMA: &str =
    "hawking.ascension.qwen30_guarded_streamed_source_oracle_executor_readiness.v1";
const PRELIGHT_STATUS: &str =
    "PREPARED_QWEN30_GUARDED_STREAMED_SOURCE_ORACLE_EXECUTOR_FIXTURE_ONLY";
const FIXTURE_CAPTURE_SCHEMA: &str =
    "hawking.ascension.qwen30_guarded_streamed_source_oracle_fixture_capture.v1";
const FIXTURE_CAPTURE_STATUS: &str =
    "CAPTURED_QWEN30_GUARDED_STREAMED_SOURCE_ORACLE_FIXTURE_ONLY_NOT_SOURCE_TEACHER";
const FIXTURE_SOURCE_TERMINAL_SCHEMA: &str =
    "hawking.ascension.qwen30_guarded_streamed_source_oracle_fixture_source_terminal.v1";
const FIXTURE_EVICTION_SCHEMA: &str =
    "hawking.ascension.qwen30_guarded_streamed_source_oracle_fixture_eviction.v1";

const SOURCE_LEASE_SCHEMA: &str =
    "hawking.ascension.qwen30_hq30gr2_source_bf16_teacher_quiet_lease.v1";
const SOURCE_LEASE_STATUS: &str =
    "GRANTED_QWEN30_HQ30GR2_SOURCE_BF16_TEACHER_RAW_LOGIT_CAPTURE_ONE_SHOT";
const OUTER_CONTROLLER_SCHEMA: &str =
    "hawking.ascension.qwen30_guarded_streamed_source_oracle_outer_controller.v1";
const SEMANTICS_CONTRACT_SCHEMA: &str =
    "hawking.ascension.qwen30_layer_streamed_source_operator_accumulation_semantics_attester.v1";
const SEMANTICS_CONTRACT_STATUS: &str =
    "PREPARED_QWEN30_LAYER_STREAMED_SOURCE_OPERATOR_ACCUMULATION_SEMANTICS_NOT_EXECUTED";
const MAX_LEASE_METADATA_BYTES: u64 = 1024 * 1024;
const MAX_FIXTURE_WINDOW_BYTES: usize = 4096;
const FIXTURE_VOCAB_ROWS: usize = 4;
const FIXTURE_LAYERS: usize = 2;
const FIXTURE_TOP_K: usize = 2;
const QWEN30_SEMANTIC_LAYERS: usize = 48;
const QWEN30_SEMANTIC_TOP_K: usize = 8;

const SOURCE_PAYLOADS: [(&str, &str); 2] = [
    ("exact_prefix", "source_bf16_exact_prefix_logits.f32le"),
    (
        "forced_shared_continuation",
        "source_bf16_forced_shared_continuation_logits.f32le",
    ),
];
const NATIVE_PAYLOADS: [(&str, &str, &str); 4] = [
    (
        "scalar_control",
        "exact_prefix",
        "scalar_control_exact_prefix_logits.f32le",
    ),
    (
        "scalar_control",
        "forced_shared_continuation",
        "scalar_control_forced_shared_continuation_logits.f32le",
    ),
    (
        "hq30gr2_candidate",
        "exact_prefix",
        "hq30gr2_candidate_exact_prefix_logits.f32le",
    ),
    (
        "hq30gr2_candidate",
        "forced_shared_continuation",
        "hq30gr2_candidate_forced_shared_continuation_logits.f32le",
    ),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CliMode {
    Preflight,
    FixtureOnly,
}

#[derive(Debug)]
struct Args {
    mode: CliMode,
    outer_lease: PathBuf,
    out: Option<PathBuf>,
    capture_dir: Option<PathBuf>,
}

fn usage() -> &'static str {
    "usage: ascension_qwen30_guarded_streamed_source_oracle_executor --mode preflight --outer-lease ABSOLUTE_SEALED_FIXTURE_LEASE_JSON --out NEW_ABSOLUTE_RECEIPT_JSON | --mode fixture-only --outer-lease ABSOLUTE_SEALED_FIXTURE_LEASE_JSON --capture-dir NEW_ABSOLUTE_DIRECTORY"
}

fn parse_args_from<I>(arguments: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut mode = None;
    let mut outer_lease = None;
    let mut out = None;
    let mut capture_dir = None;
    let mut values = arguments.into_iter();
    while let Some(flag) = values.next() {
        let next = |flag: &str, values: &mut I::IntoIter| {
            values
                .next()
                .ok_or_else(|| format!("missing value for {flag}; {}", usage()))
        };
        match flag.as_str() {
            "--mode" => {
                let value = next("--mode", &mut values)?;
                if mode.is_some() {
                    return Err(format!("--mode was supplied more than once; {}", usage()));
                }
                mode = Some(match value.as_str() {
                    "preflight" => CliMode::Preflight,
                    "fixture-only" => CliMode::FixtureOnly,
                    _ => {
                        return Err(format!(
                            "--mode must be preflight or fixture-only; {}",
                            usage()
                        ))
                    }
                });
            }
            "--outer-lease" => {
                let value = next("--outer-lease", &mut values)?;
                if outer_lease.replace(PathBuf::from(value)).is_some() {
                    return Err(format!(
                        "--outer-lease was supplied more than once; {}",
                        usage()
                    ));
                }
            }
            "--out" => {
                let value = next("--out", &mut values)?;
                if out.replace(PathBuf::from(value)).is_some() {
                    return Err(format!("--out was supplied more than once; {}", usage()));
                }
            }
            "--capture-dir" => {
                let value = next("--capture-dir", &mut values)?;
                if capture_dir.replace(PathBuf::from(value)).is_some() {
                    return Err(format!(
                        "--capture-dir was supplied more than once; {}",
                        usage()
                    ));
                }
            }
            "--help" | "-h" => return Err(usage().to_owned()),
            _ => return Err(format!("unsupported option {flag:?}; {}", usage())),
        }
    }
    let mode = mode.ok_or_else(|| format!("--mode is required; {}", usage()))?;
    let outer_lease =
        outer_lease.ok_or_else(|| format!("--outer-lease is required; {}", usage()))?;
    if !outer_lease.is_absolute() {
        return Err("--outer-lease must be absolute".into());
    }
    match mode {
        CliMode::Preflight => {
            if capture_dir.is_some() {
                return Err("--capture-dir is only valid with --mode fixture-only".into());
            }
            let out = out.ok_or_else(|| "--out is required with --mode preflight".to_owned())?;
            if !out.is_absolute() || !out.parent().is_some_and(Path::is_dir) {
                return Err("--out must be absolute and its parent must already exist".into());
            }
            Ok(Args {
                mode,
                outer_lease,
                out: Some(out),
                capture_dir: None,
            })
        }
        CliMode::FixtureOnly => {
            if out.is_some() {
                return Err("--out is only valid with --mode preflight".into());
            }
            let capture_dir = capture_dir
                .ok_or_else(|| "--capture-dir is required with --mode fixture-only".to_owned())?;
            if !capture_dir.is_absolute()
                || !capture_dir.parent().is_some_and(Path::is_dir)
                || capture_dir.exists()
            {
                return Err(
                    "--capture-dir must be a new absolute directory whose parent already exists"
                        .into(),
                );
            }
            Ok(Args {
                mode,
                outer_lease,
                out: None,
                capture_dir: Some(capture_dir),
            })
        }
    }
}

fn parse_args() -> Result<Args, String> {
    parse_args_from(env::args().skip(1))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

/// Canonical JSON compatible with `lab.receipts.seal`: sorted object keys and
/// compact separators.  Fixtures use this so lease/receipt tests exercise the
/// same tamper-evidence boundary as the outer controller.
fn canonical_json(value: &Value) -> Result<String, String> {
    match value {
        Value::Null => Ok("null".to_owned()),
        Value::Bool(value) => Ok(value.to_string()),
        Value::Number(value) => Ok(value.to_string()),
        Value::String(value) => serde_json::to_string(value)
            .map_err(|error| format!("cannot canonicalize JSON string: {error}")),
        Value::Array(values) => {
            let mut output = String::from("[");
            for (index, item) in values.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                output.push_str(&canonical_json(item)?);
            }
            output.push(']');
            Ok(output)
        }
        Value::Object(values) => {
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            let mut output = String::from("{");
            for (index, key) in keys.into_iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                output.push_str(
                    &serde_json::to_string(key)
                        .map_err(|error| format!("cannot canonicalize JSON key: {error}"))?,
                );
                output.push(':');
                output.push_str(&canonical_json(
                    values
                        .get(key)
                        .ok_or_else(|| "canonical JSON key disappeared".to_owned())?,
                )?);
            }
            output.push('}');
            Ok(output)
        }
    }
}

fn seal_value(mut value: Value) -> Result<Value, String> {
    let object = value
        .as_object_mut()
        .ok_or_else(|| "only JSON objects may be sealed".to_owned())?;
    if object.contains_key("seal_sha256") {
        return Err("refusing to reseal a document that already has seal_sha256".into());
    }
    let seal = sha256_hex(canonical_json(&Value::Object(object.clone()))?.as_bytes());
    object.insert("seal_sha256".into(), Value::String(seal));
    Ok(value)
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be a JSON object"))
}

fn required<'a>(map: &'a Map<String, Value>, key: &str, label: &str) -> Result<&'a Value, String> {
    map.get(key)
        .ok_or_else(|| format!("{label}.{key} is required"))
}

fn string<'a>(value: &'a Value, label: &str) -> Result<&'a str, String> {
    value
        .as_str()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{label} must be a non-empty string"))
}

fn bool_value(value: &Value, label: &str) -> Result<bool, String> {
    value
        .as_bool()
        .ok_or_else(|| format!("{label} must be a boolean"))
}

fn u64_value(value: &Value, label: &str) -> Result<u64, String> {
    value
        .as_u64()
        .ok_or_else(|| format!("{label} must be a non-negative integer"))
}

fn verify_seal(document: &Value, label: &str) -> Result<String, String> {
    let root = object(document, label)?;
    let recorded = string(
        required(root, "seal_sha256", label)?,
        &format!("{label}.seal_sha256"),
    )?;
    if !is_sha256(recorded) {
        return Err(format!("{label}.seal_sha256 must be a lowercase SHA-256"));
    }
    let mut unsigned = root.clone();
    unsigned.remove("seal_sha256");
    let computed = sha256_hex(canonical_json(&Value::Object(unsigned))?.as_bytes());
    if computed != recorded {
        return Err(format!("{label} seal mismatch"));
    }
    Ok(recorded.to_owned())
}

fn read_sealed_lease(path: &Path) -> Result<(Value, String), String> {
    if !path.is_absolute() || path.extension().and_then(|value| value.to_str()) != Some("json") {
        return Err("outer lease must be an absolute .json metadata receipt".into());
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat outer lease {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err("outer lease must be a regular non-symlink file".into());
    }
    if metadata.len() == 0 || metadata.len() > MAX_LEASE_METADATA_BYTES {
        return Err(format!(
            "outer lease must contain 1..={MAX_LEASE_METADATA_BYTES} bytes"
        ));
    }
    let mut bytes =
        vec![0u8; usize::try_from(metadata.len()).map_err(|_| "outer lease is too large")?];
    File::open(path)
        .and_then(|mut file| file.read_exact(&mut bytes))
        .map_err(|error| format!("cannot read outer lease {}: {error}", path.display()))?;
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("outer lease is not valid JSON: {error}"))?;
    let seal = verify_seal(&value, "outer lease")?;
    Ok((value, seal))
}

#[derive(Clone, Debug)]
struct FreshFixtureLease {
    seal_sha256: String,
    nonce: String,
    minimum_reclaimable_bytes: u64,
}

/// This deliberately accepts only a synthetic lease family.  A real Q30
/// lease is rejected rather than being accidentally interpreted as authority
/// for this fixture executable.
fn validate_fresh_fixture_outer_lease(path: &Path) -> Result<FreshFixtureLease, String> {
    let (document, seal_sha256) = read_sealed_lease(path)?;
    let root = object(&document, "outer lease")?;
    if string(
        required(root, "schema", "outer lease")?,
        "outer lease.schema",
    )? != SOURCE_LEASE_SCHEMA
        || string(
            required(root, "status", "outer lease")?,
            "outer lease.status",
        )? != SOURCE_LEASE_STATUS
    {
        return Err(
            "outer lease schema/status does not authorize the source-teacher one-shot family"
                .into(),
        );
    }
    if !bool_value(
        required(root, "fixture_only", "outer lease")?,
        "outer lease.fixture_only",
    )? || !bool_value(
        required(
            root,
            "real_source_payload_execution_forbidden",
            "outer lease",
        )?,
        "outer lease.real_source_payload_execution_forbidden",
    )? {
        return Err(
            "this executor requires an explicit fixture-only, no-real-payload lease".into(),
        );
    }
    let outer = object(
        required(root, "outer_controller_validation", "outer lease")?,
        "outer lease.outer_controller_validation",
    )?;
    if string(
        required(outer, "schema", "outer lease.outer_controller_validation")?,
        "outer lease.outer_controller_validation.schema",
    )? != OUTER_CONTROLLER_SCHEMA
        || !bool_value(
            required(
                outer,
                "fresh_validated_outer_lease",
                "outer lease.outer_controller_validation",
            )?,
            "outer lease.outer_controller_validation.fresh_validated_outer_lease",
        )?
        || !bool_value(
            required(
                outer,
                "fixture_only",
                "outer lease.outer_controller_validation",
            )?,
            "outer lease.outer_controller_validation.fixture_only",
        )?
    {
        return Err(
            "outer lease lacks a fresh validated fixture-only outer-controller binding".into(),
        );
    }
    let lifecycle = object(
        required(root, "one_shot_lifecycle", "outer lease")?,
        "outer lease.one_shot_lifecycle",
    )?;
    for key in [
        "fresh_for_this_exact_launch",
        "new_capture_root",
        "existing_output_reuse_forbidden",
        "replay_or_relaunch_forbidden",
    ] {
        if !bool_value(
            required(lifecycle, key, "outer lease.one_shot_lifecycle")?,
            &format!("outer lease.one_shot_lifecycle.{key}"),
        )? {
            return Err(format!("outer lease one-shot lifecycle {key} must be true"));
        }
    }
    if bool_value(
        required(
            lifecycle,
            "automatic_retry_allowed",
            "outer lease.one_shot_lifecycle",
        )?,
        "outer lease.one_shot_lifecycle.automatic_retry_allowed",
    )? {
        return Err("outer lease must forbid automatic retry".into());
    }
    if !required(
        lifecycle,
        "prior_terminal_receipt",
        "outer lease.one_shot_lifecycle",
    )?
    .is_null()
    {
        return Err("outer lease may not reuse a terminal receipt".into());
    }
    let nonce = string(
        required(
            lifecycle,
            "exact_launch_nonce",
            "outer lease.one_shot_lifecycle",
        )?,
        "outer lease.one_shot_lifecycle.exact_launch_nonce",
    )?;
    if !is_sha256(nonce) {
        return Err("outer lease exact launch nonce must be a lowercase SHA-256".into());
    }
    let safety = object(
        required(root, "fresh_pre_child_safety", "outer lease")?,
        "outer lease.fresh_pre_child_safety",
    )?;
    for key in [
        "observed_immediately_before_child",
        "exclusive_clean_window",
        "no_source_or_native_model_body_resident_before_child",
    ] {
        if !bool_value(
            required(safety, key, "outer lease.fresh_pre_child_safety")?,
            &format!("outer lease.fresh_pre_child_safety.{key}"),
        )? {
            return Err(format!("outer lease fresh safety {key} must be true"));
        }
    }
    if u64_value(
        required(
            safety,
            "swap_used_bytes",
            "outer lease.fresh_pre_child_safety",
        )?,
        "outer lease.fresh_pre_child_safety.swap_used_bytes",
    )? != 0
        || u64_value(
            required(
                safety,
                "swapouts_pages_delta",
                "outer lease.fresh_pre_child_safety",
            )?,
            "outer lease.fresh_pre_child_safety.swapouts_pages_delta",
        )? != 0
    {
        return Err("outer lease must show zero swap and zero swapout growth".into());
    }
    let minimum_reclaimable_bytes = u64_value(
        required(
            safety,
            "minimum_reclaimable_bytes_required",
            "outer lease.fresh_pre_child_safety",
        )?,
        "outer lease.fresh_pre_child_safety.minimum_reclaimable_bytes_required",
    )?;
    if minimum_reclaimable_bytes == 0
        || u64_value(
            required(
                safety,
                "reclaimable_bytes",
                "outer lease.fresh_pre_child_safety",
            )?,
            "outer lease.fresh_pre_child_safety.reclaimable_bytes",
        )? < minimum_reclaimable_bytes
    {
        return Err("outer lease reclaimable-memory safety bound is invalid".into());
    }
    Ok(FreshFixtureLease {
        seal_sha256,
        nonce: nonce.to_owned(),
        minimum_reclaimable_bytes,
    })
}

#[derive(Clone, Debug)]
struct FixtureRangeRecord {
    name: String,
    shard_id: String,
    offset: usize,
    bytes: usize,
    sha256: String,
}

#[derive(Clone, Debug)]
struct FixtureRangeMap {
    maximum_window_bytes: usize,
    records: BTreeMap<String, FixtureRangeRecord>,
}

impl FixtureRangeMap {
    fn new(maximum_window_bytes: usize, records: Vec<FixtureRangeRecord>) -> Result<Self, String> {
        if maximum_window_bytes == 0 || maximum_window_bytes > MAX_FIXTURE_WINDOW_BYTES {
            return Err(format!(
                "fixture range window must be 1..={MAX_FIXTURE_WINDOW_BYTES} bytes"
            ));
        }
        let mut indexed = BTreeMap::new();
        for record in records {
            if record.name.is_empty() || record.shard_id.is_empty() || record.bytes == 0 {
                return Err(
                    "fixture range records require non-empty names/shards and positive bytes"
                        .into(),
                );
            }
            if record.bytes > maximum_window_bytes || !is_sha256(&record.sha256) {
                return Err(
                    "fixture range record exceeds its bounded window or has an invalid SHA-256"
                        .into(),
                );
            }
            if indexed.insert(record.name.clone(), record).is_some() {
                return Err("fixture range map has a duplicate range name".into());
            }
        }
        if indexed.is_empty() {
            return Err("fixture range map must contain at least one range".into());
        }
        Ok(Self {
            maximum_window_bytes,
            records: indexed,
        })
    }

    fn record(&self, name: &str) -> Result<&FixtureRangeRecord, String> {
        self.records
            .get(name)
            .ok_or_else(|| format!("fixture range map does not contain {name:?}"))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct PerShardReadAccounting {
    relative_path: String,
    payload_bytes_read: usize,
    read_calls: usize,
    whole_shard_read_as_one_window: bool,
    whole_shard_cached: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct BoundedCacheStats {
    maximum_allowed_window_bytes: usize,
    maximum_observed_window_bytes: usize,
    maximum_cached_bytes: usize,
    maximum_cached_windows: usize,
    eviction_on_each_read_completion: bool,
    complete_source_shard_mapped_or_cached: bool,
    mmap_or_memory_map_used: bool,
    cache_cleared_at_sequence_eviction: bool,
}

/// The only source-like bytes this binary can see are embedded synthetic
/// fixtures.  A callback receives one verified range at a time; it cannot
/// retain a shard handle or ask the reader to map an arbitrary path.
trait ExactBoundedRangeReader {
    fn with_verified_range<R>(
        &mut self,
        range_name: &str,
        operation: impl FnOnce(&[u8]) -> Result<R, String>,
    ) -> Result<R, String>;
    fn clear_for_eviction(&mut self);
    fn cache_stats(&self) -> BoundedCacheStats;
    fn per_shard_accounting(&self) -> Vec<PerShardReadAccounting>;
    fn cache_is_clear(&self) -> bool;
}

#[derive(Debug)]
struct FixtureBoundedRangeReader {
    range_map: FixtureRangeMap,
    shards: BTreeMap<String, Vec<u8>>,
    cache: Vec<u8>,
    maximum_observed_window_bytes: usize,
    maximum_cached_bytes: usize,
    reads: BTreeMap<String, (usize, usize)>,
    cache_cleared_at_sequence_eviction: bool,
}

impl FixtureBoundedRangeReader {
    fn new(range_map: FixtureRangeMap, shards: BTreeMap<String, Vec<u8>>) -> Result<Self, String> {
        for record in range_map.records.values() {
            let shard = shards
                .get(&record.shard_id)
                .ok_or_else(|| format!("fixture range {} names a missing shard", record.name))?;
            let end = record
                .offset
                .checked_add(record.bytes)
                .ok_or_else(|| "fixture range offset overflow".to_owned())?;
            if end > shard.len() {
                return Err(format!(
                    "fixture range {} exceeds its synthetic shard",
                    record.name
                ));
            }
        }
        Ok(Self {
            range_map,
            shards,
            cache: Vec::new(),
            maximum_observed_window_bytes: 0,
            maximum_cached_bytes: 0,
            reads: BTreeMap::new(),
            cache_cleared_at_sequence_eviction: false,
        })
    }

    fn clear_cache(&mut self) {
        self.cache.fill(0);
        self.cache.clear();
    }
}

impl ExactBoundedRangeReader for FixtureBoundedRangeReader {
    fn with_verified_range<R>(
        &mut self,
        range_name: &str,
        operation: impl FnOnce(&[u8]) -> Result<R, String>,
    ) -> Result<R, String> {
        let record = self.range_map.record(range_name)?.clone();
        if record.bytes > self.range_map.maximum_window_bytes {
            return Err("fixture range exceeds the declared one-window bound".into());
        }
        self.clear_cache();
        let shard = self
            .shards
            .get(&record.shard_id)
            .ok_or_else(|| "fixture shard disappeared after reader admission".to_owned())?;
        let end = record
            .offset
            .checked_add(record.bytes)
            .ok_or_else(|| "fixture range end overflow".to_owned())?;
        let bytes = shard
            .get(record.offset..end)
            .ok_or_else(|| "fixture range moved outside its synthetic shard".to_owned())?;
        self.cache.extend_from_slice(bytes);
        self.maximum_observed_window_bytes =
            self.maximum_observed_window_bytes.max(self.cache.len());
        self.maximum_cached_bytes = self.maximum_cached_bytes.max(self.cache.len());
        let entry = self.reads.entry(record.shard_id.clone()).or_insert((0, 0));
        entry.0 = entry
            .0
            .checked_add(self.cache.len())
            .ok_or_else(|| "fixture byte accounting overflow".to_owned())?;
        entry.1 = entry
            .1
            .checked_add(1)
            .ok_or_else(|| "fixture call accounting overflow".to_owned())?;
        if sha256_hex(&self.cache) != record.sha256 {
            self.clear_cache();
            return Err(format!("fixture range {} checksum drifted", record.name));
        }
        let result = operation(&self.cache);
        // Per-read eviction is mandatory.  The callback cannot obtain a
        // second range until this one has been cleared.
        self.clear_cache();
        result
    }

    fn clear_for_eviction(&mut self) {
        self.clear_cache();
        self.cache_cleared_at_sequence_eviction = true;
    }

    fn cache_stats(&self) -> BoundedCacheStats {
        BoundedCacheStats {
            maximum_allowed_window_bytes: self.range_map.maximum_window_bytes,
            maximum_observed_window_bytes: self.maximum_observed_window_bytes,
            maximum_cached_bytes: self.maximum_cached_bytes,
            maximum_cached_windows: 1,
            eviction_on_each_read_completion: true,
            complete_source_shard_mapped_or_cached: false,
            mmap_or_memory_map_used: false,
            cache_cleared_at_sequence_eviction: self.cache_cleared_at_sequence_eviction,
        }
    }

    fn per_shard_accounting(&self) -> Vec<PerShardReadAccounting> {
        self.reads
            .iter()
            .map(
                |(shard_id, (payload_bytes_read, read_calls))| PerShardReadAccounting {
                    relative_path: shard_id.clone(),
                    payload_bytes_read: *payload_bytes_read,
                    read_calls: *read_calls,
                    whole_shard_read_as_one_window: false,
                    whole_shard_cached: false,
                },
            )
            .collect()
    }

    fn cache_is_clear(&self) -> bool {
        self.cache.is_empty()
    }
}

fn synthetic_fixture_reader() -> Result<FixtureBoundedRangeReader, String> {
    let shard_id = "fixture-shard-00001.safetensors-not-real".to_owned();
    let shard = vec![
        0x3f, 0x80, 0x40, 0x00, 0x40, 0x40, 0x40, 0x80, // embedded synthetic bytes
        0x40, 0xa0, 0x40, 0xc0, 0x40, 0xe0, 0x41, 0x00, 0x41, 0x10, 0x41, 0x20, 0x41, 0x30, 0x41,
        0x40,
    ];
    let records = vec![
        FixtureRangeRecord {
            name: "fixture.embedding".into(),
            shard_id: shard_id.clone(),
            offset: 0,
            bytes: 8,
            sha256: sha256_hex(&shard[0..8]),
        },
        FixtureRangeRecord {
            name: "fixture.linear".into(),
            shard_id: shard_id.clone(),
            offset: 8,
            bytes: 16,
            sha256: sha256_hex(&shard[8..24]),
        },
    ];
    let mut shards = BTreeMap::new();
    shards.insert(shard_id, shard);
    FixtureBoundedRangeReader::new(FixtureRangeMap::new(16, records)?, shards)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum TracePhase {
    ExactPrefix,
    ForcedSharedContinuation,
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct TraceInvocation {
    forward_index: usize,
    phase: TracePhase,
    layer_index: Option<usize>,
    route_slot: Option<usize>,
    operator: SourceOperator,
}

#[derive(Clone, Copy, Debug)]
struct SemanticGeometry {
    layers: usize,
    top_k: usize,
}

fn layer_operator_sequence() -> &'static [SourceOperator] {
    use SourceOperator::*;
    &[
        InputRmsNorm,
        QueryAndKeyRmsNorm,
        QueryKeyValueProjectionSerialK,
        RopeThenCausalKvAppendAndRead,
        AttentionOutputProjectionSerialK,
        FirstResidualAdd,
        PostAttentionRmsNorm,
        RouterProjectionSerialK,
        Top8Selection,
        SourceOrderedRouteWeightedCombine,
        SecondResidualAdd,
    ]
}

fn selected_expert_operator_sequence() -> &'static [SourceOperator] {
    use SourceOperator::*;
    &[
        SelectedExpertGateUpSerialK,
        SourceSiluStoreBoundary,
        SelectedExpertDownSerialK,
    ]
}

/// Mirrors the immutable sequence emitted by
/// `ascension_qwen30_streamed_bf16_oracle_preflight.rs`: routing selection is
/// followed by each selected expert in source route order, then combine and
/// residual, then final norm/head/full-vector retention.
fn sealed_semantics_trace(geometry: SemanticGeometry) -> Result<Vec<TraceInvocation>, String> {
    if geometry.layers == 0 || geometry.top_k == 0 {
        return Err("semantic geometry requires positive layers and top-k".into());
    }
    let mut trace = Vec::new();
    for (forward_index, phase) in [
        TracePhase::ExactPrefix,
        TracePhase::ForcedSharedContinuation,
    ]
    .into_iter()
    .enumerate()
    {
        trace.push(TraceInvocation {
            forward_index,
            phase,
            layer_index: None,
            route_slot: None,
            operator: SourceOperator::TokenEmbeddingRow,
        });
        for layer_index in 0..geometry.layers {
            for operator in layer_operator_sequence() {
                trace.push(TraceInvocation {
                    forward_index,
                    phase,
                    layer_index: Some(layer_index),
                    route_slot: None,
                    operator: *operator,
                });
                if *operator == SourceOperator::Top8Selection {
                    for route_slot in 0..geometry.top_k {
                        for selected in selected_expert_operator_sequence() {
                            trace.push(TraceInvocation {
                                forward_index,
                                phase,
                                layer_index: Some(layer_index),
                                route_slot: Some(route_slot),
                                operator: *selected,
                            });
                        }
                    }
                }
            }
        }
        for operator in [
            SourceOperator::FinalRmsNorm,
            SourceOperator::LmHeadSerialK,
            SourceOperator::RetainFullF32EndpointLogits,
        ] {
            trace.push(TraceInvocation {
                forward_index,
                phase,
                layer_index: None,
                route_slot: None,
                operator,
            });
        }
    }
    Ok(trace)
}

fn validate_sealed_semantics_order(
    observed: &[TraceInvocation],
    geometry: SemanticGeometry,
) -> Result<(), String> {
    let expected = sealed_semantics_trace(geometry)?;
    if observed != expected {
        return Err(
            "observed layer-streamed operator order differs from the sealed semantics contract"
                .into(),
        );
    }
    Ok(())
}

fn range_for_operator(operator: SourceOperator) -> Option<&'static str> {
    use SourceOperator::*;
    match operator {
        TokenEmbeddingRow => Some("fixture.embedding"),
        InputRmsNorm
        | QueryAndKeyRmsNorm
        | QueryKeyValueProjectionSerialK
        | AttentionOutputProjectionSerialK
        | PostAttentionRmsNorm
        | RouterProjectionSerialK
        | SelectedExpertGateUpSerialK
        | SelectedExpertDownSerialK
        | FinalRmsNorm
        | LmHeadSerialK => Some("fixture.linear"),
        RopeThenCausalKvAppendAndRead
        | FirstResidualAdd
        | Top8Selection
        | SourceSiluStoreBoundary
        | SourceOrderedRouteWeightedCombine
        | SecondResidualAdd
        | RetainFullF32EndpointLogits => None,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct FixtureExecutionState {
    byte_checksum: u64,
    kv_append_count: usize,
    route_slot_count: usize,
    residual_add_count: usize,
    retained_endpoint_count: usize,
}

impl Default for FixtureExecutionState {
    fn default() -> Self {
        Self {
            byte_checksum: 0,
            kv_append_count: 0,
            route_slot_count: 0,
            residual_add_count: 0,
            retained_endpoint_count: 0,
        }
    }
}

#[derive(Debug)]
struct FixtureSemanticMachine {
    state: FixtureExecutionState,
    observed: Vec<TraceInvocation>,
}

impl FixtureSemanticMachine {
    fn new() -> Self {
        Self {
            state: FixtureExecutionState::default(),
            observed: Vec::new(),
        }
    }

    /// A transaction makes failed source-shaped work rollback-safe.  It rolls
    /// back synthetic KV/residual/route state and clears the one-window cache
    /// before returning an error; it never creates a payload on failure.
    fn execute(
        &mut self,
        reader: &mut impl ExactBoundedRangeReader,
        geometry: SemanticGeometry,
        fault_after_event: Option<usize>,
    ) -> Result<(), String> {
        let starting_state = self.state.clone();
        let starting_events = self.observed.len();
        let plan = sealed_semantics_trace(geometry)?;
        let result = (|| {
            for (index, invocation) in plan.into_iter().enumerate() {
                if let Some(range) = range_for_operator(invocation.operator) {
                    reader.with_verified_range(range, |bytes| {
                        let contribution = bytes
                            .iter()
                            .fold(0u64, |sum, byte| sum.wrapping_add(u64::from(*byte)));
                        self.state.byte_checksum =
                            self.state.byte_checksum.wrapping_add(contribution);
                        Ok(())
                    })?;
                }
                match invocation.operator {
                    SourceOperator::RopeThenCausalKvAppendAndRead => {
                        self.state.kv_append_count += 1;
                    }
                    SourceOperator::SelectedExpertGateUpSerialK => {
                        self.state.route_slot_count += 1;
                    }
                    SourceOperator::FirstResidualAdd | SourceOperator::SecondResidualAdd => {
                        self.state.residual_add_count += 1;
                    }
                    SourceOperator::RetainFullF32EndpointLogits => {
                        self.state.retained_endpoint_count += 1;
                    }
                    _ => {}
                }
                self.observed.push(invocation);
                if fault_after_event == Some(index) {
                    return Err("synthetic fixture fault injected after an operator event".into());
                }
            }
            validate_sealed_semantics_order(&self.observed[starting_events..], geometry)
        })();
        if result.is_err() {
            self.state = starting_state;
            self.observed.truncate(starting_events);
            reader.clear_for_eviction();
        }
        result
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SequenceStage {
    ReadyForSource,
    SourceRunning,
    SourcePayloadsDurable,
    SourceEvicted,
    NativeRunning,
    Complete,
}

#[derive(Debug)]
struct SourceEvictNativeHooks {
    stage: SequenceStage,
    events: Vec<&'static str>,
}

impl SourceEvictNativeHooks {
    fn new() -> Self {
        Self {
            stage: SequenceStage::ReadyForSource,
            events: Vec::new(),
        }
    }

    fn begin_source(&mut self) -> Result<(), String> {
        if self.stage != SequenceStage::ReadyForSource {
            return Err("source phase may begin only once from the ready state".into());
        }
        self.stage = SequenceStage::SourceRunning;
        self.events.push("source_streamed_fixture_started");
        Ok(())
    }

    fn mark_source_payloads_durable(&mut self) -> Result<(), String> {
        if self.stage != SequenceStage::SourceRunning {
            return Err(
                "source payloads may be committed only while source phase is running".into(),
            );
        }
        self.stage = SequenceStage::SourcePayloadsDurable;
        self.events.push("source_fixture_payloads_fsynced");
        Ok(())
    }

    fn evict_source(&mut self, reader: &mut impl ExactBoundedRangeReader) -> Result<(), String> {
        if self.stage != SequenceStage::SourcePayloadsDurable {
            return Err("source eviction requires durable source payloads".into());
        }
        reader.clear_for_eviction();
        if !reader.cache_is_clear() {
            return Err("source reader cache did not clear at eviction".into());
        }
        self.stage = SequenceStage::SourceEvicted;
        self.events.push("source_fixture_evicted_and_cache_cleared");
        Ok(())
    }

    fn begin_native(&mut self) -> Result<(), String> {
        if self.stage != SequenceStage::SourceEvicted {
            return Err("native phase is forbidden before durable source eviction".into());
        }
        self.stage = SequenceStage::NativeRunning;
        self.events
            .push("native_fixture_started_after_source_eviction");
        Ok(())
    }

    fn complete_native(&mut self) -> Result<(), String> {
        if self.stage != SequenceStage::NativeRunning {
            return Err("native phase cannot complete before it begins".into());
        }
        self.stage = SequenceStage::Complete;
        self.events.push("native_fixture_payloads_fsynced");
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
struct PayloadMetadata {
    path: String,
    dtype: &'static str,
    vocab_rows: usize,
    bytes: usize,
    sha256: String,
    all_values_finite: bool,
    fixture_only: bool,
}

fn f32le(values: &[f32]) -> Result<Vec<u8>, String> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err("fixture F32LE payload requires one or more finite values".into());
    }
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

fn write_new_bytes(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if !path.is_absolute() || path.exists() {
        return Err(format!(
            "refusing to overwrite or relativize fixture artifact {}",
            path.display()
        ));
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create fixture artifact {}: {error}", path.display()))?;
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("cannot fsync fixture artifact {}: {error}", path.display()))?;
    Ok(())
}

fn sync_directory(path: &Path) -> Result<(), String> {
    File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(|error| format!("cannot fsync fixture directory {}: {error}", path.display()))
}

fn write_new_sealed_json(path: &Path, document: Value) -> Result<Value, String> {
    let sealed = seal_value(document)?;
    let bytes = serde_json::to_vec_pretty(&sealed)
        .map_err(|error| format!("cannot serialize fixture receipt: {error}"))?;
    write_new_bytes(path, &[bytes, b"\n".to_vec()].concat())?;
    sync_directory(
        path.parent()
            .ok_or_else(|| "fixture receipt lacks a parent directory".to_owned())?,
    )?;
    Ok(sealed)
}

fn write_fixture_payload(
    capture_dir: &Path,
    filename: &str,
    values: &[f32],
) -> Result<PayloadMetadata, String> {
    let path = capture_dir.join(filename);
    let bytes = f32le(values)?;
    write_new_bytes(&path, &bytes)?;
    sync_directory(capture_dir)?;
    Ok(PayloadMetadata {
        path: path.display().to_string(),
        dtype: "f32le",
        vocab_rows: values.len(),
        bytes: bytes.len(),
        sha256: sha256_hex(&bytes),
        all_values_finite: true,
        fixture_only: true,
    })
}

fn fixture_logits(seed: f32) -> [f32; FIXTURE_VOCAB_ROWS] {
    [seed, seed + 0.25, seed + 0.5, seed + 0.75]
}

fn source_payload_map(rows: &[(&str, PayloadMetadata)]) -> Value {
    let mut map = Map::new();
    for (endpoint, row) in rows {
        map.insert(
            (*endpoint).to_owned(),
            serde_json::to_value(row).expect("payload metadata must serialize"),
        );
    }
    Value::Object(map)
}

fn native_payload_map(rows: &[(&str, &str, PayloadMetadata)]) -> Value {
    let mut models = Map::new();
    for (model, endpoint, row) in rows {
        let model_row = models
            .entry((*model).to_owned())
            .or_insert_with(|| Value::Object(Map::new()));
        model_row
            .as_object_mut()
            .expect("new native payload model row must be an object")
            .insert(
                (*endpoint).to_owned(),
                serde_json::to_value(row).expect("payload metadata must serialize"),
            );
    }
    Value::Object(models)
}

fn cache_receipt(reader: &impl ExactBoundedRangeReader) -> Value {
    serde_json::to_value(reader.cache_stats()).expect("cache stats must serialize")
}

fn accounting_receipt(reader: &impl ExactBoundedRangeReader) -> Value {
    let rows = reader.per_shard_accounting();
    let payload_bytes = rows.iter().map(|row| row.payload_bytes_read).sum::<usize>();
    let read_calls = rows.iter().map(|row| row.read_calls).sum::<usize>();
    json!({
        "all_source_payload_reads_accounted": true,
        "source_tensor_payload_reads_executed": read_calls > 0,
        "source_tensor_payload_bytes_read": payload_bytes,
        "source_tensor_payload_read_calls": read_calls,
        "per_shard": rows,
        "fixture_only_synthetic_bytes_not_real_source_payload": true,
    })
}

fn preflight_receipt(lease: &FreshFixtureLease, lease_path: &Path) -> Value {
    json!({
        "schema": PRELIGHT_SCHEMA,
        "status": PRELIGHT_STATUS,
        "outer_lease": {
            "path": lease_path.display().to_string(),
            "seal_sha256": lease.seal_sha256,
            "exact_launch_nonce": lease.nonce,
            "minimum_reclaimable_bytes": lease.minimum_reclaimable_bytes,
        },
        "execution_surface": {
            "fixture_only": true,
            "only_modes": ["preflight", "fixture-only"],
            "real_source_payload_paths_accepted": false,
            "source_model_instantiated": false,
            "source_inference_executed": false,
            "gpu_or_metal_invoked": false,
            "server_started": false,
            "hcli_invoked": false,
            "benchmark_or_tps_measurement_started": false,
        },
        "prepared_interfaces": {
            "exact_bounded_range_reader": true,
            "single_window_per_read": true,
            "sealed_layer_streamed_operator_order": true,
            "six_f32le_payload_receipt_last_fixture_protocol": true,
            "source_then_evict_then_native_hooks": true,
            "rollback_clears_reader_cache_and_restores_fixture_state": true,
        },
        "sealed_semantics_contract_binding": {
            "schema": SEMANTICS_CONTRACT_SCHEMA,
            "status": SEMANTICS_CONTRACT_STATUS,
            "static_operator_order_mirrors_the_contract": true,
            "physical_qwen30_geometry_for_future_execution_only": {
                "layers": QWEN30_SEMANTIC_LAYERS,
                "top_k": QWEN30_SEMANTIC_TOP_K,
            },
            "this_fixture_run_geometry_is_not_physical_qwen30": {
                "layers": FIXTURE_LAYERS,
                "top_k": FIXTURE_TOP_K,
            },
        },
        "claim_boundary": "Executable readiness only. This receipt does not authorize or report a real Q30 source teacher, source comparison, coherence/HCLI, TPS/TG, serving, capability, promotion, or tournament result."
    })
}

/// Exercise the protocol only with fixed, in-memory fixture bytes.  The final
/// six-vector receipt is created after all six payloads were individually
/// fsynced; nothing is written after it.
fn run_fixture_capture(
    capture_dir: &Path,
    lease: &FreshFixtureLease,
    lease_path: &Path,
) -> Result<Value, String> {
    if !capture_dir.is_absolute() || capture_dir.exists() {
        return Err("fixture capture directory must be a new absolute path".into());
    }
    fs::create_dir(capture_dir).map_err(|error| {
        format!(
            "cannot create fixture capture directory {}: {error}",
            capture_dir.display()
        )
    })?;
    sync_directory(
        capture_dir
            .parent()
            .ok_or_else(|| "fixture capture directory lacks a parent".to_owned())?,
    )?;

    let result = (|| {
        let mut reader = synthetic_fixture_reader()?;
        let mut machine = FixtureSemanticMachine::new();
        let geometry = SemanticGeometry {
            layers: FIXTURE_LAYERS,
            top_k: FIXTURE_TOP_K,
        };
        let mut hooks = SourceEvictNativeHooks::new();
        hooks.begin_source()?;
        machine.execute(&mut reader, geometry, None)?;
        validate_sealed_semantics_order(&machine.observed, geometry)?;

        let mut source_rows = Vec::new();
        for (index, (endpoint, filename)) in SOURCE_PAYLOADS.iter().enumerate() {
            source_rows.push((
                *endpoint,
                write_fixture_payload(capture_dir, filename, &fixture_logits(10.0 + index as f32))?,
            ));
        }
        hooks.mark_source_payloads_durable()?;
        let source_terminal_path = capture_dir.join("source-terminal.fixture.json");
        write_new_sealed_json(
            &source_terminal_path,
            json!({
                "schema": FIXTURE_SOURCE_TERMINAL_SCHEMA,
                "status": "CAPTURED_TWO_F32LE_FIXTURE_VECTORS_RECEIPT_LAST_FOR_SOURCE_PHASE_ONLY",
                "fixture_only": true,
                "source_lease": {"seal_sha256": lease.seal_sha256, "exact_launch_nonce": lease.nonce},
                "source_payloads": source_payload_map(&source_rows),
                "receipt_written_after_payload_fsyncs": true,
                "real_source_model_or_payload_opened": false,
            }),
        )?;

        hooks.evict_source(&mut reader)?;
        let eviction_path = capture_dir.join("source-eviction.fixture.json");
        write_new_sealed_json(
            &eviction_path,
            json!({
                "schema": FIXTURE_EVICTION_SCHEMA,
                "status": "EARNED_FIXTURE_SOURCE_BYTES_AND_READER_CACHE_EVICTED_BEFORE_NATIVE_FIXTURE",
                "fixture_only": true,
                "source_terminal_path": source_terminal_path.display().to_string(),
                "eviction": {
                    "source_weights_evicted": true,
                    "source_backend_shutdown": true,
                    "source_model_residency_released": true,
                    "streamed_reader_cache_cleared": reader.cache_is_clear(),
                    "source_payloads_durable_and_immutable": true,
                    "real_source_model_never_loaded": true,
                },
            }),
        )?;

        hooks.begin_native()?;
        let mut native_rows = Vec::new();
        for (index, (model, endpoint, filename)) in NATIVE_PAYLOADS.iter().enumerate() {
            native_rows.push((
                *model,
                *endpoint,
                write_fixture_payload(capture_dir, filename, &fixture_logits(20.0 + index as f32))?,
            ));
        }
        hooks.complete_native()?;
        if hooks.stage != SequenceStage::Complete || !reader.cache_is_clear() {
            return Err("fixture sequence did not complete with a cleared reader cache".into());
        }
        let cache = cache_receipt(&reader);
        let accounting = accounting_receipt(&reader);
        let trace = serde_json::to_value(&machine.observed)
            .map_err(|error| format!("cannot serialize fixture semantic trace: {error}"))?;
        let write_order = vec![
            SOURCE_PAYLOADS[0].1,
            SOURCE_PAYLOADS[1].1,
            "source-terminal.fixture.json",
            "source-eviction.fixture.json",
            NATIVE_PAYLOADS[0].2,
            NATIVE_PAYLOADS[1].2,
            NATIVE_PAYLOADS[2].2,
            NATIVE_PAYLOADS[3].2,
            "six-vector-terminal.fixture.json",
        ];
        let final_path = capture_dir.join("six-vector-terminal.fixture.json");
        let receipt = write_new_sealed_json(
            &final_path,
            json!({
                "schema": FIXTURE_CAPTURE_SCHEMA,
                "status": FIXTURE_CAPTURE_STATUS,
                "fixture_only": true,
                "outer_lease": {
                    "path": lease_path.display().to_string(),
                    "seal_sha256": lease.seal_sha256,
                    "exact_launch_nonce": lease.nonce,
                },
                "streamed_execution": {
                    "mode": "fixture_only_layer_streamed_bf16_source_teacher_shape",
                    "real_source_payload_opened": false,
                    "source_model_instantiated": false,
                    "source_inference_executed": false,
                    "outer_reaped_child_before_terminal_receipt": true,
                    "no_child_was_launched": true,
                    "receipt_written_after_all_six_payloads_and_fsyncs": true,
                },
                "fixture_geometry": {"layers": FIXTURE_LAYERS, "top_k": FIXTURE_TOP_K, "vocab_rows": FIXTURE_VOCAB_ROWS},
                "sealed_semantics_operator_order_replayed": true,
                "semantic_trace": trace,
                "bounded_per_read_cache": cache,
                "source_payload_read_accounting": accounting,
                "source_payloads": source_payload_map(&source_rows),
                "source_eviction": {
                    "path": eviction_path.display().to_string(),
                    "source_then_evict_then_native_enforced": true,
                    "reader_cache_clear_before_native": true,
                },
                "native_payloads": native_payload_map(&native_rows),
                "sequence_hooks": hooks.events,
                "artifact_write_order": write_order,
                "claim_boundary": "All payloads are four-row synthetic fixture F32LE vectors. This is a receipt-last protocol exercise, not source-teacher execution, a source comparison, coherence/HCLI, TPS/TG, serving, capability, promotion, or tournament evidence.",
            }),
        )?;
        Ok(receipt)
    })();
    if result.is_err() {
        // The directory is intentionally retained as failed fixture evidence;
        // removing it would undermine create-new/replay diagnostics.
        sync_directory(capture_dir).ok();
    }
    result
}

fn run(args: Args) -> Result<Value, String> {
    let lease = validate_fresh_fixture_outer_lease(&args.outer_lease)?;
    match args.mode {
        CliMode::Preflight => {
            let out = args.out.expect("parser supplies --out in preflight mode");
            let receipt =
                write_new_sealed_json(&out, preflight_receipt(&lease, &args.outer_lease))?;
            Ok(receipt)
        }
        CliMode::FixtureOnly => {
            let capture_dir = args
                .capture_dir
                .expect("parser supplies --capture-dir in fixture-only mode");
            run_fixture_capture(&capture_dir, &lease, &args.outer_lease)
        }
    }
}

fn main() {
    match parse_args().and_then(run) {
        Ok(receipt) => match serde_json::to_string_pretty(&receipt) {
            Ok(rendered) => println!("{rendered}"),
            Err(error) => {
                eprintln!("cannot serialize result: {error}");
                process::exit(1);
            }
        },
        Err(error) => {
            eprintln!("Q30 guarded streamed-source fixture executor refused: {error}");
            process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use tempfile::TempDir;

    fn fixture_lease_document() -> Value {
        seal_value(json!({
            "schema": SOURCE_LEASE_SCHEMA,
            "status": SOURCE_LEASE_STATUS,
            "fixture_only": true,
            "real_source_payload_execution_forbidden": true,
            "outer_controller_validation": {
                "schema": OUTER_CONTROLLER_SCHEMA,
                "fresh_validated_outer_lease": true,
                "fixture_only": true,
            },
            "one_shot_lifecycle": {
                "fresh_for_this_exact_launch": true,
                "prior_terminal_receipt": null,
                "automatic_retry_allowed": false,
                "new_capture_root": true,
                "existing_output_reuse_forbidden": true,
                "replay_or_relaunch_forbidden": true,
                "exact_launch_nonce": "a4d29730b0c11a91559c61a7ec751dfb58bd12e73e5f7adab139ba1831bcc8c3",
            },
            "fresh_pre_child_safety": {
                "observed_immediately_before_child": true,
                "exclusive_clean_window": true,
                "no_source_or_native_model_body_resident_before_child": true,
                "swap_used_bytes": 0,
                "swapouts_pages_delta": 0,
                "reclaimable_bytes": 8192,
                "minimum_reclaimable_bytes_required": 4096,
            },
        }))
        .expect("fixture lease must seal")
    }

    fn write_fixture_lease(root: &TempDir) -> PathBuf {
        let path = root.path().join("fixture-lease.json");
        let bytes =
            serde_json::to_vec_pretty(&fixture_lease_document()).expect("serialize fixture lease");
        fs::write(&path, bytes).expect("write fixture lease");
        path
    }

    #[test]
    fn bounded_reader_proves_single_window_exact_ranges() {
        let mut reader = synthetic_fixture_reader().expect("synthetic fixture reader");
        let first = reader
            .with_verified_range("fixture.embedding", |bytes| Ok(bytes.len()))
            .expect("read fixture embedding range");
        let second = reader
            .with_verified_range("fixture.linear", |bytes| Ok(bytes.len()))
            .expect("read fixture linear range");
        assert_eq!((first, second), (8, 16));
        assert!(reader.cache_is_clear());
        let stats = reader.cache_stats();
        assert_eq!(stats.maximum_allowed_window_bytes, 16);
        assert_eq!(stats.maximum_observed_window_bytes, 16);
        assert_eq!(stats.maximum_cached_bytes, 16);
        assert_eq!(stats.maximum_cached_windows, 1);
        assert!(stats.eviction_on_each_read_completion);
        assert!(!stats.complete_source_shard_mapped_or_cached);
        assert!(!stats.mmap_or_memory_map_used);
        let accounting = reader.per_shard_accounting();
        assert_eq!(accounting.len(), 1);
        assert_eq!(accounting[0].payload_bytes_read, 24);
        assert_eq!(accounting[0].read_calls, 2);
        assert!(!accounting[0].whole_shard_read_as_one_window);
        assert!(!accounting[0].whole_shard_cached);
    }

    #[test]
    fn sealed_operator_order_rejects_a_reordered_route_sequence() {
        let geometry = SemanticGeometry {
            layers: 2,
            top_k: 2,
        };
        let mut trace = sealed_semantics_trace(geometry).expect("build semantic trace");
        validate_sealed_semantics_order(&trace, geometry).expect("exact trace must validate");
        let first_route = trace
            .iter()
            .position(|event| event.operator == SourceOperator::SelectedExpertGateUpSerialK)
            .expect("route event exists");
        trace.swap(first_route, first_route + 1);
        assert!(validate_sealed_semantics_order(&trace, geometry)
            .expect_err("reordered selected-expert events must fail")
            .contains("operator order"));
    }

    #[test]
    fn physical_qwen30_semantics_profile_is_static_not_an_inference_run() {
        let geometry = SemanticGeometry {
            layers: QWEN30_SEMANTIC_LAYERS,
            top_k: QWEN30_SEMANTIC_TOP_K,
        };
        let trace = sealed_semantics_trace(geometry).expect("build static Q30 semantic trace");
        // One forward is embedding + 48 * (11 layer events + 8 * 3 expert
        // events) + final norm/head/full-logit-retention.  This creates only
        // metadata events; it does not open a tensor or execute arithmetic.
        assert_eq!(trace.len(), 2 * (1 + 48 * (11 + 8 * 3) + 3));
        validate_sealed_semantics_order(&trace, geometry)
            .expect("static Q30 semantic profile must remain exact");
    }

    #[test]
    fn rollback_restores_fixture_state_and_clears_the_reader() {
        let geometry = SemanticGeometry {
            layers: 2,
            top_k: 2,
        };
        let mut reader = synthetic_fixture_reader().expect("synthetic fixture reader");
        let mut machine = FixtureSemanticMachine::new();
        let original = machine.state.clone();
        assert!(machine.execute(&mut reader, geometry, Some(5)).is_err());
        assert_eq!(machine.state, original);
        assert!(machine.observed.is_empty());
        assert!(reader.cache_is_clear());
        assert!(reader.cache_stats().cache_cleared_at_sequence_eviction);
        machine
            .execute(&mut reader, geometry, None)
            .expect("same fixture can execute after rollback without stale state");
        assert_eq!(machine.state.kv_append_count, 4);
        assert_eq!(machine.state.route_slot_count, 8);
        assert_eq!(machine.state.residual_add_count, 8);
        assert_eq!(machine.state.retained_endpoint_count, 2);
    }

    #[test]
    fn native_hook_refuses_before_source_eviction() {
        let mut reader = synthetic_fixture_reader().expect("synthetic fixture reader");
        let mut hooks = SourceEvictNativeHooks::new();
        assert!(hooks.begin_native().is_err());
        hooks.begin_source().expect("start source fixture");
        assert!(hooks.begin_native().is_err());
        hooks
            .mark_source_payloads_durable()
            .expect("source fixture durable");
        hooks
            .evict_source(&mut reader)
            .expect("evict fixture source");
        hooks.begin_native().expect("native only after eviction");
        hooks.complete_native().expect("finish native fixture");
        assert_eq!(hooks.stage, SequenceStage::Complete);
        assert_eq!(
            hooks.events,
            vec![
                "source_streamed_fixture_started",
                "source_fixture_payloads_fsynced",
                "source_fixture_evicted_and_cache_cleared",
                "native_fixture_started_after_source_eviction",
                "native_fixture_payloads_fsynced",
            ]
        );
    }

    #[test]
    fn absent_or_unfresh_outer_lease_fails_closed() {
        let root = TempDir::new().expect("temporary fixture root");
        assert!(validate_fresh_fixture_outer_lease(&root.path().join("missing.json")).is_err());
        let lease_path = write_fixture_lease(&root);
        let mut stale = fixture_lease_document();
        stale
            .as_object_mut()
            .expect("fixture lease object")
            .get_mut("one_shot_lifecycle")
            .expect("lifecycle object")
            .as_object_mut()
            .expect("lifecycle object map")
            .insert("fresh_for_this_exact_launch".into(), Value::Bool(false));
        let stale_path = root.path().join("stale.json");
        // Leave the old seal in place: both stale fields and seal drift must
        // be rejected before a fixture run can start.
        fs::write(
            &stale_path,
            serde_json::to_vec(&stale).expect("serialize stale lease"),
        )
        .expect("write stale lease");
        assert!(validate_fresh_fixture_outer_lease(&stale_path).is_err());
        assert!(validate_fresh_fixture_outer_lease(&lease_path).is_ok());
    }

    #[test]
    fn fixture_capture_writes_six_f32le_payloads_before_terminal_receipt() {
        let root = TempDir::new().expect("temporary fixture root");
        let lease_path = write_fixture_lease(&root);
        let lease =
            validate_fresh_fixture_outer_lease(&lease_path).expect("validate fixture lease");
        let capture_dir = root.path().join("capture");
        let receipt =
            run_fixture_capture(&capture_dir, &lease, &lease_path).expect("run fixture capture");
        verify_seal(&receipt, "fixture terminal receipt").expect("fixture terminal receipt sealed");
        assert_eq!(receipt["status"], FIXTURE_CAPTURE_STATUS);
        assert_eq!(
            receipt["fixture_geometry"]["vocab_rows"],
            FIXTURE_VOCAB_ROWS
        );
        let expected_names = SOURCE_PAYLOADS
            .iter()
            .map(|(_, filename)| *filename)
            .chain(NATIVE_PAYLOADS.iter().map(|(_, _, filename)| *filename))
            .collect::<BTreeSet<_>>();
        assert_eq!(expected_names.len(), 6);
        for filename in expected_names {
            let bytes = fs::read(capture_dir.join(filename)).expect("fixture F32LE payload exists");
            assert_eq!(bytes.len(), FIXTURE_VOCAB_ROWS * 4);
        }
        let terminal_path = capture_dir.join("six-vector-terminal.fixture.json");
        let terminal = serde_json::from_slice::<Value>(
            &fs::read(&terminal_path).expect("read terminal receipt"),
        )
        .expect("terminal receipt JSON");
        verify_seal(&terminal, "terminal receipt from disk")
            .expect("terminal receipt remains sealed");
        let order = terminal["artifact_write_order"]
            .as_array()
            .expect("write order array");
        assert_eq!(
            order.last().and_then(Value::as_str),
            Some("six-vector-terminal.fixture.json")
        );
        assert_eq!(
            terminal["sequence_hooks"][0],
            "source_streamed_fixture_started"
        );
        assert_eq!(
            terminal["sequence_hooks"][2],
            "source_fixture_evicted_and_cache_cleared"
        );
        assert_eq!(
            terminal["sequence_hooks"][3],
            "native_fixture_started_after_source_eviction"
        );
        assert_eq!(
            terminal["streamed_execution"]["receipt_written_after_all_six_payloads_and_fsyncs"],
            true
        );
    }

    #[test]
    fn cli_requires_lease_and_keeps_modes_disjoint() {
        assert!(parse_args_from(vec!["--mode".into(), "preflight".into()]).is_err());
        assert!(parse_args_from(vec![
            "--mode".into(),
            "fixture-only".into(),
            "--outer-lease".into(),
            "/tmp/lease.json".into(),
            "--out".into(),
            "/tmp/out.json".into(),
        ])
        .is_err());
    }
}
