//! Independent assessor for a future Qwen80 multi-layer same-runtime capture.
//!
//! Follows the L1 full-layer completion assessor conventions:
//! - exact claim boundary (multi-layer component, not token/decoder)
//! - composite evidence (inner/outer/release + schedule + chain oracle)
//! - seal binding with pointer() distinguishing receipt-internal seal vs wrapper json_sha
//! - adversarial negative paths
//! - producer convention tests (document_sha256 == seal on receipt pointers)
//!
//! ```text
//! cargo run -p hawking-core --example ascension_qwen80_multi_layer_completion_assessor -- \
//!   --input ABSOLUTE_SEALED_INPUT --out ABSOLUTE_NEW_ASSESSMENT
//! ```

use std::collections::BTreeMap;
use hawking_core::model::qwen80_48_layer_execution_schedule::{
    qwen80_multi_layer_structural_kernel_trace, QWEN80_DELTANET_FULL_LAYER_DISPATCHES,
    QWEN80_GRAVITY_MANIFEST_SEAL_SHA256, QWEN80_SOURCE_REVISION,
};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const INPUT_SCHEMA: &str = "hawking.ascension.qwen80_multi_layer_completion_assessor_input.v1";
const RESULT_SCHEMA: &str = "hawking.ascension.qwen80_multi_layer_completion_assessment.v1";
const EARNED_STATUS: &str =
    "EARNED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_COMPONENT_NOT_TOKEN_DECODER";
const REFUSED_STATUS: &str =
    "REFUSED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_COMPLETION_INCOMPLETE_OR_UNTRUSTED";

const SCHEDULE_SCHEMA: &str =
    "hawking.ascension.qwen80_48_layer_execution_schedule_authority.v1";
const CHAIN_ORACLE_SCHEMA: &str = "hawking.ascension.qwen80_multi_layer_chain_cpu_oracle.v1";
const HOST_PREFLIGHT_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_multi_layer_same_runtime_host_preflight.v1";
const INNER_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_multi_layer_same_runtime_capture.v1";
const INNER_STATUS: &str =
    "CAPTURED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_SAME_RUNTIME_COMPONENT_ONLY";
const OUTER_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_multi_layer_same_runtime_outer_capture.v1";
const OUTER_STATUS: &str =
    "CAPTURED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_SAME_RUNTIME_OUTER_TERMINAL_COMPONENT_ONLY";
const RELEASE_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_multi_layer_same_runtime_quiet_metal_lease_release.v1";
const RELEASE_STATUS: &str =
    "RELEASED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_COMPONENT_QUIET_METAL_LEASE_AFTER_TERMINAL_CAPTURE";

const MAX_PARITY_ERROR: f64 = 1e-3;
const HIDDEN_ELEMENTS: u64 = 2_048;

#[derive(Clone, Debug)]
struct Bound {
    document: Value,
    document_sha256: String,
    seal_sha256: String,
}

#[derive(Debug)]
struct Args {
    input: PathBuf,
    out: PathBuf,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

// Campaign-canonical JSON: sorted keys, compact separators, Python float repr.
// serde_json's default number formatting differs from Python's on exponent
// zero-padding (e-7 vs e-07) and integral floats (1 vs 1.0). Every sealed document
// in this campaign is produced or verified through the Python form, so an assessor
// using serde defaults can never accept a real input - the same defect that made a
// physically successful multi-layer capture fail its own outer's seal check.
fn python_float(number: &serde_json::Number) -> Result<String, String> {
    let value = number
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or("floating JSON number must be finite")?;
    if value == 0.0 {
        return Ok(if value.is_sign_negative() {
            "-0.0"
        } else {
            "0.0"
        }
        .into());
    }
    let raw = number.to_string();
    let (negative, unsigned) = match raw.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, raw.as_str()),
    };
    let (mantissa, exponent) = match unsigned.find('e').or_else(|| unsigned.find('E')) {
        Some(index) => (
            &unsigned[..index],
            unsigned[index + 1..]
                .parse::<i32>()
                .map_err(|error| format!("bad exponent: {error}"))?,
        ),
        None => (unsigned, 0),
    };
    let mut fractional = 0_i32;
    let mut after_decimal = false;
    let mut digits = String::new();
    for byte in mantissa.bytes() {
        match byte {
            b'.' if !after_decimal => after_decimal = true,
            b'0'..=b'9' => {
                if after_decimal {
                    fractional = fractional.checked_add(1).ok_or("float length overflow")?;
                }
                digits.push(char::from(byte));
            }
            _ => return Err("bad float mantissa".into()),
        }
    }
    let first = digits
        .bytes()
        .position(|byte| byte != b'0')
        .ok_or("zero float has no significant digit")?;
    let mut significant = digits[first..].to_owned();
    let mut power = exponent
        .checked_sub(fractional)
        .ok_or("float power overflow")?;
    while significant.len() > 1 && significant.ends_with('0') {
        significant.pop();
        power = power.checked_add(1).ok_or("float power overflow")?;
    }
    let scientific = power
        .checked_add(i32::try_from(significant.len() - 1).unwrap_or(i32::MAX))
        .ok_or("float power overflow")?;
    let sign = if negative { "-" } else { "" };
    if !(-4..16).contains(&scientific) {
        let mut rendered = significant[..1].to_owned();
        if significant.len() > 1 {
            rendered.push('.');
            rendered.push_str(&significant[1..]);
        }
        let exponent_sign = if scientific < 0 { '-' } else { '+' };
        return Ok(format!(
            "{sign}{rendered}e{exponent_sign}{:02}",
            scientific.unsigned_abs()
        ));
    }
    let position = scientific + 1;
    let rendered = if position <= 0 {
        format!(
            "0.{}{}",
            "0".repeat(usize::try_from(-position).unwrap_or(usize::MAX)),
            significant
        )
    } else if usize::try_from(position).unwrap_or(usize::MAX) >= significant.len() {
        format!(
            "{}{}.0",
            significant,
            "0".repeat(usize::try_from(position).unwrap_or(usize::MAX) - significant.len())
        )
    } else {
        let position = usize::try_from(position).map_err(|_| "negative float position")?;
        format!("{}.{}", &significant[..position], &significant[position..])
    };
    Ok(format!("{sign}{rendered}"))
}

fn canonical_into(value: &Value, output: &mut String) -> Result<(), String> {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) => {
            if value.is_i64() || value.is_u64() {
                output.push_str(&value.to_string());
            } else {
                output.push_str(&python_float(value)?);
            }
        }
        Value::String(value) => output.push_str(
            &serde_json::to_string(value)
                .map_err(|error| format!("string canonicalization: {error}"))?,
        ),
        Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                canonical_into(value, output)?;
            }
            output.push(']');
        }
        Value::Object(values) => {
            let mut ordered = BTreeMap::new();
            for (key, value) in values {
                ordered.insert(key, value);
            }
            output.push('{');
            for (index, (key, value)) in ordered.into_iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                output.push_str(
                    &serde_json::to_string(key)
                        .map_err(|error| format!("key canonicalization: {error}"))?,
                );
                output.push(':');
                canonical_into(value, output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

fn json_sha(value: &Value) -> Result<String, String> {
    let mut rendered = String::new();
    canonical_into(value, &mut rendered)?;
    let bytes = rendered.into_bytes();
    Ok(sha256(&bytes))
}

fn seal(value: &mut Value) -> Result<String, String> {
    {
        let object = value
            .as_object_mut()
            .ok_or("document must be object")?;
        object.remove("seal_sha256");
    }
    let seal = json_sha(value)?;
    value
        .as_object_mut()
        .ok_or("document must be object")?
        .insert("seal_sha256".into(), json!(seal.clone()));
    Ok(seal)
}

fn verify_seal(value: &Value, label: &str) -> Result<String, String> {
    let mut unsigned = value.clone();
    let object = unsigned
        .as_object_mut()
        .ok_or_else(|| format!("{label} must be object"))?;
    let seal = object
        .remove("seal_sha256")
        .and_then(|v| v.as_str().map(str::to_owned))
        .ok_or_else(|| format!("{label} missing seal_sha256"))?;
    if !valid_sha(&seal) {
        return Err(format!("{label} seal_sha256 malformed (len={})", seal.len()));
    }
    let expected = json_sha(&unsigned)?;
    if seal != expected {
        return Err(format!(
            "{label} seal mismatch: observed={seal}, expected={expected}"
        ));
    }
    Ok(seal)
}

fn write_new(path: &Path, value: &Value) -> Result<(), String> {
    if !path.is_absolute() {
        return Err("--out must be absolute".into());
    }
    if path.exists() {
        return Err(format!("--out must be create-new; {} exists", path.display()));
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create parent: {e}"))?;
    }
    let bytes = serde_json::to_vec_pretty(value).map_err(|e| format!("serialize: {e}"))?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("open: {e}"))?;
    file.write_all(&bytes).map_err(|e| format!("write: {e}"))?;
    file.sync_all().map_err(|e| format!("sync: {e}"))?;
    Ok(())
}

fn read_json(path: &Path) -> Result<Value, String> {
    if !path.is_absolute() {
        return Err("--input must be absolute".into());
    }
    let bytes = fs::read(path).map_err(|e| format!("read input: {e}"))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("parse input: {e}"))
}

fn parse_args(mut args: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut input = None;
    let mut out = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--input" => {
                let value = args.next().ok_or("--input requires a value")?;
                if input.replace(PathBuf::from(value)).is_some() {
                    return Err("--input may not be repeated".into());
                }
            }
            "--out" => {
                let value = args.next().ok_or("--out requires a value")?;
                if out.replace(PathBuf::from(value)).is_some() {
                    return Err("--out may not be repeated".into());
                }
            }
            other => {
                return Err(format!(
                    "unsupported {other:?}; usage: --input ABSOLUTE --out ABSOLUTE"
                ))
            }
        }
    }
    let input = input.ok_or("missing --input")?;
    let out = out.ok_or("missing --out")?;
    if !input.is_absolute() || !out.is_absolute() {
        return Err("--input and --out must be absolute".into());
    }
    Ok(Args { input, out })
}

fn obj<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be object"))
}

fn text<'a>(map: &'a Map<String, Value>, field: &str, label: &str) -> Result<&'a str, String> {
    map.get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label}.{field} must be string"))
}

fn number(map: &Map<String, Value>, field: &str, label: &str) -> Result<u64, String> {
    map.get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{label}.{field} must be unsigned integer"))
}

fn boolean(map: &Map<String, Value>, field: &str, expected: bool, label: &str) -> Result<(), String> {
    let observed = map
        .get(field)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{label}.{field} must be bool"))?;
    if observed != expected {
        return Err(format!(
            "{label}.{field} observed={observed}, expected={expected}"
        ));
    }
    Ok(())
}

fn exact(map: &Map<String, Value>, field: &str, expected: &str, label: &str) -> Result<(), String> {
    let observed = text(map, field, label)?;
    if observed != expected {
        return Err(format!(
            "{label}.{field} observed={observed}, expected={expected}"
        ));
    }
    Ok(())
}

fn sha_field(map: &Map<String, Value>, field: &str, label: &str) -> Result<String, String> {
    let value = text(map, field, label)?;
    if !valid_sha(value) {
        return Err(format!(
            "{label}.{field} is not a lowercase SHA-256 (len={})",
            value.len()
        ));
    }
    Ok(value.to_owned())
}

fn bind_wrapper(wrapper: &Value, field: &str, schema: &str, status: &str) -> Result<Bound, String> {
    let wrapper = obj(wrapper, &format!("input.{field}"))?;
    let document = wrapper
        .get("document")
        .cloned()
        .ok_or_else(|| format!("input.{field}.document is required"))?;
    let document_sha = sha_field(wrapper, "document_sha256", &format!("input.{field}"))?;
    let document_seal = sha_field(wrapper, "document_seal_sha256", &format!("input.{field}"))?;
    if json_sha(&document)? != document_sha {
        return Err(format!(
            "input.{field}.document_sha256 does not bind its document (wrapper json_sha convention)"
        ));
    }
    let sealed = verify_seal(&document, &format!("input.{field}.document"))?;
    if sealed != document_seal {
        return Err(format!(
            "input.{field}.document_seal_sha256 observed={document_seal}, document seal={sealed}"
        ));
    }
    let root = obj(&document, &format!("input.{field}.document"))?;
    exact(root, "schema", schema, &format!("input.{field}.document"))?;
    exact(root, "status", status, &format!("input.{field}.document"))?;
    Ok(Bound {
        document,
        document_sha256: document_sha,
        seal_sha256: document_seal,
    })
}

/// Receipt-internal provenance pointer (producer convention).
///
/// Producers set `document_sha256` and `document_seal_sha256` both to the seal.
/// Wrapper inputs use `document_sha256` = json_sha(full sealed document).
/// Never treat receipt-internal `document_sha256` as wrapper json_sha.
fn pointer(value: &Value, expected: &Bound, label: &str) -> Result<(), String> {
    let value = obj(value, label)?;
    boolean(value, "present", true, label)?;
    let observed_seal = value
        .get("document_seal_sha256")
        .or_else(|| value.get("seal_sha256"))
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label}.seal_sha256 is required"))?;
    if !valid_sha(observed_seal) {
        return Err(format!(
            "{label}.seal_sha256 must be lowercase SHA-256 (len={})",
            observed_seal.len()
        ));
    }
    if observed_seal != expected.seal_sha256 {
        return Err(format!(
            "{label} does not bind the expected sealed document (observed seal={observed_seal}, expected seal={})",
            expected.seal_sha256
        ));
    }
    if let Some(observed_document_sha) = value
        .get("canonical_document_sha256")
        .or_else(|| value.get("bound_document_sha256"))
        .and_then(Value::as_str)
    {
        if observed_document_sha != expected.document_sha256 {
            return Err(format!(
                "{label} canonical document identity drifted (observed={observed_document_sha}, expected={})",
                expected.document_sha256
            ));
        }
    }
    Ok(())
}

fn require_parity(value: &Map<String, Value>, label: &str) -> Result<f64, String> {
    boolean(value, "passed", true, label)?;
    sha_field(value, "cpu_f32le_sha256", label)?;
    sha_field(value, "device_f32le_sha256", label)?;
    let error = value
        .get("max_abs_error")
        .and_then(Value::as_f64)
        .filter(|e| e.is_finite() && *e >= 0.0)
        .ok_or_else(|| format!("{label}.max_abs_error must be finite nonnegative"))?;
    if error > MAX_PARITY_ERROR {
        return Err(format!(
            "{label}.max_abs_error observed={error}, max allowed={MAX_PARITY_ERROR}"
        ));
    }
    Ok(error)
}

fn assess(input: &Value) -> Value {
    match assess_inner(input) {
        Ok(report) => report,
        Err(blockers) => {
            let mut report = json!({
                "schema": RESULT_SCHEMA,
                "status": REFUSED_STATUS,
                "earned_multi_layer_component_only": false,
                "blockers": blockers,
                "claim_boundary": {
                    "multi_layer_component_only": false,
                    "token_generated": false,
                    "decoder_started": false,
                    "tps_or_tg_measured": false,
                },
            });
            let _ = seal(&mut report);
            report
        }
    }
}

fn assess_inner(input: &Value) -> Result<Value, Vec<String>> {
    let mut blockers = Vec::new();
    let root = match obj(input, "input") {
        Ok(v) => v,
        Err(e) => return Err(vec![e]),
    };
    if let Err(e) = exact(root, "schema", INPUT_SCHEMA, "input") {
        blockers.push(e);
    }
    if let Err(e) = verify_seal(input, "input") {
        blockers.push(e);
    }

    let layer_count = match number(root, "layer_count", "input") {
        Ok(n) if (2..=48).contains(&n) => n as usize,
        Ok(n) => {
            blockers.push(format!(
                "input.layer_count observed={n}, expected in 2..=48"
            ));
            return Err(blockers);
        }
        Err(e) => {
            blockers.push(e);
            return Err(blockers);
        }
    };

    let schedule = match root
        .get("execution_schedule_authority")
        .ok_or_else(|| "input.execution_schedule_authority is required".into())
        .and_then(|w| {
            bind_wrapper(
                w,
                "execution_schedule_authority",
                SCHEDULE_SCHEMA,
                "PREPARED_QWEN80_48_LAYER_EXECUTION_SCHEDULE_AUTHORITY_NOT_EXECUTED",
            )
        }) {
        Ok(b) => b,
        Err(e) => {
            blockers.push(e);
            return Err(blockers);
        }
    };

    let chain_oracle = match root
        .get("chain_cpu_oracle")
        .ok_or_else(|| "input.chain_cpu_oracle is required".into())
        .and_then(|w| {
            // Accept either structure or composed status.
            let wrapper = obj(w, "input.chain_cpu_oracle")?;
            let document = wrapper
                .get("document")
                .cloned()
                .ok_or("input.chain_cpu_oracle.document is required")?;
            let document_sha = sha_field(wrapper, "document_sha256", "input.chain_cpu_oracle")?;
            let document_seal =
                sha_field(wrapper, "document_seal_sha256", "input.chain_cpu_oracle")?;
            if json_sha(&document)? != document_sha {
                return Err(
                    "input.chain_cpu_oracle.document_sha256 does not bind its document".into(),
                );
            }
            let sealed = verify_seal(&document, "input.chain_cpu_oracle.document")?;
            if sealed != document_seal {
                return Err(format!(
                    "input.chain_cpu_oracle seal mismatch observed={document_seal}, expected={sealed}"
                ));
            }
            let droot = obj(&document, "chain oracle")?;
            exact(droot, "schema", CHAIN_ORACLE_SCHEMA, "chain oracle")?;
            Ok(Bound {
                document,
                document_sha256: document_sha,
                seal_sha256: document_seal,
            })
        }) {
        Ok(b) => b,
        Err(e) => {
            blockers.push(e);
            return Err(blockers);
        }
    };

    let host_preflight = match root
        .get("host_preflight")
        .ok_or_else(|| "input.host_preflight is required".into())
        .and_then(|w| {
            bind_wrapper(
                w,
                "host_preflight",
                HOST_PREFLIGHT_SCHEMA,
                "COMPILED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_SAME_RUNTIME_HOST_CPU_ONLY_NOT_LEASED_OR_EXECUTED",
            )
        }) {
        Ok(b) => b,
        Err(e) => {
            blockers.push(e);
            return Err(blockers);
        }
    };

    let inner = match root
        .get("fresh_multi_layer_inner")
        .ok_or_else(|| "input.fresh_multi_layer_inner is required".into())
        .and_then(|w| bind_wrapper(w, "fresh_multi_layer_inner", INNER_SCHEMA, INNER_STATUS))
    {
        Ok(b) => b,
        Err(e) => {
            blockers.push(e);
            return Err(blockers);
        }
    };

    let outer = match root
        .get("fresh_multi_layer_outer")
        .ok_or_else(|| "input.fresh_multi_layer_outer is required".into())
        .and_then(|w| bind_wrapper(w, "fresh_multi_layer_outer", OUTER_SCHEMA, OUTER_STATUS))
    {
        Ok(b) => b,
        Err(e) => {
            blockers.push(e);
            return Err(blockers);
        }
    };

    let release = match root
        .get("fresh_multi_layer_release")
        .ok_or_else(|| "input.fresh_multi_layer_release is required".into())
        .and_then(|w| {
            bind_wrapper(w, "fresh_multi_layer_release", RELEASE_SCHEMA, RELEASE_STATUS)
        }) {
        Ok(b) => b,
        Err(e) => {
            blockers.push(e);
            return Err(blockers);
        }
    };

    // --- validate chain oracle layer count ---
    {
        let oracle = obj(&chain_oracle.document, "chain oracle").map_err(|e| vec![e])?;
        match number(oracle, "layer_count", "chain oracle") {
            Ok(n) if n == layer_count as u64 => {}
            Ok(n) => blockers.push(format!(
                "chain oracle layer_count observed={n}, expected={layer_count}"
            )),
            Err(e) => blockers.push(e),
        }
    }

    // --- validate host preflight ---
    {
        let pre = obj(&host_preflight.document, "host preflight").map_err(|e| vec![e])?;
        match number(pre, "layer_count", "host preflight") {
            Ok(n) if n == layer_count as u64 => {}
            Ok(n) => blockers.push(format!(
                "host preflight layer_count observed={n}, expected={layer_count}"
            )),
            Err(e) => blockers.push(e),
        }
        if let Some(claim) = pre.get("claim_boundary").and_then(Value::as_object) {
            if claim.contains_key("fixture_or_synthetic") {
                if let Err(e) = boolean(claim, "fixture_or_synthetic", false, "host claim") {
                    blockers.push(e);
                }
            }
            if claim.contains_key("test_only_fake_child") {
                if let Err(e) = boolean(claim, "test_only_fake_child", false, "host claim") {
                    blockers.push(e);
                }
            }
        }
    }

    let expected_kernels = match qwen80_multi_layer_structural_kernel_trace(layer_count, false) {
        Ok(k) => k,
        Err(e) => {
            blockers.push(e);
            return Err(blockers);
        }
    };
    let expected_total = expected_kernels.len() as u64;

    // --- validate inner capture ---
    {
        let root = obj(&inner.document, "inner").map_err(|e| vec![e])?;
        if let Err(e) = boolean(root, "fixture_or_synthetic", false, "inner") {
            blockers.push(e);
        }
        if let Err(e) = boolean(root, "self_asserted", false, "inner") {
            blockers.push(e);
        }
        match root.get("execution_schedule_provenance") {
            Some(value) => {
                if let Err(e) = pointer(value, &schedule, "inner execution schedule provenance") {
                    blockers.push(e);
                }
            }
            None => blockers.push("inner.execution_schedule_provenance is required".into()),
        }
        match root.get("chain_cpu_oracle_provenance") {
            Some(value) => {
                if let Err(e) = pointer(value, &chain_oracle, "inner chain cpu oracle provenance") {
                    blockers.push(e);
                }
            }
            None => blockers.push("inner.chain_cpu_oracle_provenance is required".into()),
        }
        match root.get("host_preflight_provenance") {
            Some(value) => {
                if let Err(e) = pointer(value, &host_preflight, "inner host preflight provenance") {
                    blockers.push(e);
                }
            }
            None => blockers.push("inner.host_preflight_provenance is required".into()),
        }

        let execution = match root.get("fresh_same_runtime_execution") {
            Some(value) => match obj(value, "inner execution") {
                Ok(e) => e,
                Err(e) => {
                    blockers.push(e);
                    return Err(blockers);
                }
            },
            None => {
                blockers.push("inner.fresh_same_runtime_execution is required".into());
                return Err(blockers);
            }
        };
        for (field, expected) in [
            ("fresh_runtime", true),
            ("same_runtime", true),
            ("same_tcb", true),
            ("single_fence_after_all_dispatches", true),
        ] {
            if let Err(e) = boolean(execution, field, expected, "inner execution") {
                blockers.push(e);
            }
        }
        match number(execution, "layer_count", "inner execution") {
            Ok(n) if n == layer_count as u64 => {}
            Ok(n) => blockers.push(format!(
                "inner execution layer_count observed={n}, expected={layer_count}"
            )),
            Err(e) => blockers.push(e),
        }
        match number(execution, "total_dispatches", "inner execution") {
            Ok(n) if n == expected_total => {}
            Ok(n) => blockers.push(format!(
                "inner execution total_dispatches observed={n}, expected={expected_total} ({layer_count}×{QWEN80_DELTANET_FULL_LAYER_DISPATCHES})"
            )),
            Err(e) => blockers.push(e),
        }
        match number(execution, "fence_count", "inner execution") {
            Ok(1) => {}
            Ok(n) => blockers.push(format!(
                "inner execution fence_count observed={n}, expected=1"
            )),
            Err(e) => blockers.push(e),
        }

        let trace = match root.get("structural_kernel_trace") {
            Some(value) => match obj(value, "kernel trace") {
                Ok(t) => t,
                Err(e) => {
                    blockers.push(e);
                    return Err(blockers);
                }
            },
            None => {
                blockers.push("inner.structural_kernel_trace is required".into());
                return Err(blockers);
            }
        };
        if let Err(e) = boolean(trace, "exact_order", true, "kernel trace") {
            blockers.push(e);
        }
        match trace
            .get("kernel_names")
            .and_then(Value::as_array)
            .ok_or_else(|| "kernel_names must be array".into())
        {
            Ok(names) => {
                let observed: Vec<&str> = names.iter().filter_map(Value::as_str).collect();
                if observed != expected_kernels {
                    blockers.push(format!(
                        "structural kernel trace drifted: observed_len={}, expected_len={}, first_mismatch={}",
                        observed.len(),
                        expected_kernels.len(),
                        observed
                            .iter()
                            .zip(expected_kernels.iter())
                            .position(|(a, b)| a != b)
                            .map(|i| format!("index {i}: observed={}, expected={}", observed.get(i).unwrap_or(&"?"), expected_kernels.get(i).unwrap_or(&"?")))
                            .unwrap_or_else(|| "length-only".into())
                    ));
                }
            }
            Err(e) => blockers.push(e),
        }

        // Per-layer retained parity + overall retained max.
        let mut retained_max = 0.0f64;
        match root
            .get("per_layer_readbacks")
            .and_then(Value::as_array)
            .ok_or_else(|| "inner.per_layer_readbacks must be array".into())
        {
            Ok(layers) => {
                if layers.len() != layer_count {
                    blockers.push(format!(
                        "per_layer_readbacks len observed={}, expected={layer_count}",
                        layers.len()
                    ));
                }
                for (index, layer) in layers.iter().enumerate() {
                    let layer = match obj(layer, &format!("readback[{index}]")) {
                        Ok(l) => l,
                        Err(e) => {
                            blockers.push(e);
                            continue;
                        }
                    };
                    match number(layer, "layer", &format!("readback[{index}]")) {
                        Ok(n) if n == index as u64 => {}
                        Ok(n) => blockers.push(format!(
                            "readback[{index}].layer observed={n}, expected={index}"
                        )),
                        Err(e) => blockers.push(e),
                    }
                    if let Some(second) = layer.get("second_residual_output").and_then(Value::as_object)
                    {
                        match require_parity(second, &format!("readback[{index}].second_residual"))
                        {
                            Ok(err) => retained_max = retained_max.max(err),
                            Err(e) => blockers.push(e),
                        }
                    } else {
                        blockers.push(format!(
                            "readback[{index}].second_residual_output is required"
                        ));
                    }
                    match number(layer, "output_elements", &format!("readback[{index}]")) {
                        Ok(HIDDEN_ELEMENTS) => {}
                        Ok(n) => blockers.push(format!(
                            "readback[{index}].output_elements observed={n}, expected={HIDDEN_ELEMENTS}"
                        )),
                        Err(e) => blockers.push(e),
                    }
                }
            }
            Err(e) => blockers.push(e),
        }

        // Receipt must retain the measured maximum, not only a tolerance.
        match root
            .get("retained_max_abs_error")
            .and_then(Value::as_f64)
            .filter(|e| e.is_finite() && *e >= 0.0)
        {
            Some(retained) => {
                if (retained - retained_max).abs() > 1e-12 && retained_max > 0.0 {
                    // Allow retained >= measured local max.
                    if retained + 1e-15 < retained_max {
                        blockers.push(format!(
                            "retained_max_abs_error observed={retained} is less than measured per-layer max={retained_max}"
                        ));
                    }
                }
                if retained > MAX_PARITY_ERROR {
                    blockers.push(format!(
                        "retained_max_abs_error observed={retained}, max allowed={MAX_PARITY_ERROR}"
                    ));
                }
            }
            None => blockers.push(
                "inner.retained_max_abs_error must be present as finite nonnegative (measured maximum, not bare tolerance)"
                    .into(),
            ),
        }

        let claim = match root.get("claim_boundary") {
            Some(value) => match obj(value, "inner claim") {
                Ok(c) => c,
                Err(e) => {
                    blockers.push(e);
                    return Err(blockers);
                }
            },
            None => {
                blockers.push("inner.claim_boundary is required".into());
                return Err(blockers);
            }
        };
        for (field, expected) in [
            ("multi_layer_component_only", true),
            ("token_generated", false),
            ("decoder_started", false),
            ("tps_or_tg_measured", false),
            ("tournament_started", false),
        ] {
            if let Err(e) = boolean(claim, field, expected, "inner claim") {
                blockers.push(e);
            }
        }
    }

    // --- outer ---
    {
        let root = obj(&outer.document, "outer").map_err(|e| vec![e])?;
        if let Err(e) = boolean(root, "fixture_or_synthetic", false, "outer") {
            blockers.push(e);
        }
        // Publish both names consumers read.
        if let Some(claim) = root.get("claim_boundary").and_then(Value::as_object) {
            if let Err(e) = boolean(claim, "test_only_fake_child", false, "outer claim") {
                blockers.push(e);
            }
        }
        match root.get("inner_capture") {
            Some(value) => {
                if let Err(e) = pointer(value, &inner, "outer inner_capture") {
                    blockers.push(e);
                }
            }
            None => blockers.push("outer.inner_capture is required".into()),
        }
        let terminal = match root.get("child_terminal") {
            Some(value) => match obj(value, "child_terminal") {
                Ok(t) => t,
                Err(e) => {
                    blockers.push(e);
                    return Err(blockers);
                }
            },
            None => {
                blockers.push("outer.child_terminal is required".into());
                return Err(blockers);
            }
        };
        match number(terminal, "exit_code", "child_terminal") {
            Ok(0) => {}
            Ok(n) => blockers.push(format!(
                "outer child_terminal.exit_code observed={n}, expected=0"
            )),
            Err(e) => blockers.push(e),
        }
        if let Err(e) = boolean(terminal, "reaped", true, "child_terminal") {
            blockers.push(e);
        }
        if let Err(e) = boolean(
            terminal,
            "terminal_receipt_written_last",
            true,
            "child_terminal",
        ) {
            blockers.push(e);
        }
    }

    // --- release: check BOTH released_after_outer_terminal and release_after_outer_terminal ---
    {
        let root = obj(&release.document, "release").map_err(|e| vec![e])?;
        match root.get("outer_terminal") {
            Some(value) => {
                if let Err(e) = pointer(value, &outer, "release outer_terminal") {
                    blockers.push(e);
                }
            }
            None => blockers.push("release.outer_terminal is required".into()),
        }
        if let Err(e) = boolean(root, "actual_release_performed", true, "release") {
            blockers.push(e);
        }
        // Consumer-facing dual names (campaign convention).
        let released = root
            .get("released_after_outer_terminal")
            .and_then(Value::as_bool);
        let release_after = root
            .get("release_after_outer_terminal")
            .and_then(Value::as_bool);
        match (released, release_after) {
            (Some(true), Some(true)) | (Some(true), None) | (None, Some(true)) => {}
            (Some(false), _) | (_, Some(false)) => {
                blockers.push(format!(
                    "release after outer terminal observed released_after_outer_terminal={released:?} release_after_outer_terminal={release_after:?}, expected true under at least one consumer name"
                ));
            }
            (None, None) => {
                blockers.push(
                    "release must publish released_after_outer_terminal and/or release_after_outer_terminal=true"
                        .into(),
                );
            }
        }
        if let Err(e) = boolean(root, "lease_released", true, "release") {
            blockers.push(e);
        }
    }

    // Source identity on schedule
    {
        let sroot = obj(&schedule.document, "schedule").map_err(|e| vec![e])?;
        if let Some(source) = sroot.get("source_authority").and_then(Value::as_object) {
            if let Ok(rev) = text(source, "source_revision", "schedule") {
                if rev != QWEN80_SOURCE_REVISION {
                    blockers.push(format!(
                        "schedule source_revision observed={rev}, expected={QWEN80_SOURCE_REVISION}"
                    ));
                }
            }
            if let Ok(seal) = text(source, "gravity_manifest_seal_sha256", "schedule") {
                if seal != QWEN80_GRAVITY_MANIFEST_SEAL_SHA256 {
                    blockers.push(format!(
                        "schedule gravity seal observed={seal}, expected={QWEN80_GRAVITY_MANIFEST_SEAL_SHA256}"
                    ));
                }
            }
        }
    }

    if !blockers.is_empty() {
        return Err(blockers);
    }

    let mut report = json!({
        "schema": RESULT_SCHEMA,
        "status": EARNED_STATUS,
        "earned_multi_layer_component_only": true,
        "blockers": [],
        "component_scope": {
            "layer_count": layer_count,
            "layers_first": 0,
            "layers_last": layer_count - 1,
            "total_dispatches": expected_total,
            "per_layer_dispatches": QWEN80_DELTANET_FULL_LAYER_DISPATCHES,
            "single_fence": true,
            "token_or_decoder_earned": false,
        },
        "source_authority": {
            "source_revision": QWEN80_SOURCE_REVISION,
            "gravity_manifest_seal_sha256": QWEN80_GRAVITY_MANIFEST_SEAL_SHA256,
        },
        "evidence": {
            "execution_schedule_seal_sha256": schedule.seal_sha256,
            "chain_cpu_oracle_seal_sha256": chain_oracle.seal_sha256,
            "host_preflight_seal_sha256": host_preflight.seal_sha256,
            "inner_seal_sha256": inner.seal_sha256,
            "outer_seal_sha256": outer.seal_sha256,
            "release_seal_sha256": release.seal_sha256,
        },
        "claim_boundary": {
            "multi_layer_component_only": true,
            "token_generated": false,
            "decoder_started": false,
            "server_or_watcher_started": false,
            "tps_or_tg_measured": false,
            "tournament_started": false,
            "terminal_head_authorized": false,
        },
    });
    seal(&mut report).map_err(|e| vec![e])?;
    Ok(report)
}

fn main() {
    match parse_args(env::args().skip(1)).and_then(|args| {
        let input = read_json(&args.input)?;
        let output = assess(&input);
        let status = output["status"].as_str().unwrap_or("").to_owned();
        let seal = verify_seal(&output, "output")?;
        write_new(&args.out, &output)?;
        Ok((status, seal))
    }) {
        Ok((status, seal)) => {
            println!("{{\"status\":\"{status}\",\"seal_sha256\":\"{seal}\"}}");
        }
        Err(error) => {
            eprintln!("Q80 multi-layer completion assessor refused: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(character: char) -> String {
        std::iter::repeat_n(character, 64).collect()
    }

    fn sealed(mut value: Value) -> Value {
        seal(&mut value).unwrap();
        value
    }

    fn reseal(mut value: Value) -> Value {
        value.as_object_mut().unwrap().remove("seal_sha256");
        seal(&mut value).unwrap();
        value
    }

    fn wrapper(document: Value) -> Value {
        json!({
            "document": document.clone(),
            "document_sha256": json_sha(&document).unwrap(),
            "document_seal_sha256": document["seal_sha256"].clone(),
        })
    }

    fn reference(document: &Value) -> Value {
        json!({
            "present": true,
            "document_sha256": json_sha(document).unwrap(),
            "document_seal_sha256": document["seal_sha256"].clone(),
        })
    }

    /// Real producer convention: both fields carry the seal.
    fn producer_reference(document: &Value) -> Value {
        json!({
            "present": true,
            "document_sha256": document["seal_sha256"].clone(),
            "document_seal_sha256": document["seal_sha256"].clone(),
        })
    }

    fn pair(a: char, b: char, err: f64) -> Value {
        json!({
            "passed": true,
            "cpu_f32le_sha256": sha(a),
            "device_f32le_sha256": sha(b),
            "max_abs_error": err,
        })
    }

    fn schedule_doc() -> Value {
        sealed(json!({
            "schema": SCHEDULE_SCHEMA,
            "status": "PREPARED_QWEN80_48_LAYER_EXECUTION_SCHEDULE_AUTHORITY_NOT_EXECUTED",
            "source_authority": {
                "source_revision": QWEN80_SOURCE_REVISION,
                "gravity_manifest_seal_sha256": QWEN80_GRAVITY_MANIFEST_SEAL_SHA256,
            },
        }))
    }

    fn oracle_doc(layer_count: usize) -> Value {
        sealed(json!({
            "schema": CHAIN_ORACLE_SCHEMA,
            "status": "PREPARED_QWEN80_MULTI_LAYER_CHAIN_CPU_ORACLE_STRUCTURE_NOT_NUMERIC_WITHOUT_LAYER_RECEIPTS",
            "layer_count": layer_count,
            "includes_unready_gqa": false,
            "total_dispatches_physical_capture": layer_count * 23,
        }))
    }

    fn preflight_doc(layer_count: usize) -> Value {
        sealed(json!({
            "schema": HOST_PREFLIGHT_SCHEMA,
            "status": "COMPILED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_SAME_RUNTIME_HOST_CPU_ONLY_NOT_LEASED_OR_EXECUTED",
            "layer_count": layer_count,
            "claim_boundary": {
                "fixture_or_synthetic": false,
                "test_only_fake_child": false,
            },
        }))
    }

    fn inner_with(
        layer_count: usize,
        schedule: &Value,
        oracle: &Value,
        preflight: &Value,
        reference_fn: fn(&Value) -> Value,
        retained: f64,
    ) -> Value {
        let kernels = qwen80_multi_layer_structural_kernel_trace(layer_count, false).unwrap();
        let mut readbacks = Vec::new();
        for layer in 0..layer_count {
            readbacks.push(json!({
                "layer": layer,
                "output_elements": HIDDEN_ELEMENTS,
                "second_residual_output": pair('a', 'b', retained),
            }));
        }
        sealed(json!({
            "schema": INNER_SCHEMA,
            "status": INNER_STATUS,
            "fixture_or_synthetic": false,
            "self_asserted": false,
            "execution_schedule_provenance": reference_fn(schedule),
            "chain_cpu_oracle_provenance": reference_fn(oracle),
            "host_preflight_provenance": reference_fn(preflight),
            "fresh_same_runtime_execution": {
                "fresh_runtime": true,
                "same_runtime": true,
                "same_tcb": true,
                "single_fence_after_all_dispatches": true,
                "layer_count": layer_count,
                "total_dispatches": layer_count * 23,
                "fence_count": 1,
            },
            "structural_kernel_trace": {
                "exact_order": true,
                "kernel_names": kernels,
            },
            "per_layer_readbacks": readbacks,
            "retained_max_abs_error": retained,
            "claim_boundary": {
                "multi_layer_component_only": true,
                "token_generated": false,
                "decoder_started": false,
                "tps_or_tg_measured": false,
                "tournament_started": false,
            },
        }))
    }

    fn outer_with(inner: &Value, reference_fn: fn(&Value) -> Value) -> Value {
        sealed(json!({
            "schema": OUTER_SCHEMA,
            "status": OUTER_STATUS,
            "fixture_or_synthetic": false,
            "inner_capture": reference_fn(inner),
            "lease_id": sha('e'),
            "child_terminal": {
                "exit_code": 0,
                "reaped": true,
                "terminal_receipt_written_last": true,
            },
            "claim_boundary": {
                "test_only_fake_child": false,
                "multi_layer_component_only": true,
            },
        }))
    }

    fn release_with(outer: &Value, reference_fn: fn(&Value) -> Value) -> Value {
        sealed(json!({
            "schema": RELEASE_SCHEMA,
            "status": RELEASE_STATUS,
            "outer_terminal": reference_fn(outer),
            "lease_id": sha('e'),
            "actual_release_performed": true,
            "released_after_outer_terminal": true,
            "release_after_outer_terminal": true,
            "lease_released": true,
        }))
    }

    fn input_with(layer_count: usize, reference_fn: fn(&Value) -> Value) -> Value {
        let schedule = schedule_doc();
        let oracle = oracle_doc(layer_count);
        let preflight = preflight_doc(layer_count);
        let inner = inner_with(
            layer_count,
            &schedule,
            &oracle,
            &preflight,
            reference_fn,
            0.0000000317,
        );
        let outer = outer_with(&inner, reference_fn);
        let release = release_with(&outer, reference_fn);
        sealed(json!({
            "schema": INPUT_SCHEMA,
            "layer_count": layer_count,
            "execution_schedule_authority": wrapper(schedule),
            "chain_cpu_oracle": wrapper(oracle),
            "host_preflight": wrapper(preflight),
            "fresh_multi_layer_inner": wrapper(inner),
            "fresh_multi_layer_outer": wrapper(outer),
            "fresh_multi_layer_release": wrapper(release),
        }))
    }

    #[test]
    fn earns_multi_layer_component_under_fixture_and_producer_pointer_conventions() {
        let report = assess(&input_with(3, reference));
        assert_eq!(report["status"], EARNED_STATUS);
        assert_eq!(report["earned_multi_layer_component_only"], true);
        assert_eq!(report["component_scope"]["total_dispatches"], 69);
        assert_eq!(report["claim_boundary"]["decoder_started"], false);
        verify_seal(&report, "report").unwrap();

        let producer = assess(&input_with(3, producer_reference));
        assert_eq!(producer["status"], EARNED_STATUS);
        assert!(producer["blockers"].as_array().unwrap().is_empty());
    }

    #[test]
    fn rejects_token_claim_and_wrong_dispatch_count_with_values() {
        let mut value = input_with(3, producer_reference);
        let inner = value["fresh_multi_layer_inner"]["document"]
            .as_object_mut()
            .unwrap();
        inner["claim_boundary"]["token_generated"] = json!(true);
        let inner = reseal(Value::Object(inner.clone()));
        value["fresh_multi_layer_inner"] = wrapper(inner.clone());
        let outer = outer_with(&inner, producer_reference);
        let release = release_with(&outer, producer_reference);
        value["fresh_multi_layer_outer"] = wrapper(outer);
        value["fresh_multi_layer_release"] = wrapper(release);
        value = reseal(value);
        let report = assess(&value);
        assert_eq!(report["status"], REFUSED_STATUS);
        assert!(report["blockers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|b| b.as_str().unwrap().contains("token_generated")));

        let mut value = input_with(3, producer_reference);
        let inner = value["fresh_multi_layer_inner"]["document"]
            .as_object_mut()
            .unwrap();
        inner["fresh_same_runtime_execution"]["total_dispatches"] = json!(46);
        let inner = reseal(Value::Object(inner.clone()));
        value["fresh_multi_layer_inner"] = wrapper(inner.clone());
        let outer = outer_with(&inner, producer_reference);
        let release = release_with(&outer, producer_reference);
        value["fresh_multi_layer_outer"] = wrapper(outer);
        value["fresh_multi_layer_release"] = wrapper(release);
        value = reseal(value);
        let report = assess(&value);
        assert_eq!(report["status"], REFUSED_STATUS);
        assert!(report["blockers"].as_array().unwrap().iter().any(|b| {
            let s = b.as_str().unwrap();
            s.contains("total_dispatches") && s.contains("observed=46") && s.contains("expected=69")
        }));
    }

    #[test]
    fn rejects_producer_pointer_wrong_seal() {
        let mut value = input_with(3, producer_reference);
        let inner = value["fresh_multi_layer_inner"]["document"]
            .as_object_mut()
            .unwrap();
        inner["execution_schedule_provenance"]["document_seal_sha256"] = json!(sha('f'));
        inner["execution_schedule_provenance"]["document_sha256"] = json!(sha('f'));
        let inner = reseal(Value::Object(inner.clone()));
        value["fresh_multi_layer_inner"] = wrapper(inner.clone());
        let outer = outer_with(&inner, producer_reference);
        let release = release_with(&outer, producer_reference);
        value["fresh_multi_layer_outer"] = wrapper(outer);
        value["fresh_multi_layer_release"] = wrapper(release);
        value = reseal(value);
        let report = assess(&value);
        assert_eq!(report["status"], REFUSED_STATUS);
        assert!(report["blockers"].as_array().unwrap().iter().any(|b| b
            .as_str()
            .unwrap()
            .contains("does not bind the expected sealed document")));
    }

    #[test]
    fn rejects_missing_retained_max_and_release_dual_names() {
        let mut value = input_with(2, producer_reference);
        let inner = value["fresh_multi_layer_inner"]["document"]
            .as_object_mut()
            .unwrap();
        inner.remove("retained_max_abs_error");
        let inner = reseal(Value::Object(inner.clone()));
        value["fresh_multi_layer_inner"] = wrapper(inner.clone());
        let outer = outer_with(&inner, producer_reference);
        let release = release_with(&outer, producer_reference);
        value["fresh_multi_layer_outer"] = wrapper(outer);
        value["fresh_multi_layer_release"] = wrapper(release);
        value = reseal(value);
        let report = assess(&value);
        assert_eq!(report["status"], REFUSED_STATUS);
        assert!(report["blockers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|b| b.as_str().unwrap().contains("retained_max_abs_error")));

        let mut value = input_with(2, producer_reference);
        let release_doc = value["fresh_multi_layer_release"]["document"]
            .as_object_mut()
            .unwrap();
        release_doc.insert("released_after_outer_terminal".into(), json!(false));
        release_doc.insert("release_after_outer_terminal".into(), json!(false));
        let release_doc = reseal(Value::Object(release_doc.clone()));
        value["fresh_multi_layer_release"] = wrapper(release_doc);
        value = reseal(value);
        let report = assess(&value);
        assert_eq!(report["status"], REFUSED_STATUS);
        assert!(report["blockers"].as_array().unwrap().iter().any(|b| {
            b.as_str()
                .unwrap()
                .contains("released_after_outer_terminal")
        }));
    }

    #[test]
    fn rejects_outer_fixture_flag() {
        let mut value = input_with(2, producer_reference);
        let outer = value["fresh_multi_layer_outer"]["document"]
            .as_object_mut()
            .unwrap();
        outer.insert("fixture_or_synthetic".into(), json!(true));
        let outer = reseal(Value::Object(outer.clone()));
        value["fresh_multi_layer_outer"] = wrapper(outer.clone());
        let release = release_with(&outer, producer_reference);
        value["fresh_multi_layer_release"] = wrapper(release);
        value = reseal(value);
        let report = assess(&value);
        assert_eq!(report["status"], REFUSED_STATUS);
        assert!(report["blockers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|b| b.as_str().unwrap().contains("fixture_or_synthetic")));
    }
}
