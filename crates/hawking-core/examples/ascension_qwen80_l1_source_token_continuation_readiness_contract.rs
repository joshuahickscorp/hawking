//! CPU-only Qwen80 source-token Layer-1 continuation readiness contract.
//!
//! This is an authoritative checklist for one future, bounded Layer-1
//! DeltaNet prefix capture.  It consumes sealed copies of the permanent
//! 48-layer schedule authority and the existing L0-to-L1 child preflight,
//! then waits for one future sealed L0 handoff receipt.  No branch here can
//! encode Layer 1: a valid handoff receipt explicitly says
//! `L1_BINDING_NOT_EXECUTED`, and the output only reserves the next capture.
//! The received L0 device identities are baseline evidence, not transferable
//! execution inputs: a PinnedBuffer cannot cross process boundaries.  Any
//! future Layer-1 success must therefore be one fresh same-runtime,
//! same-TCB joint L0-to-L1 capture that produces its own L0 output identity.
//!
//! The contract does not open model artifacts, scan a directory, construct a
//! Metal context, acquire a lease, start a process/server/watcher, dispatch a
//! kernel, or measure TPS/TG.  It returns only a prepared or incomplete
//! checklist.  Neither status is decoder, layer, token, server, HCLI, TG, or
//! tournament evidence.

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const INPUT_SCHEMA: &str =
    "hawking.ascension.qwen80_l1_source_token_continuation_readiness_input.v1";
const RESULT_SCHEMA: &str =
    "hawking.ascension.qwen80_l1_source_token_continuation_readiness_contract.v1";
const PREPARED_STATUS: &str =
    "PREPARED_QWEN80_SOURCE_TOKEN_L1_SLOT1_DELTANET_PREFIX_CAPTURE_RESERVED_NOT_EXECUTED";
const INCOMPLETE_STATUS: &str =
    "INCOMPLETE_QWEN80_SOURCE_TOKEN_L1_CONTINUATION_MISSING_TRUSTED_L0_HANDOFF_OR_AUTHORITY";

const SCHEDULE_SCHEMA: &str = "hawking.ascension.qwen80_48_layer_payload_schedule_authority.v1";
const SCHEDULE_STATUS: &str = "PREPARED_QWEN80_48_LAYER_PAYLOAD_SCHEDULE_AUTHORITY_NOT_EXECUTED";
const SCHEDULE_WRAPPER_SCHEMA: &str =
    "hawking.ascension.qwen80_48_layer_schedule_sealed_wrapper.v1";
const SCHEDULE_WRAPPER_STATUS: &str =
    "SEALED_QWEN80_48_LAYER_PAYLOAD_SCHEDULE_AUTHORITY_BOUND_NOT_EXECUTED";
const RAW_SCHEDULE_SHA256: &str =
    "8302deb6beece8c04773ece19ae27baea67749014552b0b946516146b5e2282e";
const RAW_SCHEDULE_BYTES: u64 = 88_551_859;
const MANIFEST_DOCUMENT_SHA256: &str =
    "a0fcac0401a7962402bb8cb87d5055c83667b39575f9e0f4c7470d080758aa10";
const MANIFEST_SEAL_SHA256: &str =
    "14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b";
const SOURCE_CONFIG_SHA256: &str =
    "a7b8098d3b05777f12bb5677a26bf1240a1bb09def1b06b29e6be86cae2e84f8";
const SOURCE_CONFIG_AUTHORITY_DOCUMENT_SHA256: &str =
    "c21f5ac489d58d91ba2eb43c3daf34e2412f39925632b30e147e5de28780596b";
const SOURCE_CONFIG_AUTHORITY_SEAL_SHA256: &str =
    "3d062ca5a8acdcc3c2c018e4ded049fd6647210b8161dfcedd37e99363c8fafd";
const L0_TO_L1_PREFLIGHT_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_to_layer1_state_handoff_device.v1";
const L0_TO_L1_PREFLIGHT_STATUS: &str =
    "PREPARED_QWEN80_SOURCE_TOKEN_L0_STATE_COMMIT_ROLLBACK_AND_LAYER1_HANDOFF_CHILD_NOT_EXECUTED";
const L0_HANDOFF_RECEIPT_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_state_handoff_capture.v1";
const L0_HANDOFF_RECEIPT_STATUS: &str =
    "CAPTURED_QWEN80_SOURCE_TOKEN_L0_POST_STATE_ROLLBACK_RETAINED_OUTPUT_L1_BINDING_NOT_EXECUTED_COMPONENT_ONLY";

const MODEL_KEY: &str = "qwen80";
const MODEL_ID: &str = "Qwen3-Coder-Next-80B";
const SOURCE_REPOSITORY: &str = "Qwen/Qwen3-Coder-Next";
const SOURCE_REVISION: &str = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb";
const SOURCE_TOKEN_ID: u64 = 1;
const HIDDEN_ELEMENTS: u64 = 2_048;
const HIDDEN_BYTES: u64 = HIDDEN_ELEMENTS * 4;
const L0_SLOT: u64 = 0;
const L1_SLOT: u64 = 1;
const L1_LAYER: u64 = 1;
const L0_CONV_BYTES: u64 = 98_304;
const L0_RECURRENT_BYTES: u64 = 2_097_152;
const L1_CONV_OFFSET_BYTES: u64 = L0_CONV_BYTES;
const L1_RECURRENT_OFFSET_BYTES: u64 = L0_RECURRENT_BYTES;

const L1_PREFIX: [(&str, &str); 9] = [
    ("input_rmsnorm", "qwen_next_direct_packed_input_rmsnorm"),
    ("qkvz_projection", "qwen_binary_sign_scale_matvec"),
    ("ba_projection", "qwen_binary_sign_scale_matvec"),
    ("qkvz_rearrange_conv", "qwen_next_qkvz_rearrange_conv_l2"),
    ("ba_decay_beta", "qwen_next_ba_to_decay_beta"),
    ("deltanet_recurrent", "qwen_next_gated_delta_decode_single"),
    ("deltanet_gated_rmsnorm", "qwen_next_deltanet_gated_rmsnorm"),
    ("out_projection", "qwen_binary_sign_scale_matvec"),
    ("first_residual", "qwen_next_add_residual"),
];

#[derive(Clone, Debug)]
struct Args {
    input: Option<PathBuf>,
    schedule_authority: Option<PathBuf>,
    l0_to_l1_preflight: Option<PathBuf>,
    l0_state_handoff_receipt: Option<PathBuf>,
    out: PathBuf,
}

#[derive(Clone, Debug)]
struct BoundDocument {
    document: Value,
    document_sha256: String,
    document_seal_sha256: String,
}

#[derive(Clone, Debug)]
struct L0HandoffFacts {
    session_id: String,
    output_sha256: String,
    output_device_buffer_id: String,
    receipt_seal_sha256: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Render finite JSON floats with CPython's `json.dumps` spelling.
///
/// The evidence receipts are sealed with Python's sorted, compact JSON form.
/// Rust's Ryu spelling differs for scientific exponents, so this lexical form
/// is required to verify authentic receipts rather than only local fixtures.
fn python_json_float(number: &serde_json::Number) -> Result<String, String> {
    let value = number
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or("canonical JSON floating number must be finite")?;
    if value == 0.0 {
        return Ok(if value.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        });
    }

    let raw = number.to_string();
    let (negative, unsigned) = match raw.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, raw.as_str()),
    };
    let (mantissa, exponent) = match unsigned.find('e').or_else(|| unsigned.find('E')) {
        Some(index) => {
            let exponent = unsigned[index + 1..]
                .parse::<i32>()
                .map_err(|error| format!("canonical JSON exponent is invalid: {error}"))?;
            (&unsigned[..index], exponent)
        }
        None => (unsigned, 0),
    };
    let mut fractional_digits = 0i32;
    let mut after_decimal = false;
    let mut digits = String::new();
    for byte in mantissa.bytes() {
        match byte {
            b'.' if !after_decimal => after_decimal = true,
            b'0'..=b'9' => {
                if after_decimal {
                    fractional_digits = fractional_digits
                        .checked_add(1)
                        .ok_or("canonical JSON fractional digit count overflows")?;
                }
                digits.push(char::from(byte));
            }
            _ => return Err(format!("canonical JSON mantissa is invalid: {raw:?}")),
        }
    }
    let first_significant = digits
        .bytes()
        .position(|byte| byte != b'0')
        .ok_or("nonzero canonical JSON float has no significant digit")?;
    let mut significant = digits[first_significant..].to_owned();
    let mut decimal_power = exponent
        .checked_sub(fractional_digits)
        .ok_or("canonical JSON decimal exponent overflows")?;
    while significant.len() > 1 && significant.ends_with('0') {
        significant.pop();
        decimal_power = decimal_power
            .checked_add(1)
            .ok_or("canonical JSON decimal exponent overflows")?;
    }
    let scientific_exponent = decimal_power
        .checked_add(i32::try_from(significant.len() - 1).unwrap_or(i32::MAX))
        .ok_or("canonical JSON decimal exponent overflows")?;
    let sign = if negative { "-" } else { "" };
    if !(-4..16).contains(&scientific_exponent) {
        let mut rendered_mantissa = significant[..1].to_owned();
        if significant.len() > 1 {
            rendered_mantissa.push('.');
            rendered_mantissa.push_str(&significant[1..]);
        }
        let exponent_sign = if scientific_exponent < 0 { '-' } else { '+' };
        return Ok(format!(
            "{sign}{rendered_mantissa}e{exponent_sign}{:02}",
            scientific_exponent.unsigned_abs()
        ));
    }

    let decimal_position = scientific_exponent + 1;
    let rendered = if decimal_position <= 0 {
        format!(
            "0.{}{}",
            "0".repeat(usize::try_from(-decimal_position).unwrap_or(usize::MAX)),
            significant
        )
    } else if usize::try_from(decimal_position).unwrap_or(usize::MAX) >= significant.len() {
        format!(
            "{}{}.0",
            significant,
            "0".repeat(usize::try_from(decimal_position).unwrap_or(usize::MAX) - significant.len())
        )
    } else {
        let position = usize::try_from(decimal_position).unwrap();
        format!("{}.{}", &significant[..position], &significant[position..])
    };
    Ok(format!("{sign}{rendered}"))
}

fn canonical_json_into(value: &Value, output: &mut String) -> Result<(), String> {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(number) => {
            if number.is_i64() || number.is_u64() {
                output.push_str(&number.to_string());
            } else {
                output.push_str(&python_json_float(number)?);
            }
        }
        Value::String(value) => output.push_str(
            &serde_json::to_string(value)
                .map_err(|error| format!("cannot canonicalize JSON string: {error}"))?,
        ),
        Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                canonical_json_into(value, output)?;
            }
            output.push(']');
        }
        Value::Object(values) => {
            let mut keys: Vec<&String> = values.keys().collect();
            keys.sort_unstable();
            output.push('{');
            for (index, key) in keys.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                output.push_str(
                    &serde_json::to_string(*key)
                        .map_err(|error| format!("cannot canonicalize JSON key: {error}"))?,
                );
                output.push(':');
                canonical_json_into(&values[*key], output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

fn canonical_json(value: &Value) -> Result<Vec<u8>, String> {
    let mut output = String::new();
    canonical_json_into(value, &mut output)?;
    Ok(output.into_bytes())
}

fn sha256_json(value: &Value) -> Result<String, String> {
    canonical_json(value).map(|encoded| sha256_hex(&encoded))
}

fn seal(value: &mut Value) -> Result<String, String> {
    let object = object(value, "output")?;
    if object.contains_key("seal_sha256") {
        return Err("output must not already contain seal_sha256".into());
    }
    let seal = sha256_json(value)?;
    value
        .as_object_mut()
        .expect("validated object")
        .insert("seal_sha256".into(), Value::String(seal.clone()));
    Ok(seal)
}

fn verify_seal(value: &Value, label: &str) -> Result<String, String> {
    let object = object(value, label)?;
    let observed = require_sha(object, "seal_sha256", label)?;
    let mut unsigned = object.clone();
    unsigned.remove("seal_sha256");
    let expected = sha256_json(&Value::Object(unsigned))?;
    if observed != expected {
        return Err(format!("{label} seal mismatch"));
    }
    Ok(observed)
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be a JSON object"))
}

fn object_field<'a>(
    object: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a Map<String, Value>, String> {
    object
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{label}.{field} must be an object"))
}

fn array_field<'a>(
    object: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a [Value], String> {
    object
        .get(field)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| format!("{label}.{field} must be an array"))
}

fn require_string<'a>(
    object: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a str, String> {
    object
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{label}.{field} must be a non-empty string"))
}

fn require_sha(object: &Map<String, Value>, field: &str, label: &str) -> Result<String, String> {
    let value = require_string(object, field, label)?;
    if !is_lower_sha256(value) {
        return Err(format!("{label}.{field} must be a lowercase SHA-256"));
    }
    Ok(value.into())
}

fn require_u64(object: &Map<String, Value>, field: &str, label: &str) -> Result<u64, String> {
    object
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{label}.{field} must be an unsigned integer"))
}

fn require_bool(
    object: &Map<String, Value>,
    field: &str,
    expected: bool,
    label: &str,
) -> Result<(), String> {
    if object.get(field).and_then(Value::as_bool) != Some(expected) {
        return Err(format!("{label}.{field} must be {expected}"));
    }
    Ok(())
}

fn require_exact_string(
    object: &Map<String, Value>,
    field: &str,
    expected: &str,
    label: &str,
) -> Result<(), String> {
    let observed = require_string(object, field, label)?;
    if observed != expected {
        return Err(format!("{label}.{field} drifted"));
    }
    Ok(())
}

fn parse_bound_document(
    input: &Map<String, Value>,
    field: &str,
    expected_schema: &str,
    expected_status: &str,
) -> Result<BoundDocument, String> {
    parse_bound_document_with_identities(input, field, &[(expected_schema, expected_status)])
}

fn parse_bound_document_with_identities(
    input: &Map<String, Value>,
    field: &str,
    expected_identities: &[(&str, &str)],
) -> Result<BoundDocument, String> {
    let binding = object_field(input, field, "input")?;
    let document = binding
        .get("document")
        .cloned()
        .ok_or_else(|| format!("input.{field}.document is required"))?;
    let document_object = object(&document, &format!("input.{field}.document"))?;
    let document_sha256 = require_sha(binding, "document_sha256", &format!("input.{field}"))?;
    let document_seal_sha256 =
        require_sha(binding, "document_seal_sha256", &format!("input.{field}"))?;
    if sha256_json(&document)? != document_sha256 {
        return Err(format!(
            "input.{field}.document_sha256 does not bind document"
        ));
    }
    if verify_seal(&document, &format!("input.{field}.document"))? != document_seal_sha256 {
        return Err(format!(
            "input.{field}.document_seal_sha256 does not bind document"
        ));
    }
    let schema = require_string(
        document_object,
        "schema",
        &format!("input.{field}.document"),
    )?;
    let status = require_string(
        document_object,
        "status",
        &format!("input.{field}.document"),
    )?;
    if !expected_identities
        .iter()
        .any(|(expected_schema, expected_status)| {
            schema == *expected_schema && status == *expected_status
        })
    {
        return Err(format!("input.{field}.document schema or status drifted"));
    }
    Ok(BoundDocument {
        document,
        document_sha256,
        document_seal_sha256,
    })
}

fn parse_schedule_document(input: &Map<String, Value>) -> Result<BoundDocument, String> {
    parse_bound_document_with_identities(
        input,
        "schedule_authority",
        &[
            (SCHEDULE_SCHEMA, SCHEDULE_STATUS),
            (SCHEDULE_WRAPPER_SCHEMA, SCHEDULE_WRAPPER_STATUS),
        ],
    )
}

fn validate_raw_schedule(schedule: &BoundDocument) -> Result<(), String> {
    let root = object(&schedule.document, "48-layer schedule authority")?;
    require_bool(
        root,
        "all_48_layers_scheduled",
        true,
        "48-layer schedule authority",
    )?;
    let layers = array_field(root, "layers", "48-layer schedule authority")?;
    if layers.len() != 48 {
        return Err("48-layer schedule authority must retain exactly 48 layers".into());
    }
    let layer1 = layers
        .get(L1_LAYER as usize)
        .and_then(Value::as_object)
        .ok_or("48-layer schedule authority lacks layer 1")?;
    if require_u64(layer1, "layer", "48-layer schedule authority.layers[1]")? != L1_LAYER
        || require_string(layer1, "mixer", "48-layer schedule authority.layers[1]")? != "delta_net"
    {
        return Err("48-layer schedule authority does not reserve Layer 1 DeltaNet".into());
    }
    let slot = object_field(
        layer1,
        "state_slot",
        "48-layer schedule authority.layers[1]",
    )?;
    if require_u64(
        slot,
        "slot",
        "48-layer schedule authority.layers[1].state_slot",
    )? != L1_SLOT
        || require_string(
            slot,
            "domain",
            "48-layer schedule authority.layers[1].state_slot",
        )? != "delta_net_conv_and_recurrent"
    {
        return Err("48-layer schedule authority Layer 1 state slot drifted".into());
    }
    let boundary = object_field(root, "claim_boundary", "48-layer schedule authority")?;
    for field in [
        "artifact_payload_open_or_scan_performed",
        "metal_device_or_dispatch_performed",
        "runtime_watcher_registry_server_or_hcli_changed",
        "model_execution_performed",
        "token_generation_or_feedback_performed",
        "tps_or_tg_measured",
    ] {
        require_bool(
            boundary,
            field,
            false,
            "48-layer schedule authority.claim_boundary",
        )?;
    }
    Ok(())
}

fn validate_sealed_schedule_wrapper(schedule: &BoundDocument) -> Result<(), String> {
    let root = object(&schedule.document, "sealed 48-layer schedule wrapper")?;
    let raw = object_field(
        root,
        "raw_schedule_authority",
        "sealed 48-layer schedule wrapper",
    )?;
    require_bool(
        raw,
        "present",
        true,
        "sealed 48-layer schedule wrapper.raw_schedule_authority",
    )?;
    if require_u64(
        raw,
        "bytes",
        "sealed 48-layer schedule wrapper.raw_schedule_authority",
    )? != RAW_SCHEDULE_BYTES
        || require_sha(
            raw,
            "sha256",
            "sealed 48-layer schedule wrapper.raw_schedule_authority",
        )? != RAW_SCHEDULE_SHA256
    {
        return Err("sealed schedule wrapper does not bind the canonical raw static plan".into());
    }
    require_exact_string(
        raw,
        "schema",
        SCHEDULE_SCHEMA,
        "sealed 48-layer schedule wrapper.raw_schedule_authority",
    )?;
    require_exact_string(
        raw,
        "status",
        SCHEDULE_STATUS,
        "sealed 48-layer schedule wrapper.raw_schedule_authority",
    )?;
    if raw.get("raw_schedule_seal_sha256") != Some(&Value::Null) {
        return Err(
            "sealed schedule wrapper must preserve the raw schedule's unsealed identity".into(),
        );
    }
    require_bool(
        raw,
        "raw_schedule_is_static_and_unmodified",
        true,
        "sealed 48-layer schedule wrapper.raw_schedule_authority",
    )?;
    require_string(
        raw,
        "path",
        "sealed 48-layer schedule wrapper.raw_schedule_authority",
    )?;

    let source = object_field(root, "source_authority", "sealed 48-layer schedule wrapper")?;
    for (field, expected) in [
        ("model_key", MODEL_KEY),
        ("model_id", MODEL_ID),
        ("source_repository", SOURCE_REPOSITORY),
        ("source_revision", SOURCE_REVISION),
        ("source_config_sha256", SOURCE_CONFIG_SHA256),
        (
            "descriptor_inventory_document_sha256",
            MANIFEST_DOCUMENT_SHA256,
        ),
        ("descriptor_inventory_seal_sha256", MANIFEST_SEAL_SHA256),
        (
            "source_config_authority_document_sha256",
            SOURCE_CONFIG_AUTHORITY_DOCUMENT_SHA256,
        ),
        (
            "source_config_authority_seal_sha256",
            SOURCE_CONFIG_AUTHORITY_SEAL_SHA256,
        ),
    ] {
        require_exact_string(
            source,
            field,
            expected,
            "sealed 48-layer schedule wrapper.source_authority",
        )?;
    }

    let facts = object_field(root, "schedule_facts", "sealed 48-layer schedule wrapper")?;
    for (field, expected) in [
        ("layer_count", 48),
        ("delta_net_layer_count", 36),
        ("gqa_layer_count", 12),
        ("delta_net_state_slot_count", 36),
        ("gqa_state_slot_count", 12),
        ("full_command_graph_item_count", 54),
    ] {
        if require_u64(
            facts,
            field,
            "sealed 48-layer schedule wrapper.schedule_facts",
        )? != expected
        {
            return Err(format!("sealed 48-layer schedule wrapper {field} drifted"));
        }
    }
    require_bool(
        facts,
        "all_48_layers_scheduled",
        true,
        "sealed 48-layer schedule wrapper.schedule_facts",
    )?;
    let layer1 = object_field(
        facts,
        "layer_1",
        "sealed 48-layer schedule wrapper.schedule_facts",
    )?;
    if require_u64(
        layer1,
        "layer",
        "sealed 48-layer schedule wrapper.schedule_facts.layer_1",
    )? != L1_LAYER
        || require_u64(
            layer1,
            "state_slot",
            "sealed 48-layer schedule wrapper.schedule_facts.layer_1",
        )? != L1_SLOT
    {
        return Err("sealed schedule wrapper Layer 1 position drifted".into());
    }
    require_exact_string(
        layer1,
        "mixer",
        "delta_net",
        "sealed 48-layer schedule wrapper.schedule_facts.layer_1",
    )?;
    require_exact_string(
        layer1,
        "state_domain",
        "delta_net_conv_and_recurrent",
        "sealed 48-layer schedule wrapper.schedule_facts.layer_1",
    )?;

    let boundary = object_field(root, "claim_boundary", "sealed 48-layer schedule wrapper")?;
    require_bool(
        boundary,
        "wrapper_is_read_only",
        true,
        "sealed 48-layer schedule wrapper.claim_boundary",
    )?;
    for field in [
        "raw_schedule_rewritten_or_resealed",
        "artifact_payload_open_or_scan_performed",
        "metal_device_or_dispatch_performed",
        "runtime_server_watcher_or_hcli_changed",
        "lease_issued_or_released",
        "token_generation_or_feedback_performed",
        "tps_or_tg_measured",
        "future_joint_l0_to_l1_capture_authorized",
    ] {
        require_bool(
            boundary,
            field,
            false,
            "sealed 48-layer schedule wrapper.claim_boundary",
        )?;
    }
    Ok(())
}

fn validate_schedule(schedule: &BoundDocument) -> Result<(), String> {
    let root = object(&schedule.document, "schedule authority")?;
    match (
        require_string(root, "schema", "schedule authority")?,
        require_string(root, "status", "schedule authority")?,
    ) {
        (SCHEDULE_SCHEMA, SCHEDULE_STATUS) => validate_raw_schedule(schedule),
        (SCHEDULE_WRAPPER_SCHEMA, SCHEDULE_WRAPPER_STATUS) => {
            validate_sealed_schedule_wrapper(schedule)
        }
        _ => Err("schedule authority schema or status drifted".into()),
    }
}

fn validate_preflight(preflight: &BoundDocument) -> Result<(), String> {
    let root = object(&preflight.document, "L0-to-L1 child preflight")?;
    require_exact_string(
        root,
        "mode",
        "cpu_only_preflight",
        "L0-to-L1 child preflight",
    )?;
    let source = object_field(root, "source_binding", "L0-to-L1 child preflight")?;
    require_exact_string(
        source,
        "model_key",
        MODEL_KEY,
        "L0-to-L1 child preflight.source_binding",
    )?;
    if require_u64(
        source,
        "source_token_id",
        "L0-to-L1 child preflight.source_binding",
    )? != SOURCE_TOKEN_ID
    {
        return Err("L0-to-L1 child preflight source token drifted".into());
    }
    let planned = object_field(
        root,
        "planned_pre_l1_handoff_capture",
        "L0-to-L1 child preflight",
    )?;
    require_exact_string(
        planned,
        "schema",
        L0_HANDOFF_RECEIPT_SCHEMA,
        "L0-to-L1 child preflight.planned_pre_l1_handoff_capture",
    )?;
    require_exact_string(
        planned,
        "status",
        L0_HANDOFF_RECEIPT_STATUS,
        "L0-to-L1 child preflight.planned_pre_l1_handoff_capture",
    )?;
    require_bool(
        planned,
        "l1_binding_not_executed",
        true,
        "L0-to-L1 child preflight.planned_pre_l1_handoff_capture",
    )?;
    if require_u64(
        planned,
        "l1_prefix_dispatches",
        "L0-to-L1 child preflight.planned_pre_l1_handoff_capture",
    )? != 0
    {
        return Err("L0-to-L1 child preflight may not execute an L1 prefix".into());
    }
    let witness = object_field(
        root,
        "required_next_layer_handoff_witness",
        "L0-to-L1 child preflight",
    )?;
    let retained = object_field(
        witness,
        "retained_l0_second_residual",
        "L0-to-L1 child preflight.required_next_layer_handoff_witness",
    )?;
    if require_u64(
        retained,
        "elements",
        "L0-to-L1 child preflight.required_next_layer_handoff_witness.retained_l0_second_residual",
    )? != HIDDEN_ELEMENTS
        || require_u64(
            retained,
            "bytes",
            "L0-to-L1 child preflight.required_next_layer_handoff_witness.retained_l0_second_residual",
        )? != HIDDEN_BYTES
    {
        return Err("L0-to-L1 child preflight retained-output geometry drifted".into());
    }
    let layer1 = object_field(
        witness,
        "layer1_input_binding",
        "L0-to-L1 child preflight.required_next_layer_handoff_witness",
    )?;
    if require_u64(
        layer1,
        "layer",
        "L0-to-L1 child preflight.required_next_layer_handoff_witness.layer1_input_binding",
    )? != L1_LAYER
        || require_u64(
            layer1,
            "linear_state_slot",
            "L0-to-L1 child preflight.required_next_layer_handoff_witness.layer1_input_binding",
        )? != L1_SLOT
    {
        return Err("L0-to-L1 child preflight Layer 1 slot drifted".into());
    }
    Ok(())
}

fn validate_state_record(
    state: &Map<String, Value>,
    field: &str,
    expected_slot: u64,
    expected_capacity: u64,
    hash_field: &str,
    label: &str,
) -> Result<(), String> {
    let record = object_field(state, field, label)?;
    if require_string(record, "allocation_id", &format!("{label}.{field}"))?.is_empty()
        || require_u64(record, "slot", &format!("{label}.{field}"))? != expected_slot
        || require_u64(record, "capacity_bytes", &format!("{label}.{field}"))? != expected_capacity
    {
        return Err(format!("{label}.{field} state layout drifted"));
    }
    require_sha(record, "device_buffer_id", &format!("{label}.{field}"))?;
    require_sha(record, hash_field, &format!("{label}.{field}"))?;
    Ok(())
}

fn validate_l0_handoff(receipt: &BoundDocument) -> Result<L0HandoffFacts, String> {
    let root = object(&receipt.document, "L0 handoff receipt")?;
    require_exact_string(root, "mode", "metal", "L0 handoff receipt")?;
    require_bool(
        root,
        "metal_device_or_dispatch_performed",
        true,
        "L0 handoff receipt",
    )?;
    require_bool(root, "component_only", true, "L0 handoff receipt")?;
    require_bool(
        root,
        "complete_layer_or_token_performed",
        false,
        "L0 handoff receipt",
    )?;
    require_bool(root, "l1_binding_not_executed", true, "L0 handoff receipt")?;
    if require_u64(root, "l1_prefix_dispatches", "L0 handoff receipt")? != 0 {
        return Err("L0 handoff receipt must not contain L1 prefix dispatches".into());
    }

    let handoff = object_field(root, "l0_state_handoff", "L0 handoff receipt")?;
    let session_id =
        require_string(handoff, "session_id", "L0 handoff receipt.l0_state_handoff")?.to_owned();
    if require_u64(
        handoff,
        "source_token_id",
        "L0 handoff receipt.l0_state_handoff",
    )? != SOURCE_TOKEN_ID
    {
        return Err("L0 handoff receipt source token drifted".into());
    }
    require_bool(
        handoff,
        "same_command_graph_retained",
        true,
        "L0 handoff receipt.l0_state_handoff",
    )?;
    require_bool(
        handoff,
        "l1_binding_not_executed",
        true,
        "L0 handoff receipt.l0_state_handoff",
    )?;
    if require_u64(
        handoff,
        "l1_prefix_dispatches",
        "L0 handoff receipt.l0_state_handoff",
    )? != 0
    {
        return Err("L0 handoff receipt must not contain L1 prefix dispatches".into());
    }
    let retained = object_field(
        handoff,
        "retained_l0_second_residual",
        "L0 handoff receipt.l0_state_handoff",
    )?;
    if require_u64(
        retained,
        "elements",
        "L0 handoff receipt.l0_state_handoff.retained_l0_second_residual",
    )? != HIDDEN_ELEMENTS
        || require_u64(
            retained,
            "bytes",
            "L0 handoff receipt.l0_state_handoff.retained_l0_second_residual",
        )? != HIDDEN_BYTES
    {
        return Err(
            "L0 handoff receipt retained output must be exactly 2048 f32 / 8192 bytes".into(),
        );
    }
    let output_sha256 = require_sha(
        retained,
        "f32le_sha256",
        "L0 handoff receipt.l0_state_handoff.retained_l0_second_residual",
    )?;
    let output_device_buffer_id = require_sha(
        retained,
        "device_buffer_id",
        "L0 handoff receipt.l0_state_handoff.retained_l0_second_residual",
    )?;
    require_bool(
        retained,
        "retained_for_future_layer1_encode",
        true,
        "L0 handoff receipt.l0_state_handoff.retained_l0_second_residual",
    )?;

    let l0_state = object_field(
        handoff,
        "l0_post_state_commit",
        "L0 handoff receipt.l0_state_handoff",
    )?;
    if require_u64(
        l0_state,
        "layer",
        "L0 handoff receipt.l0_state_handoff.l0_post_state_commit",
    )? != 0
        || require_u64(
            l0_state,
            "linear_state_slot",
            "L0 handoff receipt.l0_state_handoff.l0_post_state_commit",
        )? != L0_SLOT
    {
        return Err("L0 handoff receipt must bind layer 0 / state slot 0".into());
    }
    require_bool(
        l0_state,
        "checkpoint_before_mutation",
        true,
        "L0 handoff receipt.l0_state_handoff.l0_post_state_commit",
    )?;
    validate_state_record(
        l0_state,
        "active_conv",
        L0_SLOT,
        L0_CONV_BYTES,
        "post_state_f32le_sha256",
        "L0 handoff receipt.l0_state_handoff.l0_post_state_commit",
    )?;
    validate_state_record(
        l0_state,
        "active_recurrent",
        L0_SLOT,
        L0_RECURRENT_BYTES,
        "post_state_f32le_sha256",
        "L0 handoff receipt.l0_state_handoff.l0_post_state_commit",
    )?;
    validate_state_record(
        l0_state,
        "rollback_conv",
        L0_SLOT,
        L0_CONV_BYTES,
        "checkpoint_f32le_sha256",
        "L0 handoff receipt.l0_state_handoff.l0_post_state_commit",
    )?;
    validate_state_record(
        l0_state,
        "rollback_recurrent",
        L0_SLOT,
        L0_RECURRENT_BYTES,
        "checkpoint_f32le_sha256",
        "L0 handoff receipt.l0_state_handoff.l0_post_state_commit",
    )?;

    let layer1 = object_field(
        handoff,
        "layer1_input_binding",
        "L0 handoff receipt.l0_state_handoff",
    )?;
    if require_string(
        layer1,
        "session_id",
        "L0 handoff receipt.l0_state_handoff.layer1_input_binding",
    )? != session_id
        || require_u64(layer1, "layer", "L0 handoff receipt.layer1_input_binding")? != L1_LAYER
        || require_u64(
            layer1,
            "linear_state_slot",
            "L0 handoff receipt.l0_state_handoff.layer1_input_binding",
        )? != L1_SLOT
        || require_sha(
            layer1,
            "input_device_buffer_id",
            "L0 handoff receipt.l0_state_handoff.layer1_input_binding",
        )? != output_device_buffer_id
        || require_sha(
            layer1,
            "input_f32le_sha256",
            "L0 handoff receipt.l0_state_handoff.layer1_input_binding",
        )? != output_sha256
    {
        return Err(
            "L0 handoff receipt Layer 1 input does not retain same session/output identity".into(),
        );
    }
    require_bool(
        layer1,
        "same_command_graph_retained",
        true,
        "L0 handoff receipt.l0_state_handoff.layer1_input_binding",
    )?;
    require_bool(
        layer1,
        "l1_binding_executed",
        false,
        "L0 handoff receipt.l0_state_handoff.layer1_input_binding",
    )?;
    let claim = object_field(
        handoff,
        "claim_boundary",
        "L0 handoff receipt.l0_state_handoff",
    )?;
    for field in [
        "component_only",
        "layer1_not_encoded",
        "retention_binding_is_not_a_layer1_execution_claim",
        "may_not_satisfy_next_layer_execution_dependency",
    ] {
        require_bool(
            claim,
            field,
            true,
            "L0 handoff receipt.l0_state_handoff.claim_boundary",
        )?;
    }
    Ok(L0HandoffFacts {
        session_id,
        output_sha256,
        output_device_buffer_id,
        receipt_seal_sha256: receipt.document_seal_sha256.clone(),
    })
}

fn binding_summary(binding: Option<&BoundDocument>, valid: bool, blockers: &[String]) -> Value {
    json!({
        "present": binding.is_some(),
        "valid": valid,
        "document_sha256": binding.map(|value| value.document_sha256.clone()),
        "document_seal_sha256": binding.map(|value| value.document_seal_sha256.clone()),
        "blockers": blockers,
    })
}

fn l1_prefix_scope(facts: Option<&L0HandoffFacts>) -> Value {
    let dispatches = L1_PREFIX
        .iter()
        .enumerate()
        .map(|(ordinal, (stage, kernel))| {
            json!({
                "ordinal": ordinal + 1,
                "stage": stage,
                "kernel": kernel,
            })
        })
        .collect::<Vec<_>>();
    json!({
        "layer": L1_LAYER,
        "mixer": "delta_net",
        "linear_state_slot": L1_SLOT,
        "l0_baseline_evidence_only": true,
        "baseline_l0_handoff_receipt_seal_sha256": facts.map(|value| value.receipt_seal_sha256.clone()),
        "baseline_l0_session_id": facts.map(|value| value.session_id.clone()),
        "baseline_l0_output_f32le_sha256": facts.map(|value| value.output_sha256.clone()),
        "baseline_l0_output_device_buffer_id": facts.map(|value| value.output_device_buffer_id.clone()),
        "cross_process_or_prior_capture_pinned_buffer_reuse_authorized": false,
        "fresh_same_runtime_same_tcb_joint_l0_to_l1_capture_required": true,
        "fresh_joint_capture_same_session_required": true,
        "fresh_joint_capture_same_runtime_required": true,
        "fresh_joint_capture_same_tcb_required": true,
        "fresh_joint_l0_component_dispatch_count": 23,
        "fresh_joint_l1_slot1_prefix_dispatch_count": L1_PREFIX.len(),
        "fresh_joint_total_dispatch_count": 32,
        "fresh_joint_capture_dispatch_sequence": "L0_component(23)+L1_slot1_DeltaNet_prefix(9)",
        "fresh_joint_l0_output_identity_required": true,
        "fresh_joint_l0_output_baseline_parity_required": true,
        "future_l1_not_authorized_by_baseline_receipt_alone": true,
        "exact_prefix_dispatch_count": L1_PREFIX.len(),
        "exact_prefix_dispatches": dispatches,
        "no_l1_suffix_or_moe_dispatch_authorized": true,
        "required_post_l1_state_rollback_and_output_parity": {
            "layer": L1_LAYER,
            "linear_state_slot": L1_SLOT,
            "active_conv": {
                "slot": L1_SLOT,
                "offset_bytes": L1_CONV_OFFSET_BYTES,
                "capacity_bytes": L1_CONV_OFFSET_BYTES + L0_CONV_BYTES,
                "device_buffer_identity_required": true,
                "post_state_f32le_sha256_required": true,
            },
            "active_recurrent": {
                "slot": L1_SLOT,
                "offset_bytes": L1_RECURRENT_OFFSET_BYTES,
                "capacity_bytes": L1_RECURRENT_OFFSET_BYTES + L0_RECURRENT_BYTES,
                "device_buffer_identity_required": true,
                "post_state_f32le_sha256_required": true,
            },
            "rollback_conv": {
                "slot": L1_SLOT,
                "offset_bytes": L1_CONV_OFFSET_BYTES,
                "capacity_bytes": L1_CONV_OFFSET_BYTES + L0_CONV_BYTES,
                "device_buffer_identity_required": true,
                "checkpoint_f32le_sha256_required": true,
            },
            "rollback_recurrent": {
                "slot": L1_SLOT,
                "offset_bytes": L1_RECURRENT_OFFSET_BYTES,
                "capacity_bytes": L1_RECURRENT_OFFSET_BYTES + L0_RECURRENT_BYTES,
                "device_buffer_identity_required": true,
                "checkpoint_f32le_sha256_required": true,
            },
            "output": {
                "elements": HIDDEN_ELEMENTS,
                "bytes": HIDDEN_BYTES,
                "device_buffer_identity_required": true,
                "f32le_sha256_required": true,
                "strict_cpu_device_parity_required": true,
                "same_fresh_joint_capture_session_runtime_tcb_and_l0_input_identity_required": true,
                "prior_capture_pinned_buffer_reuse_authorized": false,
                "fresh_joint_l0_output_baseline_parity_required": true,
            },
        },
    })
}

fn parse_optional_binding(
    input: &Map<String, Value>,
    field: &str,
    expected_schema: &str,
    expected_status: &str,
) -> Result<Option<BoundDocument>, String> {
    match input.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(_) => parse_bound_document(input, field, expected_schema, expected_status).map(Some),
    }
}

fn build_report(input: &Value) -> Value {
    let mut blockers: Vec<String> = Vec::new();
    let mut schedule: Option<BoundDocument> = None;
    let mut preflight: Option<BoundDocument> = None;
    let mut l0_receipt: Option<BoundDocument> = None;
    let mut l0_facts: Option<L0HandoffFacts> = None;

    let input_object = match object(input, "input") {
        Ok(value) => value,
        Err(error) => {
            blockers.push(error);
            &Map::new()
        }
    };
    if input_object.get("schema").and_then(Value::as_str) != Some(INPUT_SCHEMA) {
        blockers.push(format!("input.schema must be {INPUT_SCHEMA}"));
    }
    if input_object
        .get("l1_execution_requested")
        .and_then(Value::as_bool)
        == Some(true)
    {
        blockers.push("this checklist refuses a requested Layer-1 execution".into());
    }
    if input_object
        .get("l1_execution_evidence")
        .is_some_and(|value| !value.is_null())
    {
        blockers.push("Layer-1 execution evidence is outside this readiness checklist".into());
    }
    if input_object
        .get("cross_process_or_prior_capture_pinned_buffer_reuse_requested")
        .and_then(Value::as_bool)
        == Some(true)
    {
        blockers.push(
            "a prior-capture PinnedBuffer cannot authorize a detached or cross-process Layer-1 execution"
                .into(),
        );
    }

    match parse_schedule_document(input_object) {
        Ok(value) => match validate_schedule(&value) {
            Ok(()) => schedule = Some(value),
            Err(error) => blockers.push(format!("schedule authority: {error}")),
        },
        Err(error) => blockers.push(format!("schedule authority: {error}")),
    }
    match parse_bound_document(
        input_object,
        "l0_to_l1_preflight",
        L0_TO_L1_PREFLIGHT_SCHEMA,
        L0_TO_L1_PREFLIGHT_STATUS,
    ) {
        Ok(value) => match validate_preflight(&value) {
            Ok(()) => preflight = Some(value),
            Err(error) => blockers.push(format!("L0-to-L1 preflight: {error}")),
        },
        Err(error) => blockers.push(format!("L0-to-L1 preflight: {error}")),
    }
    match parse_optional_binding(
        input_object,
        "l0_state_handoff_receipt",
        L0_HANDOFF_RECEIPT_SCHEMA,
        L0_HANDOFF_RECEIPT_STATUS,
    ) {
        Ok(Some(value)) => match validate_l0_handoff(&value) {
            Ok(facts) => {
                l0_facts = Some(facts);
                l0_receipt = Some(value);
            }
            Err(error) => blockers.push(format!("L0 state-handoff receipt: {error}")),
        },
        Ok(None) => blockers.push("future sealed L0 state-handoff receipt is not present".into()),
        Err(error) => blockers.push(format!("L0 state-handoff receipt: {error}")),
    }

    blockers.sort();
    blockers.dedup();
    let prepared = blockers.is_empty();
    let mut output = json!({
        "schema": RESULT_SCHEMA,
        "status": if prepared { PREPARED_STATUS } else { INCOMPLETE_STATUS },
        "prepared": prepared,
        "l1_execution_performed_by_this_contract": false,
        "l1_prefix_dispatches_executed_by_this_contract": 0,
        "schedule_authority": binding_summary(schedule.as_ref(), schedule.is_some(), &blockers),
        "l0_to_l1_preflight": binding_summary(preflight.as_ref(), preflight.is_some(), &blockers),
        "l0_state_handoff_receipt": binding_summary(l0_receipt.as_ref(), l0_facts.is_some(), &blockers),
        "future_l1_slot1_deltanet_prefix_scope": l1_prefix_scope(l0_facts.as_ref()),
        "blockers": blockers,
        "authority_boundary": {
            "new_physical_model_processes_authorized": 0,
            "server_starts_authorized": 0,
            "port_binds_authorized": 0,
            "gpu_leases_authorized": 0,
            "watcher_changes_authorized": 0,
            "tournament_state_mutations_authorized": 0,
        },
        "claim_boundary": {
            "cpu_only_checklist": true,
            "l0_handoff_receipt_must_state_l1_binding_not_executed": true,
            "l0_handoff_is_evidence_baseline_only": true,
            "cross_process_or_prior_capture_pinned_buffer_reuse_authorized": false,
            "future_l1_requires_fresh_same_runtime_same_tcb_joint_l0_to_l1_capture": true,
            "does_not_execute_l1": true,
            "does_not_open_or_scan_artifacts": true,
            "does_not_construct_metal_or_dispatch": true,
            "does_not_start_runtime_server_or_watcher": true,
            "does_not_acquire_lease": true,
            "does_not_measure_tps_or_tg": true,
            "does_not_claim_complete_layer_token_decoder_or_tournament": true,
        },
    });
    seal(&mut output).expect("contract output must be sealable");
    output
}

fn canonical_regular(path: &Path, label: &str) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err(format!("{label} must be absolute"));
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    fs::canonicalize(path).map_err(|error| format!("cannot canonicalize {label}: {error}"))
}

fn read_json(path: &Path, label: &str) -> Result<Value, String> {
    let path = canonical_regular(path, label)?;
    let raw = fs::read(path).map_err(|error| format!("cannot read {label}: {error}"))?;
    let value: Value =
        serde_json::from_slice(&raw).map_err(|error| format!("cannot parse {label}: {error}"))?;
    object(&value, label)?;
    Ok(value)
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if !path.is_absolute() {
        return Err("--out must be absolute".into());
    }
    let parent = path.parent().ok_or("--out has no parent")?;
    let metadata = fs::symlink_metadata(parent)
        .map_err(|error| format!("cannot stat --out parent {}: {error}", parent.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err("--out parent must be an existing non-symlink directory".into());
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("refusing to overwrite --out {}: {error}", path.display()))?;
    file.write_all(bytes)
        .map_err(|error| format!("cannot write --out: {error}"))
}

fn usage() -> &'static str {
    "usage: ascension_qwen80_l1_source_token_continuation_readiness_contract \\\n+--input ABSOLUTE_PATH --out ABSOLUTE_NEW_PATH"
}

fn parse_args<I>(arguments: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut values = BTreeMap::new();
    let mut iterator = arguments.into_iter();
    while let Some(flag) = iterator.next() {
        let value = iterator
            .next()
            .ok_or_else(|| format!("missing value after {flag}; {}", usage()))?;
        if !matches!(
            flag.as_str(),
            "--input"
                | "--out"
                | "--schedule-authority"
                | "--l0-to-l1-preflight"
                | "--l0-state-handoff-receipt"
        ) {
            return Err(format!("unsupported option {flag:?}; {}", usage()));
        }
        if values.insert(flag.clone(), value).is_some() {
            return Err(format!("duplicate {flag}; {}", usage()));
        }
    }
    let required = |flag: &str| {
        values
            .get(flag)
            .map(PathBuf::from)
            .ok_or_else(|| format!("missing {flag}; {}", usage()))
    };
    let optional = |flag: &str| values.get(flag).map(PathBuf::from);
    let input = optional("--input");
    let schedule_authority = optional("--schedule-authority");
    let l0_to_l1_preflight = optional("--l0-to-l1-preflight");
    let l0_state_handoff_receipt = optional("--l0-state-handoff-receipt");
    if input.is_some()
        && (schedule_authority.is_some()
            || l0_to_l1_preflight.is_some()
            || l0_state_handoff_receipt.is_some())
    {
        return Err(format!(
            "--input cannot be combined with explicit evidence paths; {}",
            usage()
        ));
    }
    if input.is_none() && (schedule_authority.is_none() || l0_to_l1_preflight.is_none()) {
        return Err(format!(
            "explicit evidence mode requires --schedule-authority and --l0-to-l1-preflight; {}",
            usage()
        ));
    }
    let args = Args {
        input,
        schedule_authority,
        l0_to_l1_preflight,
        l0_state_handoff_receipt,
        out: required("--out")?,
    };
    if !args.out.is_absolute()
        || args.input.as_ref().is_some_and(|path| !path.is_absolute())
        || args
            .schedule_authority
            .as_ref()
            .is_some_and(|path| !path.is_absolute())
        || args
            .l0_to_l1_preflight
            .as_ref()
            .is_some_and(|path| !path.is_absolute())
        || args
            .l0_state_handoff_receipt
            .as_ref()
            .is_some_and(|path| !path.is_absolute())
    {
        return Err(format!("all supplied paths must be absolute; {}", usage()));
    }
    Ok(args)
}

fn live_binding(document: Value) -> Result<Value, String> {
    let document_sha256 = sha256_json(&document)?;
    let document_seal_sha256 = document.get("seal_sha256").cloned().unwrap_or(Value::Null);
    Ok(json!({
        "document": document,
        "document_sha256": document_sha256,
        "document_seal_sha256": document_seal_sha256,
    }))
}

fn build_explicit_evidence_input(args: &Args) -> Result<Value, String> {
    let schedule_path = args
        .schedule_authority
        .as_ref()
        .ok_or("missing --schedule-authority")?;
    let preflight_path = args
        .l0_to_l1_preflight
        .as_ref()
        .ok_or("missing --l0-to-l1-preflight")?;
    let mut input = json!({
        "schema": INPUT_SCHEMA,
        "schedule_authority": live_binding(read_json(schedule_path, "--schedule-authority")?)?,
        "l0_to_l1_preflight": live_binding(read_json(preflight_path, "--l0-to-l1-preflight")?)?,
        "l0_state_handoff_receipt": Value::Null,
        "l1_execution_requested": false,
        "l1_execution_evidence": Value::Null,
        "cross_process_or_prior_capture_pinned_buffer_reuse_requested": false,
    });
    if let Some(path) = args.l0_state_handoff_receipt.as_ref() {
        input["l0_state_handoff_receipt"] =
            live_binding(read_json(path, "--l0-state-handoff-receipt")?)?;
    }
    Ok(input)
}

fn run(args: Args) -> Result<(PathBuf, String, String), String> {
    let input = match args.input.as_ref() {
        Some(path) => read_json(path, "--input")?,
        None => build_explicit_evidence_input(&args)?,
    };
    let output = build_report(&input);
    let seal = verify_seal(&output, "continuation readiness output")?;
    let status = object(&output, "continuation readiness output")
        .and_then(|root| require_string(root, "status", "continuation readiness output"))?
        .to_owned();
    let bytes = serde_json::to_vec_pretty(&output)
        .map_err(|error| format!("cannot encode output: {error}"))?;
    write_new(&args.out, &bytes)?;
    Ok((args.out, seal, status))
}

fn main() {
    match parse_args(env::args().skip(1)).and_then(run) {
        Ok((path, seal, status)) => println!(
            "{{\"status\":\"{}\",\"out\":\"{}\",\"seal_sha256\":\"{}\"}}",
            status,
            path.display(),
            seal
        ),
        Err(error) => {
            eprintln!("{error}");
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

    fn seal_document(mut value: Value) -> Value {
        seal(&mut value).unwrap();
        value
    }

    fn bind(document: Value) -> Value {
        json!({
            "document_sha256": sha256_json(&document).unwrap(),
            "document_seal_sha256": document["seal_sha256"].clone(),
            "document": document,
        })
    }

    fn schedule_document() -> Value {
        let layers = (0_u64..48)
            .map(|layer| {
                json!({
                    "layer": layer,
                    "mixer": if layer % 4 == 3 { "gqa" } else { "delta_net" },
                    "state_slot": {
                        "slot": layer,
                        "domain": if layer % 4 == 3 { "gqa_kv" } else { "delta_net_conv_and_recurrent" },
                    },
                })
            })
            .collect::<Vec<_>>();
        seal_document(json!({
            "schema": SCHEDULE_SCHEMA,
            "status": SCHEDULE_STATUS,
            "all_48_layers_scheduled": true,
            "layers": layers,
            "claim_boundary": {
                "artifact_payload_open_or_scan_performed": false,
                "metal_device_or_dispatch_performed": false,
                "runtime_watcher_registry_server_or_hcli_changed": false,
                "model_execution_performed": false,
                "token_generation_or_feedback_performed": false,
                "tps_or_tg_measured": false,
            },
        }))
    }

    fn sealed_schedule_wrapper_document() -> Value {
        seal_document(json!({
            "schema": SCHEDULE_WRAPPER_SCHEMA,
            "status": SCHEDULE_WRAPPER_STATUS,
            "raw_schedule_authority": {
                "path": "/sealed/static/QWEN80_48_LAYER_PAYLOAD_SCHEDULE_AUTHORITY_20260809T033554Z.json",
                "present": true,
                "bytes": RAW_SCHEDULE_BYTES,
                "sha256": RAW_SCHEDULE_SHA256,
                "schema": SCHEDULE_SCHEMA,
                "status": SCHEDULE_STATUS,
                "raw_schedule_seal_sha256": null,
                "raw_schedule_is_static_and_unmodified": true,
            },
            "source_authority": {
                "model_key": MODEL_KEY,
                "model_id": MODEL_ID,
                "source_repository": SOURCE_REPOSITORY,
                "source_revision": SOURCE_REVISION,
                "source_config_sha256": SOURCE_CONFIG_SHA256,
                "descriptor_inventory_document_sha256": MANIFEST_DOCUMENT_SHA256,
                "descriptor_inventory_seal_sha256": MANIFEST_SEAL_SHA256,
                "source_config_authority_document_sha256": SOURCE_CONFIG_AUTHORITY_DOCUMENT_SHA256,
                "source_config_authority_seal_sha256": SOURCE_CONFIG_AUTHORITY_SEAL_SHA256,
            },
            "schedule_facts": {
                "all_48_layers_scheduled": true,
                "layer_count": 48,
                "delta_net_layer_count": 36,
                "gqa_layer_count": 12,
                "delta_net_state_slot_count": 36,
                "gqa_state_slot_count": 12,
                "full_command_graph_item_count": 54,
                "layer_1": {
                    "layer": L1_LAYER,
                    "mixer": "delta_net",
                    "state_slot": L1_SLOT,
                    "state_domain": "delta_net_conv_and_recurrent",
                },
            },
            "claim_boundary": {
                "wrapper_is_read_only": true,
                "raw_schedule_rewritten_or_resealed": false,
                "artifact_payload_open_or_scan_performed": false,
                "metal_device_or_dispatch_performed": false,
                "runtime_server_watcher_or_hcli_changed": false,
                "lease_issued_or_released": false,
                "token_generation_or_feedback_performed": false,
                "tps_or_tg_measured": false,
                "future_joint_l0_to_l1_capture_authorized": false,
            },
        }))
    }

    fn preflight_document() -> Value {
        seal_document(json!({
            "schema": L0_TO_L1_PREFLIGHT_SCHEMA,
            "status": L0_TO_L1_PREFLIGHT_STATUS,
            "mode": "cpu_only_preflight",
            "source_binding": {"model_key": MODEL_KEY, "source_token_id": SOURCE_TOKEN_ID},
            "planned_pre_l1_handoff_capture": {
                "schema": L0_HANDOFF_RECEIPT_SCHEMA,
                "status": L0_HANDOFF_RECEIPT_STATUS,
                "l1_binding_not_executed": true,
                "l1_prefix_dispatches": 0,
            },
            "required_next_layer_handoff_witness": {
                "retained_l0_second_residual": {"elements": HIDDEN_ELEMENTS, "bytes": HIDDEN_BYTES},
                "layer1_input_binding": {"layer": L1_LAYER, "linear_state_slot": L1_SLOT},
            },
        }))
    }

    fn state_record(
        allocation: &str,
        slot: u64,
        capacity: u64,
        hash_field: &str,
        character: char,
    ) -> Value {
        json!({
            "allocation_id": allocation,
            "slot": slot,
            "capacity_bytes": capacity,
            "device_buffer_id": sha(character),
            hash_field: sha(if character == 'f' { 'e' } else { 'f' }),
        })
    }

    fn l0_receipt_document() -> Value {
        let output_buffer = sha('a');
        let output_hash = sha('b');
        seal_document(json!({
            "schema": L0_HANDOFF_RECEIPT_SCHEMA,
            "status": L0_HANDOFF_RECEIPT_STATUS,
            "mode": "metal",
            "metal_device_or_dispatch_performed": true,
            "component_only": true,
            "l1_binding_not_executed": true,
            "l1_prefix_dispatches": 0,
            "complete_layer_or_token_performed": false,
            "l0_state_handoff": {
                "session_id": "qwen80-source-token-session-01",
                "source_token_id": SOURCE_TOKEN_ID,
                "same_command_graph_retained": true,
                "l1_binding_not_executed": true,
                "l1_prefix_dispatches": 0,
                "retained_l0_second_residual": {
                    "elements": HIDDEN_ELEMENTS,
                    "bytes": HIDDEN_BYTES,
                    "f32le_sha256": output_hash,
                    "device_buffer_id": output_buffer,
                    "retained_for_future_layer1_encode": true,
                },
                "l0_post_state_commit": {
                    "layer": 0,
                    "linear_state_slot": L0_SLOT,
                    "checkpoint_before_mutation": true,
                    "active_conv": state_record("l0-active-conv", L0_SLOT, L0_CONV_BYTES, "post_state_f32le_sha256", '1'),
                    "active_recurrent": state_record("l0-active-recurrent", L0_SLOT, L0_RECURRENT_BYTES, "post_state_f32le_sha256", '2'),
                    "rollback_conv": state_record("l0-rollback-conv", L0_SLOT, L0_CONV_BYTES, "checkpoint_f32le_sha256", '3'),
                    "rollback_recurrent": state_record("l0-rollback-recurrent", L0_SLOT, L0_RECURRENT_BYTES, "checkpoint_f32le_sha256", '4'),
                },
                "layer1_input_binding": {
                    "session_id": "qwen80-source-token-session-01",
                    "layer": L1_LAYER,
                    "linear_state_slot": L1_SLOT,
                    "input_device_buffer_id": output_buffer,
                    "input_f32le_sha256": output_hash,
                    "same_command_graph_retained": true,
                    "l1_binding_executed": false,
                },
                "claim_boundary": {
                    "component_only": true,
                    "layer1_not_encoded": true,
                    "retention_binding_is_not_a_layer1_execution_claim": true,
                    "may_not_satisfy_next_layer_execution_dependency": true,
                },
            },
            "claim_boundary": {
                "l0_post_state_rollback_retained_output_component_only": true,
                "l1_binding_not_executed": true,
                "may_not_satisfy_next_layer_execution_dependency": true,
                "no_complete_layer_token_decoder_generation_server_hcli_tps_tg_or_tournament_claim": true,
            },
        }))
    }

    fn input_with_schedule(schedule: Value, l0_receipt: Option<Value>) -> Value {
        json!({
            "schema": INPUT_SCHEMA,
            "schedule_authority": bind(schedule),
            "l0_to_l1_preflight": bind(preflight_document()),
            "l0_state_handoff_receipt": l0_receipt.map(bind),
            "l1_execution_requested": false,
            "l1_execution_evidence": null,
        })
    }

    fn input(l0_receipt: Option<Value>) -> Value {
        input_with_schedule(schedule_document(), l0_receipt)
    }

    #[test]
    fn absent_future_l0_receipt_stays_incomplete_and_never_encodes_l1() {
        let report = build_report(&input(None));
        assert_eq!(
            verify_seal(&report, "report").unwrap(),
            report["seal_sha256"]
        );
        assert_eq!(report["status"], INCOMPLETE_STATUS);
        assert_eq!(report["prepared"], false);
        assert_eq!(report["l1_execution_performed_by_this_contract"], false);
        assert_eq!(report["l1_prefix_dispatches_executed_by_this_contract"], 0);
        assert_eq!(
            report["future_l1_slot1_deltanet_prefix_scope"]["exact_prefix_dispatch_count"],
            L1_PREFIX.len()
        );
        assert_eq!(
            report["future_l1_slot1_deltanet_prefix_scope"]["layer"],
            L1_LAYER
        );
        assert!(report["blockers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value.as_str().unwrap().contains("future sealed L0")));
    }

    #[test]
    fn sealed_l0_handoff_prepares_only_the_future_slot1_prefix_checklist() {
        let report = build_report(&input(Some(l0_receipt_document())));
        assert_eq!(report["status"], PREPARED_STATUS);
        assert_eq!(report["prepared"], true);
        assert_eq!(report["l1_execution_performed_by_this_contract"], false);
        let scope = &report["future_l1_slot1_deltanet_prefix_scope"];
        assert_eq!(scope["mixer"], "delta_net");
        assert_eq!(scope["linear_state_slot"], L1_SLOT);
        assert_eq!(scope["baseline_l0_output_f32le_sha256"], sha('b'));
        assert_eq!(scope["baseline_l0_output_device_buffer_id"], sha('a'));
        assert_eq!(scope["l0_baseline_evidence_only"], true);
        assert_eq!(
            scope["fresh_same_runtime_same_tcb_joint_l0_to_l1_capture_required"],
            true
        );
        assert_eq!(scope["fresh_joint_l0_component_dispatch_count"], 23);
        assert_eq!(scope["fresh_joint_l1_slot1_prefix_dispatch_count"], 9);
        assert_eq!(scope["fresh_joint_total_dispatch_count"], 32);
        assert_eq!(
            scope["cross_process_or_prior_capture_pinned_buffer_reuse_authorized"],
            false
        );
        assert_eq!(
            scope["exact_prefix_dispatches"].as_array().unwrap().len(),
            9
        );
        assert_eq!(scope["no_l1_suffix_or_moe_dispatch_authorized"], true);
        assert_eq!(
            scope["required_post_l1_state_rollback_and_output_parity"]["output"]["bytes"],
            HIDDEN_BYTES
        );
    }

    #[test]
    fn canonical_sealed_schedule_wrapper_prepares_the_same_future_23_plus_9_scope() {
        let report = build_report(&input_with_schedule(
            sealed_schedule_wrapper_document(),
            Some(l0_receipt_document()),
        ));
        assert_eq!(report["status"], PREPARED_STATUS);
        assert_eq!(report["prepared"], true);
        let scope = &report["future_l1_slot1_deltanet_prefix_scope"];
        assert_eq!(
            scope["fresh_joint_capture_dispatch_sequence"],
            "L0_component(23)+L1_slot1_DeltaNet_prefix(9)"
        );
        assert_eq!(scope["fresh_joint_total_dispatch_count"], 32);
        assert_eq!(
            scope["cross_process_or_prior_capture_pinned_buffer_reuse_authorized"],
            false
        );
    }

    #[test]
    fn sealed_schedule_wrapper_refuses_a_rebound_noncanonical_raw_sha() {
        let mut wrapper = sealed_schedule_wrapper_document();
        wrapper["raw_schedule_authority"]["sha256"] = json!(sha('e'));
        wrapper.as_object_mut().unwrap().remove("seal_sha256");
        seal(&mut wrapper).unwrap();
        let report = build_report(&input_with_schedule(wrapper, Some(l0_receipt_document())));
        assert_eq!(report["status"], INCOMPLETE_STATUS);
        assert!(report["blockers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value
                .as_str()
                .unwrap()
                .contains("canonical raw static plan")));
    }

    #[test]
    fn l0_receipt_must_truthfully_exclude_l1_prefix_execution() {
        let mut receipt = l0_receipt_document();
        receipt["l1_binding_not_executed"] = json!(false);
        receipt["l1_prefix_dispatches"] = json!(1);
        receipt.as_object_mut().unwrap().remove("seal_sha256");
        seal(&mut receipt).unwrap();
        let report = build_report(&input(Some(receipt)));
        assert_eq!(report["status"], INCOMPLETE_STATUS);
        assert!(report["blockers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value.as_str().unwrap().contains("l1_binding_not_executed")));
    }

    #[test]
    fn l0_output_identity_must_equal_layer1_input_identity_in_the_captured_session() {
        let mut receipt = l0_receipt_document();
        receipt["l0_state_handoff"]["layer1_input_binding"]["input_device_buffer_id"] =
            json!(sha('d'));
        receipt.as_object_mut().unwrap().remove("seal_sha256");
        seal(&mut receipt).unwrap();
        let report = build_report(&input(Some(receipt)));
        assert_eq!(report["status"], INCOMPLETE_STATUS);
        assert!(report["blockers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value
                .as_str()
                .unwrap()
                .contains("same session/output identity")));
    }

    #[test]
    fn detached_or_cross_process_pinned_buffer_reuse_is_refused() {
        let mut request = input(Some(l0_receipt_document()));
        request["cross_process_or_prior_capture_pinned_buffer_reuse_requested"] = json!(true);
        let report = build_report(&request);
        assert_eq!(report["status"], INCOMPLETE_STATUS);
        assert_eq!(
            report["claim_boundary"]
                ["cross_process_or_prior_capture_pinned_buffer_reuse_authorized"],
            false
        );
        assert!(report["blockers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value
                .as_str()
                .unwrap()
                .contains("cannot authorize a detached or cross-process")));
    }

    #[test]
    fn schedule_and_current_preflight_are_non_substitutable_sealed_inputs() {
        let mut invalid = input(Some(l0_receipt_document()));
        invalid["schedule_authority"]["document"]["layers"][1]["mixer"] = json!("gqa");
        let document = invalid["schedule_authority"]["document"].clone();
        let mut document = document;
        document.as_object_mut().unwrap().remove("seal_sha256");
        seal(&mut document).unwrap();
        invalid["schedule_authority"]["document_sha256"] = json!(sha256_json(&document).unwrap());
        invalid["schedule_authority"]["document_seal_sha256"] = document["seal_sha256"].clone();
        invalid["schedule_authority"]["document"] = document;
        let report = build_report(&invalid);
        assert_eq!(report["status"], INCOMPLETE_STATUS);
        assert!(report["blockers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value
                .as_str()
                .unwrap()
                .contains("does not reserve Layer 1 DeltaNet")));
    }

    #[test]
    fn requested_or_supplied_l1_execution_is_refused_by_the_readiness_contract() {
        let mut request = input(Some(l0_receipt_document()));
        request["l1_execution_requested"] = json!(true);
        request["l1_execution_evidence"] = json!({"claim": "not accepted"});
        let report = build_report(&request);
        assert_eq!(report["status"], INCOMPLETE_STATUS);
        assert_eq!(report["l1_execution_performed_by_this_contract"], false);
        assert!(report["blockers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value
                .as_str()
                .unwrap()
                .contains("refuses a requested Layer-1 execution")));
        assert!(report["blockers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value
                .as_str()
                .unwrap()
                .contains("outside this readiness checklist")));
    }
}
