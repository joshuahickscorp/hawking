//! Read-only post-capture assessor for the Qwen80 source-token L0 handoff.
//!
//! This program deliberately has no model, Metal, lease-issuance, server,
//! watcher, benchmark, or tournament path.  It consumes only sealed evidence
//! supplied in one input document and returns a sealed assessment.  A positive
//! result proves exactly one completed L0 9+14 component handoff and still
//! states that Layer 1 was not executed.  The retained device identities are
//! historical evidence only: a PinnedBuffer cannot cross a process boundary,
//! so a future Layer-1 capture must be a fresh joint L0-to-L1 capture in one
//! runtime and TCB, never a standalone Layer-1 process fed by this receipt.
//!
//! In particular, an output hash alone is not enough: the assessor requires a
//! retained 8,192-byte device output, committed active and rollback state
//! witnesses, a same-session distinct Layer-1 slot-1 input binding, a reaped
//! receipt-last outer terminal, and a distinct lease-release receipt.  It
//! rejects fixtures, partial evidence, self-assertions, and any nonzero L1
//! dispatch claim.

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const INPUT_SCHEMA: &str =
    "hawking.ascension.qwen80_l0_state_handoff_post_capture_assessor_input.v1";
const INPUT_STATUS: &str = "SUBMITTED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_POST_CAPTURE_ASSESSMENT";
const RESULT_SCHEMA: &str = "hawking.ascension.qwen80_l0_state_handoff_post_capture_assessment.v1";
const EARNED_STATUS: &str =
    "EARNED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_COMPONENT_L1_BINDING_NOT_EXECUTED";
const REFUSED_STATUS: &str =
    "REFUSED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_POST_CAPTURE_EVIDENCE_INCOMPLETE_OR_UNTRUSTED";

const HANDOFF_AUTHORITY_SCHEMA: &str = "hawking.ascension.qwen80_l0_to_layer1_handoff_authority.v1";
const HANDOFF_AUTHORITY_STATUS: &str =
    "ASSESSED_QWEN80_SOURCE_TOKEN_L0_TO_LAYER1_HANDOFF_INCOMPLETE_MISSING_RETAINED_DEVICE_OUTPUT_AND_POST_STATE_WITNESSES";
const OUTER_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_state_handoff_outer_capture.v1";
const OUTER_STATUS: &str =
    "CAPTURED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_OUTER_TERMINAL_PRE_L1_COMPONENT_ONLY";
const INNER_SCHEMA: &str = "hawking.ascension.qwen80_source_token_l0_state_handoff_capture.v1";
const INNER_STATUS: &str =
    "CAPTURED_QWEN80_SOURCE_TOKEN_L0_POST_STATE_ROLLBACK_RETAINED_OUTPUT_L1_BINDING_NOT_EXECUTED_COMPONENT_ONLY";
const LEASE_RELEASE_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_state_handoff_quiet_metal_lease_release.v1";
const LEASE_RELEASE_STATUS: &str =
    "RELEASED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_COMPONENT_QUIET_METAL_LEASE_AFTER_TERMINAL_CAPTURE";
const RELEASE_RECOMMENDATION_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_state_handoff_recommended_lease_release_contract.v1";
const RELEASE_RECOMMENDATION_STATUS: &str =
    "RECOMMENDED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_LEASE_RELEASE_AFTER_OUTER_TERMINAL";
const CONTINUATION_SCHEMA: &str =
    "hawking.ascension.qwen80_l1_source_token_continuation_readiness_contract.v1";
const CONTINUATION_PREPARED_STATUS: &str =
    "PREPARED_QWEN80_SOURCE_TOKEN_L1_SLOT1_DELTANET_PREFIX_CAPTURE_RESERVED_NOT_EXECUTED";
const CONTINUATION_INCOMPLETE_STATUS: &str =
    "INCOMPLETE_QWEN80_SOURCE_TOKEN_L1_CONTINUATION_MISSING_TRUSTED_L0_HANDOFF_OR_AUTHORITY";

const MODEL_KEY: &str = "qwen80";
const SOURCE_REVISION: &str = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb";
const MANIFEST_DOCUMENT_SHA: &str =
    "a0fcac0401a7962402bb8cb87d5055c83667b39575f9e0f4c7470d080758aa10";
const MANIFEST_SEAL: &str = "14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b";
const ADMISSION_RECEIPT_SEAL: &str =
    "939b41322363da3db774a2530b207bf380ed641d23cae671fc6438c0eecbf628";
const SOURCE_TOKEN_ID: u64 = 1;
const L0_LAYER: u64 = 0;
const L1_LAYER: u64 = 1;
const L0_SLOT: u64 = 0;
const L1_SLOT: u64 = 1;
const HIDDEN_ELEMENTS: u64 = 2_048;
const HIDDEN_BYTES: u64 = HIDDEN_ELEMENTS * 4;
const PREFIX_DISPATCHES: u64 = 9;
const SUFFIX_DISPATCHES: u64 = 14;
const TOTAL_DISPATCHES: u64 = PREFIX_DISPATCHES + SUFFIX_DISPATCHES;
const L0_CONV_BYTES: u64 = 98_304;
const L0_RECURRENT_BYTES: u64 = 2_097_152;
const L1_CONV_OFFSET_BYTES: u64 = L0_CONV_BYTES;
const L1_RECURRENT_OFFSET_BYTES: u64 = L0_RECURRENT_BYTES;
const L1_CONV_CAPACITY_BYTES: u64 = L1_CONV_OFFSET_BYTES + L0_CONV_BYTES;
const L1_RECURRENT_CAPACITY_BYTES: u64 = L1_RECURRENT_OFFSET_BYTES + L0_RECURRENT_BYTES;

const L0_KERNELS: [&str; TOTAL_DISPATCHES as usize] = [
    "qwen_next_direct_packed_input_rmsnorm",
    "qwen_binary_sign_scale_matvec",
    "qwen_binary_sign_scale_matvec",
    "qwen_next_qkvz_rearrange_conv_l2",
    "qwen_next_ba_to_decay_beta",
    "qwen_next_gated_delta_decode_single",
    "qwen_next_deltanet_gated_rmsnorm",
    "qwen_binary_sign_scale_matvec",
    "qwen_next_add_residual",
    "qwen80_postnorm_router_top10_rmsnorm",
    "qwen80_postnorm_router_top10_matvec",
    "qwen80_postnorm_router_top10_select",
    "qwen80_all_ten_routed_wave_route_guard",
    "qwen80_all_ten_routed_wave_gate_up",
    "qwen80_all_ten_routed_wave_swiglu",
    "qwen80_all_ten_routed_wave_down_weighted",
    "qwen80_shared_expert_wave_gate_up",
    "qwen80_shared_expert_wave_swiglu",
    "qwen80_shared_expert_wave_down",
    "qwen80_shared_expert_wave_scalar_gate",
    "qwen80_shared_expert_wave_apply_sigmoid_gate",
    "qwen80_moe_wave_aggregate_second_residual_route_sum",
    "qwen80_moe_wave_aggregate_second_residual_add_shared_residual",
];

#[derive(Clone, Debug)]
struct BoundDocument {
    document: Value,
    document_sha256: String,
    document_seal_sha256: String,
}

#[derive(Clone, Debug)]
struct AuthorityFacts {
    second_residual_sha256: String,
}

#[derive(Clone, Debug)]
struct InnerFacts {
    session_id: String,
    output_sha256: String,
    output_device_buffer_id: String,
    outer_launch_authority_seal: String,
    lease_seal: String,
    lease_id: String,
}

#[derive(Clone, Debug)]
struct OuterFacts {
    lease_id: String,
    lease_seal: String,
    recorded_at: String,
}

#[derive(Clone, Debug)]
struct Args {
    input: Option<PathBuf>,
    handoff_authority: Option<PathBuf>,
    l0_outer_terminal: Option<PathBuf>,
    l0_inner_receipt: Option<PathBuf>,
    lease_release_receipt: Option<PathBuf>,
    lease_release_recommendation_contract: Option<PathBuf>,
    l1_continuation_contract: Option<PathBuf>,
    out: PathBuf,
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
/// The live capture receipts use Python's sorted, compact JSON sealing form.
/// Rust's Ryu spelling differs for scientific exponents, so assessment must
/// use this lexical form to verify authentic receipts, not just fixtures.
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
        return Err("output already has a seal".into());
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
    let recorded = sha_field(object, "seal_sha256", label)?;
    let mut unsigned = object.clone();
    unsigned.remove("seal_sha256");
    let expected = sha256_json(&Value::Object(unsigned))?;
    if recorded != expected {
        return Err(format!("{label} seal mismatch"));
    }
    Ok(recorded)
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

fn string_field<'a>(
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

fn sha_field(object: &Map<String, Value>, field: &str, label: &str) -> Result<String, String> {
    let value = string_field(object, field, label)?;
    if !is_lower_sha256(value) {
        return Err(format!("{label}.{field} must be a lowercase SHA-256"));
    }
    Ok(value.into())
}

fn u64_field(object: &Map<String, Value>, field: &str, label: &str) -> Result<u64, String> {
    object
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{label}.{field} must be an unsigned integer"))
}

fn nonzero_u64_field(object: &Map<String, Value>, field: &str, label: &str) -> Result<u64, String> {
    let value = u64_field(object, field, label)?;
    if value == 0 {
        return Err(format!("{label}.{field} must be nonzero"));
    }
    Ok(value)
}

fn utc_timestamp_field(
    object: &Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<String, String> {
    let value = string_field(object, field, label)?;
    let bytes = value.as_bytes();
    let required_digits = [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18];
    if bytes.len() < 20
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
        || *bytes.last().expect("length checked") != b'Z'
        || required_digits
            .iter()
            .any(|index| !bytes[*index].is_ascii_digit())
    {
        return Err(format!("{label}.{field} must be a UTC ISO-8601 timestamp"));
    }
    let fractional = &bytes[19..bytes.len() - 1];
    if !fractional.is_empty()
        && (fractional[0] != b'.'
            || fractional.len() == 1
            || fractional[1..].iter().any(|byte| !byte.is_ascii_digit()))
    {
        return Err(format!("{label}.{field} must be a UTC ISO-8601 timestamp"));
    }
    Ok(value.into())
}

fn bool_field(
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

fn exact_string(
    object: &Map<String, Value>,
    field: &str,
    expected: &str,
    label: &str,
) -> Result<(), String> {
    let value = string_field(object, field, label)?;
    if value != expected {
        return Err(format!("{label}.{field} drifted"));
    }
    Ok(())
}

fn reject_fixture_or_self_assertion(root: &Map<String, Value>, label: &str) -> Result<(), String> {
    fn inspect(value: &Value, path: &str) -> Option<String> {
        match value {
            Value::Array(values) => values
                .iter()
                .enumerate()
                .find_map(|(index, value)| inspect(value, &format!("{path}[{index}]"))),
            Value::Object(values) => {
                for (field, value) in values {
                    let field_path = format!("{path}.{field}");
                    if matches!(
                        field.as_str(),
                        "fixture"
                            | "test_fixture"
                            | "synthetic"
                            | "simulated"
                            | "partial"
                            | "self_asserted"
                            | "generated_by_assessor"
                    ) && value.as_bool() == Some(true)
                    {
                        return Some(format!(
                            "{field_path} is forbidden for an actual post-capture receipt"
                        ));
                    }
                    if matches!(
                        field.as_str(),
                        "evidence_kind" | "provenance" | "issuer_role"
                    ) {
                        if let Some(text) = value.as_str() {
                            let normalized = text.to_ascii_lowercase();
                            if normalized.contains("fixture")
                                || normalized.contains("self_assert")
                                || normalized.contains("synthetic")
                            {
                                return Some(format!("{field_path} declares non-actual evidence"));
                            }
                        }
                    }
                    if let Some(error) = inspect(value, &field_path) {
                        return Some(error);
                    }
                }
                None
            }
            _ => None,
        }
    }

    inspect(&Value::Object(root.clone()), label).map_or(Ok(()), Err)
}

fn parse_bound_document(
    input: &Map<String, Value>,
    field: &str,
    expected_schema: &str,
    expected_status: &str,
) -> Result<BoundDocument, String> {
    parse_bound_document_with_statuses(input, field, expected_schema, &[expected_status])
}

fn parse_bound_document_with_statuses(
    input: &Map<String, Value>,
    field: &str,
    expected_schema: &str,
    expected_statuses: &[&str],
) -> Result<BoundDocument, String> {
    let binding = object_field(input, field, "input")?;
    let document = binding
        .get("document")
        .cloned()
        .ok_or_else(|| format!("input.{field}.document is required"))?;
    let root = object(&document, &format!("input.{field}.document"))?;
    let claimed_document_sha = sha_field(binding, "document_sha256", &format!("input.{field}"))?;
    let claimed_seal = sha_field(binding, "document_seal_sha256", &format!("input.{field}"))?;
    if sha256_json(&document)? != claimed_document_sha {
        return Err(format!(
            "input.{field}.document_sha256 does not bind its document"
        ));
    }
    if verify_seal(&document, &format!("input.{field}.document"))? != claimed_seal {
        return Err(format!(
            "input.{field}.document_seal_sha256 does not bind its document"
        ));
    }
    exact_string(
        root,
        "schema",
        expected_schema,
        &format!("input.{field}.document"),
    )?;
    let observed_status = string_field(root, "status", &format!("input.{field}.document"))?;
    if !expected_statuses.contains(&observed_status) {
        return Err(format!("input.{field}.document.status drifted"));
    }
    Ok(BoundDocument {
        document,
        document_sha256: claimed_document_sha,
        document_seal_sha256: claimed_seal,
    })
}

fn require_binding(
    binding: &Map<String, Value>,
    expected: &BoundDocument,
    label: &str,
) -> Result<(), String> {
    let observed_seal = binding
        .get("document_seal_sha256")
        .or_else(|| binding.get("seal_sha256"))
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label}.seal_sha256 is required"))?;
    if observed_seal != expected.document_seal_sha256 {
        return Err(format!(
            "{label} does not bind the expected sealed document"
        ));
    }
    if let Some(observed_document_sha) = binding
        .get("canonical_document_sha256")
        .or_else(|| binding.get("bound_document_sha256"))
        .and_then(Value::as_str)
    {
        if observed_document_sha != expected.document_sha256 {
            return Err(format!("{label} canonical document identity drifted"));
        }
    }
    Ok(())
}

fn identity_summary(document: Option<&BoundDocument>) -> Value {
    match document {
        Some(value) => json!({
            "present": true,
            "document_sha256": value.document_sha256,
            "document_seal_sha256": value.document_seal_sha256,
        }),
        None => json!({
            "present": false,
            "document_sha256": Value::Null,
            "document_seal_sha256": Value::Null,
        }),
    }
}

fn validate_authority(authority: &BoundDocument) -> Result<AuthorityFacts, String> {
    let root = object(&authority.document, "handoff authority")?;
    reject_fixture_or_self_assertion(root, "handoff authority")?;
    bool_field(
        root,
        "ready_for_l1_device_handoff",
        false,
        "handoff authority",
    )?;
    bool_field(root, "component_only", true, "handoff authority")?;
    let source = object_field(root, "source_binding", "handoff authority")?;
    exact_string(
        source,
        "model_key",
        MODEL_KEY,
        "handoff authority.source_binding",
    )?;
    exact_string(
        source,
        "source_revision",
        SOURCE_REVISION,
        "handoff authority.source_binding",
    )?;
    exact_string(
        source,
        "manifest_document_sha256",
        MANIFEST_DOCUMENT_SHA,
        "handoff authority.source_binding",
    )?;
    exact_string(
        source,
        "manifest_seal_sha256",
        MANIFEST_SEAL,
        "handoff authority.source_binding",
    )?;
    exact_string(
        source,
        "admission_receipt_seal_sha256",
        ADMISSION_RECEIPT_SEAL,
        "handoff authority.source_binding",
    )?;
    if u64_field(
        source,
        "source_token_id",
        "handoff authority.source_binding",
    )? != SOURCE_TOKEN_ID
    {
        return Err("handoff authority source token drifted".into());
    }
    let consumed = object_field(root, "consumed_component_capture", "handoff authority")?;
    if u64_field(
        consumed,
        "layer",
        "handoff authority.consumed_component_capture",
    )? != L0_LAYER
        || u64_field(
            consumed,
            "linear_state_slot",
            "handoff authority.consumed_component_capture",
        )? != L0_SLOT
    {
        return Err("handoff authority layer/slot drifted".into());
    }
    let graph = object_field(
        consumed,
        "same_command_graph",
        "handoff authority.consumed_component_capture",
    )?;
    for (field, expected) in [
        ("prefix_dispatches", PREFIX_DISPATCHES),
        ("suffix_dispatches", SUFFIX_DISPATCHES),
        ("total_dispatches", TOTAL_DISPATCHES),
    ] {
        if u64_field(
            graph,
            field,
            "handoff authority.consumed_component_capture.same_command_graph",
        )? != expected
        {
            return Err(format!("handoff authority {field} drifted"));
        }
    }
    let residual = object_field(
        consumed,
        "second_residual",
        "handoff authority.consumed_component_capture",
    )?;
    if u64_field(
        residual,
        "elements",
        "handoff authority.consumed_component_capture.second_residual",
    )? != HIDDEN_ELEMENTS
        || u64_field(
            residual,
            "bytes",
            "handoff authority.consumed_component_capture.second_residual",
        )? != HIDDEN_BYTES
    {
        return Err("handoff authority second residual geometry drifted".into());
    }
    Ok(AuthorityFacts {
        second_residual_sha256: sha_field(
            residual,
            "f32le_sha256",
            "handoff authority.consumed_component_capture.second_residual",
        )?,
    })
}

fn validate_l0_state_record(
    state: &Map<String, Value>,
    field: &str,
    expected_offset: u64,
    expected_capacity: u64,
    hash_field: &str,
    label: &str,
) -> Result<(String, String), String> {
    let record = object_field(state, field, label)?;
    if u64_field(record, "slot", &format!("{label}.{field}"))? != L0_SLOT
        || u64_field(record, "offset_bytes", &format!("{label}.{field}"))? != expected_offset
        || u64_field(record, "capacity_bytes", &format!("{label}.{field}"))? != expected_capacity
    {
        return Err(format!("{label}.{field} layout drifted"));
    }
    let allocation = string_field(record, "allocation_id", &format!("{label}.{field}"))?.to_owned();
    let buffer = sha_field(record, "device_buffer_id", &format!("{label}.{field}"))?;
    sha_field(record, hash_field, &format!("{label}.{field}"))?;
    Ok((allocation, buffer))
}

fn validate_l1_active_record(
    l1: &Map<String, Value>,
    field: &str,
    expected_offset: u64,
    expected_capacity: u64,
    label: &str,
) -> Result<(String, String), String> {
    let record = object_field(l1, field, label)?;
    if u64_field(record, "slot", &format!("{label}.{field}"))? != L1_SLOT
        || u64_field(record, "offset_bytes", &format!("{label}.{field}"))? != expected_offset
        || u64_field(record, "capacity_bytes", &format!("{label}.{field}"))? != expected_capacity
    {
        return Err(format!("{label}.{field} L1 slot layout drifted"));
    }
    let allocation = string_field(record, "allocation_id", &format!("{label}.{field}"))?.to_owned();
    let buffer = sha_field(record, "device_buffer_id", &format!("{label}.{field}"))?;
    sha_field(
        record,
        "device_buffer_identity_sha256",
        &format!("{label}.{field}"),
    )?;
    Ok((allocation, buffer))
}

fn validate_inner(inner: &BoundDocument, authority: &AuthorityFacts) -> Result<InnerFacts, String> {
    let root = object(&inner.document, "L0 inner receipt")?;
    reject_fixture_or_self_assertion(root, "L0 inner receipt")?;
    exact_string(root, "mode", "metal", "L0 inner receipt")?;
    for (field, expected) in [
        ("metal_device_or_dispatch_performed", true),
        ("component_only", true),
        ("l1_binding_not_executed", true),
        ("complete_layer_or_token_performed", false),
        ("raw_bf16_or_safetensors_opened", false),
    ] {
        bool_field(root, field, expected, "L0 inner receipt")?;
    }
    if u64_field(root, "l1_prefix_dispatches", "L0 inner receipt")? != 0 {
        return Err("L0 inner receipt claims Layer-1 dispatches".into());
    }
    let artifact = object_field(root, "artifact_binding", "L0 inner receipt")?;
    for (field, expected) in [
        ("manifest_document_sha256", MANIFEST_DOCUMENT_SHA),
        ("manifest_seal_sha256", MANIFEST_SEAL),
        ("admission_receipt_seal_sha256", ADMISSION_RECEIPT_SEAL),
        ("source_revision", SOURCE_REVISION),
    ] {
        exact_string(
            artifact,
            field,
            expected,
            "L0 inner receipt.artifact_binding",
        )?;
    }
    if u64_field(artifact, "layer", "L0 inner receipt.artifact_binding")? != L0_LAYER
        || u64_field(
            artifact,
            "linear_state_slot",
            "L0 inner receipt.artifact_binding",
        )? != L0_SLOT
    {
        return Err("L0 inner artifact layer/slot drifted".into());
    }
    let graph = object_field(root, "same_command_graph", "L0 inner receipt")?;
    for (field, expected) in [
        ("source_token_id", SOURCE_TOKEN_ID),
        ("prefix_dispatches", PREFIX_DISPATCHES),
        ("suffix_dispatches", SUFFIX_DISPATCHES),
        ("total_dispatches", TOTAL_DISPATCHES),
    ] {
        if u64_field(graph, field, "L0 inner receipt.same_command_graph")? != expected {
            return Err(format!(
                "L0 inner receipt.same_command_graph.{field} drifted"
            ));
        }
    }
    for field in [
        "same_command_graph_retained",
        "fenced_once_after_prefix_and_suffix",
        "structural_kernel_trace_non_timed",
    ] {
        bool_field(graph, field, true, "L0 inner receipt.same_command_graph")?;
    }
    let kernel_names = array_field(
        graph,
        "encoded_kernel_names",
        "L0 inner receipt.same_command_graph",
    )?;
    if kernel_names.len() != L0_KERNELS.len()
        || kernel_names
            .iter()
            .zip(L0_KERNELS)
            .any(|(actual, expected)| actual.as_str() != Some(expected))
    {
        return Err("L0 inner receipt must retain the exact 9+14 structural kernel order".into());
    }
    let phase = object_field(root, "execution_phase", "L0 inner receipt")?;
    for field in [
        "strict_artifact_admission_started",
        "strict_artifact_admission_succeeded",
        "metal_context_construction_attempted",
        "metal_context_constructed",
        "structural_kernel_trace_enabled",
        "command_commit_attempted",
        "command_fence_succeeded",
        "readback_started",
        "device_dispatch_may_have_occurred",
    ] {
        bool_field(phase, field, true, "L0 inner receipt.execution_phase")?;
    }
    if u64_field(
        phase,
        "dispatches_encoded",
        "L0 inner receipt.execution_phase",
    )? != TOTAL_DISPATCHES
    {
        return Err("L0 inner receipt execution phase is partial".into());
    }
    let handoff = object_field(root, "l0_state_handoff", "L0 inner receipt")?;
    exact_string(
        handoff,
        "schema",
        INNER_SCHEMA,
        "L0 inner receipt.l0_state_handoff",
    )?;
    exact_string(
        handoff,
        "status",
        INNER_STATUS,
        "L0 inner receipt.l0_state_handoff",
    )?;
    let session_id =
        string_field(handoff, "session_id", "L0 inner receipt.l0_state_handoff")?.to_owned();
    if u64_field(
        handoff,
        "source_token_id",
        "L0 inner receipt.l0_state_handoff",
    )? != SOURCE_TOKEN_ID
    {
        return Err("L0 inner handoff source token drifted".into());
    }
    bool_field(
        handoff,
        "same_command_graph_retained",
        true,
        "L0 inner receipt.l0_state_handoff",
    )?;
    bool_field(
        handoff,
        "l1_binding_not_executed",
        true,
        "L0 inner receipt.l0_state_handoff",
    )?;
    if u64_field(
        handoff,
        "l1_prefix_dispatches",
        "L0 inner receipt.l0_state_handoff",
    )? != 0
    {
        return Err("L0 inner handoff claims Layer-1 work".into());
    }
    let retained = object_field(
        handoff,
        "retained_l0_second_residual",
        "L0 inner receipt.l0_state_handoff",
    )?;
    if u64_field(retained, "elements", "L0 retained output")? != HIDDEN_ELEMENTS
        || u64_field(retained, "bytes", "L0 retained output")? != HIDDEN_BYTES
    {
        return Err("L0 retained output must be exactly 2048 f32 / 8192 bytes".into());
    }
    let output_sha256 = sha_field(retained, "f32le_sha256", "L0 retained output")?;
    if output_sha256 != authority.second_residual_sha256 {
        return Err("L0 retained output hash does not match immutable handoff authority".into());
    }
    let output_device_buffer_id = sha_field(retained, "device_buffer_id", "L0 retained output")?;
    bool_field(
        retained,
        "retained_for_future_layer1_encode",
        true,
        "L0 retained output",
    )?;
    let state = object_field(
        handoff,
        "l0_post_state_commit",
        "L0 inner receipt.l0_state_handoff",
    )?;
    if u64_field(state, "layer", "L0 post-state")? != L0_LAYER
        || u64_field(state, "linear_state_slot", "L0 post-state")? != L0_SLOT
    {
        return Err("L0 post-state layer/slot drifted".into());
    }
    bool_field(state, "checkpoint_before_mutation", true, "L0 post-state")?;
    let records = [
        validate_l0_state_record(
            state,
            "active_conv",
            0,
            L0_CONV_BYTES,
            "post_state_f32le_sha256",
            "L0 post-state",
        )?,
        validate_l0_state_record(
            state,
            "active_recurrent",
            0,
            L0_RECURRENT_BYTES,
            "post_state_f32le_sha256",
            "L0 post-state",
        )?,
        validate_l0_state_record(
            state,
            "rollback_conv",
            0,
            L0_CONV_BYTES,
            "checkpoint_f32le_sha256",
            "L0 post-state",
        )?,
        validate_l0_state_record(
            state,
            "rollback_recurrent",
            0,
            L0_RECURRENT_BYTES,
            "checkpoint_f32le_sha256",
            "L0 post-state",
        )?,
    ];
    let mut allocation_ids = std::collections::BTreeSet::new();
    let mut buffer_ids = std::collections::BTreeSet::new();
    for (allocation, buffer) in &records {
        allocation_ids.insert(allocation);
        buffer_ids.insert(buffer);
    }
    if allocation_ids.len() != records.len() || buffer_ids.len() != records.len() {
        return Err("L0 active/rollback state identities alias".into());
    }
    let l1 = object_field(
        handoff,
        "layer1_input_binding",
        "L0 inner receipt.l0_state_handoff",
    )?;
    if string_field(l1, "session_id", "L0 Layer-1 input binding")? != session_id
        || u64_field(l1, "layer", "L0 Layer-1 input binding")? != L1_LAYER
        || u64_field(l1, "linear_state_slot", "L0 Layer-1 input binding")? != L1_SLOT
        || sha_field(l1, "input_device_buffer_id", "L0 Layer-1 input binding")?
            != output_device_buffer_id
        || sha_field(l1, "input_f32le_sha256", "L0 Layer-1 input binding")? != output_sha256
    {
        return Err("Layer-1 input does not retain the same-session L0 output identity".into());
    }
    bool_field(
        l1,
        "same_command_graph_retained",
        true,
        "L0 Layer-1 input binding",
    )?;
    bool_field(l1, "l1_binding_executed", false, "L0 Layer-1 input binding")?;
    let l1_records = [
        validate_l1_active_record(
            l1,
            "active_conv",
            L1_CONV_OFFSET_BYTES,
            L1_CONV_CAPACITY_BYTES,
            "L0 Layer-1 input binding",
        )?,
        validate_l1_active_record(
            l1,
            "active_recurrent",
            L1_RECURRENT_OFFSET_BYTES,
            L1_RECURRENT_CAPACITY_BYTES,
            "L0 Layer-1 input binding",
        )?,
    ];
    let mut l1_allocations = std::collections::BTreeSet::new();
    let mut l1_buffers = std::collections::BTreeSet::new();
    for (allocation, buffer) in &l1_records {
        l1_allocations.insert(allocation);
        l1_buffers.insert(buffer);
    }
    if l1_allocations.len() != l1_records.len()
        || l1_buffers.len() != l1_records.len()
        || l1_buffers.contains(&output_device_buffer_id)
        || buffer_ids.iter().any(|buffer| l1_buffers.contains(buffer))
    {
        return Err("Layer-1 slot-1 state alias is not distinct from retained/L0 state".into());
    }
    let policy = object_field(root, "metal_execution_policy", "L0 inner receipt")?;
    for field in [
        "timing_or_benchmarking_allowed",
        "l1_prefix_execution_allowed",
        "complete_layer_or_token_allowed",
        "tps_or_tg_claim_allowed",
    ] {
        bool_field(
            policy,
            field,
            false,
            "L0 inner receipt.metal_execution_policy",
        )?;
    }
    bool_field(
        policy,
        "strict_math_required",
        true,
        "L0 inner receipt.metal_execution_policy",
    )?;
    let lease = object_field(
        policy,
        "lease_binding",
        "L0 inner receipt.metal_execution_policy",
    )?;
    let lease_seal = sha_field(lease, "seal_sha256", "L0 inner receipt lease binding")?;
    let lease_id = sha_field(lease, "lease_id", "L0 inner receipt lease binding")?;
    let outer_launch = object_field(root, "outer_launch_authority_binding", "L0 inner receipt")?;
    let outer_launch_authority_seal =
        sha_field(outer_launch, "seal_sha256", "L0 inner outer-launch binding")?;
    let terminal_child = object_field(root, "terminal_child", "L0 inner receipt")?;
    if u64_field(terminal_child, "exit_code", "L0 inner terminal child")? != 0 {
        return Err("L0 inner terminal child did not exit successfully".into());
    }
    bool_field(
        terminal_child,
        "receipt_written_last_is_completion_marker",
        true,
        "L0 inner terminal child",
    )?;
    let durable = object_field(root, "durable_capture", "L0 inner receipt")?;
    for field in [
        "receipt_written_last_is_completion_marker",
        "outer_reaped_capture_required",
        "replay_guarded",
    ] {
        bool_field(durable, field, true, "L0 inner durable capture")?;
    }
    let boundary = object_field(root, "claim_boundary", "L0 inner receipt")?;
    for field in [
        "l0_post_state_rollback_retained_output_component_only",
        "l1_binding_not_executed",
        "may_not_satisfy_next_layer_execution_dependency",
        "no_complete_layer_token_decoder_generation_server_hcli_tps_tg_or_tournament_claim",
    ] {
        bool_field(boundary, field, true, "L0 inner claim boundary")?;
    }
    Ok(InnerFacts {
        session_id,
        output_sha256,
        output_device_buffer_id,
        outer_launch_authority_seal,
        lease_seal,
        lease_id,
    })
}

fn validate_outer(
    outer: &BoundDocument,
    inner: &BoundDocument,
    inner_facts: &InnerFacts,
) -> Result<OuterFacts, String> {
    let root = object(&outer.document, "L0 outer terminal")?;
    reject_fixture_or_self_assertion(root, "L0 outer terminal")?;
    let lease_id = sha_field(root, "lease_id", "L0 outer terminal")?;
    let recorded_at = utc_timestamp_field(root, "recorded_at", "L0 outer terminal")?;
    let one_shot = object_field(root, "one_shot", "L0 outer terminal")?;
    for field in [
        "automatic_retry_disabled",
        "same_capture_dir_never_starts_a_second_child",
        "terminal_receipt_written_last",
        "lease_reuse_prohibited_after_terminal",
        "outer_reaped_child",
    ] {
        bool_field(one_shot, field, true, "L0 outer terminal.one_shot")?;
    }
    let child = object_field(root, "child", "L0 outer terminal")?;
    let terminal = object_field(child, "terminal", "L0 outer terminal.child")?;
    bool_field(terminal, "reaped", true, "L0 outer terminal.child.terminal")?;
    bool_field(
        terminal,
        "timed_out",
        false,
        "L0 outer terminal.child.terminal",
    )?;
    if u64_field(terminal, "exit_code", "L0 outer terminal.child.terminal")? != 0 {
        return Err("L0 outer terminal child did not exit successfully".into());
    }
    let source = object_field(root, "source_binding", "L0 outer terminal")?;
    let lease = object_field(source, "lease_receipt", "L0 outer terminal.source_binding")?;
    let lease_seal = sha_field(lease, "seal_sha256", "L0 outer terminal source lease")?;
    let outer_launch = object_field(
        source,
        "outer_launch_authority",
        "L0 outer terminal.source_binding",
    )?;
    let outer_launch_seal = sha_field(
        outer_launch,
        "seal_sha256",
        "L0 outer terminal outer-launch authority",
    )?;
    if outer_launch_seal != inner_facts.outer_launch_authority_seal {
        return Err(
            "L0 outer terminal and inner receipt disagree on outer launch authority".into(),
        );
    }
    let contract = object_field(
        source,
        "handoff_contract",
        "L0 outer terminal.source_binding",
    )?;
    for (field, expected) in [
        ("source_token_id", SOURCE_TOKEN_ID),
        ("prefix_dispatches", PREFIX_DISPATCHES),
        ("suffix_dispatches", SUFFIX_DISPATCHES),
        ("total_dispatches", TOTAL_DISPATCHES),
        ("l1_prefix_dispatches", 0),
    ] {
        if u64_field(contract, field, "L0 outer terminal handoff contract")? != expected {
            return Err(format!(
                "L0 outer terminal handoff contract {field} drifted"
            ));
        }
    }
    bool_field(
        contract,
        "l1_binding_not_executed",
        true,
        "L0 outer terminal handoff contract",
    )?;
    let capture = object_field(root, "inner_probe_capture", "L0 outer terminal")?;
    bool_field(
        capture,
        "present",
        true,
        "L0 outer terminal.inner_probe_capture",
    )?;
    bool_field(
        capture,
        "binding_valid",
        true,
        "L0 outer terminal.inner_probe_capture",
    )?;
    exact_string(
        capture,
        "schema",
        INNER_SCHEMA,
        "L0 outer terminal.inner_probe_capture",
    )?;
    exact_string(
        capture,
        "status",
        INNER_STATUS,
        "L0 outer terminal.inner_probe_capture",
    )?;
    let receipt = object_field(capture, "receipt", "L0 outer terminal.inner_probe_capture")?;
    require_binding(
        receipt,
        inner,
        "L0 outer terminal.inner_probe_capture.receipt",
    )?;
    let pointer = object_field(root, "versioned_current_admission", "L0 outer terminal")?;
    bool_field(
        pointer,
        "terminal_current_pointer_valid",
        true,
        "L0 outer terminal.versioned_current_admission",
    )?;
    let release = object_field(root, "release", "L0 outer terminal")?;
    bool_field(
        release,
        "actual_release_performed",
        false,
        "L0 outer terminal.release",
    )?;
    let boundary = object_field(root, "claim_boundary", "L0 outer terminal")?;
    for field in [
        "l0_post_state_rollback_retained_output_pre_l1_component_only",
        "l1_binding_not_executed",
        "no_complete_layer_token_decoder_generation_server_hcli_tps_tg_or_tournament_claim",
    ] {
        bool_field(boundary, field, true, "L0 outer terminal.claim_boundary")?;
    }
    bool_field(
        boundary,
        "l1_prefix_executed",
        false,
        "L0 outer terminal.claim_boundary",
    )?;
    if lease_id != inner_facts.lease_id || lease_seal != inner_facts.lease_seal {
        return Err("L0 outer terminal and inner receipt disagree on the exact lease".into());
    }
    Ok(OuterFacts {
        lease_id,
        lease_seal,
        recorded_at,
    })
}

fn validate_release_recommendation(
    recommendation: &BoundDocument,
    outer_facts: &OuterFacts,
    release_lease: &Map<String, Value>,
    release_outer: &Map<String, Value>,
    release_recommendation: &Map<String, Value>,
) -> Result<(), String> {
    let root = object(
        &recommendation.document,
        "lease-release recommendation contract",
    )?;
    reject_fixture_or_self_assertion(root, "lease-release recommendation contract")?;
    bool_field(
        root,
        "outer_terminal_must_be_sealed_and_terminal_before_release",
        true,
        "lease-release recommendation contract",
    )?;

    let recommendation_lease =
        object_field(root, "lease", "lease-release recommendation contract")?;
    if sha_field(
        recommendation_lease,
        "lease_id",
        "lease-release recommendation contract.lease",
    )? != outer_facts.lease_id
        || sha_field(
            recommendation_lease,
            "seal_sha256",
            "lease-release recommendation contract.lease",
        )? != outer_facts.lease_seal
    {
        return Err("lease-release recommendation does not bind the exact terminal lease".into());
    }
    bool_field(
        recommendation_lease,
        "present",
        true,
        "lease-release recommendation contract.lease",
    )?;
    nonzero_u64_field(
        recommendation_lease,
        "bytes",
        "lease-release recommendation contract.lease",
    )?;
    sha_field(
        recommendation_lease,
        "sha256",
        "lease-release recommendation contract.lease",
    )?;

    let release_lease_id = sha_field(release_lease, "lease_id", "lease-release receipt.lease")?;
    let release_lease_seal =
        sha_field(release_lease, "seal_sha256", "lease-release receipt.lease")?;
    if release_lease_id != outer_facts.lease_id
        || release_lease_seal != outer_facts.lease_seal
        || release_lease_id
            != sha_field(
                recommendation_lease,
                "lease_id",
                "lease-release recommendation contract.lease",
            )?
        || release_lease_seal
            != sha_field(
                recommendation_lease,
                "seal_sha256",
                "lease-release recommendation contract.lease",
            )?
    {
        return Err(
            "lease-release recommendation and actual release disagree on the exact lease".into(),
        );
    }

    let recommendation_outer_path = string_field(
        root,
        "outer_terminal_path",
        "lease-release recommendation contract",
    )?;
    let release_outer_path = string_field(
        release_outer,
        "path",
        "lease-release receipt.outer_terminal",
    )?;
    if recommendation_outer_path != release_outer_path {
        return Err(
            "lease-release recommendation and actual release disagree on outer terminal path"
                .into(),
        );
    }
    let recommendation_output_path = string_field(
        root,
        "recommended_release_output_path",
        "lease-release recommendation contract",
    )?;
    let release_recommendation_path = string_field(
        release_recommendation,
        "path",
        "lease-release receipt.recommended_release_contract",
    )?;
    if recommendation_output_path != release_recommendation_path {
        return Err(
            "lease-release recommendation output path does not bind actual release evidence".into(),
        );
    }

    let coordination = object_field(
        root,
        "coordination",
        "lease-release recommendation contract",
    )?;
    for field in [
        "actual_release_not_performed_by_outer_reaper",
        "automatic_retry_prohibited",
        "new_qwen80_gpu_work_requires_a_fresh_explicit_lease",
        "watcher_hold_must_remain_active",
    ] {
        bool_field(
            coordination,
            field,
            true,
            "lease-release recommendation contract.coordination",
        )?;
    }
    bool_field(
        coordination,
        "watcher_restart_or_transition_authorized",
        false,
        "lease-release recommendation contract.coordination",
    )?;
    let boundary = object_field(
        root,
        "claim_boundary",
        "lease-release recommendation contract",
    )?;
    for field in [
        "recommendation_is_gpu_coordination_only",
        "does_not_promote_pre_l1_component_to_layer_token_decoder_hcli_tps_tg_or_tournament",
    ] {
        bool_field(
            boundary,
            field,
            true,
            "lease-release recommendation contract.claim_boundary",
        )?;
    }
    Ok(())
}

fn validate_release(
    release: &BoundDocument,
    outer: &BoundDocument,
    outer_facts: &OuterFacts,
    recommendation: &BoundDocument,
) -> Result<(), String> {
    let root = object(&release.document, "lease-release receipt")?;
    reject_fixture_or_self_assertion(root, "lease-release receipt")?;
    let recorded_at = utc_timestamp_field(root, "recorded_at", "lease-release receipt")?;
    if recorded_at <= outer_facts.recorded_at {
        return Err(
            "lease-release receipt predates or is not newer than the terminal outer receipt".into(),
        );
    }

    let lease = object_field(root, "lease", "lease-release receipt")?;
    if sha_field(lease, "lease_id", "lease-release receipt.lease")? != outer_facts.lease_id {
        return Err("lease-release receipt does not bind the terminal lease ID".into());
    }
    if sha_field(lease, "seal_sha256", "lease-release receipt.lease")? != outer_facts.lease_seal {
        return Err("lease-release receipt does not bind the exact outer lease".into());
    }
    bool_field(lease, "present", true, "lease-release receipt.lease")?;
    nonzero_u64_field(lease, "bytes", "lease-release receipt.lease")?;
    sha_field(lease, "sha256", "lease-release receipt.lease")?;
    string_field(lease, "path", "lease-release receipt.lease")?;

    let outer_binding = object_field(root, "outer_terminal", "lease-release receipt")?;
    bool_field(
        outer_binding,
        "present",
        true,
        "lease-release receipt.outer_terminal",
    )?;
    exact_string(
        outer_binding,
        "status",
        OUTER_STATUS,
        "lease-release receipt.outer_terminal",
    )?;
    nonzero_u64_field(
        outer_binding,
        "bytes",
        "lease-release receipt.outer_terminal",
    )?;
    sha_field(
        outer_binding,
        "sha256",
        "lease-release receipt.outer_terminal",
    )?;
    string_field(
        outer_binding,
        "path",
        "lease-release receipt.outer_terminal",
    )?;
    require_binding(outer_binding, outer, "lease-release receipt.outer_terminal")?;

    let recommendation_binding = object_field(
        root,
        "recommended_release_contract",
        "lease-release receipt",
    )?;
    bool_field(
        recommendation_binding,
        "present",
        true,
        "lease-release receipt.recommended_release_contract",
    )?;
    nonzero_u64_field(
        recommendation_binding,
        "bytes",
        "lease-release receipt.recommended_release_contract",
    )?;
    sha_field(
        recommendation_binding,
        "sha256",
        "lease-release receipt.recommended_release_contract",
    )?;
    string_field(
        recommendation_binding,
        "path",
        "lease-release receipt.recommended_release_contract",
    )?;
    require_binding(
        recommendation_binding,
        recommendation,
        "lease-release receipt.recommended_release_contract",
    )?;
    validate_release_recommendation(
        recommendation,
        outer_facts,
        lease,
        outer_binding,
        recommendation_binding,
    )?;

    let coordination = object_field(root, "coordination", "lease-release receipt")?;
    for field in [
        "quiet_qwen80_component_lease_released",
        "new_qwen80_gpu_work_requires_a_fresh_explicit_lease",
        "automatic_retry_prohibited",
        "watcher_hold_remains_active",
    ] {
        bool_field(
            coordination,
            field,
            true,
            "lease-release receipt.coordination",
        )?;
    }
    bool_field(
        coordination,
        "watcher_restart_or_transition_authorized",
        false,
        "lease-release receipt.coordination",
    )?;
    let boundary = object_field(root, "claim_boundary", "lease-release receipt")?;
    bool_field(
        boundary,
        "release_is_gpu_coordination_only",
        true,
        "lease-release receipt.claim_boundary",
    )?;
    bool_field(
        boundary,
        "does_not_promote_component_to_layer_token_decoder_hcli_tps_tg_or_tournament",
        true,
        "lease-release receipt.claim_boundary",
    )?;
    Ok(())
}

fn validate_continuation(
    continuation: &BoundDocument,
    inner: &BoundDocument,
) -> Result<bool, String> {
    let root = object(&continuation.document, "L1 continuation contract")?;
    reject_fixture_or_self_assertion(root, "L1 continuation contract")?;
    let status = string_field(root, "status", "L1 continuation contract")?;
    let prepared = match status {
        CONTINUATION_PREPARED_STATUS => true,
        CONTINUATION_INCOMPLETE_STATUS => false,
        _ => return Err("L1 continuation contract status drifted".into()),
    };
    bool_field(root, "prepared", prepared, "L1 continuation contract")?;
    bool_field(
        root,
        "l1_execution_performed_by_this_contract",
        false,
        "L1 continuation contract",
    )?;
    if u64_field(
        root,
        "l1_prefix_dispatches_executed_by_this_contract",
        "L1 continuation contract",
    )? != 0
    {
        return Err("L1 continuation contract claims dispatches".into());
    }
    let handoff = object_field(root, "l0_state_handoff_receipt", "L1 continuation contract")?;
    require_binding(
        handoff,
        inner,
        "L1 continuation contract.l0_state_handoff_receipt",
    )?;
    let scope = object_field(
        root,
        "future_l1_slot1_deltanet_prefix_scope",
        "L1 continuation contract",
    )?;
    if u64_field(scope, "layer", "L1 continuation contract scope")? != L1_LAYER
        || u64_field(scope, "linear_state_slot", "L1 continuation contract scope")? != L1_SLOT
        || u64_field(
            scope,
            "exact_prefix_dispatch_count",
            "L1 continuation contract scope",
        )? != 9
    {
        return Err("L1 continuation contract scope drifted".into());
    }
    bool_field(
        scope,
        "no_l1_suffix_or_moe_dispatch_authorized",
        true,
        "L1 continuation contract scope",
    )?;
    for field in [
        "l0_baseline_evidence_only",
        "fresh_same_runtime_same_tcb_joint_l0_to_l1_capture_required",
        "fresh_joint_l0_output_identity_required",
    ] {
        bool_field(scope, field, true, "L1 continuation contract scope")?;
    }
    bool_field(
        scope,
        "cross_process_or_prior_capture_pinned_buffer_reuse_authorized",
        false,
        "L1 continuation contract scope",
    )?;
    let boundary = object_field(root, "authority_boundary", "L1 continuation contract")?;
    for field in [
        "new_physical_model_processes_authorized",
        "server_starts_authorized",
        "port_binds_authorized",
        "gpu_leases_authorized",
        "watcher_changes_authorized",
        "tournament_state_mutations_authorized",
    ] {
        if u64_field(
            boundary,
            field,
            "L1 continuation contract authority boundary",
        )? != 0
        {
            return Err(format!("L1 continuation contract authorizes {field}"));
        }
    }
    Ok(prepared)
}

fn parse_optional(
    input: &Map<String, Value>,
    field: &str,
    schema: &str,
    status: &str,
    blockers: &mut Vec<String>,
) -> Option<BoundDocument> {
    match parse_bound_document(input, field, schema, status) {
        Ok(value) => Some(value),
        Err(error) => {
            blockers.push(format!("{field}: {error}"));
            None
        }
    }
}

fn parse_optional_continuation(
    input: &Map<String, Value>,
    blockers: &mut Vec<String>,
) -> Option<BoundDocument> {
    match parse_bound_document_with_statuses(
        input,
        "l1_continuation_contract",
        CONTINUATION_SCHEMA,
        &[CONTINUATION_PREPARED_STATUS, CONTINUATION_INCOMPLETE_STATUS],
    ) {
        Ok(value) => Some(value),
        Err(error) => {
            blockers.push(format!("l1_continuation_contract: {error}"));
            None
        }
    }
}

fn build_report(input: &Value) -> Value {
    let mut blockers = Vec::new();
    let empty = Map::new();
    let root = match object(input, "input") {
        Ok(value) => value,
        Err(error) => {
            blockers.push(error);
            &empty
        }
    };
    if root.get("schema").and_then(Value::as_str) != Some(INPUT_SCHEMA) {
        blockers.push(format!("input.schema must be {INPUT_SCHEMA}"));
    }
    if root.get("status").and_then(Value::as_str) != Some(INPUT_STATUS) {
        blockers.push(format!("input.status must be {INPUT_STATUS}"));
    }
    if let Err(error) = verify_seal(input, "input") {
        blockers.push(format!("input must be sealed: {error}"));
    }
    if root.get("l1_execution_requested").and_then(Value::as_bool) != Some(false) {
        blockers.push("input must state l1_execution_requested=false".into());
    }
    if root
        .get("l1_execution_evidence")
        .is_some_and(|value| !value.is_null())
    {
        blockers.push("input may not include L1 execution evidence".into());
    }

    let authority = parse_optional(
        root,
        "handoff_authority",
        HANDOFF_AUTHORITY_SCHEMA,
        HANDOFF_AUTHORITY_STATUS,
        &mut blockers,
    );
    let outer = parse_optional(
        root,
        "l0_outer_terminal",
        OUTER_SCHEMA,
        OUTER_STATUS,
        &mut blockers,
    );
    let inner = parse_optional(
        root,
        "l0_inner_receipt",
        INNER_SCHEMA,
        INNER_STATUS,
        &mut blockers,
    );
    let release = parse_optional(
        root,
        "lease_release_receipt",
        LEASE_RELEASE_SCHEMA,
        LEASE_RELEASE_STATUS,
        &mut blockers,
    );
    let release_recommendation = parse_optional(
        root,
        "lease_release_recommendation_contract",
        RELEASE_RECOMMENDATION_SCHEMA,
        RELEASE_RECOMMENDATION_STATUS,
        &mut blockers,
    );
    let continuation = parse_optional_continuation(root, &mut blockers);

    let authority_facts = authority
        .as_ref()
        .and_then(|value| match validate_authority(value) {
            Ok(facts) => Some(facts),
            Err(error) => {
                blockers.push(format!("handoff_authority: {error}"));
                None
            }
        });
    let inner_facts = match (inner.as_ref(), authority_facts.as_ref()) {
        (Some(inner), Some(authority)) => match validate_inner(inner, authority) {
            Ok(facts) => Some(facts),
            Err(error) => {
                blockers.push(format!("l0_inner_receipt: {error}"));
                None
            }
        },
        _ => None,
    };
    let outer_facts = match (outer.as_ref(), inner.as_ref(), inner_facts.as_ref()) {
        (Some(outer), Some(inner), Some(inner_facts)) => {
            match validate_outer(outer, inner, inner_facts) {
                Ok(facts) => Some(facts),
                Err(error) => {
                    blockers.push(format!("l0_outer_terminal: {error}"));
                    None
                }
            }
        }
        _ => None,
    };
    if let (Some(release), Some(outer), Some(outer_facts), Some(recommendation)) = (
        release.as_ref(),
        outer.as_ref(),
        outer_facts.as_ref(),
        release_recommendation.as_ref(),
    ) {
        if let Err(error) = validate_release(release, outer, outer_facts, recommendation) {
            blockers.push(format!("lease_release_receipt: {error}"));
        }
    }
    let continuation_prepared = match (continuation.as_ref(), inner.as_ref()) {
        (Some(continuation), Some(inner)) => match validate_continuation(continuation, inner) {
            Ok(prepared) => Some(prepared),
            Err(error) => {
                blockers.push(format!("l1_continuation_contract: {error}"));
                None
            }
        },
        _ => None,
    };

    blockers.sort();
    blockers.dedup();
    let earned = blockers.is_empty()
        && authority_facts.is_some()
        && inner_facts.is_some()
        && outer_facts.is_some()
        && release.is_some()
        && release_recommendation.is_some()
        && continuation.is_some();
    let handoff = inner_facts.as_ref().map(|facts| {
        json!({
            "session_id": facts.session_id,
            "retained_l0_second_residual_f32le_sha256": facts.output_sha256,
            "retained_l0_second_residual_device_buffer_id": facts.output_device_buffer_id,
            "elements": HIDDEN_ELEMENTS,
            "bytes": HIDDEN_BYTES,
            "l1_layer": L1_LAYER,
            "l1_linear_state_slot": L1_SLOT,
        })
    });
    let mut output = json!({
        "schema": RESULT_SCHEMA,
        "status": if earned { EARNED_STATUS } else { REFUSED_STATUS },
        "earned_l0_state_handoff_component": earned,
        "l1_binding_not_executed": true,
        "l1_prefix_dispatches": 0,
        "may_not_satisfy_next_layer_execution_dependency": true,
        "l0_handoff_is_evidence_baseline_only": true,
        "future_l1_requires_fresh_same_runtime_same_tcb_joint_l0_to_l1_capture": true,
        "cross_process_or_prior_capture_pinned_buffer_reuse_authorized": false,
        "handoff_authority": identity_summary(authority.as_ref()),
        "l0_outer_terminal": identity_summary(outer.as_ref()),
        "l0_inner_receipt": identity_summary(inner.as_ref()),
        "lease_release_receipt": identity_summary(release.as_ref()),
        "lease_release_recommendation_contract": identity_summary(release_recommendation.as_ref()),
        "l1_continuation_contract": identity_summary(continuation.as_ref()),
        "l1_continuation_prepared": continuation_prepared,
        "l1_continuation_remains_non_executing": true,
        "validated_l0_handoff": handoff,
        "blockers": blockers,
        "authority_boundary": {
            "new_model_processes_authorized": 0,
            "metal_or_gpu_actions_authorized": 0,
            "lease_actions_authorized": 0,
            "server_or_watcher_actions_authorized": 0,
            "tps_or_tg_measurements_authorized": 0,
            "tournament_actions_authorized": 0,
            "l1_dispatches_authorized": 0,
        },
        "claim_boundary": {
            "cpu_only_post_capture_assessment": true,
            "does_not_open_or_scan_artifacts": true,
            "does_not_construct_metal_or_dispatch": true,
            "does_not_issue_or_release_a_lease": true,
            "does_not_start_server_or_watcher": true,
            "does_not_measure_tps_or_tg": true,
            "does_not_claim_complete_layer_token_decoder_or_tournament": true,
            "l0_device_buffers_are_not_transferable_execution_authority": true,
            "future_l1_requires_a_fresh_joint_same_runtime_same_tcb_capture": true,
        },
    });
    seal(&mut output).expect("assessment output must be sealable");
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
    let raw = fs::read(&path).map_err(|error| format!("cannot read {label}: {error}"))?;
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
    "usage: ascension_qwen80_l0_state_handoff_post_capture_assessor --input ABSOLUTE_PATH --out ABSOLUTE_NEW_PATH"
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
                | "--handoff-authority"
                | "--l0-outer-terminal"
                | "--l0-inner-receipt"
                | "--lease-release-receipt"
                | "--lease-release-recommendation-contract"
                | "--l1-continuation-contract"
        ) {
            return Err(format!("unsupported option {flag:?}; {}", usage()));
        }
        if values.insert(flag.clone(), value).is_some() {
            return Err(format!("duplicate {flag}; {}", usage()));
        }
    }
    let take = |flag: &str| {
        values
            .get(flag)
            .map(PathBuf::from)
            .ok_or_else(|| format!("missing {flag}; {}", usage()))
    };
    let optional = |flag: &str| values.get(flag).map(PathBuf::from);
    let input = optional("--input");
    let handoff_authority = optional("--handoff-authority");
    let l0_outer_terminal = optional("--l0-outer-terminal");
    let l0_inner_receipt = optional("--l0-inner-receipt");
    let lease_release_receipt = optional("--lease-release-receipt");
    let lease_release_recommendation_contract = optional("--lease-release-recommendation-contract");
    let l1_continuation_contract = optional("--l1-continuation-contract");
    let explicit_required = [
        handoff_authority.as_ref(),
        l0_outer_terminal.as_ref(),
        l0_inner_receipt.as_ref(),
        lease_release_receipt.as_ref(),
        lease_release_recommendation_contract.as_ref(),
    ];
    if input.is_some() && explicit_required.iter().any(Option::is_some)
        || input.is_some() && l1_continuation_contract.is_some()
    {
        return Err(format!(
            "--input cannot be combined with explicit evidence paths; {}",
            usage()
        ));
    }
    if input.is_none() && explicit_required.iter().any(Option::is_none) {
        return Err(format!(
            "explicit evidence mode requires the authority, outer, inner, release, and recommendation paths; {}",
            usage()
        ));
    }
    let args = Args {
        input,
        handoff_authority,
        l0_outer_terminal,
        l0_inner_receipt,
        lease_release_receipt,
        lease_release_recommendation_contract,
        l1_continuation_contract,
        out: take("--out")?,
    };
    let paths = [
        args.input.as_ref(),
        args.handoff_authority.as_ref(),
        args.l0_outer_terminal.as_ref(),
        args.l0_inner_receipt.as_ref(),
        args.lease_release_receipt.as_ref(),
        args.lease_release_recommendation_contract.as_ref(),
        args.l1_continuation_contract.as_ref(),
    ];
    if !args.out.is_absolute() || paths.iter().flatten().any(|path| !path.is_absolute()) {
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
    let handoff_authority = args
        .handoff_authority
        .as_ref()
        .ok_or("missing --handoff-authority")?;
    let l0_outer_terminal = args
        .l0_outer_terminal
        .as_ref()
        .ok_or("missing --l0-outer-terminal")?;
    let l0_inner_receipt = args
        .l0_inner_receipt
        .as_ref()
        .ok_or("missing --l0-inner-receipt")?;
    let lease_release_receipt = args
        .lease_release_receipt
        .as_ref()
        .ok_or("missing --lease-release-receipt")?;
    let lease_release_recommendation_contract = args
        .lease_release_recommendation_contract
        .as_ref()
        .ok_or("missing --lease-release-recommendation-contract")?;
    let mut input = json!({
        "schema": INPUT_SCHEMA,
        "status": INPUT_STATUS,
        "l1_execution_requested": false,
        "l1_execution_evidence": Value::Null,
        "handoff_authority": live_binding(read_json(handoff_authority, "--handoff-authority")?)?,
        "l0_outer_terminal": live_binding(read_json(l0_outer_terminal, "--l0-outer-terminal")?)?,
        "l0_inner_receipt": live_binding(read_json(l0_inner_receipt, "--l0-inner-receipt")?)?,
        "lease_release_receipt": live_binding(read_json(lease_release_receipt, "--lease-release-receipt")?)?,
        "lease_release_recommendation_contract": live_binding(read_json(lease_release_recommendation_contract, "--lease-release-recommendation-contract")?)?,
    });
    if let Some(path) = args.l1_continuation_contract.as_ref() {
        input["l1_continuation_contract"] =
            live_binding(read_json(path, "--l1-continuation-contract")?)?;
    }
    seal(&mut input)?;
    Ok(input)
}

fn run(args: Args) -> Result<(), String> {
    let input = match args.input.as_ref() {
        Some(path) => read_json(path, "--input")?,
        None => build_explicit_evidence_input(&args)?,
    };
    let output = build_report(&input);
    let bytes = serde_json::to_vec_pretty(&output)
        .map_err(|error| format!("cannot serialize assessment: {error}"))?;
    write_new(&args.out, &bytes)
}

fn main() {
    match parse_args(env::args().skip(1)).and_then(run) {
        Ok(()) => {}
        Err(error) => {
            eprintln!("ascension_qwen80_l0_state_handoff_post_capture_assessor: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn seal_document(mut value: Value) -> Value {
        seal(&mut value).unwrap();
        value
    }

    fn bind(value: Value) -> Value {
        json!({
            "document": value,
            "document_sha256": sha256_json(&value).unwrap(),
            "document_seal_sha256": value["seal_sha256"],
        })
    }

    fn l0_record(allocation: &str, capacity: u64, hash_field: &str, seed: char) -> Value {
        json!({
            "allocation_id": allocation,
            "slot": L0_SLOT,
            "offset_bytes": 0,
            "capacity_bytes": capacity,
            "device_buffer_id": sha(seed),
            hash_field: sha(if seed == 'f' { 'e' } else { 'f' }),
        })
    }

    fn l1_record(allocation: &str, offset: u64, capacity: u64, seed: char) -> Value {
        json!({
            "allocation_id": allocation,
            "slot": L1_SLOT,
            "offset_bytes": offset,
            "capacity_bytes": capacity,
            "device_buffer_id": sha(seed),
            "device_buffer_identity_sha256": sha(if seed == 'd' { 'c' } else { 'd' }),
        })
    }

    fn authority() -> Value {
        seal_document(json!({
            "schema": HANDOFF_AUTHORITY_SCHEMA,
            "status": HANDOFF_AUTHORITY_STATUS,
            "ready_for_l1_device_handoff": false,
            "component_only": true,
            "source_binding": {
                "model_key": MODEL_KEY,
                "source_revision": SOURCE_REVISION,
                "manifest_document_sha256": MANIFEST_DOCUMENT_SHA,
                "manifest_seal_sha256": MANIFEST_SEAL,
                "admission_receipt_seal_sha256": ADMISSION_RECEIPT_SEAL,
                "source_token_id": SOURCE_TOKEN_ID,
            },
            "consumed_component_capture": {
                "layer": L0_LAYER,
                "linear_state_slot": L0_SLOT,
                "same_command_graph": {
                    "prefix_dispatches": PREFIX_DISPATCHES,
                    "suffix_dispatches": SUFFIX_DISPATCHES,
                    "total_dispatches": TOTAL_DISPATCHES,
                },
                "second_residual": {"elements": HIDDEN_ELEMENTS, "bytes": HIDDEN_BYTES, "f32le_sha256": sha('a')},
            },
        }))
    }

    fn inner(lease_id: &str, lease_seal: &str, outer_launch_seal: &str) -> Value {
        seal_document(json!({
            "schema": INNER_SCHEMA,
            "status": INNER_STATUS,
            "mode": "metal",
            "metal_device_or_dispatch_performed": true,
            "component_only": true,
            "l1_binding_not_executed": true,
            "l1_prefix_dispatches": 0,
            "complete_layer_or_token_performed": false,
            "raw_bf16_or_safetensors_opened": false,
            "artifact_binding": {
                "manifest_document_sha256": MANIFEST_DOCUMENT_SHA,
                "manifest_seal_sha256": MANIFEST_SEAL,
                "admission_receipt_seal_sha256": ADMISSION_RECEIPT_SEAL,
                "source_revision": SOURCE_REVISION,
                "layer": L0_LAYER,
                "linear_state_slot": L0_SLOT,
            },
            "same_command_graph": {
                "source_token_id": SOURCE_TOKEN_ID,
                "prefix_dispatches": PREFIX_DISPATCHES,
                "suffix_dispatches": SUFFIX_DISPATCHES,
                "total_dispatches": TOTAL_DISPATCHES,
                "same_command_graph_retained": true,
                "fenced_once_after_prefix_and_suffix": true,
                "structural_kernel_trace_non_timed": true,
                "encoded_kernel_names": L0_KERNELS,
            },
            "execution_phase": {
                "strict_artifact_admission_started": true,
                "strict_artifact_admission_succeeded": true,
                "metal_context_construction_attempted": true,
                "metal_context_constructed": true,
                "structural_kernel_trace_enabled": true,
                "command_commit_attempted": true,
                "command_fence_succeeded": true,
                "readback_started": true,
                "device_dispatch_may_have_occurred": true,
                "dispatches_encoded": TOTAL_DISPATCHES,
            },
            "l0_state_handoff": {
                "schema": INNER_SCHEMA,
                "status": INNER_STATUS,
                "session_id": "qwen80-source-token-session-01",
                "source_token_id": SOURCE_TOKEN_ID,
                "same_command_graph_retained": true,
                "l1_binding_not_executed": true,
                "l1_prefix_dispatches": 0,
                "retained_l0_second_residual": {
                    "elements": HIDDEN_ELEMENTS,
                    "bytes": HIDDEN_BYTES,
                    "f32le_sha256": sha('a'),
                    "device_buffer_id": sha('b'),
                    "retained_for_future_layer1_encode": true,
                },
                "l0_post_state_commit": {
                    "layer": L0_LAYER,
                    "linear_state_slot": L0_SLOT,
                    "checkpoint_before_mutation": true,
                    "active_conv": l0_record("l0-active-conv", L0_CONV_BYTES, "post_state_f32le_sha256", '1'),
                    "active_recurrent": l0_record("l0-active-recurrent", L0_RECURRENT_BYTES, "post_state_f32le_sha256", '2'),
                    "rollback_conv": l0_record("l0-rollback-conv", L0_CONV_BYTES, "checkpoint_f32le_sha256", '3'),
                    "rollback_recurrent": l0_record("l0-rollback-recurrent", L0_RECURRENT_BYTES, "checkpoint_f32le_sha256", '4'),
                },
                "layer1_input_binding": {
                    "session_id": "qwen80-source-token-session-01",
                    "layer": L1_LAYER,
                    "linear_state_slot": L1_SLOT,
                    "input_device_buffer_id": sha('b'),
                    "input_f32le_sha256": sha('a'),
                    "same_command_graph_retained": true,
                    "l1_binding_executed": false,
                    "active_conv": l1_record("l1-active-conv", L1_CONV_OFFSET_BYTES, L1_CONV_CAPACITY_BYTES, '5'),
                    "active_recurrent": l1_record("l1-active-recurrent", L1_RECURRENT_OFFSET_BYTES, L1_RECURRENT_CAPACITY_BYTES, '6'),
                },
            },
            "metal_execution_policy": {
                "strict_math_required": true,
                "timing_or_benchmarking_allowed": false,
                "l1_prefix_execution_allowed": false,
                "complete_layer_or_token_allowed": false,
                "tps_or_tg_claim_allowed": false,
                "lease_binding": {"seal_sha256": lease_seal, "lease_id": lease_id},
            },
            "outer_launch_authority_binding": {"seal_sha256": outer_launch_seal},
            "terminal_child": {"exit_code": 0, "receipt_written_last_is_completion_marker": true},
            "durable_capture": {"receipt_written_last_is_completion_marker": true, "outer_reaped_capture_required": true, "replay_guarded": true},
            "claim_boundary": {
                "l0_post_state_rollback_retained_output_component_only": true,
                "l1_binding_not_executed": true,
                "may_not_satisfy_next_layer_execution_dependency": true,
                "no_complete_layer_token_decoder_generation_server_hcli_tps_tg_or_tournament_claim": true,
            },
        }))
    }

    fn outer(inner: &Value, lease_id: &str, lease_seal: &str, outer_launch_seal: &str) -> Value {
        seal_document(json!({
            "schema": OUTER_SCHEMA,
            "status": OUTER_STATUS,
            "recorded_at": "2026-08-09T08:19:36.779358Z",
            "lease_id": lease_id,
            "one_shot": {
                "automatic_retry_disabled": true,
                "same_capture_dir_never_starts_a_second_child": true,
                "terminal_receipt_written_last": true,
                "lease_reuse_prohibited_after_terminal": true,
                "outer_reaped_child": true,
            },
            "child": {"terminal": {"reaped": true, "timed_out": false, "exit_code": 0}},
            "source_binding": {
                "lease_receipt": {"seal_sha256": lease_seal},
                "outer_launch_authority": {"seal_sha256": outer_launch_seal},
                "handoff_contract": {
                    "source_token_id": SOURCE_TOKEN_ID,
                    "prefix_dispatches": PREFIX_DISPATCHES,
                    "suffix_dispatches": SUFFIX_DISPATCHES,
                    "total_dispatches": TOTAL_DISPATCHES,
                    "l1_prefix_dispatches": 0,
                    "l1_binding_not_executed": true,
                },
            },
            "inner_probe_capture": {
                "present": true,
                "binding_valid": true,
                "schema": INNER_SCHEMA,
                "status": INNER_STATUS,
                "receipt": {"seal_sha256": inner["seal_sha256"]},
            },
            "versioned_current_admission": {"terminal_current_pointer_valid": true},
            "release": {"actual_release_performed": false},
            "claim_boundary": {
                "l0_post_state_rollback_retained_output_pre_l1_component_only": true,
                "l1_binding_not_executed": true,
                "l1_prefix_executed": false,
                "no_complete_layer_token_decoder_generation_server_hcli_tps_tg_or_tournament_claim": true,
            },
        }))
    }

    fn recommendation(lease_id: &str, lease_seal: &str) -> Value {
        seal_document(json!({
            "schema": RELEASE_RECOMMENDATION_SCHEMA,
            "status": RELEASE_RECOMMENDATION_STATUS,
            "recorded_at": "2026-08-09T08:19:36.739350Z",
            "lease": {
                "present": true,
                "bytes": 8213,
                "path": "/records/qwen80/lease.json",
                "sha256": sha('c'),
                "seal_sha256": lease_seal,
                "lease_id": lease_id,
            },
            "outer_terminal_must_be_sealed_and_terminal_before_release": true,
            "outer_terminal_path": "/records/qwen80/outer-terminal.json",
            "recommended_release_output_path": "/records/qwen80/release-recommendation.json",
            "coordination": {
                "actual_release_not_performed_by_outer_reaper": true,
                "automatic_retry_prohibited": true,
                "new_qwen80_gpu_work_requires_a_fresh_explicit_lease": true,
                "watcher_hold_must_remain_active": true,
                "watcher_restart_or_transition_authorized": false,
            },
            "claim_boundary": {
                "recommendation_is_gpu_coordination_only": true,
                "does_not_promote_pre_l1_component_to_layer_token_decoder_hcli_tps_tg_or_tournament": true,
            },
        }))
    }

    fn release(outer: &Value, lease_id: &str, lease_seal: &str, recommendation: &Value) -> Value {
        seal_document(json!({
            "schema": LEASE_RELEASE_SCHEMA,
            "status": LEASE_RELEASE_STATUS,
            "recorded_at": "2026-08-09T08:20:10.506580Z",
            "lease": {
                "present": true,
                "bytes": 8213,
                "path": "/records/qwen80/lease.json",
                "sha256": sha('c'),
                "seal_sha256": lease_seal,
                "lease_id": lease_id,
            },
            "outer_terminal": {
                "present": true,
                "bytes": 11483,
                "path": "/records/qwen80/outer-terminal.json",
                "sha256": sha('d'),
                "seal_sha256": outer["seal_sha256"],
                "status": OUTER_STATUS,
            },
            "recommended_release_contract": {
                "present": true,
                "bytes": 1837,
                "path": "/records/qwen80/release-recommendation.json",
                "sha256": sha('e'),
                "seal_sha256": recommendation["seal_sha256"],
            },
            "coordination": {
                "quiet_qwen80_component_lease_released": true,
                "new_qwen80_gpu_work_requires_a_fresh_explicit_lease": true,
                "automatic_retry_prohibited": true,
                "watcher_hold_remains_active": true,
                "watcher_restart_or_transition_authorized": false,
            },
            "claim_boundary": {
                "release_is_gpu_coordination_only": true,
                "does_not_promote_component_to_layer_token_decoder_hcli_tps_tg_or_tournament": true,
            },
        }))
    }

    fn continuation(inner: &Value) -> Value {
        seal_document(json!({
            "schema": CONTINUATION_SCHEMA,
            "status": CONTINUATION_PREPARED_STATUS,
            "prepared": true,
            "l1_execution_performed_by_this_contract": false,
            "l1_prefix_dispatches_executed_by_this_contract": 0,
            "l0_state_handoff_receipt": {
                "document_sha256": sha256_json(inner).unwrap(),
                "document_seal_sha256": inner["seal_sha256"],
            },
            "future_l1_slot1_deltanet_prefix_scope": {
                "layer": L1_LAYER,
                "linear_state_slot": L1_SLOT,
                "exact_prefix_dispatch_count": 9,
                "no_l1_suffix_or_moe_dispatch_authorized": true,
                "l0_baseline_evidence_only": true,
                "fresh_same_runtime_same_tcb_joint_l0_to_l1_capture_required": true,
                "fresh_joint_l0_output_identity_required": true,
                "cross_process_or_prior_capture_pinned_buffer_reuse_authorized": false,
            },
            "authority_boundary": {
                "new_physical_model_processes_authorized": 0,
                "server_starts_authorized": 0,
                "port_binds_authorized": 0,
                "gpu_leases_authorized": 0,
                "watcher_changes_authorized": 0,
                "tournament_state_mutations_authorized": 0,
            },
        }))
    }

    fn input() -> Value {
        let authority = authority();
        let lease_id = sha('8');
        let lease_seal = sha('9');
        let outer_launch_seal = sha('a');
        let inner = inner(&lease_id, &lease_seal, &outer_launch_seal);
        let outer = outer(&inner, &lease_id, &lease_seal, &outer_launch_seal);
        let release_recommendation = recommendation(&lease_id, &lease_seal);
        let release = release(&outer, &lease_id, &lease_seal, &release_recommendation);
        let continuation = continuation(&inner);
        seal_document(json!({
            "schema": INPUT_SCHEMA,
            "status": INPUT_STATUS,
            "l1_execution_requested": false,
            "handoff_authority": bind(authority),
            "l0_outer_terminal": bind(outer),
            "l0_inner_receipt": bind(inner),
            "lease_release_receipt": bind(release),
            "lease_release_recommendation_contract": bind(release_recommendation),
            "l1_continuation_contract": bind(continuation),
        }))
    }

    #[test]
    fn absent_evidence_refuses_without_authorizing_l1_or_device_work() {
        let result = build_report(&json!({}));
        assert_eq!(
            verify_seal(&result, "result").unwrap(),
            result["seal_sha256"]
        );
        assert_eq!(result["status"], REFUSED_STATUS);
        assert_eq!(result["earned_l0_state_handoff_component"], false);
        assert_eq!(result["l1_binding_not_executed"], true);
        assert_eq!(result["l1_prefix_dispatches"], 0);
        assert_eq!(
            result["authority_boundary"]["metal_or_gpu_actions_authorized"],
            0
        );
    }

    #[test]
    fn authentic_nested_lease_release_chain_earns_only_l0_baseline_component() {
        let result = build_report(&input());
        assert_eq!(
            verify_seal(&result, "result").unwrap(),
            result["seal_sha256"]
        );
        assert_eq!(result["status"], EARNED_STATUS);
        assert_eq!(result["earned_l0_state_handoff_component"], true);
        assert_eq!(result["l1_binding_not_executed"], true);
        assert_eq!(result["l1_prefix_dispatches"], 0);
        assert_eq!(result["validated_l0_handoff"]["bytes"], HIDDEN_BYTES);
        assert_eq!(result["authority_boundary"]["l1_dispatches_authorized"], 0);
        assert_eq!(result["l0_handoff_is_evidence_baseline_only"], true);
        assert_eq!(
            result["future_l1_requires_fresh_same_runtime_same_tcb_joint_l0_to_l1_capture"],
            true
        );
        assert_eq!(
            result["cross_process_or_prior_capture_pinned_buffer_reuse_authorized"],
            false
        );
    }

    #[test]
    fn sealed_incomplete_continuation_can_preserve_the_earned_l0_baseline_but_not_l1_readiness() {
        let mut value = input();
        let continuation = &mut value["l1_continuation_contract"]["document"];
        continuation["status"] = json!(CONTINUATION_INCOMPLETE_STATUS);
        continuation["prepared"] = json!(false);
        continuation.as_object_mut().unwrap().remove("seal_sha256");
        seal(continuation).unwrap();
        let continuation_copy = continuation.clone();
        value["l1_continuation_contract"]["document_sha256"] =
            json!(sha256_json(&continuation_copy).unwrap());
        value["l1_continuation_contract"]["document_seal_sha256"] =
            continuation_copy["seal_sha256"].clone();
        value.as_object_mut().unwrap().remove("seal_sha256");
        seal(&mut value).unwrap();
        let result = build_report(&value);
        assert_eq!(result["status"], EARNED_STATUS);
        assert_eq!(result["earned_l0_state_handoff_component"], true);
        assert_eq!(result["l1_continuation_prepared"], false);
        assert_eq!(result["l1_prefix_dispatches"], 0);
    }

    #[test]
    fn nested_release_lease_or_outer_binding_cannot_be_substituted() {
        let mut value = input();
        let release = &mut value["lease_release_receipt"]["document"];
        release["lease"]["lease_id"] = json!(sha('0'));
        release.as_object_mut().unwrap().remove("seal_sha256");
        seal(release).unwrap();
        let release_copy = release.clone();
        value["lease_release_receipt"]["document_sha256"] =
            json!(sha256_json(&release_copy).unwrap());
        value["lease_release_receipt"]["document_seal_sha256"] =
            release_copy["seal_sha256"].clone();
        value.as_object_mut().unwrap().remove("seal_sha256");
        seal(&mut value).unwrap();
        let result = build_report(&value);
        assert_eq!(result["status"], REFUSED_STATUS);
        assert!(result["blockers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value.as_str().unwrap().contains("terminal lease ID")));
    }

    #[test]
    fn release_that_predates_outer_terminal_or_lacks_recommendation_is_refused() {
        let mut value = input();
        let release = &mut value["lease_release_receipt"]["document"];
        release["recorded_at"] = json!("2026-08-09T08:19:36.000000Z");
        release.as_object_mut().unwrap().remove("seal_sha256");
        seal(release).unwrap();
        let release_copy = release.clone();
        value["lease_release_receipt"]["document_sha256"] =
            json!(sha256_json(&release_copy).unwrap());
        value["lease_release_receipt"]["document_seal_sha256"] =
            release_copy["seal_sha256"].clone();
        value.as_object_mut().unwrap().remove("seal_sha256");
        seal(&mut value).unwrap();
        let result = build_report(&value);
        assert_eq!(result["status"], REFUSED_STATUS);
        assert!(result["blockers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value.as_str().unwrap().contains("predates")));

        let mut missing = input();
        missing
            .as_object_mut()
            .unwrap()
            .remove("lease_release_recommendation_contract");
        missing.as_object_mut().unwrap().remove("seal_sha256");
        seal(&mut missing).unwrap();
        let result = build_report(&missing);
        assert_eq!(result["status"], REFUSED_STATUS);
        assert!(result["blockers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value
                .as_str()
                .unwrap()
                .contains("lease_release_recommendation_contract")));
    }

    #[test]
    fn l1_dispatch_claim_in_l0_receipt_is_refused() {
        let mut value = input();
        let inner = &mut value["l0_inner_receipt"]["document"];
        inner["l1_prefix_dispatches"] = json!(1);
        inner["l0_state_handoff"]["l1_prefix_dispatches"] = json!(1);
        inner.as_object_mut().unwrap().remove("seal_sha256");
        seal(inner).unwrap();
        let inner_copy = inner.clone();
        value["l0_inner_receipt"]["document_sha256"] = json!(sha256_json(&inner_copy).unwrap());
        value["l0_inner_receipt"]["document_seal_sha256"] = inner_copy["seal_sha256"].clone();
        value.as_object_mut().unwrap().remove("seal_sha256");
        seal(&mut value).unwrap();
        let result = build_report(&value);
        assert_eq!(result["status"], REFUSED_STATUS);
        assert!(result["blockers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value.as_str().unwrap().contains("Layer-1 dispatches")));
    }

    #[test]
    fn partial_or_aliased_handoff_is_refused() {
        let mut value = input();
        let inner = &mut value["l0_inner_receipt"]["document"];
        inner["l0_state_handoff"]["retained_l0_second_residual"]["bytes"] = json!(8196);
        inner["l0_state_handoff"]["layer1_input_binding"]["active_conv"]["device_buffer_id"] =
            json!(sha('b'));
        inner.as_object_mut().unwrap().remove("seal_sha256");
        seal(inner).unwrap();
        let inner_copy = inner.clone();
        value["l0_inner_receipt"]["document_sha256"] = json!(sha256_json(&inner_copy).unwrap());
        value["l0_inner_receipt"]["document_seal_sha256"] = inner_copy["seal_sha256"].clone();
        value.as_object_mut().unwrap().remove("seal_sha256");
        seal(&mut value).unwrap();
        let result = build_report(&value);
        assert_eq!(result["status"], REFUSED_STATUS);
        assert!(result["blockers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value.as_str().unwrap().contains("8192")));
    }

    #[test]
    fn outer_must_be_reaped_and_release_must_be_independent() {
        let mut value = input();
        let outer = &mut value["l0_outer_terminal"]["document"];
        outer["one_shot"]["outer_reaped_child"] = json!(false);
        outer.as_object_mut().unwrap().remove("seal_sha256");
        seal(outer).unwrap();
        let outer_copy = outer.clone();
        value["l0_outer_terminal"]["document_sha256"] = json!(sha256_json(&outer_copy).unwrap());
        value["l0_outer_terminal"]["document_seal_sha256"] = outer_copy["seal_sha256"].clone();
        value.as_object_mut().unwrap().remove("seal_sha256");
        seal(&mut value).unwrap();
        let result = build_report(&value);
        assert_eq!(result["status"], REFUSED_STATUS);
        assert!(result["blockers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value.as_str().unwrap().contains("outer_reaped_child")));
    }

    #[test]
    fn fixture_or_self_asserted_release_is_refused() {
        let mut value = input();
        let release = &mut value["lease_release_receipt"]["document"];
        release["evidence_kind"] = json!("synthetic_fixture");
        release.as_object_mut().unwrap().remove("seal_sha256");
        seal(release).unwrap();
        let release_copy = release.clone();
        value["lease_release_receipt"]["document_sha256"] =
            json!(sha256_json(&release_copy).unwrap());
        value["lease_release_receipt"]["document_seal_sha256"] =
            release_copy["seal_sha256"].clone();
        value.as_object_mut().unwrap().remove("seal_sha256");
        seal(&mut value).unwrap();
        let result = build_report(&value);
        assert_eq!(result["status"], REFUSED_STATUS);
        assert!(result["blockers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value.as_str().unwrap().contains("non-actual evidence")));
    }
}
