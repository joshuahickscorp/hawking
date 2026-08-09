//! CPU/static assessor for a future Qwen80 source-token Layer-1 completion.
//!
//! The earned L0(23)+L1-prefix(9) assessment is historical provenance only.
//! A positive result requires a fresh same-runtime L0(23)+L1-prefix(9)+
//! L1-MoE-suffix(14) trace. It never claims a token, decoder, server, TPS,
//! TG, or tournament result and has no GPU, process, lease, or artifact scan
//! surface.

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const INPUT_SCHEMA: &str = "hawking.ascension.qwen80_l1_full_layer_completion_assessor_input.v1";
const RESULT_SCHEMA: &str = "hawking.ascension.qwen80_l1_full_layer_completion_assessment.v1";
const EARNED_STATUS: &str =
    "EARNED_QWEN80_SOURCE_TOKEN_L1_COMPLETE_LAYER_COMPONENT_NOT_TOKEN_DECODER";
const REFUSED_STATUS: &str =
    "REFUSED_QWEN80_SOURCE_TOKEN_L1_FULL_LAYER_COMPLETION_INCOMPLETE_OR_UNTRUSTED";

const HISTORICAL_SCHEMA: &str = "hawking.ascension.qwen80_l0_l1_joint_post_capture_assessment.v1";
const HISTORICAL_STATUS: &str =
    "EARNED_QWEN80_SOURCE_TOKEN_L0_L1_COMPONENT_NOT_FULL_LAYER_TOKEN_DECODER";
const HISTORICAL_SEAL: &str = "d1b2893135287e282987e7d35609db3d44cd6c42846f79518f58f7ed5684829d";

const INNER_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_l1_full_layer_same_runtime_capture.v1";
const INNER_STATUS: &str =
    "CAPTURED_QWEN80_SOURCE_TOKEN_L0_REENCODE_AND_L1_COMPLETE_MOE_LAYER_SAME_RUNTIME_COMPONENT_ONLY";
const OUTER_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_l1_full_layer_same_runtime_outer_capture.v1";
const OUTER_STATUS: &str =
    "CAPTURED_QWEN80_SOURCE_TOKEN_L0_L1_COMPLETE_LAYER_SAME_RUNTIME_OUTER_TERMINAL_COMPONENT_ONLY";
const RELEASE_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_l1_full_layer_same_runtime_quiet_metal_lease_release.v1";
const RELEASE_STATUS: &str =
    "RELEASED_QWEN80_SOURCE_TOKEN_L0_L1_COMPLETE_LAYER_COMPONENT_QUIET_METAL_LEASE_AFTER_TERMINAL_CAPTURE";

const L0_DISPATCHES: u64 = 23;
const L1_PREFIX_DISPATCHES: u64 = 9;
const L1_SUFFIX_DISPATCHES: u64 = 14;
const TOTAL_DISPATCHES: u64 = 46;
const MAX_PARITY_ERROR: f64 = 1e-3;
const L1_LAYER: u64 = 1;
const L1_SLOT: u64 = 1;
const HIDDEN_ELEMENTS: u64 = 2_048;
const HIDDEN_BYTES: u64 = 8_192;
const L0_CONV_BYTES: u64 = 98_304;
const L0_RECURRENT_BYTES: u64 = 2_097_152;
const L1_CONV_CAPACITY: u64 = 196_608;
const L1_RECURRENT_CAPACITY: u64 = 4_194_304;

const L0_KERNELS: [&str; 23] = [
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
const L1_PREFIX: [&str; 9] = [
    "qwen_next_direct_packed_input_rmsnorm",
    "qwen_binary_sign_scale_matvec",
    "qwen_binary_sign_scale_matvec",
    "qwen_next_qkvz_rearrange_conv_l2",
    "qwen_next_ba_to_decay_beta",
    "qwen_next_gated_delta_decode_single",
    "qwen_next_deltanet_gated_rmsnorm",
    "qwen_binary_sign_scale_matvec",
    "qwen_next_add_residual",
];
const L1_SUFFIX: [&str; 14] = [
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
    Ok(sha256(rendered.as_bytes()))
}

fn obj<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))
}

fn obj_field<'a>(
    parent: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a Map<String, Value>, String> {
    parent
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{label}.{field} must be an object"))
}

fn array_field<'a>(
    parent: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a [Value], String> {
    parent
        .get(field)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| format!("{label}.{field} must be an array"))
}

fn text<'a>(parent: &'a Map<String, Value>, field: &str, label: &str) -> Result<&'a str, String> {
    parent
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{label}.{field} must be non-empty text"))
}

fn sha_field(parent: &Map<String, Value>, field: &str, label: &str) -> Result<String, String> {
    let value = text(parent, field, label)?;
    if !valid_sha(value) {
        return Err(format!("{label}.{field} must be a lowercase SHA-256"));
    }
    Ok(value.into())
}

fn number(parent: &Map<String, Value>, field: &str, label: &str) -> Result<u64, String> {
    parent
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{label}.{field} must be an unsigned integer"))
}

fn boolean(
    parent: &Map<String, Value>,
    field: &str,
    expected: bool,
    label: &str,
) -> Result<(), String> {
    if parent.get(field).and_then(Value::as_bool) != Some(expected) {
        return Err(format!("{label}.{field} must be {expected}"));
    }
    Ok(())
}

fn exact(
    parent: &Map<String, Value>,
    field: &str,
    expected: &str,
    label: &str,
) -> Result<(), String> {
    if text(parent, field, label)? != expected {
        return Err(format!("{label}.{field} drifted"));
    }
    Ok(())
}

fn seal(value: &mut Value) -> Result<String, String> {
    let root = obj(value, "output")?;
    if root.contains_key("seal_sha256") {
        return Err("output already contains seal_sha256".into());
    }
    let value_seal = json_sha(value)?;
    value
        .as_object_mut()
        .expect("checked object")
        .insert("seal_sha256".into(), Value::String(value_seal.clone()));
    Ok(value_seal)
}

fn verify_seal(value: &Value, label: &str) -> Result<String, String> {
    let root = obj(value, label)?;
    let value_seal = sha_field(root, "seal_sha256", label)?;
    let mut unsigned = root.clone();
    unsigned.remove("seal_sha256");
    if json_sha(&Value::Object(unsigned))? != value_seal {
        return Err(format!("{label} seal mismatch"));
    }
    Ok(value_seal)
}

fn bind(
    input: &Map<String, Value>,
    field: &str,
    schema: &str,
    status: &str,
) -> Result<Bound, String> {
    let wrapper = obj_field(input, field, "input")?;
    let document = wrapper
        .get("document")
        .cloned()
        .ok_or_else(|| format!("input.{field}.document is required"))?;
    let document_sha = sha_field(wrapper, "document_sha256", &format!("input.{field}"))?;
    let document_seal = sha_field(wrapper, "document_seal_sha256", &format!("input.{field}"))?;
    if json_sha(&document)? != document_sha {
        return Err(format!(
            "input.{field}.document_sha256 does not bind its document"
        ));
    }
    let observed_seal = verify_seal(&document, &format!("input.{field}.document"))?;
    if observed_seal != document_seal {
        return Err(format!(
            "input.{field}.document_seal_sha256 does not bind its document"
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

/// Receipt-internal provenance pointer check.
///
/// Producers record upstream sealed documents as seal-identity pointers
/// (`document_sha256` and `document_seal_sha256` both carry the seal — the
/// unsigned canonical hash). That is distinct from the assessor-input
/// *wrapper* convention, where `document_sha256` is `json_sha` of the full
/// sealed document. Mirror the L0 state-handoff assessor's proven semantics:
/// seal binding is mandatory; full-document identity is checked only when the
/// pointer actually carries it under `canonical_document_sha256` /
/// `bound_document_sha256`. Never treat receipt-internal `document_sha256` as
/// the wrapper-style full-document hash.
fn pointer(value: &Value, expected: &Bound, label: &str) -> Result<(), String> {
    let value = obj(value, label)?;
    boolean(value, "present", true, label)?;
    let observed_seal = value
        .get("document_seal_sha256")
        .or_else(|| value.get("seal_sha256"))
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label}.seal_sha256 is required"))?;
    if !valid_sha(observed_seal) {
        return Err(format!("{label}.seal_sha256 must be a lowercase SHA-256"));
    }
    if observed_seal != expected.seal_sha256 {
        return Err(format!(
            "{label} does not bind the expected sealed document"
        ));
    }
    if let Some(observed_document_sha) = value
        .get("canonical_document_sha256")
        .or_else(|| value.get("bound_document_sha256"))
        .and_then(Value::as_str)
    {
        if observed_document_sha != expected.document_sha256 {
            return Err(format!("{label} canonical document identity drifted"));
        }
    }
    Ok(())
}

fn forbidden(value: &Value) -> Option<&'static str> {
    const FIELDS: [&str; 11] = [
        "old_l0_receipt",
        "historical_l0_receipt",
        "prior_l0_receipt",
        "raw_pinned_buffer",
        "raw_l0_buffer",
        "raw_l1_prefix_buffer",
        "input_device_buffer_id",
        "input_f32le_sha256",
        "raw_dispatch_count",
        "detached_l0_execution_input",
        "historical_component_execution_input",
    ];
    match value {
        Value::Array(values) => values.iter().find_map(forbidden),
        Value::Object(values) => {
            for (key, value) in values {
                if let Some(found) = FIELDS.iter().find(|field| key == **field) {
                    return Some(*found);
                }
                if let Some(found) = forbidden(value) {
                    return Some(found);
                }
            }
            None
        }
        _ => None,
    }
}

fn require_parity(value: &Map<String, Value>, label: &str) -> Result<(), String> {
    boolean(value, "passed", true, label)?;
    sha_field(value, "cpu_f32le_sha256", label)?;
    sha_field(value, "device_f32le_sha256", label)?;
    let error = value
        .get("max_abs_error")
        .and_then(Value::as_f64)
        .filter(|error| error.is_finite() && *error >= 0.0)
        .ok_or_else(|| format!("{label}.max_abs_error must be finite and nonnegative"))?;
    if error > MAX_PARITY_ERROR {
        return Err(format!("{label}.max_abs_error exceeds component tolerance"));
    }
    Ok(())
}

fn require_state(
    parent: &Map<String, Value>,
    field: &str,
    offset: u64,
    capacity: u64,
    rollback: bool,
) -> Result<(), String> {
    let value = obj_field(parent, field, "L1 readbacks")?;
    let label = format!("L1 readbacks.{field}");
    boolean(value, "passed", true, &label)?;
    if number(value, "slot", &label)? != L1_SLOT
        || number(value, "offset_bytes", &label)? != offset
        || number(value, "capacity_bytes", &label)? != capacity
    {
        return Err(format!("{label} geometry drifted"));
    }
    sha_field(value, "state_identity_sha256", &label)?;
    sha_field(value, "device_buffer_identity_sha256", &label)?;
    sha_field(value, "f32le_sha256", &label)?;
    let error = value
        .get("max_abs_error")
        .and_then(Value::as_f64)
        .filter(|error| error.is_finite() && *error >= 0.0)
        .ok_or_else(|| format!("{label}.max_abs_error must be finite"))?;
    if rollback {
        boolean(value, "rollback_is_exact_zero", true, &label)?;
        if error != 0.0 {
            return Err(format!("{label}.max_abs_error must be exact zero"));
        }
    } else if error > MAX_PARITY_ERROR {
        return Err(format!("{label}.max_abs_error exceeds component tolerance"));
    }
    Ok(())
}

fn expected_kernels() -> Vec<&'static str> {
    L0_KERNELS
        .into_iter()
        .chain(L1_PREFIX)
        .chain(L1_SUFFIX)
        .collect()
}

fn validate_historical(bound: &Bound, pin_live: bool) -> Result<(), String> {
    if pin_live && bound.seal_sha256 != HISTORICAL_SEAL {
        return Err("historical assessment is not the earned d1b289 authority".into());
    }
    let root = obj(&bound.document, "historical assessment")?;
    boolean(root, "earned_component_only", true, "historical assessment")?;
    let scope = obj_field(root, "component_scope", "historical assessment")?;
    if number(scope, "fresh_l0_dispatches", "historical scope")? != L0_DISPATCHES
        || number(
            scope,
            "fresh_l1_slot1_prefix_dispatches",
            "historical scope",
        )? != L1_PREFIX_DISPATCHES
        || number(scope, "fresh_total_dispatches", "historical scope")? != 32
    {
        return Err("historical assessment 23+9 scope drifted".into());
    }
    boolean(
        scope,
        "full_layer_or_token_decoder_earned",
        false,
        "historical scope",
    )
}

fn validate_route_authority(root: &Map<String, Value>) -> Result<(), String> {
    let authority = obj_field(root, "l1_route_payload_authority", "inner")?;
    let guard = obj_field(authority, "route_guard", "route authority")?;
    boolean(guard, "passed", true, "route guard")?;
    if number(guard, "value", "route guard")? != 1 {
        return Err("route guard value must be one".into());
    }
    let expected_ids = array_field(guard, "expected_route_ids", "route guard")?;
    let observed_ids = array_field(guard, "observed_route_ids", "route guard")?;
    let expected_weights = array_field(guard, "expected_route_weights", "route guard")?;
    let observed_weights = array_field(guard, "observed_route_weights", "route guard")?;
    if expected_ids.len() != 10
        || observed_ids != expected_ids
        || expected_weights.len() != 10
        || observed_weights.len() != 10
    {
        return Err("route guard must bind exactly ten route IDs and weights".into());
    }
    let mut ids = BTreeSet::new();
    for value in expected_ids {
        ids.insert(value.as_u64().ok_or("route IDs must be unsigned")?);
    }
    if ids.len() != 10 {
        return Err("route guard IDs must be unique".into());
    }
    let weights_error = guard
        .get("weights_max_abs_error")
        .and_then(Value::as_f64)
        .filter(|error| error.is_finite() && *error >= 0.0)
        .ok_or("route guard weights error must be finite")?;
    if weights_error > MAX_PARITY_ERROR {
        return Err("route guard weights error exceeds tolerance".into());
    }
    let payloads = array_field(authority, "route_payloads", "route authority")?;
    if payloads.len() != 30 {
        return Err("route payload authority must contain exactly 30 payload identities".into());
    }
    for (index, payload) in payloads.iter().enumerate() {
        let payload = obj(payload, "route payload")?;
        let route = index / 3;
        let expected_kind = ["gate", "up", "down"][index % 3];
        if number(payload, "route_index", "route payload")? != route as u64
            || number(payload, "expert_id", "route payload")?
                != expected_ids[route]
                    .as_u64()
                    .ok_or("route ID must be unsigned")?
            || text(payload, "payload_kind", "route payload")? != expected_kind
        {
            return Err("route payload authority ordering drifted".into());
        }
        sha_field(payload, "payload_identity_sha256", "route payload")?;
        sha_field(payload, "tensor_sha256", "route payload")?;
    }
    Ok(())
}

fn validate_inner(inner: &Bound, historical: &Bound) -> Result<(), String> {
    if let Some(field) = forbidden(&inner.document) {
        return Err(format!("inner may not accept {field}"));
    }
    let root = obj(&inner.document, "inner")?;
    boolean(root, "fixture_or_synthetic", false, "inner")?;
    boolean(root, "self_asserted", false, "inner")?;
    pointer(
        root.get("historical_component_provenance")
            .ok_or("inner historical provenance is required")?,
        historical,
        "inner historical provenance",
    )?;
    boolean(root, "historical_provenance_only", true, "inner")?;
    boolean(
        root,
        "prior_process_or_buffer_reuse_accepted",
        false,
        "inner",
    )?;

    let execution = obj_field(root, "fresh_same_runtime_execution", "inner")?;
    for field in [
        "fresh_runtime",
        "fresh_session",
        "same_runtime",
        "same_tcb",
        "l0_reencoded_in_this_capture",
        "l1_prefix_and_moe_suffix_in_this_capture",
        "route_guard_enforced_before_l1_moe_suffix",
    ] {
        boolean(execution, field, true, "inner fresh execution")?;
    }
    if number(execution, "source_token_id", "inner fresh execution")? != 1
        || number(execution, "l0_dispatches", "inner fresh execution")? != L0_DISPATCHES
        || number(execution, "l1_prefix_dispatches", "inner fresh execution")?
            != L1_PREFIX_DISPATCHES
        || number(
            execution,
            "l1_moe_suffix_dispatches",
            "inner fresh execution",
        )? != L1_SUFFIX_DISPATCHES
        || number(execution, "total_dispatches", "inner fresh execution")? != TOTAL_DISPATCHES
        || number(execution, "fence_count", "inner fresh execution")? != 1
    {
        return Err("inner fresh execution must be exact 23+9+14=46 with one fence".into());
    }
    let runtime = sha_field(
        execution,
        "runtime_identity_sha256",
        "inner fresh execution",
    )?;
    let tcb = sha_field(execution, "tcb_identity_sha256", "inner fresh execution")?;

    let custody = obj_field(root, "opaque_same_runtime_continuation", "inner")?;
    for field in [
        "opaque",
        "same_runtime_state_arena_bound",
        "same_command_buffer_bound",
        "non_transferable_across_processes",
    ] {
        boolean(custody, field, true, "opaque custody")?;
    }
    boolean(
        custody,
        "raw_pinned_buffer_or_dispatch_count_input_accepted",
        false,
        "opaque custody",
    )?;
    if sha_field(custody, "runtime_identity_sha256", "opaque custody")? != runtime
        || sha_field(custody, "tcb_identity_sha256", "opaque custody")? != tcb
    {
        return Err("opaque custody does not bind fresh runtime/TCB identity".into());
    }

    let trace = obj_field(root, "structural_kernel_trace", "inner")?;
    boolean(trace, "non_timed", true, "inner trace")?;
    boolean(trace, "exact_order", true, "inner trace")?;
    let observed = array_field(trace, "kernel_names", "inner trace")?;
    let expected = expected_kernels();
    if observed.len() != expected.len()
        || observed
            .iter()
            .zip(expected)
            .any(|(value, expected)| value.as_str() != Some(expected))
    {
        return Err("inner trace is not exact fresh L0(23)+L1(9)+suffix(14)".into());
    }

    let fence = obj_field(root, "single_fence", "inner")?;
    for (field, expected) in [
        ("only_command_buffer_consumed", true),
        ("fence_succeeded", true),
        ("readbacks_after_fence", true),
        ("append_after_fence_possible", false),
    ] {
        boolean(fence, field, expected, "inner fence")?;
    }
    if number(fence, "fence_count", "inner fence")? != 1 {
        return Err("inner must have exactly one fence".into());
    }
    validate_route_authority(root)?;

    let readbacks = obj_field(root, "l1_completion_readbacks", "inner")?;
    if number(readbacks, "layer", "L1 readbacks")? != L1_LAYER
        || number(readbacks, "slot", "L1 readbacks")? != L1_SLOT
        || number(readbacks, "output_elements", "L1 readbacks")? != HIDDEN_ELEMENTS
        || number(readbacks, "output_bytes", "L1 readbacks")? != HIDDEN_BYTES
    {
        return Err("L1 readback geometry drifted".into());
    }
    for field in [
        "input",
        "prefix_first_residual",
        "postnorm",
        "router_logits",
        "shared_output",
        "routed_sum",
        "second_residual_output",
    ] {
        require_parity(
            obj_field(readbacks, field, "L1 readbacks")?,
            &format!("L1 readbacks.{field}"),
        )?;
    }
    require_state(
        readbacks,
        "active_conv",
        L0_CONV_BYTES,
        L1_CONV_CAPACITY,
        false,
    )?;
    require_state(
        readbacks,
        "active_recurrent",
        L0_RECURRENT_BYTES,
        L1_RECURRENT_CAPACITY,
        false,
    )?;
    require_state(
        readbacks,
        "rollback_conv",
        L0_CONV_BYTES,
        L1_CONV_CAPACITY,
        true,
    )?;
    require_state(
        readbacks,
        "rollback_recurrent",
        L0_RECURRENT_BYTES,
        L1_RECURRENT_CAPACITY,
        true,
    )?;

    let boundary = obj_field(root, "claim_boundary", "inner")?;
    boolean(
        boundary,
        "complete_l1_component_only",
        true,
        "inner boundary",
    )?;
    for field in [
        "token_generated",
        "decoder_started",
        "server_or_watcher_started",
        "tps_or_tg_measured",
        "tournament_started",
        "next_layer_executed",
    ] {
        boolean(boundary, field, false, "inner boundary")?;
    }
    Ok(())
}

fn validate_outer(outer: &Bound, inner: &Bound) -> Result<String, String> {
    let root = obj(&outer.document, "outer")?;
    boolean(root, "fixture_or_synthetic", false, "outer")?;
    boolean(root, "self_asserted", false, "outer")?;
    pointer(
        root.get("inner_capture")
            .ok_or("outer inner_capture is required")?,
        inner,
        "outer inner_capture",
    )?;
    let terminal = obj_field(root, "child_terminal", "outer")?;
    if number(terminal, "exit_code", "outer terminal")? != 0 {
        return Err("outer child did not exit zero".into());
    }
    for field in [
        "reaped",
        "terminal_receipt_written_last",
        "automatic_retry_disabled",
        "lease_reuse_prohibited",
    ] {
        boolean(terminal, field, true, "outer terminal")?;
    }
    boolean(terminal, "timed_out", false, "outer terminal")?;
    let boundary = obj_field(root, "claim_boundary", "outer")?;
    boolean(
        boundary,
        "complete_l1_component_only",
        true,
        "outer boundary",
    )?;
    for field in [
        "token_generated",
        "decoder_started",
        "server_or_watcher_started",
        "tps_or_tg_measured",
        "tournament_started",
    ] {
        boolean(boundary, field, false, "outer boundary")?;
    }
    sha_field(root, "lease_id", "outer")
}

fn validate_release(release: &Bound, outer: &Bound, lease_id: &str) -> Result<(), String> {
    let root = obj(&release.document, "release")?;
    pointer(
        root.get("outer_terminal")
            .ok_or("release outer_terminal is required")?,
        outer,
        "release outer_terminal",
    )?;
    if sha_field(root, "lease_id", "release")? != lease_id {
        return Err("release lease ID drifted".into());
    }
    for field in [
        "actual_release_performed",
        "released_after_outer_terminal",
        "lease_released",
        "automatic_retry_prohibited",
        "fresh_lease_required_for_any_future_gpu_work",
    ] {
        boolean(root, field, true, "release")?;
    }
    boolean(
        root,
        "watcher_restart_or_transition_authorized",
        false,
        "release",
    )
}

fn summary(bound: Option<&Bound>) -> Value {
    json!({
        "present": bound.is_some(),
        "document_sha256": bound.map(|value| value.document_sha256.clone()),
        "document_seal_sha256": bound.map(|value| value.seal_sha256.clone()),
    })
}

fn assess(input: &Value, pin_live_historical: bool) -> Value {
    let mut blockers = Vec::new();
    if obj(input, "input").is_err() {
        blockers.push("input must be an object".to_owned());
    }
    if input.get("schema").and_then(Value::as_str) != Some(INPUT_SCHEMA) {
        blockers.push(format!("input.schema must be {INPUT_SCHEMA}"));
    }
    if let Err(error) = verify_seal(input, "input") {
        blockers.push(error);
    }
    let root = input.as_object();
    let mut historical = None;
    let mut inner = None;
    let mut outer = None;
    let mut release = None;
    if let Some(root) = root {
        match bind(
            root,
            "historical_joint_assessment",
            HISTORICAL_SCHEMA,
            HISTORICAL_STATUS,
        ) {
            Ok(value) => match validate_historical(&value, pin_live_historical) {
                Ok(()) => historical = Some(value),
                Err(error) => blockers.push(format!("historical: {error}")),
            },
            Err(error) => blockers.push(format!("historical: {error}")),
        }
        match bind(root, "fresh_full_layer_inner", INNER_SCHEMA, INNER_STATUS) {
            Ok(value) => {
                if let Some(historical) = historical.as_ref() {
                    match validate_inner(&value, historical) {
                        Ok(()) => inner = Some(value),
                        Err(error) => blockers.push(format!("inner: {error}")),
                    }
                } else {
                    blockers.push("inner cannot be trusted without historical provenance".into());
                }
            }
            Err(error) => blockers.push(format!("inner: {error}")),
        }
        match bind(root, "fresh_full_layer_outer", OUTER_SCHEMA, OUTER_STATUS) {
            Ok(value) => {
                if let Some(inner) = inner.as_ref() {
                    match validate_outer(&value, inner) {
                        Ok(_) => outer = Some(value),
                        Err(error) => blockers.push(format!("outer: {error}")),
                    }
                } else {
                    blockers.push("outer cannot be trusted without valid fresh inner".into());
                }
            }
            Err(error) => blockers.push(format!("outer: {error}")),
        }
        match bind(
            root,
            "fresh_full_layer_release",
            RELEASE_SCHEMA,
            RELEASE_STATUS,
        ) {
            Ok(value) => {
                if let (Some(inner), Some(outer)) = (inner.as_ref(), outer.as_ref()) {
                    match validate_outer(outer, inner)
                        .and_then(|lease_id| validate_release(&value, outer, &lease_id))
                    {
                        Ok(()) => release = Some(value),
                        Err(error) => blockers.push(format!("release: {error}")),
                    }
                } else {
                    blockers.push("release cannot be trusted without valid outer".into());
                }
            }
            Err(error) => blockers.push(format!("release: {error}")),
        }
    }
    blockers.sort();
    blockers.dedup();
    let earned = blockers.is_empty();
    let mut output = json!({
        "schema": RESULT_SCHEMA,
        "status": if earned { EARNED_STATUS } else { REFUSED_STATUS },
        "earned_complete_l1_component_only": earned,
        "component_scope": {
            "historical_joint_assessment_seal_sha256": HISTORICAL_SEAL,
            "historical_assessment_is_provenance_only": true,
            "fresh_l0_dispatches": L0_DISPATCHES,
            "fresh_l1_prefix_dispatches": L1_PREFIX_DISPATCHES,
            "fresh_l1_moe_suffix_dispatches": L1_SUFFIX_DISPATCHES,
            "fresh_total_dispatches": TOTAL_DISPATCHES,
            "exact_route_payload_authority_count": 30,
            "token_decoder_server_tps_or_tournament_earned": false,
        },
        "sealed_inputs": {
            "historical_joint_assessment": summary(historical.as_ref()),
            "fresh_full_layer_inner": summary(inner.as_ref()),
            "fresh_full_layer_outer": summary(outer.as_ref()),
            "fresh_full_layer_release": summary(release.as_ref()),
        },
        "blockers": blockers,
        "authority_boundary": {
            "artifact_payload_or_directory_scan_performed": false,
            "gpu_or_metal_action_performed": false,
            "lease_or_process_action_performed": false,
            "server_or_watcher_action_performed": false,
            "tps_or_tg_measurement_performed": false,
            "tournament_action_performed": false,
        },
        "claim_boundary": {
            "complete_l1_component_only": true,
            "historical_component_is_not_execution_input": true,
            "cross_process_buffer_reuse_accepted": false,
            "token_generated": false,
            "decoder_started": false,
            "server_started": false,
            "tps_or_tg_measured": false,
            "tournament_started": false,
        },
    });
    seal(&mut output).expect("output must seal");
    output
}

fn parse_args<I>(arguments: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut values = BTreeMap::new();
    let mut it = arguments.into_iter();
    while let Some(flag) = it.next() {
        let value = it.next().ok_or("missing argument value")?;
        if !matches!(flag.as_str(), "--input" | "--out")
            || values.insert(flag.clone(), value).is_some()
        {
            return Err(
                "usage: assessor --input ABSOLUTE_SEALED_INPUT --out NEW_ABSOLUTE_OUTPUT".into(),
            );
        }
    }
    let input = values
        .remove("--input")
        .map(PathBuf::from)
        .ok_or("missing --input")?;
    let out = values
        .remove("--out")
        .map(PathBuf::from)
        .ok_or("missing --out")?;
    if !values.is_empty() || !input.is_absolute() || !out.is_absolute() {
        return Err("input and out must be absolute".into());
    }
    Ok(Args { input, out })
}

fn read_json(path: &Path) -> Result<Value, String> {
    let metadata =
        fs::symlink_metadata(path).map_err(|error| format!("cannot stat input: {error}"))?;
    if !path.is_absolute() || metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err("input must be an absolute regular non-symlink file".into());
    }
    let bytes = fs::read(path).map_err(|error| format!("cannot read input: {error}"))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("cannot parse input: {error}"))
}

fn write_new(path: &Path, value: &Value) -> Result<(), String> {
    if !path.is_absolute() || path.exists() || !path.parent().is_some_and(Path::is_dir) {
        return Err("out must be a new absolute path below an existing directory".into());
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create out: {error}"))?;
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?;
    file.write_all(&bytes)
        .and_then(|_| file.write_all(b"\n"))
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("cannot sync out: {error}"))
}

fn main() {
    match parse_args(env::args().skip(1)).and_then(|args| {
        let output = assess(&read_json(&args.input)?, true);
        let status = text(obj(&output, "output")?, "status", "output")?.to_owned();
        let output_seal = verify_seal(&output, "output")?;
        write_new(&args.out, &output)?;
        Ok((status, output_seal))
    }) {
        Ok((status, output_seal)) => {
            println!("{{\"status\":\"{status}\",\"seal_sha256\":\"{output_seal}\"}}")
        }
        Err(error) => {
            eprintln!("Q80 L1 full-layer completion assessor refused: {error}");
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

    /// Wrapper-style pointer: `document_sha256` = full sealed-document hash.
    /// Used by older synthetic fixtures; still valid because `pointer()` only
    /// mandatorily checks the seal field.
    fn reference(document: &Value) -> Value {
        json!({
            "present": true,
            "document_sha256": json_sha(document).unwrap(),
            "document_seal_sha256": document["seal_sha256"].clone(),
        })
    }

    /// Real producer/receipt-internal convention: both identity fields carry
    /// the seal (unsigned canonical hash). This is what the capture host and
    /// Python `_evidence()` emit.
    fn producer_reference(document: &Value) -> Value {
        json!({
            "present": true,
            "document_sha256": document["seal_sha256"].clone(),
            "document_seal_sha256": document["seal_sha256"].clone(),
        })
    }

    fn pair(a: char, b: char) -> Value {
        json!({
            "passed": true,
            "cpu_f32le_sha256": sha(a),
            "device_f32le_sha256": sha(b),
            "max_abs_error": 0.0000000317,
        })
    }

    fn state(marker: char, offset: u64, capacity: u64, rollback: bool) -> Value {
        json!({
            "passed": true,
            "slot": L1_SLOT,
            "offset_bytes": offset,
            "capacity_bytes": capacity,
            "state_identity_sha256": sha(marker),
            "device_buffer_identity_sha256": sha(marker),
            "f32le_sha256": sha(marker),
            "rollback_is_exact_zero": rollback,
            "max_abs_error": if rollback { 0.0 } else { 0.0000000317 },
        })
    }

    fn historical() -> Value {
        sealed(json!({
            "schema": HISTORICAL_SCHEMA,
            "status": HISTORICAL_STATUS,
            "earned_component_only": true,
            "component_scope": {
                "fresh_l0_dispatches": L0_DISPATCHES,
                "fresh_l1_slot1_prefix_dispatches": L1_PREFIX_DISPATCHES,
                "fresh_total_dispatches": 32,
                "full_layer_or_token_decoder_earned": false,
            },
        }))
    }

    fn route_authority() -> Value {
        let ids = (0..10).map(|value| json!(value)).collect::<Vec<_>>();
        let weights = (0..10)
            .map(|value| json!(value as f64 / 10.0))
            .collect::<Vec<_>>();
        let mut payloads = Vec::new();
        for route in 0..10 {
            for kind in ["gate", "up", "down"] {
                payloads.push(json!({
                    "route_index": route,
                    "expert_id": route,
                    "payload_kind": kind,
                    "payload_identity_sha256": sha('a'),
                    "tensor_sha256": sha('b'),
                }));
            }
        }
        json!({
            "route_guard": {
                "passed": true,
                "value": 1,
                "expected_route_ids": ids,
                "observed_route_ids": (0..10).map(|value| json!(value)).collect::<Vec<_>>(),
                "expected_route_weights": weights,
                "observed_route_weights": (0..10).map(|value| json!(value as f64 / 10.0)).collect::<Vec<_>>(),
                "weights_max_abs_error": 0.0000000317,
            },
            "route_payloads": payloads,
        })
    }

    fn inner_with(historical: &Value, reference_fn: fn(&Value) -> Value) -> Value {
        sealed(json!({
            "schema": INNER_SCHEMA,
            "status": INNER_STATUS,
            "fixture_or_synthetic": false,
            "self_asserted": false,
            "historical_component_provenance": reference_fn(historical),
            "historical_provenance_only": true,
            "prior_process_or_buffer_reuse_accepted": false,
            "fresh_same_runtime_execution": {
                "fresh_runtime": true,
                "fresh_session": true,
                "same_runtime": true,
                "same_tcb": true,
                "l0_reencoded_in_this_capture": true,
                "l1_prefix_and_moe_suffix_in_this_capture": true,
                "route_guard_enforced_before_l1_moe_suffix": true,
                "source_token_id": 1,
                "l0_dispatches": L0_DISPATCHES,
                "l1_prefix_dispatches": L1_PREFIX_DISPATCHES,
                "l1_moe_suffix_dispatches": L1_SUFFIX_DISPATCHES,
                "total_dispatches": TOTAL_DISPATCHES,
                "fence_count": 1,
                "runtime_identity_sha256": sha('c'),
                "tcb_identity_sha256": sha('d'),
            },
            "opaque_same_runtime_continuation": {
                "opaque": true,
                "same_runtime_state_arena_bound": true,
                "same_command_buffer_bound": true,
                "non_transferable_across_processes": true,
                "raw_pinned_buffer_or_dispatch_count_input_accepted": false,
                "runtime_identity_sha256": sha('c'),
                "tcb_identity_sha256": sha('d'),
            },
            "structural_kernel_trace": {
                "non_timed": true,
                "exact_order": true,
                "kernel_names": expected_kernels(),
            },
            "single_fence": {
                "only_command_buffer_consumed": true,
                "fence_succeeded": true,
                "readbacks_after_fence": true,
                "append_after_fence_possible": false,
                "fence_count": 1,
            },
            "l1_route_payload_authority": route_authority(),
            "l1_completion_readbacks": {
                "layer": L1_LAYER,
                "slot": L1_SLOT,
                "output_elements": HIDDEN_ELEMENTS,
                "output_bytes": HIDDEN_BYTES,
                "input": pair('a', 'b'),
                "prefix_first_residual": pair('b', 'c'),
                "postnorm": pair('c', 'd'),
                "router_logits": pair('d', 'e'),
                "shared_output": pair('e', 'f'),
                "routed_sum": pair('f', 'a'),
                "second_residual_output": pair('a', 'c'),
                "active_conv": state('a', L0_CONV_BYTES, L1_CONV_CAPACITY, false),
                "active_recurrent": state('b', L0_RECURRENT_BYTES, L1_RECURRENT_CAPACITY, false),
                "rollback_conv": state('c', L0_CONV_BYTES, L1_CONV_CAPACITY, true),
                "rollback_recurrent": state('d', L0_RECURRENT_BYTES, L1_RECURRENT_CAPACITY, true),
            },
            "claim_boundary": {
                "complete_l1_component_only": true,
                "token_generated": false,
                "decoder_started": false,
                "server_or_watcher_started": false,
                "tps_or_tg_measured": false,
                "tournament_started": false,
                "next_layer_executed": false,
            },
        }))
    }

    fn outer_with(inner: &Value, reference_fn: fn(&Value) -> Value) -> Value {
        sealed(json!({
            "schema": OUTER_SCHEMA,
            "status": OUTER_STATUS,
            "fixture_or_synthetic": false,
            "self_asserted": false,
            "inner_capture": reference_fn(inner),
            "lease_id": sha('e'),
            "child_terminal": {
                "exit_code": 0,
                "reaped": true,
                "terminal_receipt_written_last": true,
                "automatic_retry_disabled": true,
                "lease_reuse_prohibited": true,
                "timed_out": false,
            },
            "claim_boundary": {
                "complete_l1_component_only": true,
                "token_generated": false,
                "decoder_started": false,
                "server_or_watcher_started": false,
                "tps_or_tg_measured": false,
                "tournament_started": false,
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
            "lease_released": true,
            "automatic_retry_prohibited": true,
            "fresh_lease_required_for_any_future_gpu_work": true,
            "watcher_restart_or_transition_authorized": false,
        }))
    }

    fn outer(inner: &Value) -> Value {
        outer_with(inner, reference)
    }

    fn release(outer: &Value) -> Value {
        release_with(outer, reference)
    }

    fn input() -> Value {
        input_with_references(reference)
    }

    fn input_with_references(reference_fn: fn(&Value) -> Value) -> Value {
        let historical = historical();
        let inner = inner_with(&historical, reference_fn);
        let outer = outer_with(&inner, reference_fn);
        let release = release_with(&outer, reference_fn);
        sealed(json!({
            "schema": INPUT_SCHEMA,
            "historical_joint_assessment": wrapper(historical),
            "fresh_full_layer_inner": wrapper(inner),
            "fresh_full_layer_outer": wrapper(outer),
            "fresh_full_layer_release": wrapper(release),
        }))
    }

    #[test]
    fn exact_fresh_trace_route_payloads_and_lifecycle_earn_component_only_under_fixture_policy() {
        let report = assess(&input(), false);
        assert_eq!(report["status"], EARNED_STATUS);
        assert_eq!(report["earned_complete_l1_component_only"], true);
        assert_eq!(report["component_scope"]["fresh_total_dispatches"], 46);
        assert_eq!(
            report["component_scope"]["exact_route_payload_authority_count"],
            30
        );
        assert_eq!(report["claim_boundary"]["decoder_started"], false);
        verify_seal(&report, "report").unwrap();

        let production = assess(&input(), true);
        assert_eq!(production["status"], REFUSED_STATUS);
        assert!(production["blockers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value.as_str().unwrap().contains("d1b289")));
    }

    #[test]
    fn rejects_missing_suffix_or_nonzero_rollback() {
        let mut value = input();
        let inner = value["fresh_full_layer_inner"]["document"]
            .as_object_mut()
            .unwrap();
        inner["fresh_same_runtime_execution"]["total_dispatches"] = json!(32);
        let inner = reseal(Value::Object(inner.clone()));
        value["fresh_full_layer_inner"] = wrapper(inner.clone());
        let first_outer = outer(&inner);
        let first_release = release(&first_outer);
        value["fresh_full_layer_outer"] = wrapper(first_outer);
        value["fresh_full_layer_release"] = wrapper(first_release);
        value = reseal(value);
        let report = assess(&value, false);
        assert_eq!(report["status"], REFUSED_STATUS);
        let blockers = report["blockers"].as_array().unwrap();
        assert!(blockers
            .iter()
            .any(|value| value.as_str().unwrap().contains("23+9+14")));

        let mut value = input();
        let inner = value["fresh_full_layer_inner"]["document"]
            .as_object_mut()
            .unwrap();
        inner["l1_completion_readbacks"]["rollback_conv"]["max_abs_error"] = json!(0.00001);
        let inner = reseal(Value::Object(inner.clone()));
        value["fresh_full_layer_inner"] = wrapper(inner.clone());
        let outer = outer(&inner);
        let release = release(&outer);
        value["fresh_full_layer_outer"] = wrapper(outer);
        value["fresh_full_layer_release"] = wrapper(release);
        value = reseal(value);
        let report = assess(&value, false);
        assert_eq!(report["status"], REFUSED_STATUS);
        let blockers = report["blockers"].as_array().unwrap();
        assert!(blockers
            .iter()
            .any(|value| value.as_str().unwrap().contains("exact zero")));
    }

    #[test]
    fn rejects_route_payload_drift_and_historical_buffer_reuse() {
        let mut value = input();
        let inner = value["fresh_full_layer_inner"]["document"]
            .as_object_mut()
            .unwrap();
        inner["l1_route_payload_authority"]["route_payloads"][29]["payload_kind"] = json!("gate");
        inner.insert("historical_l0_receipt".to_owned(), json!(sha('f')));
        let inner = reseal(Value::Object(inner.clone()));
        value["fresh_full_layer_inner"] = wrapper(inner.clone());
        let outer = outer(&inner);
        let release = release(&outer);
        value["fresh_full_layer_outer"] = wrapper(outer);
        value["fresh_full_layer_release"] = wrapper(release);
        value = reseal(value);
        let report = assess(&value, false);
        assert_eq!(report["status"], REFUSED_STATUS);
        assert!(report["blockers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value.as_str().unwrap().contains("historical_l0_receipt")));
    }

    #[test]
    fn accepts_real_producer_seal_only_provenance_pointers() {
        let report = assess(&input_with_references(producer_reference), false);
        assert_eq!(report["status"], EARNED_STATUS);
        assert_eq!(report["earned_complete_l1_component_only"], true);
        assert!(report["blockers"].as_array().unwrap().is_empty());
        verify_seal(&report, "report").unwrap();
    }

    #[test]
    fn rejects_producer_pointer_with_wrong_seal() {
        let mut value = input_with_references(producer_reference);
        let inner = value["fresh_full_layer_inner"]["document"]
            .as_object_mut()
            .unwrap();
        // Valid hex shape, but not the historical seal.
        inner["historical_component_provenance"]["document_seal_sha256"] = json!(sha('f'));
        // Keep document_sha256 on the same wrong-seal path producers use: both fields equal.
        inner["historical_component_provenance"]["document_sha256"] = json!(sha('f'));
        let inner = reseal(Value::Object(inner.clone()));
        value["fresh_full_layer_inner"] = wrapper(inner.clone());
        let outer = outer_with(&inner, producer_reference);
        let release = release_with(&outer, producer_reference);
        value["fresh_full_layer_outer"] = wrapper(outer);
        value["fresh_full_layer_release"] = wrapper(release);
        value = reseal(value);
        let report = assess(&value, false);
        assert_eq!(report["status"], REFUSED_STATUS);
        assert!(report["blockers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| {
                value
                    .as_str()
                    .unwrap()
                    .contains("does not bind the expected sealed document")
            }));
    }

    #[test]
    fn rejects_producer_pointer_with_wrong_canonical_document_sha() {
        let mut value = input_with_references(producer_reference);
        let historical = value["historical_joint_assessment"]["document"].clone();
        let correct_full = json_sha(&historical).unwrap();
        assert_ne!(correct_full, historical["seal_sha256"].as_str().unwrap());

        let inner = value["fresh_full_layer_inner"]["document"]
            .as_object_mut()
            .unwrap();
        // Seal still correct; optional full-document identity is present and wrong.
        inner["historical_component_provenance"]["canonical_document_sha256"] = json!(sha('e'));
        let inner = reseal(Value::Object(inner.clone()));
        value["fresh_full_layer_inner"] = wrapper(inner.clone());
        let outer = outer_with(&inner, producer_reference);
        let release = release_with(&outer, producer_reference);
        value["fresh_full_layer_outer"] = wrapper(outer);
        value["fresh_full_layer_release"] = wrapper(release);
        value = reseal(value);
        let report = assess(&value, false);
        assert_eq!(report["status"], REFUSED_STATUS);
        assert!(report["blockers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| {
                value
                    .as_str()
                    .unwrap()
                    .contains("canonical document identity drifted")
            }));
    }

    #[test]
    fn rejects_tampered_historical_with_resealed_different_content() {
        // Build a producer-shaped chain, then replace the historical document
        // with different content resealed under a new seal while leaving the
        // receipt pointer bound to the original seal. Tamper via an extra
        // field so validate_historical still accepts the document body — only
        // the seal-binding pointer must refuse.
        let original_historical = historical();
        let original_seal = original_historical["seal_sha256"].as_str().unwrap().to_owned();
        let mut value = input_with_references(producer_reference);

        let mut tampered = original_historical.clone();
        tampered
            .as_object_mut()
            .unwrap()
            .insert("tamper_marker".into(), json!("different content"));
        let tampered = reseal(tampered);
        let tampered_seal = tampered["seal_sha256"].as_str().unwrap().to_owned();
        assert_ne!(original_seal, tampered_seal);

        value["historical_joint_assessment"] = wrapper(tampered);
        value = reseal(value);

        let report = assess(&value, false);
        assert_eq!(report["status"], REFUSED_STATUS);
        assert!(report["blockers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| {
                value
                    .as_str()
                    .unwrap()
                    .contains("does not bind the expected sealed document")
            }));
    }
}
