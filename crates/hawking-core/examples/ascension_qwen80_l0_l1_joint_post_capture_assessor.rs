//! CPU-only assessor for one future Qwen80 same-runtime L0→L1 component capture.
//!
//! This example validates sealed receipt data only.  It neither creates a
//! Metal context nor launches a process, acquires/releases a lease, opens a
//! model artifact, starts a server/watcher, or measures TPS/TG.  A positive
//! result is deliberately bounded to the fresh L0 true-MoE body plus Layer-1
//! DeltaNet prefix: it is never a complete Layer-1, token, decoder, or
//! tournament result.

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const INPUT_SCHEMA: &str = "hawking.ascension.qwen80_l0_l1_joint_post_capture_assessor_input.v1";
const RESULT_SCHEMA: &str = "hawking.ascension.qwen80_l0_l1_joint_post_capture_assessment.v1";
const EARNED_STATUS: &str =
    "EARNED_QWEN80_SOURCE_TOKEN_L0_L1_COMPONENT_NOT_FULL_LAYER_TOKEN_DECODER";
const REFUSED_STATUS: &str =
    "REFUSED_QWEN80_SOURCE_TOKEN_L0_L1_JOINT_POST_CAPTURE_INCOMPLETE_OR_UNTRUSTED";

const SCHEDULE_SCHEMA: &str = "hawking.ascension.qwen80_48_layer_schedule_sealed_wrapper.v1";
const SCHEDULE_STATUS: &str =
    "SEALED_QWEN80_48_LAYER_PAYLOAD_SCHEDULE_AUTHORITY_BOUND_NOT_EXECUTED";
const CONTINUATION_SCHEMA: &str =
    "hawking.ascension.qwen80_l1_source_token_continuation_readiness_contract.v1";
const CONTINUATION_STATUS: &str =
    "PREPARED_QWEN80_SOURCE_TOKEN_L1_SLOT1_DELTANET_PREFIX_CAPTURE_RESERVED_NOT_EXECUTED";
const BINDING_SCHEMA: &str =
    "hawking.ascension.qwen80_l0_state_handoff_post_capture_assessor_binding.v1";
const BINDING_STATUS: &str =
    "REQUIRED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_POST_CAPTURE_ASSESSMENT_BEFORE_L1_JOINT_CAPTURE";
const INNER_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_l1_same_runtime_prefix_capture.v1";
const INNER_STATUS: &str =
    "CAPTURED_QWEN80_SOURCE_TOKEN_L0_REENCODE_AND_L1_SLOT1_PREFIX_SAME_RUNTIME_COMPONENT_ONLY";
const OUTER_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_l1_same_runtime_prefix_outer_capture.v1";
const OUTER_STATUS: &str =
    "CAPTURED_QWEN80_SOURCE_TOKEN_L0_L1_SAME_RUNTIME_PREFIX_OUTER_TERMINAL_COMPONENT_ONLY";
const RELEASE_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_l1_same_runtime_prefix_quiet_metal_lease_release.v1";
const RELEASE_STATUS: &str =
    "RELEASED_QWEN80_SOURCE_TOKEN_L0_L1_SAME_RUNTIME_PREFIX_COMPONENT_QUIET_METAL_LEASE_AFTER_TERMINAL_CAPTURE";

const SCHEDULE_DOCUMENT_SHA256: &str =
    "4ab2c83d9df8573d95a65a1498ddd45dc6d812d47c123700981ffd87be1e5e39";
const SCHEDULE_SEAL_SHA256: &str =
    "5ed1367c4e5e0680967dde35ae583aec09f4417b5ecbae14d0d54184cd4a8554";
const CONTINUATION_DOCUMENT_SHA256: &str =
    "cad60fdb800b5ba2f4202d3c61095ee501e17f79f19abefd87c1e66c83e7cb7e";
const CONTINUATION_SEAL_SHA256: &str =
    "fdb3ec5a9ca400517ab958ef1a26ce56567c00ed9705eff30ecf725b98187f93";
const BINDING_DOCUMENT_SHA256: &str =
    "75444bee046d533ff6eefc3ed7df8b41e5b359e552ec9e17e98e40ad7a1cbc1c";
const BINDING_SEAL_SHA256: &str =
    "523518a8fdc3d43416dfdfdf62488da9032275b41b8fe2e71c6c8cc29e27e403";

const CAPABILITY_FACTORY: &str =
    "Qwen80CompleteNativeRuntime::certify_source_token_l0_true_moe_continuation";
const L1_ENCODER: &str = "Qwen80CompleteNativeRuntime::encode_source_token_l1_deltanet_prefix_from_canonical_l0_continuation_into";
const FINALIZER: &str =
    "Qwen80SameRuntimeLayer1DeltaNetPrefixEncoder::finalize_after_exact_joint_fence";

const SOURCE_TOKEN_ID: u64 = 1;
const L0_DISPATCHES: u64 = 23;
const L1_PREFIX_DISPATCHES: u64 = 9;
const TOTAL_DISPATCHES: u64 = 32;
const HIDDEN_ELEMENTS: u64 = 2_048;
const HIDDEN_BYTES: u64 = 8_192;
const L1_LAYER: u64 = 1;
const L1_SLOT: u64 = 1;
const L0_CONV_BYTES: u64 = 98_304;
const L0_RECURRENT_BYTES: u64 = 2_097_152;
const L1_CONV_CAPACITY_BYTES: u64 = 196_608;
const L1_RECURRENT_CAPACITY_BYTES: u64 = 4_194_304;
const MAX_PARITY_ERROR: f64 = 1.0e-3;

const STRUCTURAL_KERNELS: [&str; 32] = [
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

#[derive(Clone, Debug)]
struct Args {
    input: PathBuf,
    out: PathBuf,
}

#[derive(Clone, Debug)]
struct BoundDocument {
    document: Value,
    document_sha256: String,
    document_seal_sha256: String,
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

/// Match the sorted, compact Python receipt canonicalization for real receipt
/// seals, including its exponent spelling for finite floating values.
fn python_json_float(number: &serde_json::Number) -> Result<String, String> {
    let value = number
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or("canonical JSON floating number must be finite")?;
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
                .map_err(|error| format!("invalid canonical exponent: {error}"))?,
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
                    fractional = fractional
                        .checked_add(1)
                        .ok_or("fractional length overflow")?;
                }
                digits.push(char::from(byte));
            }
            _ => return Err(format!("invalid canonical float mantissa {raw:?}")),
        }
    }
    let first = digits
        .bytes()
        .position(|byte| byte != b'0')
        .ok_or("nonzero float has no significant digit")?;
    let mut significant = digits[first..].to_owned();
    let mut decimal_power = exponent
        .checked_sub(fractional)
        .ok_or("decimal exponent overflow")?;
    while significant.len() > 1 && significant.ends_with('0') {
        significant.pop();
        decimal_power = decimal_power
            .checked_add(1)
            .ok_or("decimal exponent overflow")?;
    }
    let scientific = decimal_power
        .checked_add(i32::try_from(significant.len() - 1).unwrap_or(i32::MAX))
        .ok_or("decimal exponent overflow")?;
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
        let position = usize::try_from(position).map_err(|_| "decimal position is negative")?;
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
                        .map_err(|error| format!("cannot canonicalize JSON key: {error}"))?,
                );
                output.push(':');
                canonical_json_into(value, output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

fn sha256_json(value: &Value) -> Result<String, String> {
    let mut canonical = String::new();
    canonical_json_into(value, &mut canonical)?;
    Ok(sha256_hex(canonical.as_bytes()))
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
    if require_string(object, field, label)? != expected {
        return Err(format!("{label}.{field} drifted"));
    }
    Ok(())
}

fn seal(value: &mut Value) -> Result<String, String> {
    let root = object(value, "output")?;
    if root.contains_key("seal_sha256") {
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
    let root = object(value, label)?;
    let observed = require_sha(root, "seal_sha256", label)?;
    let mut unsigned = root.clone();
    unsigned.remove("seal_sha256");
    if sha256_json(&Value::Object(unsigned))? != observed {
        return Err(format!("{label} seal mismatch"));
    }
    Ok(observed)
}

fn parse_bound(
    input: &Map<String, Value>,
    field: &str,
    schema: &str,
    status: &str,
    exact_document_sha256: Option<&str>,
    exact_seal_sha256: Option<&str>,
) -> Result<BoundDocument, String> {
    let binding = object_field(input, field, "input")?;
    let document = binding
        .get("document")
        .cloned()
        .ok_or_else(|| format!("input.{field}.document is required"))?;
    let root = object(&document, &format!("input.{field}.document"))?;
    let document_sha256 = require_sha(binding, "document_sha256", &format!("input.{field}"))?;
    let document_seal_sha256 =
        require_sha(binding, "document_seal_sha256", &format!("input.{field}"))?;
    let actual_document_sha256 = sha256_json(&document)?;
    if actual_document_sha256 != document_sha256 {
        return Err(format!(
            "input.{field}.document_sha256 does not bind document"
        ));
    }
    let actual_seal = verify_seal(&document, &format!("input.{field}.document"))?;
    if actual_seal != document_seal_sha256 {
        return Err(format!(
            "input.{field}.document_seal_sha256 does not bind document"
        ));
    }
    require_exact_string(root, "schema", schema, &format!("input.{field}.document"))?;
    require_exact_string(root, "status", status, &format!("input.{field}.document"))?;
    if let Some(expected) = exact_document_sha256 {
        if actual_document_sha256 != expected {
            return Err(format!(
                "input.{field} is not the exact immutable authority"
            ));
        }
    }
    if let Some(expected) = exact_seal_sha256 {
        if actual_seal != expected {
            return Err(format!(
                "input.{field} seal is not the exact immutable authority"
            ));
        }
    }
    Ok(BoundDocument {
        document,
        document_sha256: actual_document_sha256,
        document_seal_sha256: actual_seal,
    })
}

fn require_bound_identity(
    reference: &Value,
    expected: &BoundDocument,
    label: &str,
) -> Result<(), String> {
    let value = object(reference, label)?;
    require_bool(value, "present", true, label)?;
    if require_sha(value, "document_sha256", label)? != expected.document_sha256 {
        return Err(format!(
            "{label}.document_sha256 does not bind supplied authority"
        ));
    }
    let seal = value
        .get("document_seal_sha256")
        .or_else(|| value.get("seal_sha256"))
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label}.document_seal_sha256 is required"))?;
    if seal != expected.document_seal_sha256 {
        return Err(format!(
            "{label}.document_seal_sha256 does not bind supplied authority"
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ParityHashRequirement {
    SingleF32leHash,
    RecordedCpuDevicePair,
}

fn require_nonnegative_parity(
    object: &Map<String, Value>,
    label: &str,
    hash_requirement: ParityHashRequirement,
) -> Result<(), String> {
    require_bool(object, "passed", true, label)?;
    let error = object
        .get("max_abs_error")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite() && *value >= 0.0)
        .ok_or_else(|| format!("{label}.max_abs_error must be a finite non-negative number"))?;
    if error > MAX_PARITY_ERROR {
        return Err(format!(
            "{label}.max_abs_error exceeds strict component tolerance"
        ));
    }
    match hash_requirement {
        ParityHashRequirement::SingleF32leHash => {
            require_sha(object, "f32le_sha256", label)?;
        }
        ParityHashRequirement::RecordedCpuDevicePair => {
            require_sha(object, "cpu_f32le_sha256", label)?;
            require_sha(object, "device_f32le_sha256", label)?;
        }
    }
    Ok(())
}

fn require_state_readback(
    object: &Map<String, Value>,
    label: &str,
    expected_slot: u64,
    expected_offset_bytes: u64,
    expected_capacity_bytes: u64,
) -> Result<(), String> {
    require_bool(object, "passed", true, label)?;
    if require_u64(object, "slot", label)? != expected_slot
        || require_u64(object, "offset_bytes", label)? != expected_offset_bytes
        || require_u64(object, "capacity_bytes", label)? != expected_capacity_bytes
    {
        return Err(format!("{label} state geometry drifted"));
    }
    require_sha(object, "device_buffer_identity_sha256", label)?;
    require_sha(object, "f32le_sha256", label)?;
    let error = object
        .get("max_abs_error")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite() && *value >= 0.0)
        .ok_or_else(|| format!("{label}.max_abs_error must be finite and non-negative"))?;
    if error > MAX_PARITY_ERROR {
        return Err(format!(
            "{label}.max_abs_error exceeds strict component tolerance"
        ));
    }
    Ok(())
}

fn no_forbidden_execution_input(object: &Map<String, Value>, label: &str) -> Result<(), String> {
    for field in [
        "old_l0_receipt",
        "historical_l0_receipt",
        "prior_l0_receipt",
        "input_device_buffer_id",
        "input_f32le_sha256",
        "raw_pinned_buffer",
        "raw_l0_buffer",
        "raw_dispatch_count",
        "preceding_l0_dispatch_count",
        "detached_l0_execution_input",
    ] {
        if object.contains_key(field) {
            return Err(format!("{label} may not accept {field} as execution input"));
        }
    }
    Ok(())
}

fn validate_schedule(schedule: &BoundDocument) -> Result<(), String> {
    let root = object(&schedule.document, "schedule wrapper")?;
    let raw = object_field(root, "raw_schedule_authority", "schedule wrapper")?;
    if require_sha(raw, "sha256", "schedule wrapper.raw_schedule_authority")?
        != "8302deb6beece8c04773ece19ae27baea67749014552b0b946516146b5e2282e"
        || require_u64(raw, "bytes", "schedule wrapper.raw_schedule_authority")? != 88_551_859
    {
        return Err("schedule wrapper does not pin the canonical raw 48-layer plan".into());
    }
    require_bool(
        raw,
        "raw_schedule_is_static_and_unmodified",
        true,
        "schedule wrapper.raw_schedule_authority",
    )?;
    let facts = object_field(root, "schedule_facts", "schedule wrapper")?;
    for (field, expected) in [
        ("layer_count", 48),
        ("delta_net_layer_count", 36),
        ("gqa_layer_count", 12),
        ("delta_net_state_slot_count", 36),
        ("gqa_state_slot_count", 12),
    ] {
        if require_u64(facts, field, "schedule wrapper.schedule_facts")? != expected {
            return Err(format!("schedule wrapper {field} drifted"));
        }
    }
    let layer1 = object_field(facts, "layer_1", "schedule wrapper.schedule_facts")?;
    if require_u64(layer1, "layer", "schedule wrapper.schedule_facts.layer_1")? != L1_LAYER
        || require_u64(
            layer1,
            "state_slot",
            "schedule wrapper.schedule_facts.layer_1",
        )? != L1_SLOT
        || require_string(layer1, "mixer", "schedule wrapper.schedule_facts.layer_1")?
            != "delta_net"
    {
        return Err("schedule wrapper Layer-1 DeltaNet slot drifted".into());
    }
    let boundary = object_field(root, "claim_boundary", "schedule wrapper")?;
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
        require_bool(boundary, field, false, "schedule wrapper.claim_boundary")?;
    }
    Ok(())
}

fn validate_continuation(continuation: &BoundDocument) -> Result<(), String> {
    let root = object(&continuation.document, "continuation")?;
    require_bool(root, "prepared", true, "continuation")?;
    require_bool(
        root,
        "l1_execution_performed_by_this_contract",
        false,
        "continuation",
    )?;
    if require_u64(
        root,
        "l1_prefix_dispatches_executed_by_this_contract",
        "continuation",
    )? != 0
    {
        return Err("continuation may not claim an executed L1 prefix".into());
    }
    let scope = object_field(
        root,
        "future_l1_slot1_deltanet_prefix_scope",
        "continuation",
    )?;
    for (field, expected) in [
        ("fresh_joint_l0_component_dispatch_count", L0_DISPATCHES),
        (
            "fresh_joint_l1_slot1_prefix_dispatch_count",
            L1_PREFIX_DISPATCHES,
        ),
        ("fresh_joint_total_dispatch_count", TOTAL_DISPATCHES),
        ("exact_prefix_dispatch_count", L1_PREFIX_DISPATCHES),
    ] {
        if require_u64(
            scope,
            field,
            "continuation.future_l1_slot1_deltanet_prefix_scope",
        )? != expected
        {
            return Err(format!("continuation {field} drifted"));
        }
    }
    require_bool(
        scope,
        "fresh_same_runtime_same_tcb_joint_l0_to_l1_capture_required",
        true,
        "continuation.future_l1_slot1_deltanet_prefix_scope",
    )?;
    require_bool(
        scope,
        "cross_process_or_prior_capture_pinned_buffer_reuse_authorized",
        false,
        "continuation.future_l1_slot1_deltanet_prefix_scope",
    )?;
    Ok(())
}

fn validate_assessor_binding(binding: &BoundDocument) -> Result<(), String> {
    let root = object(&binding.document, "assessor binding")?;
    require_bool(root, "assessment_result_bound", true, "assessor binding")?;
    require_bool(
        root,
        "assessment_required_before_joint_child_launch",
        true,
        "assessor binding",
    )?;
    require_bool(
        root,
        "baseline_l0_evidence_is_provenance_only",
        true,
        "assessor binding",
    )?;
    require_bool(
        root,
        "cross_process_pinned_buffer_transfer_allowed",
        false,
        "assessor binding",
    )?;
    require_bool(root, "joint_l0_reencode_required", true, "assessor binding")?;
    let assessment = object_field(root, "post_capture_assessment", "assessor binding")?;
    if require_sha(
        assessment,
        "document_seal_sha256",
        "assessor binding.post_capture_assessment",
    )? != "23b6021b8403b9403a9b11044d43b2ba712fbcb2b99c431936d93b16e75ddba5"
    {
        return Err("assessor binding does not retain the earned L0 assessment".into());
    }
    let facts = object_field(root, "retained_l0_state_handoff", "assessor binding")?;
    if require_u64(
        facts,
        "source_token_id",
        "assessor binding.retained_l0_state_handoff",
    )? != SOURCE_TOKEN_ID
    {
        return Err("assessor binding historical L0 boundary drifted".into());
    }
    require_bool(
        facts,
        "l1_binding_not_executed",
        true,
        "assessor binding.retained_l0_state_handoff",
    )?;
    if require_u64(
        facts,
        "l1_prefix_dispatches",
        "assessor binding.retained_l0_state_handoff",
    )? != 0
    {
        return Err("assessor binding historical L0 boundary drifted".into());
    }
    Ok(())
}

fn validate_inner(
    inner: &BoundDocument,
    schedule: &BoundDocument,
    continuation: &BoundDocument,
    binding: &BoundDocument,
) -> Result<(), String> {
    let root = object(&inner.document, "joint inner capture")?;
    no_forbidden_execution_input(root, "joint inner capture")?;
    require_bool(root, "fixture_or_synthetic", false, "joint inner capture")?;
    require_bool(root, "self_asserted", false, "joint inner capture")?;
    let issuer = object_field(root, "issuer", "joint inner capture")?;
    require_exact_string(
        issuer,
        "role",
        "joint_component_capture_child",
        "joint inner capture.issuer",
    )?;
    require_sha(
        issuer,
        "issuer_identity_sha256",
        "joint inner capture.issuer",
    )?;
    let authorities = object_field(root, "upstream_authorities", "joint inner capture")?;
    require_bound_identity(
        authorities
            .get("schedule_wrapper")
            .ok_or("joint inner capture upstream schedule wrapper is required")?,
        schedule,
        "joint inner capture.upstream_authorities.schedule_wrapper",
    )?;
    require_bound_identity(
        authorities
            .get("continuation")
            .ok_or("joint inner capture upstream continuation is required")?,
        continuation,
        "joint inner capture.upstream_authorities.continuation",
    )?;
    require_bound_identity(
        authorities
            .get("assessor_binding")
            .ok_or("joint inner capture upstream assessor binding is required")?,
        binding,
        "joint inner capture.upstream_authorities.assessor_binding",
    )?;

    let capability = object_field(root, "opaque_l0_continuation", "joint inner capture")?;
    no_forbidden_execution_input(capability, "joint inner capture.opaque_l0_continuation")?;
    require_exact_string(
        capability,
        "factory",
        CAPABILITY_FACTORY,
        "joint inner capture.opaque_l0_continuation",
    )?;
    require_exact_string(
        capability,
        "l1_encoder",
        L1_ENCODER,
        "joint inner capture.opaque_l0_continuation",
    )?;
    require_exact_string(
        capability,
        "consuming_finalizer",
        FINALIZER,
        "joint inner capture.opaque_l0_continuation",
    )?;
    for field in [
        "opaque",
        "freshly_derived_from_l0_23_dispatch_graph",
        "same_runtime_state_arena_bound",
        "same_command_buffer_bound",
        "non_transferable_across_processes",
    ] {
        require_bool(
            capability,
            field,
            true,
            "joint inner capture.opaque_l0_continuation",
        )?;
    }
    require_bool(
        capability,
        "raw_pinned_buffer_or_dispatch_count_input_accepted",
        false,
        "joint inner capture.opaque_l0_continuation",
    )?;
    for field in [
        "capability_identity_sha256",
        "runtime_identity_sha256",
        "runtime_state_arena_identity_sha256",
        "command_buffer_identity_sha256",
    ] {
        require_sha(
            capability,
            field,
            "joint inner capture.opaque_l0_continuation",
        )?;
    }

    let execution = object_field(root, "fresh_joint_execution", "joint inner capture")?;
    no_forbidden_execution_input(execution, "joint inner capture.fresh_joint_execution")?;
    for field in [
        "fresh_runtime",
        "fresh_session",
        "same_runtime",
        "same_tcb",
        "structural_trace_non_timed",
        "route_guard_enforced_before_l1",
    ] {
        require_bool(
            execution,
            field,
            true,
            "joint inner capture.fresh_joint_execution",
        )?;
    }
    for field in [
        "runtime_identity_sha256",
        "session_identity_sha256",
        "tcb_identity_sha256",
    ] {
        require_sha(
            execution,
            field,
            "joint inner capture.fresh_joint_execution",
        )?;
    }
    for (field, expected) in [
        ("source_token_id", SOURCE_TOKEN_ID),
        ("l0_dispatches", L0_DISPATCHES),
        ("l1_prefix_dispatches", L1_PREFIX_DISPATCHES),
        ("total_dispatches", TOTAL_DISPATCHES),
        ("fence_count", 1),
    ] {
        if require_u64(
            execution,
            field,
            "joint inner capture.fresh_joint_execution",
        )? != expected
        {
            return Err(format!(
                "joint inner capture fresh execution {field} drifted"
            ));
        }
    }
    let trace = object_field(root, "structural_kernel_trace", "joint inner capture")?;
    require_bool(
        trace,
        "non_timed",
        true,
        "joint inner capture.structural_kernel_trace",
    )?;
    require_bool(
        trace,
        "exact_order",
        true,
        "joint inner capture.structural_kernel_trace",
    )?;
    let names = array_field(
        trace,
        "kernel_names",
        "joint inner capture.structural_kernel_trace",
    )?;
    if names.len() != STRUCTURAL_KERNELS.len()
        || names
            .iter()
            .zip(STRUCTURAL_KERNELS)
            .any(|(value, expected)| value.as_str() != Some(expected))
    {
        return Err(
            "joint inner capture structural trace is not the exact fresh 23+9 graph".into(),
        );
    }
    let fence = object_field(root, "single_fence", "joint inner capture")?;
    require_exact_string(
        fence,
        "consuming_finalizer",
        FINALIZER,
        "joint inner capture.single_fence",
    )?;
    for field in [
        "only_command_buffer_consumed",
        "fence_succeeded",
        "readbacks_after_fence",
        "append_after_fence_possible",
    ] {
        require_bool(
            fence,
            field,
            field != "append_after_fence_possible",
            "joint inner capture.single_fence",
        )?;
    }
    if require_u64(fence, "fence_count", "joint inner capture.single_fence")? != 1 {
        return Err("joint inner capture must have exactly one fence".into());
    }
    validate_joint_readbacks(object_field(
        root,
        "fresh_readbacks",
        "joint inner capture",
    )?)?;
    let boundary = object_field(root, "claim_boundary", "joint inner capture")?;
    for field in [
        "l1_suffix_or_moe_executed",
        "complete_layer_executed",
        "token_generated",
        "decoder_started",
        "server_or_watcher_started",
    ] {
        require_bool(boundary, field, false, "joint inner capture.claim_boundary")?;
    }
    require_bool(
        boundary,
        "component_only",
        true,
        "joint inner capture.claim_boundary",
    )?;
    Ok(())
}

fn validate_joint_readbacks(readbacks: &Map<String, Value>) -> Result<(), String> {
    let l0 = object_field(readbacks, "l0_suffix", "joint readbacks")?;
    let route_guard = object_field(l0, "route_guard", "joint readbacks.l0_suffix")?;
    require_bool(
        route_guard,
        "passed",
        true,
        "joint readbacks.l0_suffix.route_guard",
    )?;
    if require_u64(
        route_guard,
        "value",
        "joint readbacks.l0_suffix.route_guard",
    )? != 1
    {
        return Err("joint readbacks route_guard must equal one".into());
    }
    let expected_ids = array_field(
        route_guard,
        "expected_route_ids",
        "joint readbacks.l0_suffix.route_guard",
    )?;
    let observed_ids = array_field(
        route_guard,
        "observed_route_ids",
        "joint readbacks.l0_suffix.route_guard",
    )?;
    let expected_weights = array_field(
        route_guard,
        "expected_route_weights",
        "joint readbacks.l0_suffix.route_guard",
    )?;
    let observed_weights = array_field(
        route_guard,
        "observed_route_weights",
        "joint readbacks.l0_suffix.route_guard",
    )?;
    if expected_ids.len() != 10
        || observed_ids != expected_ids
        || expected_weights.len() != 10
        || observed_weights.len() != 10
    {
        return Err("joint readbacks route guard does not bind the exact ten routes".into());
    }
    let weights_error = route_guard
        .get("weights_max_abs_error")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite() && *value >= 0.0)
        .ok_or("joint readbacks route_guard.weights_max_abs_error must be finite")?;
    if weights_error > MAX_PARITY_ERROR {
        return Err("joint readbacks route weights exceed strict component tolerance".into());
    }
    for field in [
        "postnorm",
        "router_logits",
        "shared_output",
        "routed_sum",
        "second_residual",
    ] {
        require_nonnegative_parity(
            object_field(l0, field, "joint readbacks.l0_suffix")?,
            &format!("joint readbacks.l0_suffix.{field}"),
            ParityHashRequirement::RecordedCpuDevicePair,
        )?;
    }
    let routes = array_field(
        l0,
        "all_ten_weighted_route_witnesses",
        "joint readbacks.l0_suffix",
    )?;
    if routes.len() != 10 {
        return Err("joint readbacks must contain exactly ten weighted route witnesses".into());
    }
    for (index, witness) in routes.iter().enumerate() {
        let witness = object(witness, "joint readbacks weighted route witness")?;
        if require_u64(
            witness,
            "wave_index",
            "joint readbacks weighted route witness",
        )? != index as u64
        {
            return Err("joint readbacks weighted route witnesses are not ordered".into());
        }
        require_u64(
            witness,
            "expert_id",
            "joint readbacks weighted route witness",
        )?;
        require_nonnegative_parity(
            witness,
            "joint readbacks weighted route witness",
            ParityHashRequirement::SingleF32leHash,
        )?;
    }

    let l0_state = object_field(readbacks, "fresh_l0_state", "joint readbacks")?;
    for (field, capacity) in [
        ("active_conv", L0_CONV_BYTES),
        ("active_recurrent", L0_RECURRENT_BYTES),
        ("rollback_conv", L0_CONV_BYTES),
        ("rollback_recurrent", L0_RECURRENT_BYTES),
    ] {
        require_state_readback(
            object_field(l0_state, field, "joint readbacks.fresh_l0_state")?,
            &format!("joint readbacks.fresh_l0_state.{field}"),
            0,
            0,
            capacity,
        )?;
    }
    let l1 = object_field(readbacks, "fresh_l1_slot1", "joint readbacks")?;
    if require_u64(l1, "layer", "joint readbacks.fresh_l1_slot1")? != L1_LAYER
        || require_u64(l1, "linear_state_slot", "joint readbacks.fresh_l1_slot1")? != L1_SLOT
        || require_u64(l1, "output_elements", "joint readbacks.fresh_l1_slot1")? != HIDDEN_ELEMENTS
        || require_u64(l1, "output_bytes", "joint readbacks.fresh_l1_slot1")? != HIDDEN_BYTES
    {
        return Err("joint readbacks Layer-1 slot-one geometry drifted".into());
    }
    // The opaque continuation plus exact fresh 23+9 structural trace, one
    // consuming fence, same-runtime/TCB lineage, and fresh readback geometry
    // establish custody of L0's retained output. CPU/device floating-point
    // snapshots can therefore have distinct byte hashes while a sealed
    // numerical parity witness remains within the strict component bound.
    require_nonnegative_parity(
        object_field(l1, "input", "joint readbacks.fresh_l1_slot1")?,
        "joint readbacks.fresh_l1_slot1.input",
        ParityHashRequirement::RecordedCpuDevicePair,
    )?;
    require_nonnegative_parity(
        object_field(
            l1,
            "first_residual_output",
            "joint readbacks.fresh_l1_slot1",
        )?,
        "joint readbacks.fresh_l1_slot1.first_residual_output",
        ParityHashRequirement::RecordedCpuDevicePair,
    )?;
    for (field, capacity) in [
        ("active_conv", L1_CONV_CAPACITY_BYTES),
        ("active_recurrent", L1_RECURRENT_CAPACITY_BYTES),
        ("rollback_conv", L1_CONV_CAPACITY_BYTES),
        ("rollback_recurrent", L1_RECURRENT_CAPACITY_BYTES),
    ] {
        let offset = if field.contains("conv") {
            L0_CONV_BYTES
        } else {
            L0_RECURRENT_BYTES
        };
        require_state_readback(
            object_field(l1, field, "joint readbacks.fresh_l1_slot1")?,
            &format!("joint readbacks.fresh_l1_slot1.{field}"),
            L1_SLOT,
            offset,
            capacity,
        )?;
    }
    Ok(())
}

fn validate_outer(outer: &BoundDocument, inner: &BoundDocument) -> Result<String, String> {
    let root = object(&outer.document, "joint outer terminal")?;
    no_forbidden_execution_input(root, "joint outer terminal")?;
    require_bool(root, "fixture_or_synthetic", false, "joint outer terminal")?;
    require_bool(root, "self_asserted", false, "joint outer terminal")?;
    let issuer = object_field(root, "issuer", "joint outer terminal")?;
    require_exact_string(
        issuer,
        "role",
        "joint_component_outer_reaper",
        "joint outer terminal.issuer",
    )?;
    let issuer_id = require_sha(
        issuer,
        "issuer_identity_sha256",
        "joint outer terminal.issuer",
    )?;
    let inner_root = object(&inner.document, "joint inner capture")?;
    let inner_issuer = object_field(inner_root, "issuer", "joint inner capture")?;
    if issuer_id
        == require_sha(
            inner_issuer,
            "issuer_identity_sha256",
            "joint inner capture.issuer",
        )?
    {
        return Err("joint outer terminal may not self-verify the inner capture".into());
    }
    require_bound_identity(
        root.get("inner_capture")
            .ok_or("joint outer terminal.inner_capture is required")?,
        inner,
        "joint outer terminal.inner_capture",
    )?;
    let terminal = object_field(root, "child_terminal", "joint outer terminal")?;
    if require_u64(terminal, "exit_code", "joint outer terminal.child_terminal")? != 0 {
        return Err("joint outer terminal child did not exit successfully".into());
    }
    for field in [
        "reaped",
        "terminal_receipt_written_last",
        "automatic_retry_disabled",
        "lease_reuse_prohibited",
    ] {
        require_bool(terminal, field, true, "joint outer terminal.child_terminal")?;
    }
    require_bool(
        terminal,
        "timed_out",
        false,
        "joint outer terminal.child_terminal",
    )?;
    let lease_id = require_sha(root, "lease_id", "joint outer terminal")?;
    let boundary = object_field(root, "claim_boundary", "joint outer terminal")?;
    for field in [
        "l1_suffix_or_moe_executed",
        "complete_layer_executed",
        "token_generated",
        "decoder_started",
        "server_or_watcher_started",
    ] {
        require_bool(
            boundary,
            field,
            false,
            "joint outer terminal.claim_boundary",
        )?;
    }
    require_bool(
        boundary,
        "component_only",
        true,
        "joint outer terminal.claim_boundary",
    )?;
    Ok(lease_id)
}

fn validate_release(
    release: &BoundDocument,
    outer: &BoundDocument,
    lease_id: &str,
) -> Result<(), String> {
    let root = object(&release.document, "joint lease release")?;
    no_forbidden_execution_input(root, "joint lease release")?;
    require_bool(root, "fixture_or_synthetic", false, "joint lease release")?;
    require_bool(root, "self_asserted", false, "joint lease release")?;
    let issuer = object_field(root, "issuer", "joint lease release")?;
    require_exact_string(
        issuer,
        "role",
        "joint_component_lease_release_authority",
        "joint lease release.issuer",
    )?;
    let outer_root = object(&outer.document, "joint outer terminal")?;
    let outer_issuer = object_field(outer_root, "issuer", "joint outer terminal")?;
    if require_sha(
        issuer,
        "issuer_identity_sha256",
        "joint lease release.issuer",
    )? == require_sha(
        outer_issuer,
        "issuer_identity_sha256",
        "joint outer terminal.issuer",
    )? {
        return Err("joint release may not be self-issued by the outer reaper".into());
    }
    require_bound_identity(
        root.get("outer_terminal")
            .ok_or("joint lease release.outer_terminal is required")?,
        outer,
        "joint lease release.outer_terminal",
    )?;
    if require_sha(root, "lease_id", "joint lease release")? != lease_id {
        return Err("joint lease release lease_id does not match outer terminal".into());
    }
    for field in [
        "actual_release_performed",
        "released_after_outer_terminal",
        "lease_released",
        "automatic_retry_prohibited",
        "fresh_lease_required_for_any_future_gpu_work",
    ] {
        require_bool(root, field, true, "joint lease release")?;
    }
    require_bool(
        root,
        "watcher_restart_or_transition_authorized",
        false,
        "joint lease release",
    )?;
    Ok(())
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

fn build_report(input: &Value) -> Value {
    let mut blockers = Vec::new();
    let input_root = match object(input, "input") {
        Ok(root) => root,
        Err(error) => {
            blockers.push(error);
            &Map::new()
        }
    };
    if input_root.get("schema").and_then(Value::as_str) != Some(INPUT_SCHEMA) {
        blockers.push(format!("input.schema must be {INPUT_SCHEMA}"));
    }
    if let Err(error) = verify_seal(input, "input") {
        blockers.push(error);
    }
    if let Err(error) = no_forbidden_execution_input(input_root, "input") {
        blockers.push(error);
    }

    let mut schedule = None;
    let mut continuation = None;
    let mut binding = None;
    let mut inner = None;
    let mut outer = None;
    let mut release = None;

    match parse_bound(
        input_root,
        "schedule_wrapper",
        SCHEDULE_SCHEMA,
        SCHEDULE_STATUS,
        Some(SCHEDULE_DOCUMENT_SHA256),
        Some(SCHEDULE_SEAL_SHA256),
    ) {
        Ok(value) => match validate_schedule(&value) {
            Ok(()) => schedule = Some(value),
            Err(error) => blockers.push(format!("schedule wrapper: {error}")),
        },
        Err(error) => blockers.push(format!("schedule wrapper: {error}")),
    }
    match parse_bound(
        input_root,
        "continuation",
        CONTINUATION_SCHEMA,
        CONTINUATION_STATUS,
        Some(CONTINUATION_DOCUMENT_SHA256),
        Some(CONTINUATION_SEAL_SHA256),
    ) {
        Ok(value) => match validate_continuation(&value) {
            Ok(()) => continuation = Some(value),
            Err(error) => blockers.push(format!("continuation: {error}")),
        },
        Err(error) => blockers.push(format!("continuation: {error}")),
    }
    match parse_bound(
        input_root,
        "assessor_binding",
        BINDING_SCHEMA,
        BINDING_STATUS,
        Some(BINDING_DOCUMENT_SHA256),
        Some(BINDING_SEAL_SHA256),
    ) {
        Ok(value) => match validate_assessor_binding(&value) {
            Ok(()) => binding = Some(value),
            Err(error) => blockers.push(format!("assessor binding: {error}")),
        },
        Err(error) => blockers.push(format!("assessor binding: {error}")),
    }
    match parse_bound(
        input_root,
        "joint_inner_capture",
        INNER_SCHEMA,
        INNER_STATUS,
        None,
        None,
    ) {
        Ok(value) => {
            if let (Some(schedule), Some(continuation), Some(binding)) =
                (schedule.as_ref(), continuation.as_ref(), binding.as_ref())
            {
                match validate_inner(&value, schedule, continuation, binding) {
                    Ok(()) => inner = Some(value),
                    Err(error) => blockers.push(format!("joint inner capture: {error}")),
                }
            } else {
                blockers.push("joint inner capture cannot be trusted until all immutable upstream authorities validate".into());
            }
        }
        Err(error) => blockers.push(format!("joint inner capture: {error}")),
    }
    match parse_bound(
        input_root,
        "joint_outer_terminal",
        OUTER_SCHEMA,
        OUTER_STATUS,
        None,
        None,
    ) {
        Ok(value) => {
            if let Some(inner) = inner.as_ref() {
                match validate_outer(&value, inner) {
                    Ok(_) => outer = Some(value),
                    Err(error) => blockers.push(format!("joint outer terminal: {error}")),
                }
            } else {
                blockers.push(
                    "joint outer terminal cannot be trusted without a valid inner capture".into(),
                );
            }
        }
        Err(error) => blockers.push(format!("joint outer terminal: {error}")),
    }
    match parse_bound(
        input_root,
        "joint_lease_release",
        RELEASE_SCHEMA,
        RELEASE_STATUS,
        None,
        None,
    ) {
        Ok(value) => {
            if let Some(outer) = outer.as_ref() {
                match validate_outer(outer, inner.as_ref().expect("outer requires inner"))
                    .and_then(|lease_id| validate_release(&value, outer, &lease_id))
                {
                    Ok(()) => release = Some(value),
                    Err(error) => blockers.push(format!("joint lease release: {error}")),
                }
            } else {
                blockers.push(
                    "joint lease release cannot be trusted without a valid outer terminal".into(),
                );
            }
        }
        Err(error) => blockers.push(format!("joint lease release: {error}")),
    }

    blockers.sort();
    blockers.dedup();
    let earned = blockers.is_empty();
    let mut output = json!({
        "schema": RESULT_SCHEMA,
        "status": if earned { EARNED_STATUS } else { REFUSED_STATUS },
        "earned_component_only": earned,
        "component_scope": {
            "source_token_id": SOURCE_TOKEN_ID,
            "fresh_l0_dispatches": L0_DISPATCHES,
            "fresh_l1_slot1_prefix_dispatches": L1_PREFIX_DISPATCHES,
            "fresh_total_dispatches": TOTAL_DISPATCHES,
            "opaque_same_runtime_continuation_required": true,
            "single_fence_required": true,
            "full_layer_or_token_decoder_earned": false,
        },
        "sealed_inputs": {
            "schedule_wrapper": binding_summary(schedule.as_ref(), schedule.is_some(), &blockers),
            "continuation": binding_summary(continuation.as_ref(), continuation.is_some(), &blockers),
            "assessor_binding": binding_summary(binding.as_ref(), binding.is_some(), &blockers),
            "joint_inner_capture": binding_summary(inner.as_ref(), inner.is_some(), &blockers),
            "joint_outer_terminal": binding_summary(outer.as_ref(), outer.is_some(), &blockers),
            "joint_lease_release": binding_summary(release.as_ref(), release.is_some(), &blockers),
        },
        "blockers": blockers,
        "authority_boundary": {
            "new_model_processes_authorized": 0,
            "metal_or_gpu_actions_authorized": 0,
            "lease_actions_authorized": 0,
            "server_or_watcher_actions_authorized": 0,
            "tps_or_tg_measurements_authorized": 0,
            "tournament_actions_authorized": 0,
        },
        "claim_boundary": {
            "cpu_only_post_capture_assessment": true,
            "l0_l1_component_not_full_layer_token_decoder": true,
            "does_not_reuse_historical_l0_receipt_as_execution_input": true,
            "does_not_accept_raw_pinned_buffer_or_dispatch_count_input": true,
            "does_not_construct_metal_or_dispatch": true,
            "does_not_issue_or_release_lease": true,
            "does_not_start_runtime_server_or_watcher": true,
            "does_not_measure_tps_or_tg": true,
            "does_not_claim_decoder_token_or_tournament": true,
        },
    });
    seal(&mut output).expect("assessor output must be sealable");
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
    let bytes = fs::read(path).map_err(|error| format!("cannot read {label}: {error}"))?;
    let value: Value =
        serde_json::from_slice(&bytes).map_err(|error| format!("cannot parse {label}: {error}"))?;
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
        .map_err(|error| format!("cannot write --out: {error}"))?;
    file.sync_all()
        .map_err(|error| format!("cannot sync --out: {error}"))
}

fn usage() -> &'static str {
    "usage: ascension_qwen80_l0_l1_joint_post_capture_assessor --input ABSOLUTE_PATH --out ABSOLUTE_NEW_PATH"
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
        if !matches!(flag.as_str(), "--input" | "--out") {
            return Err(format!("unsupported option {flag:?}; {}", usage()));
        }
        if values.insert(flag.clone(), value).is_some() {
            return Err(format!("duplicate {flag}; {}", usage()));
        }
    }
    let input = values
        .get("--input")
        .map(PathBuf::from)
        .ok_or_else(|| format!("missing --input; {}", usage()))?;
    let out = values
        .get("--out")
        .map(PathBuf::from)
        .ok_or_else(|| format!("missing --out; {}", usage()))?;
    if !input.is_absolute() || !out.is_absolute() {
        return Err(format!("all paths must be absolute; {}", usage()));
    }
    Ok(Args { input, out })
}

fn run(args: Args) -> Result<(String, String), String> {
    let input = read_json(&args.input, "--input")?;
    let output = build_report(&input);
    let seal = verify_seal(&output, "joint post-capture assessment output")?;
    let status = require_string(
        object(&output, "joint post-capture assessment output")?,
        "status",
        "joint post-capture assessment output",
    )?
    .to_owned();
    let bytes = serde_json::to_vec_pretty(&output)
        .map_err(|error| format!("cannot encode output: {error}"))?;
    write_new(&args.out, &bytes)?;
    Ok((seal, status))
}

fn main() {
    match parse_args(env::args().skip(1)).and_then(run) {
        Ok((seal, status)) => println!("{{\"status\":\"{status}\",\"seal_sha256\":\"{seal}\"}}"),
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

    fn reseal_document(mut value: Value) -> Value {
        value.as_object_mut().unwrap().remove("seal_sha256");
        seal(&mut value).unwrap();
        value
    }

    fn bound(document: Value) -> BoundDocument {
        BoundDocument {
            document_sha256: sha256_json(&document).unwrap(),
            document_seal_sha256: document["seal_sha256"].as_str().unwrap().to_owned(),
            document,
        }
    }

    fn bind(document: Value) -> Value {
        json!({
            "document": document.clone(),
            "document_sha256": sha256_json(&document).unwrap(),
            "document_seal_sha256": document["seal_sha256"].clone(),
        })
    }

    fn ref_identity(document: &Value) -> Value {
        json!({
            "present": true,
            "document_sha256": sha256_json(document).unwrap(),
            "document_seal_sha256": document["seal_sha256"].clone(),
        })
    }

    fn live_schedule() -> Value {
        serde_json::from_str(include_str!("../../../workspace/campaign/records/ascension-sandbox/physical/qwen80/complete-runtime/QWEN80_48_LAYER_PAYLOAD_SCHEDULE_SEALED_WRAPPER_20260809T083400Z.json")).unwrap()
    }

    fn live_continuation() -> Value {
        serde_json::from_str(include_str!("../../../workspace/campaign/records/ascension-sandbox/physical/qwen80/complete-runtime/QWEN80_L1_SOURCE_TOKEN_CONTINUATION_READINESS_WITH_SEALED_SCHEDULE_20260809T084000Z.json")).unwrap()
    }

    fn live_binding() -> Value {
        serde_json::from_str(include_str!("../../../workspace/campaign/records/ascension-sandbox/physical/qwen80/complete-runtime/QWEN80_L0_STATE_HANDOFF_POST_CAPTURE_ASSESSOR_BINDING_20260809T085000Z.json")).unwrap()
    }

    fn parity_pair(character: char) -> Value {
        json!({
            "passed": true,
            "cpu_f32le_sha256": sha(character),
            "device_f32le_sha256": sha(character),
            "max_abs_error": 0.0,
        })
    }

    fn state(character: char, slot: u64, offset: u64, capacity: u64) -> Value {
        json!({
            "passed": true,
            "slot": slot,
            "offset_bytes": offset,
            "capacity_bytes": capacity,
            "device_buffer_identity_sha256": sha(character),
            "f32le_sha256": sha(if character == 'f' { 'e' } else { 'f' }),
            "max_abs_error": 0.0,
        })
    }

    fn fresh_readbacks() -> Value {
        let routes = (0_u64..10)
            .map(|wave_index| {
                let character = ["a", "b", "c", "d", "e", "f"]
                    [usize::try_from(wave_index % 6).unwrap()]
                .chars()
                .next()
                .unwrap();
                json!({
                    "wave_index": wave_index,
                    "expert_id": wave_index,
                    "passed": true,
                    "f32le_sha256": sha(character),
                    "max_abs_error": 0.0,
                })
            })
            .collect::<Vec<_>>();
        json!({
            "l0_suffix": {
                "route_guard": {
                    "passed": true,
                    "value": 1,
                    "expected_route_ids": (0_u64..10).collect::<Vec<_>>(),
                    "observed_route_ids": (0_u64..10).collect::<Vec<_>>(),
                    "expected_route_weights": vec![0.1_f64; 10],
                    "observed_route_weights": vec![0.1_f64; 10],
                    "weights_max_abs_error": 0.0,
                },
                "postnorm": parity_pair('a'),
                "router_logits": parity_pair('b'),
                "all_ten_weighted_route_witnesses": routes,
                "shared_output": parity_pair('c'),
                "routed_sum": parity_pair('d'),
                "second_residual": parity_pair('e'),
            },
            "fresh_l0_state": {
                "active_conv": state('a', 0, 0, L0_CONV_BYTES),
                "active_recurrent": state('b', 0, 0, L0_RECURRENT_BYTES),
                "rollback_conv": state('c', 0, 0, L0_CONV_BYTES),
                "rollback_recurrent": state('d', 0, 0, L0_RECURRENT_BYTES),
            },
            "fresh_l1_slot1": {
                "layer": L1_LAYER,
                "linear_state_slot": L1_SLOT,
                "output_elements": HIDDEN_ELEMENTS,
                "output_bytes": HIDDEN_BYTES,
                "input": parity_pair('e'),
                "first_residual_output": parity_pair('f'),
                "active_conv": state('a', L1_SLOT, L0_CONV_BYTES, L1_CONV_CAPACITY_BYTES),
                "active_recurrent": state('b', L1_SLOT, L0_RECURRENT_BYTES, L1_RECURRENT_CAPACITY_BYTES),
                "rollback_conv": state('c', L1_SLOT, L0_CONV_BYTES, L1_CONV_CAPACITY_BYTES),
                "rollback_recurrent": state('d', L1_SLOT, L0_RECURRENT_BYTES, L1_RECURRENT_CAPACITY_BYTES),
            },
        })
    }

    fn inner_document(schedule: &Value, continuation: &Value, binding: &Value) -> Value {
        seal_document(json!({
            "schema": INNER_SCHEMA,
            "status": INNER_STATUS,
            "fixture_or_synthetic": false,
            "self_asserted": false,
            "issuer": {"role": "joint_component_capture_child", "issuer_identity_sha256": sha('a')},
            "upstream_authorities": {
                "schedule_wrapper": ref_identity(schedule),
                "continuation": ref_identity(continuation),
                "assessor_binding": ref_identity(binding),
            },
            "opaque_l0_continuation": {
                "factory": CAPABILITY_FACTORY,
                "l1_encoder": L1_ENCODER,
                "consuming_finalizer": FINALIZER,
                "opaque": true,
                "freshly_derived_from_l0_23_dispatch_graph": true,
                "same_runtime_state_arena_bound": true,
                "same_command_buffer_bound": true,
                "non_transferable_across_processes": true,
                "raw_pinned_buffer_or_dispatch_count_input_accepted": false,
                "capability_identity_sha256": sha('b'),
                "runtime_identity_sha256": sha('c'),
                "runtime_state_arena_identity_sha256": sha('d'),
                "command_buffer_identity_sha256": sha('e'),
            },
            "fresh_joint_execution": {
                "fresh_runtime": true,
                "fresh_session": true,
                "same_runtime": true,
                "same_tcb": true,
                "structural_trace_non_timed": true,
                "route_guard_enforced_before_l1": true,
                "runtime_identity_sha256": sha('c'),
                "session_identity_sha256": sha('f'),
                "tcb_identity_sha256": sha('e'),
                "source_token_id": SOURCE_TOKEN_ID,
                "l0_dispatches": L0_DISPATCHES,
                "l1_prefix_dispatches": L1_PREFIX_DISPATCHES,
                "total_dispatches": TOTAL_DISPATCHES,
                "fence_count": 1,
            },
            "structural_kernel_trace": {
                "non_timed": true,
                "exact_order": true,
                "kernel_names": STRUCTURAL_KERNELS,
            },
            "single_fence": {
                "consuming_finalizer": FINALIZER,
                "only_command_buffer_consumed": true,
                "fence_succeeded": true,
                "readbacks_after_fence": true,
                "append_after_fence_possible": false,
                "fence_count": 1,
            },
            "fresh_readbacks": fresh_readbacks(),
            "claim_boundary": {
                "component_only": true,
                "l1_suffix_or_moe_executed": false,
                "complete_layer_executed": false,
                "token_generated": false,
                "decoder_started": false,
                "server_or_watcher_started": false,
            },
        }))
    }

    fn outer_document(inner: &Value) -> Value {
        seal_document(json!({
            "schema": OUTER_SCHEMA,
            "status": OUTER_STATUS,
            "fixture_or_synthetic": false,
            "self_asserted": false,
            "issuer": {"role": "joint_component_outer_reaper", "issuer_identity_sha256": sha('b')},
            "inner_capture": ref_identity(inner),
            "lease_id": sha('c'),
            "child_terminal": {
                "exit_code": 0,
                "reaped": true,
                "timed_out": false,
                "terminal_receipt_written_last": true,
                "automatic_retry_disabled": true,
                "lease_reuse_prohibited": true,
            },
            "claim_boundary": {
                "component_only": true,
                "l1_suffix_or_moe_executed": false,
                "complete_layer_executed": false,
                "token_generated": false,
                "decoder_started": false,
                "server_or_watcher_started": false,
            },
        }))
    }

    fn release_document(outer: &Value) -> Value {
        seal_document(json!({
            "schema": RELEASE_SCHEMA,
            "status": RELEASE_STATUS,
            "fixture_or_synthetic": false,
            "self_asserted": false,
            "issuer": {"role": "joint_component_lease_release_authority", "issuer_identity_sha256": sha('d')},
            "outer_terminal": ref_identity(outer),
            "lease_id": outer["lease_id"].clone(),
            "actual_release_performed": true,
            "released_after_outer_terminal": true,
            "lease_released": true,
            "automatic_retry_prohibited": true,
            "fresh_lease_required_for_any_future_gpu_work": true,
            "watcher_restart_or_transition_authorized": false,
        }))
    }

    fn input_document(inner: Value, outer: Value, release: Value) -> Value {
        seal_document(json!({
            "schema": INPUT_SCHEMA,
            "schedule_wrapper": bind(live_schedule()),
            "continuation": bind(live_continuation()),
            "assessor_binding": bind(live_binding()),
            "joint_inner_capture": bind(inner),
            "joint_outer_terminal": bind(outer),
            "joint_lease_release": bind(release),
        }))
    }

    fn valid_capture_input() -> Value {
        let schedule = live_schedule();
        let continuation = live_continuation();
        let binding = live_binding();
        let inner = inner_document(&schedule, &continuation, &binding);
        let outer = outer_document(&inner);
        let release = release_document(&outer);
        input_document(inner, outer, release)
    }

    #[test]
    fn exact_future_opaque_23_plus_9_component_can_be_earned_but_not_promoted() {
        let report = build_report(&valid_capture_input());
        assert_eq!(
            verify_seal(&report, "report").unwrap(),
            report["seal_sha256"]
        );
        assert_eq!(report["status"], EARNED_STATUS);
        assert_eq!(report["earned_component_only"], true);
        assert_eq!(
            report["component_scope"]["fresh_total_dispatches"],
            TOTAL_DISPATCHES
        );
        assert_eq!(
            report["component_scope"]["full_layer_or_token_decoder_earned"],
            false
        );
        assert_eq!(
            report["authority_boundary"]["metal_or_gpu_actions_authorized"],
            0
        );
    }

    #[test]
    fn floating_cpu_device_pairs_accept_distinct_hashes_only_with_complete_strict_parity_evidence()
    {
        let schedule = bound(live_schedule());
        let continuation = bound(live_continuation());
        let binding = bound(live_binding());

        let mut tolerated = inner_document(
            &schedule.document,
            &continuation.document,
            &binding.document,
        );
        tolerated["fresh_readbacks"]["l0_suffix"]["postnorm"]["device_f32le_sha256"] =
            json!(sha('f'));
        tolerated["fresh_readbacks"]["l0_suffix"]["postnorm"]["max_abs_error"] = json!(3.17e-8_f64);
        tolerated["fresh_readbacks"]["fresh_l1_slot1"]["input"]["device_f32le_sha256"] =
            json!(sha('a'));
        tolerated["fresh_readbacks"]["fresh_l1_slot1"]["input"]["max_abs_error"] =
            json!(0.0005_f64);
        tolerated["fresh_readbacks"]["fresh_l1_slot1"]["first_residual_output"]
            ["device_f32le_sha256"] = json!(sha('a'));
        tolerated["fresh_readbacks"]["fresh_l1_slot1"]["first_residual_output"]["max_abs_error"] =
            json!(0.0005_f64);
        let tolerated = reseal_document(tolerated);
        assert!(validate_inner(
            &bound(tolerated.clone()),
            &schedule,
            &continuation,
            &binding
        )
        .is_ok());
        let outer = outer_document(&tolerated);
        let release = release_document(&outer);
        assert_eq!(
            build_report(&input_document(tolerated, outer, release))["status"],
            EARNED_STATUS
        );

        let mut over_tolerance = inner_document(
            &schedule.document,
            &continuation.document,
            &binding.document,
        );
        over_tolerance["fresh_readbacks"]["l0_suffix"]["postnorm"]["device_f32le_sha256"] =
            json!(sha('f'));
        over_tolerance["fresh_readbacks"]["l0_suffix"]["postnorm"]["max_abs_error"] =
            json!(0.0011_f64);
        let error = validate_inner(
            &bound(reseal_document(over_tolerance)),
            &schedule,
            &continuation,
            &binding,
        )
        .unwrap_err();
        assert!(error.contains("max_abs_error exceeds strict component tolerance"));

        let mut missing_device_hash = inner_document(
            &schedule.document,
            &continuation.document,
            &binding.document,
        );
        missing_device_hash["fresh_readbacks"]["fresh_l1_slot1"]["first_residual_output"]
            .as_object_mut()
            .unwrap()
            .remove("device_f32le_sha256");
        let error = validate_inner(
            &bound(reseal_document(missing_device_hash)),
            &schedule,
            &continuation,
            &binding,
        )
        .unwrap_err();
        assert!(error.contains("device_f32le_sha256"));
    }

    #[test]
    fn old_l0_receipt_or_raw_execution_input_is_refused() {
        let mut input = valid_capture_input();
        input["historical_l0_receipt"] = json!({"receipt": "not execution input"});
        input.as_object_mut().unwrap().remove("seal_sha256");
        seal(&mut input).unwrap();
        let report = build_report(&input);
        assert_eq!(report["status"], REFUSED_STATUS);
        assert!(report["blockers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value.as_str().unwrap().contains("historical_l0_receipt")));

        let schedule = bound(live_schedule());
        let continuation = bound(live_continuation());
        let binding = bound(live_binding());
        let mut inner = inner_document(
            &schedule.document,
            &continuation.document,
            &binding.document,
        );
        inner["opaque_l0_continuation"]["raw_pinned_buffer"] = json!("forbidden");
        inner.as_object_mut().unwrap().remove("seal_sha256");
        seal(&mut inner).unwrap();
        assert!(
            validate_inner(&bound(inner), &schedule, &continuation, &binding)
                .unwrap_err()
                .contains("raw_pinned_buffer")
        );
    }

    #[test]
    fn appended_kernel_or_l1_only_trace_is_refused() {
        let schedule = bound(live_schedule());
        let continuation = bound(live_continuation());
        let binding = bound(live_binding());
        let mut appended = inner_document(
            &schedule.document,
            &continuation.document,
            &binding.document,
        );
        appended["structural_kernel_trace"]["kernel_names"]
            .as_array_mut()
            .unwrap()
            .push(json!("unexpected_suffix"));
        appended.as_object_mut().unwrap().remove("seal_sha256");
        seal(&mut appended).unwrap();
        assert!(
            validate_inner(&bound(appended), &schedule, &continuation, &binding)
                .unwrap_err()
                .contains("exact fresh 23+9")
        );

        let mut l1_only = inner_document(
            &schedule.document,
            &continuation.document,
            &binding.document,
        );
        l1_only["structural_kernel_trace"]["kernel_names"] = json!(STRUCTURAL_KERNELS[23..]);
        l1_only.as_object_mut().unwrap().remove("seal_sha256");
        seal(&mut l1_only).unwrap();
        assert!(
            validate_inner(&bound(l1_only), &schedule, &continuation, &binding)
                .unwrap_err()
                .contains("exact fresh 23+9")
        );
    }

    #[test]
    fn fixture_and_incomplete_release_are_refused() {
        let schedule = live_schedule();
        let continuation = live_continuation();
        let binding = live_binding();
        let mut fixture = inner_document(&schedule, &continuation, &binding);
        fixture["fixture_or_synthetic"] = json!(true);
        fixture.as_object_mut().unwrap().remove("seal_sha256");
        seal(&mut fixture).unwrap();
        assert!(validate_inner(
            &bound(fixture),
            &bound(schedule.clone()),
            &bound(continuation.clone()),
            &bound(binding.clone()),
        )
        .unwrap_err()
        .contains("fixture_or_synthetic"));

        let inner = inner_document(&schedule, &continuation, &binding);
        let outer = outer_document(&inner);
        let mut release = release_document(&outer);
        release["lease_released"] = json!(false);
        release.as_object_mut().unwrap().remove("seal_sha256");
        seal(&mut release).unwrap();
        let outer_bound = bound(outer);
        let lease_id = validate_outer(&outer_bound, &bound(inner)).unwrap();
        assert!(validate_release(&bound(release), &outer_bound, &lease_id)
            .unwrap_err()
            .contains("lease_released"));
    }
}
