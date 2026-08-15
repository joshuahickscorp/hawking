//! CPU-only source-token Qwen80 Layer-1 MoE-completion preflight contract.
//!
//! The earned source-token L0(23) + Layer-1 DeltaNet-prefix(9) component is
//! an antecedent, not a transferable device buffer.  A future Layer-1
//! postnorm/router/top-10/all-ten/shared/second-residual capture must re-run
//! the L0 graph and the L1 prefix in one fresh runtime and command buffer,
//! then append this Layer-1 MoE suffix before the one fence.  This program
//! makes that requirement machine-checkable without opening model payloads,
//! constructing Metal, issuing a lease, or dispatching work.
//!
//! It validates only explicitly supplied sealed receipt documents.  In
//! particular it never discovers route tensors from filenames or accepts the
//! historical Layer-0 fixture route plan.  A future CPU route authority must
//! bind the exact Layer-1 source-token CPU oracle, all six fixed L1 payload
//! descriptors, and all thirty top-10 route descriptors before this preflight
//! can say that its *future* component capture is structurally prepared.

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process;

const SCHEMA: &str = "hawking.ascension.qwen80_source_token_l1_moe_completion_preflight.v1";
const PREPARED_STATUS: &str =
    "PREPARED_QWEN80_SOURCE_TOKEN_L1_MOE_COMPLETION_ROUTE_AUTHORITY_REQUIRED_NOT_LEASED_OR_EXECUTED";
const PREFLIGHTED_STATUS: &str =
    "PREFLIGHTED_QWEN80_SOURCE_TOKEN_L1_MOE_COMPLETION_COMPONENT_NOT_LEASED_OR_EXECUTED";

const JOINT_ASSESSMENT_SCHEMA: &str =
    "hawking.ascension.qwen80_l0_l1_joint_post_capture_assessment.v1";
const JOINT_ASSESSMENT_STATUS: &str =
    "EARNED_QWEN80_SOURCE_TOKEN_L0_L1_COMPONENT_NOT_FULL_LAYER_TOKEN_DECODER";

const ROUTE_AUTHORITY_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l1_all_ten_route_authority.v1";
const ROUTE_AUTHORITY_STATUS: &str =
    "SEALED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_L1_ALL_TEN_ROUTE_AUTHORITY_READY_FOR_SAME_RUNTIME_MOE_SUFFIX";

const FUTURE_INNER_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_l1_full_layer_same_runtime_capture.v1";
const FUTURE_INNER_STATUS: &str =
    "CAPTURED_QWEN80_SOURCE_TOKEN_L0_REENCODE_AND_L1_COMPLETE_MOE_LAYER_SAME_RUNTIME_COMPONENT_ONLY";
const FUTURE_OUTER_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_l1_full_layer_same_runtime_outer_capture.v1";
const FUTURE_OUTER_STATUS: &str =
    "CAPTURED_QWEN80_SOURCE_TOKEN_L0_L1_COMPLETE_LAYER_SAME_RUNTIME_OUTER_TERMINAL_COMPONENT_ONLY";
const FUTURE_RELEASE_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_l1_full_layer_same_runtime_quiet_metal_lease_release.v1";
const FUTURE_RELEASE_STATUS: &str =
    "RELEASED_QWEN80_SOURCE_TOKEN_L0_L1_COMPLETE_LAYER_COMPONENT_QUIET_METAL_LEASE_AFTER_TERMINAL_CAPTURE";

const MODEL_ID: &str = "Qwen3-Coder-Next-80B";
const MODEL_KEY: &str = "qwen80";
const SOURCE_REPOSITORY: &str = "Qwen/Qwen3-Coder-Next";
const SOURCE_REVISION: &str = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb";
const MANIFEST_DOCUMENT_SHA256: &str =
    "a0fcac0401a7962402bb8cb87d5055c83667b39575f9e0f4c7470d080758aa10";
const MANIFEST_SEAL_SHA256: &str =
    "14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b";
const ADMISSION_RECEIPT_SEAL_SHA256: &str =
    "939b41322363da3db774a2530b207bf380ed641d23cae671fc6438c0eecbf628";

const SOURCE_TOKEN_ID: u64 = 1;
const L0_LAYER: u64 = 0;
const L1_LAYER: u64 = 1;
const L1_LINEAR_STATE_SLOT: u64 = 1;
const HIDDEN: u64 = 2_048;
const HIDDEN_BYTES: u64 = HIDDEN * 4;
const INTERMEDIATE: u64 = 512;
const EXPERTS: u64 = 512;
const TOP_K: usize = 10;
const GROUP_SIZE: u64 = 128;
const L0_DISPATCHES: u64 = 23;
const L1_PREFIX_DISPATCHES: u64 = 9;
const L1_MOE_SUFFIX_DISPATCHES: u64 = 14;
const TOTAL_DISPATCHES: u64 = L0_DISPATCHES + L1_PREFIX_DISPATCHES + L1_MOE_SUFFIX_DISPATCHES;
const MAX_COMPONENT_PARITY_ERROR: f64 = 1.0e-3;
const MAX_ROUTE_WEIGHT_ERROR: f64 = 2.0e-5;
const MAX_ROUTE_WEIGHT_SUM_ERROR: f64 = 2.0e-6;

const L0_REENCODE_KERNELS: [&str; 23] = [
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

const L1_PREFIX_KERNELS: [&str; 9] = [
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

const L1_MOE_SUFFIX_KERNELS: [&str; 14] = [
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

#[derive(Debug)]
struct Args {
    joint_assessment: PathBuf,
    l1_route_authority: Option<PathBuf>,
    out: PathBuf,
}

#[derive(Clone, Debug)]
struct BoundDocument {
    path: PathBuf,
    raw_sha256: String,
    document_sha256: String,
    document_seal_sha256: String,
    value: Value,
}

fn usage() -> &'static str {
    "usage: ascension_qwen80_source_token_l1_moe_completion_preflight \\
--joint-assessment ABSOLUTE_SEALED_ASSESSMENT \\
[--l1-route-authority ABSOLUTE_SEALED_ROUTE_AUTHORITY] \\
--out ABSOLUTE_NEW_JSON"
}

fn parse_args<I>(arguments: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut joint_assessment = None;
    let mut l1_route_authority = None;
    let mut out = None;
    let mut arguments = arguments.into_iter();
    while let Some(flag) = arguments.next() {
        let slot = match flag.as_str() {
            "--joint-assessment" => &mut joint_assessment,
            "--l1-route-authority" => &mut l1_route_authority,
            "--out" => &mut out,
            "--help" | "-h" => return Err(usage().to_owned()),
            _ => return Err(format!("unsupported argument {flag:?}; {}", usage())),
        };
        let value = arguments
            .next()
            .ok_or_else(|| format!("{flag} requires a value; {}", usage()))?;
        if slot.replace(PathBuf::from(value)).is_some() {
            return Err(format!("{flag} may not be repeated; {}", usage()));
        }
    }
    let require_absolute = |path: Option<PathBuf>, label: &str| -> Result<PathBuf, String> {
        let path = path.ok_or_else(|| format!("missing {label}; {}", usage()))?;
        if !path.is_absolute() {
            return Err(format!("{label} must be absolute"));
        }
        Ok(path)
    };
    let joint_assessment = require_absolute(joint_assessment, "--joint-assessment")?;
    let out = require_absolute(out, "--out")?;
    if out.exists() || !out.parent().is_some_and(Path::is_dir) {
        return Err("--out must be a new file beneath an existing parent".into());
    }
    if let Some(path) = &l1_route_authority {
        if !path.is_absolute() {
            return Err("--l1-route-authority must be absolute".into());
        }
    }
    Ok(Args {
        joint_assessment,
        l1_route_authority,
        out,
    })
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

fn require_sha256(value: &str, label: &str) -> Result<(), String> {
    if !is_lower_sha256(value) {
        return Err(format!("{label} must be a lowercase SHA-256"));
    }
    Ok(())
}

/// Match Python's sorted compact JSON spelling for finite floating values.
/// Receipt seals are exchanged with Python lifecycle controllers, so a
/// serde_json-only rendering would be insufficient for exponent edge cases.
fn python_json_float(number: &serde_json::Number) -> Result<String, String> {
    let value = number
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or("canonical JSON float must be finite")?;
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
        Some(index) => (
            &unsigned[..index],
            unsigned[index + 1..]
                .parse::<i32>()
                .map_err(|error| format!("invalid float exponent: {error}"))?,
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
                        .ok_or("canonical float fractional length overflow")?;
                }
                digits.push(char::from(byte));
            }
            _ => return Err(format!("invalid canonical float mantissa {raw:?}")),
        }
    }
    let first = digits
        .bytes()
        .position(|byte| byte != b'0')
        .ok_or("nonzero canonical float has no significant digit")?;
    let mut significant = digits[first..].to_owned();
    let mut decimal_power = exponent
        .checked_sub(fractional)
        .ok_or("canonical decimal exponent overflow")?;
    while significant.len() > 1 && significant.ends_with('0') {
        significant.pop();
        decimal_power = decimal_power
            .checked_add(1)
            .ok_or("canonical decimal exponent overflow")?;
    }
    let scientific = decimal_power
        .checked_add(i32::try_from(significant.len() - 1).unwrap_or(i32::MAX))
        .ok_or("canonical scientific exponent overflow")?;
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
        Value::String(value) => {
            output.push_str(
                &serde_json::to_string(value)
                    .map_err(|error| format!("cannot render JSON string: {error}"))?,
            );
        }
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
            output.push('{');
            let mut first = true;
            for (key, value) in values {
                if !first {
                    output.push(',');
                }
                first = false;
                output.push_str(
                    &serde_json::to_string(key)
                        .map_err(|error| format!("cannot render JSON key: {error}"))?,
                );
                output.push(':');
                canonical_json_into(value, output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

fn canonical_json_bytes(value: &Value) -> Result<Vec<u8>, String> {
    let mut rendered = String::new();
    canonical_json_into(value, &mut rendered)?;
    Ok(rendered.into_bytes())
}

fn document_sha256(value: &Value, label: &str) -> Result<String, String> {
    let root = object(value, label)?;
    let mut unsigned = root.clone();
    unsigned.remove("seal_sha256");
    Ok(sha256_hex(&canonical_json_bytes(&Value::Object(unsigned))?))
}

fn verify_seal(value: &Value, label: &str) -> Result<String, String> {
    let root = object(value, label)?;
    let observed = string(root, "seal_sha256", label)?;
    require_sha256(observed, &format!("{label}.seal_sha256"))?;
    let computed = document_sha256(value, label)?;
    if observed != computed {
        return Err(format!("{label} seal does not match canonical document"));
    }
    Ok(observed.to_owned())
}

fn seal(value: &mut Value) -> Result<String, String> {
    if object(value, "preflight output")?.contains_key("seal_sha256") {
        return Err("preflight output is already sealed".into());
    }
    let seal = document_sha256(value, "preflight output")?;
    value
        .as_object_mut()
        .expect("preflight output has been checked as an object")
        .insert("seal_sha256".into(), Value::String(seal.clone()));
    Ok(seal)
}

fn canonical_regular(path: &Path, label: &str) -> Result<PathBuf, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    path.canonicalize()
        .map_err(|error| format!("cannot canonicalize {label} {}: {error}", path.display()))
}

fn read_bound_document(path: &Path, label: &str) -> Result<BoundDocument, String> {
    if !path.is_absolute() {
        return Err(format!("{label} must be absolute"));
    }
    let path = canonical_regular(path, label)?;
    let raw = fs::read(&path).map_err(|error| format!("cannot read {label}: {error}"))?;
    let value: Value =
        serde_json::from_slice(&raw).map_err(|error| format!("{label} is not JSON: {error}"))?;
    if !value.is_object() {
        return Err(format!("{label} must be a JSON object"));
    }
    let document_seal_sha256 = verify_seal(&value, label)?;
    Ok(BoundDocument {
        path,
        raw_sha256: sha256_hex(&raw),
        document_sha256: document_sha256(&value, label)?,
        document_seal_sha256,
        value,
    })
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))
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

fn string<'a>(object: &'a Map<String, Value>, field: &str, label: &str) -> Result<&'a str, String> {
    object
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{label}.{field} must be a non-empty string"))
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

fn u64_field(object: &Map<String, Value>, field: &str, label: &str) -> Result<u64, String> {
    object
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{label}.{field} must be an unsigned integer"))
}

fn f64_field(object: &Map<String, Value>, field: &str, label: &str) -> Result<f64, String> {
    object
        .get(field)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .ok_or_else(|| format!("{label}.{field} must be a finite number"))
}

fn require_exact_string(
    object: &Map<String, Value>,
    field: &str,
    expected: &str,
    label: &str,
) -> Result<(), String> {
    let actual = string(object, field, label)?;
    if actual != expected {
        return Err(format!("{label}.{field}={actual:?}, expected {expected:?}"));
    }
    Ok(())
}

fn require_sha_field(
    object: &Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<String, String> {
    let value = string(object, field, label)?;
    require_sha256(value, &format!("{label}.{field}"))?;
    Ok(value.to_owned())
}

fn binding_json(document: &BoundDocument) -> Value {
    json!({
        "path": document.path,
        "present": true,
        "bytes": fs::metadata(&document.path).map(|metadata| metadata.len()).unwrap_or_default(),
        "raw_sha256": document.raw_sha256,
        "document_sha256": document.document_sha256,
        "document_seal_sha256": document.document_seal_sha256,
    })
}

fn validate_joint_assessment(document: &BoundDocument) -> Result<(), String> {
    let root = object(&document.value, "joint assessment")?;
    require_exact_string(root, "schema", JOINT_ASSESSMENT_SCHEMA, "joint assessment")?;
    require_exact_string(root, "status", JOINT_ASSESSMENT_STATUS, "joint assessment")?;
    bool_field(root, "earned_component_only", true, "joint assessment")?;

    let scope = object_field(root, "component_scope", "joint assessment")?;
    for (field, expected) in [
        ("source_token_id", SOURCE_TOKEN_ID),
        ("fresh_l0_dispatches", L0_DISPATCHES),
        ("fresh_l1_slot1_prefix_dispatches", L1_PREFIX_DISPATCHES),
        (
            "fresh_total_dispatches",
            L0_DISPATCHES + L1_PREFIX_DISPATCHES,
        ),
    ] {
        if u64_field(scope, field, "joint assessment.component_scope")? != expected {
            return Err(format!("joint assessment.component_scope.{field} drifted"));
        }
    }
    bool_field(
        scope,
        "opaque_same_runtime_continuation_required",
        true,
        "joint assessment.component_scope",
    )?;
    bool_field(
        scope,
        "single_fence_required",
        true,
        "joint assessment.component_scope",
    )?;
    bool_field(
        scope,
        "full_layer_or_token_decoder_earned",
        false,
        "joint assessment.component_scope",
    )?;

    let inputs = object_field(root, "sealed_inputs", "joint assessment")?;
    for field in [
        "joint_inner_capture",
        "joint_outer_terminal",
        "joint_lease_release",
    ] {
        let input = object_field(inputs, field, "joint assessment.sealed_inputs")?;
        bool_field(input, "present", true, "joint assessment sealed input")?;
        bool_field(input, "valid", true, "joint assessment sealed input")?;
        require_sha_field(input, "document_sha256", "joint assessment sealed input")?;
        require_sha_field(
            input,
            "document_seal_sha256",
            "joint assessment sealed input",
        )?;
    }
    let boundary = object_field(root, "claim_boundary", "joint assessment")?;
    for field in [
        "cpu_only_post_capture_assessment",
        "l0_l1_component_not_full_layer_token_decoder",
        "does_not_reuse_historical_l0_receipt_as_execution_input",
        "does_not_accept_raw_pinned_buffer_or_dispatch_count_input",
        "does_not_construct_metal_or_dispatch",
        "does_not_issue_or_release_lease",
        "does_not_start_runtime_server_or_watcher",
        "does_not_measure_tps_or_tg",
        "does_not_claim_decoder_token_or_tournament",
    ] {
        bool_field(boundary, field, true, "joint assessment.claim_boundary")?;
    }
    Ok(())
}

fn expected_fixed_payloads() -> Vec<Value> {
    [
        (
            "post_attention_layernorm",
            "model.layers.1.post_attention_layernorm.weight",
            vec![HIDDEN],
        ),
        (
            "router",
            "model.layers.1.mlp.gate.weight",
            vec![EXPERTS, HIDDEN],
        ),
        (
            "shared_gate_proj",
            "model.layers.1.mlp.shared_expert.gate_proj.weight",
            vec![INTERMEDIATE, HIDDEN],
        ),
        (
            "shared_up_proj",
            "model.layers.1.mlp.shared_expert.up_proj.weight",
            vec![INTERMEDIATE, HIDDEN],
        ),
        (
            "shared_down_proj",
            "model.layers.1.mlp.shared_expert.down_proj.weight",
            vec![HIDDEN, INTERMEDIATE],
        ),
        (
            "shared_expert_gate",
            "model.layers.1.mlp.shared_expert_gate.weight",
            vec![1, HIDDEN],
        ),
    ]
    .into_iter()
    .map(|(role, tensor_name, shape)| {
        json!({
            "role": role,
            "tensor_name": tensor_name,
            "shape": shape,
            "group_size": GROUP_SIZE,
            "required_descriptor_fields": [
                "artifact_sha256",
                "direct_packed_payload_sha256",
                "header_sha256",
                "payload_bytes",
                "layout",
            ],
            "required_layout": {
                "magic": "HQ30G1B1",
                "version": 1,
                "group_size": GROUP_SIZE,
                "scale_dtype": "float16",
                "sign_bit_order": "little",
            },
        })
    })
    .collect()
}

fn expected_route_descriptor(role: &str, expert_id: u64) -> Value {
    let (suffix, shape) = match role {
        "gate" => ("gate_proj.weight", vec![INTERMEDIATE, HIDDEN]),
        "up" => ("up_proj.weight", vec![INTERMEDIATE, HIDDEN]),
        "down" => ("down_proj.weight", vec![HIDDEN, INTERMEDIATE]),
        _ => unreachable!("route descriptor role is fixed"),
    };
    json!({
        "tensor_name": format!("model.layers.1.mlp.experts.{expert_id}.{suffix}"),
        "shape": shape,
        "group_size": GROUP_SIZE,
        "required_descriptor_fields": [
            "artifact_sha256",
            "direct_packed_payload_sha256",
            "header_sha256",
            "payload_bytes",
            "layout",
        ],
        "required_layout": {
            "magic": "HQ30G1B1",
            "version": 1,
            "group_size": GROUP_SIZE,
            "scale_dtype": "float16",
            "sign_bit_order": "little",
        },
    })
}

fn dispatch_trace() -> Vec<Value> {
    let mut output = Vec::with_capacity(TOTAL_DISPATCHES as usize);
    for (index, kernel) in L0_REENCODE_KERNELS.iter().enumerate() {
        output.push(json!({
            "ordinal": index,
            "phase": "fresh_l0_reencode",
            "kernel": kernel,
        }));
    }
    for (index, kernel) in L1_PREFIX_KERNELS.iter().enumerate() {
        output.push(json!({
            "ordinal": L0_DISPATCHES as usize + index,
            "phase": "fresh_l1_deltanet_prefix",
            "kernel": kernel,
        }));
    }
    for (index, kernel) in L1_MOE_SUFFIX_KERNELS.iter().enumerate() {
        output.push(json!({
            "ordinal": (L0_DISPATCHES + L1_PREFIX_DISPATCHES) as usize + index,
            "phase": "fresh_l1_moe_suffix",
            "kernel": kernel,
        }));
    }
    output
}

fn route_authority_requirements() -> Value {
    json!({
        "schema": ROUTE_AUTHORITY_SCHEMA,
        "status": ROUTE_AUTHORITY_STATUS,
        "sealed_document_required": true,
        "source_binding_required": {
            "model_id": MODEL_ID,
            "model_key": MODEL_KEY,
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
            "manifest_document_sha256": MANIFEST_DOCUMENT_SHA256,
            "manifest_seal_sha256": MANIFEST_SEAL_SHA256,
            "admission_receipt_seal_sha256": ADMISSION_RECEIPT_SEAL_SHA256,
            "joint_l0_l1_assessment_binding_required": true,
            "prior_joint_assessment_is_provenance_only": true,
            "cross_process_pinned_buffer_import_allowed": false,
        },
        "same_source_token_cpu_oracle_required": {
            "source_token_id": SOURCE_TOKEN_ID,
            "layer": L1_LAYER,
            "linear_state_slot": L1_LINEAR_STATE_SLOT,
            "fresh_l0_reencode_dispatches": L0_DISPATCHES,
            "fresh_l1_prefix_dispatches": L1_PREFIX_DISPATCHES,
            "cpu_oracle_reencodes_l0_then_l1_prefix": true,
            "zero_initial_l0_state_required": true,
            "zero_initial_l1_slot1_state_required": true,
            "required_hash_fields": [
                "source_input_f32le_sha256",
                "l0_second_residual_cpu_f32le_sha256",
                "l1_prefix_input_cpu_f32le_sha256",
                "l1_first_residual_cpu_f32le_sha256",
                "l1_post_attention_normalized_hidden_cpu_f32le_sha256",
                "l1_router_logits_cpu_f32le_sha256",
                "l1_post_conv_state_cpu_f32le_sha256",
                "l1_post_recurrent_state_cpu_f32le_sha256",
            ],
        },
        "router_policy_required": {
            "logit_count": EXPERTS,
            "top_k": TOP_K,
            "softmax": "subtract_max_exp_f32",
            "selection": "source_qwen80_topk_router",
            "tie_break": "lowest_expert_id_within_route_tie_epsilon",
            "route_tie_epsilon_source": "HAWKING_DS_ROUTE_TIE_EPS",
            "route_tie_epsilon_f32_bits_hex_required": true,
            "selected_probabilities_renormalized": true,
            "route_weight_sum_target": 1.0,
            "route_weight_sum_tolerance": MAX_ROUTE_WEIGHT_SUM_ERROR,
            "device_weight_parity_tolerance": MAX_ROUTE_WEIGHT_ERROR,
            "ordered_unique_route_ids_required": true,
        },
        "fixed_l1_payloads_required": expected_fixed_payloads(),
        "all_ten_waves_required": {
            "count": TOP_K,
            "ordered_wave_indices": (0..TOP_K).collect::<Vec<_>>(),
            "each_wave_required_fields": [
                "wave_index",
                "layer",
                "expert_id",
                "normalized_weight",
                "normalized_weight_bits_hex",
                "gate",
                "up",
                "down",
            ],
            "all_thirty_payload_artifact_sha256s_unique": true,
            "route_order_must_match_source_stable_route_ids": true,
            "route_weights_must_match_source_stable_normalized_weights": true,
            "route_reorder_substitution_or_duplication_allowed": false,
        },
        "execution_boundary": {
            "cpu_route_authority_may_not_claim_metal_or_device_execution": true,
            "route_execution_performed": false,
            "shared_expert_performed": false,
            "second_residual_performed": false,
            "full_layer_or_token_or_decoder_claim_earned": false,
        },
    })
}

fn future_receipt_contract() -> Value {
    json!({
        "inner_schema": FUTURE_INNER_SCHEMA,
        "inner_status": FUTURE_INNER_STATUS,
        "outer_schema": FUTURE_OUTER_SCHEMA,
        "outer_status": FUTURE_OUTER_STATUS,
        "release_schema": FUTURE_RELEASE_SCHEMA,
        "release_status": FUTURE_RELEASE_STATUS,
        "fresh_execution_required": {
            "source_token_id": SOURCE_TOKEN_ID,
            "same_runtime_required": true,
            "same_token_command_buffer_required": true,
            "single_fence_after_all_dispatches_required": true,
            "non_timed_trace_required": true,
            "l0_reencode_dispatches": L0_DISPATCHES,
            "l1_prefix_dispatches": L1_PREFIX_DISPATCHES,
            "l1_moe_suffix_dispatches": L1_MOE_SUFFIX_DISPATCHES,
            "total_dispatches": TOTAL_DISPATCHES,
            "no_cross_process_buffer_or_state_import": true,
            "no_l1_suffix_beyond_moe_completion": true,
        },
        "required_readbacks": {
            "route_guard": {
                "required_value": 1,
                "ordered_ids_exact": true,
                "weights_max_abs_error_tolerance": MAX_ROUTE_WEIGHT_ERROR,
            },
            "cpu_device_parity": {
                "max_abs_error_tolerance": MAX_COMPONENT_PARITY_ERROR,
                "retain_distinct_cpu_and_device_hashes": true,
                "required_outputs": [
                    "l1_post_attention_normalized_hidden",
                    "l1_router_logits",
                    "all_ten_weighted_route_outputs",
                    "l1_shared_gated_output",
                    "l1_routed_sum",
                    "l1_second_residual",
                    "l1_active_conv_state",
                    "l1_active_recurrent_state",
                    "l1_rollback_conv_state",
                    "l1_rollback_recurrent_state",
                ],
            },
            "retained_l1_second_residual": {
                "elements": HIDDEN,
                "bytes": HIDDEN_BYTES,
                "same_runtime_next_layer_input_candidate_only": true,
                "cross_process_transfer_authorized": false,
            },
        },
        "claim_boundary": {
            "component_only": true,
            "l1_moe_operator_chain_completed_if_capture_passes": true,
            "complete_token_decoder_server_hcli_tps_tg_tournament": false,
            "automatic_retry": false,
        },
    })
}

fn validate_fixed_descriptor(
    value: &Map<String, Value>,
    expected: &Value,
    label: &str,
    seen_artifacts: &mut BTreeSet<String>,
    seen_payloads: &mut BTreeSet<String>,
) -> Result<(), String> {
    let expected = object(expected, "fixed descriptor requirement")?;
    for field in ["tensor_name", "group_size"] {
        if value.get(field) != expected.get(field) {
            return Err(format!(
                "{label}.{field} drifted from source Layer-1 contract"
            ));
        }
    }
    if value.get("shape") != expected.get("shape") {
        return Err(format!(
            "{label}.shape drifted from source Layer-1 contract"
        ));
    }
    let layout = object_field(value, "layout", label)?;
    let expected_layout =
        object_field(expected, "required_layout", "fixed descriptor requirement")?;
    for field in [
        "magic",
        "version",
        "group_size",
        "scale_dtype",
        "sign_bit_order",
    ] {
        if layout.get(field) != expected_layout.get(field) {
            return Err(format!("{label}.layout.{field} drifted"));
        }
    }
    for field in [
        "artifact_sha256",
        "direct_packed_payload_sha256",
        "header_sha256",
    ] {
        let hash = require_sha_field(value, field, label)?;
        if field == "artifact_sha256" {
            if !seen_artifacts.insert(hash) {
                return Err(format!("{label} reuses an artifact SHA"));
            }
        } else if field == "direct_packed_payload_sha256" && !seen_payloads.insert(hash) {
            return Err(format!("{label} reuses a direct-packed payload SHA"));
        }
    }
    if u64_field(value, "payload_bytes", label)? == 0 {
        return Err(format!("{label}.payload_bytes must be positive"));
    }
    Ok(())
}

fn validate_route_authority(
    authority: &BoundDocument,
    assessment: &BoundDocument,
) -> Result<(), String> {
    let root = object(&authority.value, "Layer-1 route authority")?;
    require_exact_string(
        root,
        "schema",
        ROUTE_AUTHORITY_SCHEMA,
        "Layer-1 route authority",
    )?;
    require_exact_string(
        root,
        "status",
        ROUTE_AUTHORITY_STATUS,
        "Layer-1 route authority",
    )?;
    bool_field(
        root,
        "fixture_or_synthetic",
        false,
        "Layer-1 route authority",
    )?;
    bool_field(
        root,
        "metal_or_gpu_activity_performed",
        false,
        "Layer-1 route authority",
    )?;

    let source = object_field(root, "source_binding", "Layer-1 route authority")?;
    for (field, expected) in [
        ("model_id", MODEL_ID),
        ("model_key", MODEL_KEY),
        ("source_repository", SOURCE_REPOSITORY),
        ("source_revision", SOURCE_REVISION),
        ("manifest_document_sha256", MANIFEST_DOCUMENT_SHA256),
        ("manifest_seal_sha256", MANIFEST_SEAL_SHA256),
        (
            "admission_receipt_seal_sha256",
            ADMISSION_RECEIPT_SEAL_SHA256,
        ),
    ] {
        require_exact_string(
            source,
            field,
            expected,
            "Layer-1 route authority.source_binding",
        )?;
    }
    let assessment_binding = object_field(
        source,
        "joint_l0_l1_assessment",
        "Layer-1 route authority.source_binding",
    )?;
    if assessment_binding.get("document_sha256")
        != Some(&Value::String(assessment.document_sha256.clone()))
        || assessment_binding.get("document_seal_sha256")
            != Some(&Value::String(assessment.document_seal_sha256.clone()))
    {
        return Err(
            "Layer-1 route authority does not bind the supplied earned L0-L1 assessment".into(),
        );
    }
    bool_field(
        source,
        "prior_joint_assessment_is_provenance_only",
        true,
        "Layer-1 route authority.source_binding",
    )?;
    bool_field(
        source,
        "cross_process_pinned_buffer_import_allowed",
        false,
        "Layer-1 route authority.source_binding",
    )?;

    let cpu = object_field(
        root,
        "source_token_l1_cpu_oracle",
        "Layer-1 route authority",
    )?;
    for (field, expected) in [
        ("source_token_id", SOURCE_TOKEN_ID),
        ("layer", L1_LAYER),
        ("linear_state_slot", L1_LINEAR_STATE_SLOT),
        ("fresh_l0_reencode_dispatches", L0_DISPATCHES),
        ("fresh_l1_prefix_dispatches", L1_PREFIX_DISPATCHES),
    ] {
        if u64_field(
            cpu,
            field,
            "Layer-1 route authority.source_token_l1_cpu_oracle",
        )? != expected
        {
            return Err(format!(
                "Layer-1 route authority.source_token_l1_cpu_oracle.{field} drifted"
            ));
        }
    }
    for field in [
        "cpu_oracle_reencodes_l0_then_l1_prefix",
        "zero_initial_l0_state",
        "zero_initial_l1_slot1_state",
    ] {
        bool_field(
            cpu,
            field,
            true,
            "Layer-1 route authority.source_token_l1_cpu_oracle",
        )?;
    }
    for field in [
        "source_input_f32le_sha256",
        "l0_second_residual_cpu_f32le_sha256",
        "l1_prefix_input_cpu_f32le_sha256",
        "l1_first_residual_cpu_f32le_sha256",
        "l1_post_attention_normalized_hidden_cpu_f32le_sha256",
        "l1_router_logits_cpu_f32le_sha256",
        "l1_post_conv_state_cpu_f32le_sha256",
        "l1_post_recurrent_state_cpu_f32le_sha256",
    ] {
        require_sha_field(
            cpu,
            field,
            "Layer-1 route authority.source_token_l1_cpu_oracle",
        )?;
    }

    let route = object_field(
        root,
        "source_token_router_evidence",
        "Layer-1 route authority",
    )?;
    for (field, expected) in [("logit_count", EXPERTS), ("top_k", TOP_K as u64)] {
        if u64_field(
            route,
            field,
            "Layer-1 route authority.source_token_router_evidence",
        )? != expected
        {
            return Err(format!(
                "Layer-1 route authority.source_token_router_evidence.{field} drifted"
            ));
        }
    }
    for (field, expected) in [
        ("selection", "source_qwen80_topk_router"),
        ("tie_break", "lowest_expert_id_within_route_tie_epsilon"),
        ("softmax", "subtract_max_exp_f32"),
        ("route_tie_epsilon_source", "HAWKING_DS_ROUTE_TIE_EPS"),
    ] {
        require_exact_string(
            route,
            field,
            expected,
            "Layer-1 route authority.source_token_router_evidence",
        )?;
    }
    bool_field(
        route,
        "selected_probabilities_renormalized",
        true,
        "Layer-1 route authority.source_token_router_evidence",
    )?;
    let route_tie_epsilon = f64_field(
        route,
        "route_tie_epsilon",
        "Layer-1 route authority.source_token_router_evidence",
    )?;
    if route_tie_epsilon < 0.0 {
        return Err("Layer-1 route authority route_tie_epsilon must be non-negative".into());
    }
    let route_tie_epsilon_bits = string(
        route,
        "route_tie_epsilon_f32_bits_hex",
        "Layer-1 route authority.source_token_router_evidence",
    )?;
    let route_tie_epsilon_bits = route_tie_epsilon_bits
        .strip_prefix("0x")
        .filter(|bits| {
            bits.len() == 8
                && bits
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        })
        .ok_or("Layer-1 route authority route_tie_epsilon_f32_bits_hex is invalid")?;
    let route_tie_epsilon_bits =
        u32::from_str_radix(route_tie_epsilon_bits, 16).map_err(|error| {
            format!("Layer-1 route authority route_tie_epsilon bits are invalid: {error}")
        })?;
    if f64::from(f32::from_bits(route_tie_epsilon_bits)).to_bits() != route_tie_epsilon.to_bits() {
        return Err(
            "Layer-1 route authority route_tie_epsilon does not match its exact f32 bits".into(),
        );
    }
    let ids = array_field(
        route,
        "source_stable_route_ids",
        "Layer-1 route authority.source_token_router_evidence",
    )?;
    let weights = array_field(
        route,
        "source_stable_normalized_weights",
        "Layer-1 route authority.source_token_router_evidence",
    )?;
    if ids.len() != TOP_K || weights.len() != TOP_K {
        return Err("Layer-1 route authority must provide exactly ten IDs and weights".into());
    }
    let mut ids_u64 = Vec::with_capacity(TOP_K);
    let mut weights_f64 = Vec::with_capacity(TOP_K);
    let mut seen_ids = BTreeSet::new();
    for (index, (id, weight)) in ids.iter().zip(weights).enumerate() {
        let id = id
            .as_u64()
            .filter(|value| *value < EXPERTS)
            .ok_or_else(|| format!("Layer-1 route authority route id {index} is invalid"))?;
        if !seen_ids.insert(id) {
            return Err("Layer-1 route authority has duplicate route IDs".into());
        }
        let weight = weight
            .as_f64()
            .filter(|value| value.is_finite() && *value >= 0.0)
            .ok_or_else(|| format!("Layer-1 route authority route weight {index} is invalid"))?;
        ids_u64.push(id);
        weights_f64.push(weight);
    }
    let weight_sum = weights_f64.iter().sum::<f64>();
    if (weight_sum - 1.0).abs() > MAX_ROUTE_WEIGHT_SUM_ERROR {
        return Err(format!(
            "Layer-1 route authority normalized weights sum to {weight_sum}, not one"
        ));
    }
    let declared_weight_sum = f64_field(
        route,
        "weights_sum",
        "Layer-1 route authority.source_token_router_evidence",
    )?;
    if (declared_weight_sum - weight_sum).abs() > MAX_ROUTE_WEIGHT_SUM_ERROR {
        return Err(
            "Layer-1 route authority declared weights_sum does not match its ordered weights within the source route tolerance".into(),
        );
    }

    let fixed = array_field(root, "fixed_l1_payloads", "Layer-1 route authority")?;
    let expected_fixed = expected_fixed_payloads();
    if fixed.len() != expected_fixed.len() {
        return Err("Layer-1 route authority must bind exactly six fixed payloads".into());
    }
    let mut seen_artifacts = BTreeSet::new();
    let mut seen_payloads = BTreeSet::new();
    for (index, (actual, expected)) in fixed.iter().zip(&expected_fixed).enumerate() {
        let actual = object(actual, "Layer-1 fixed payload")?;
        validate_fixed_descriptor(
            actual,
            expected,
            &format!("Layer-1 fixed payload {index}"),
            &mut seen_artifacts,
            &mut seen_payloads,
        )?;
    }

    let waves = array_field(root, "deterministic_waves", "Layer-1 route authority")?;
    if waves.len() != TOP_K {
        return Err("Layer-1 route authority must bind exactly ten deterministic waves".into());
    }
    for (index, wave) in waves.iter().enumerate() {
        let wave = object(wave, "Layer-1 route authority deterministic wave")?;
        if u64_field(wave, "wave_index", "Layer-1 deterministic wave")? != index as u64
            || u64_field(wave, "layer", "Layer-1 deterministic wave")? != L1_LAYER
            || u64_field(wave, "expert_id", "Layer-1 deterministic wave")? != ids_u64[index]
        {
            return Err(format!(
                "Layer-1 deterministic wave {index} route identity drifted"
            ));
        }
        let weight = f64_field(wave, "normalized_weight", "Layer-1 deterministic wave")?;
        if (weight - weights_f64[index]).abs() > f64::EPSILON {
            return Err(format!("Layer-1 deterministic wave {index} weight drifted"));
        }
        let expected_bits = format!("0x{:016x}", weights_f64[index].to_bits());
        require_exact_string(
            wave,
            "normalized_weight_bits_hex",
            &expected_bits,
            "Layer-1 deterministic wave",
        )?;
        for role in ["gate", "up", "down"] {
            let descriptor = object_field(wave, role, "Layer-1 deterministic wave")?;
            validate_fixed_descriptor(
                descriptor,
                &expected_route_descriptor(role, ids_u64[index]),
                &format!("Layer-1 deterministic wave {index} {role}"),
                &mut seen_artifacts,
                &mut seen_payloads,
            )?;
        }
    }
    if seen_artifacts.len() != 36 || seen_payloads.len() != 36 {
        return Err(
            "Layer-1 route authority must bind 36 distinct fixed/route artifact and direct-packed payload identities".into(),
        );
    }

    let gate = object_field(
        root,
        "rawls_real_all_ten_provenance_gate",
        "Layer-1 route authority",
    )?;
    for field in [
        "all_ten_source_bindings_complete",
        "execution_receipt_required_for_each_wave",
        "direct_packed_execution_required_for_each_wave",
        "source_bound_input_required_for_each_wave",
        "route_combine_receipt_required_separately",
        "shared_expert_receipt_required_separately",
        "first_and_second_residual_receipts_required_separately",
        "rejects_tensor_substitution",
        "rejects_route_reorder",
        "rejects_duplicate_experts",
        "rejects_missing_tensor_or_weight",
    ] {
        bool_field(gate, field, true, "Layer-1 route authority provenance gate")?;
    }
    if u64_field(
        gate,
        "expected_layer",
        "Layer-1 route authority provenance gate",
    )? != L1_LAYER
    {
        return Err("Layer-1 route authority provenance gate must target layer one".into());
    }
    for field in [
        "route_execution_performed",
        "route_combine_performed",
        "shared_expert_performed",
        "residual_combine_performed",
        "metal_device_or_dispatch_performed",
        "model_execution_performed",
        "hcli_execution_performed",
        "tps_or_tg_measurement_performed",
        "complete_layer_or_decoder_claim_earned",
    ] {
        bool_field(root, field, false, "Layer-1 route authority")?;
    }
    Ok(())
}

/// File/seal evidence returned by the CPU-only completion validator for an
/// already-existing receipt.  Its `value` is provided only for an in-process
/// structural check; consumers must retain a shallow binding rather than
/// serialize it into a future device receipt.
#[derive(Clone, Debug)]
pub struct VerifiedSealedDocument {
    pub path: PathBuf,
    pub raw_sha256: String,
    pub document_sha256: String,
    pub document_seal_sha256: String,
    pub value: Value,
}

/// Read and seal-verify one existing JSON receipt without performing any
/// artifact, catalog, Metal, or process operation.
pub fn read_verified_sealed_document(
    path: &Path,
    label: &str,
) -> Result<VerifiedSealedDocument, String> {
    let document = read_bound_document(path, label)?;
    Ok(VerifiedSealedDocument {
        path: document.path,
        raw_sha256: document.raw_sha256,
        document_sha256: document.document_sha256,
        document_seal_sha256: document.document_seal_sha256,
        value: document.value,
    })
}

/// A narrowly typed view of the already-sealed Layer-1 CPU route authority.
///
/// This is deliberately an *in-process validation result*, not a serializable
/// continuation.  The raw authority contains source-input CPU hashes and must
/// never be embedded wholesale in a future Metal receipt, where it could be
/// mistaken for a transferable device-buffer authority.  A same-runtime host
/// may consume this view only to reconstruct the exact ordered route selection
/// and to bind shallow file/seal evidence in its own static preflight.
#[derive(Clone, Debug)]
pub struct ValidatedSourceTokenL1RouteAuthority {
    pub joint_assessment_path: PathBuf,
    pub joint_assessment_raw_sha256: String,
    pub joint_assessment_document_sha256: String,
    pub joint_assessment_document_seal_sha256: String,
    pub route_authority_path: PathBuf,
    pub route_authority_raw_sha256: String,
    pub route_authority_document_sha256: String,
    pub route_authority_document_seal_sha256: String,
    pub route_ids: [u16; TOP_K],
    pub route_weights: [f32; TOP_K],
    pub fixed_l1_payloads: Vec<Value>,
    pub deterministic_waves: Vec<Value>,
}

/// Read and fully validate the earned L0→L1 assessor plus one original
/// source-token Layer-1 route authority.  This accepts the raw authoritative
/// inner receipt directly; a recovery/canonicalization wrapper is audit
/// provenance only and intentionally cannot stand in for the route authority.
///
/// The helper is shared by the future same-runtime 46-dispatch host so its
/// route order/weights cannot drift from the standalone CPU preflight.  It is
/// file-only and never opens a model payload, constructs Metal, or creates a
/// command buffer.
pub fn validate_source_token_l1_route_authority_files(
    joint_assessment_path: &Path,
    route_authority_path: &Path,
) -> Result<ValidatedSourceTokenL1RouteAuthority, String> {
    let assessment = read_bound_document(joint_assessment_path, "joint assessment")?;
    let authority = read_bound_document(route_authority_path, "Layer-1 route authority")?;
    validate_joint_assessment(&assessment)?;
    validate_route_authority(&authority, &assessment)?;

    let root = object(&authority.value, "validated Layer-1 route authority")?;
    let route = object_field(
        root,
        "source_token_router_evidence",
        "validated Layer-1 route authority",
    )?;
    let ids = array_field(
        route,
        "source_stable_route_ids",
        "validated Layer-1 route authority.source_token_router_evidence",
    )?
    .iter()
    .map(|value| {
        value
            .as_u64()
            .and_then(|value| u16::try_from(value).ok())
            .ok_or_else(|| "validated Layer-1 route authority has an invalid route ID".to_owned())
    })
    .collect::<Result<Vec<_>, _>>()?;
    let weights = array_field(
        route,
        "source_stable_normalized_weights",
        "validated Layer-1 route authority.source_token_router_evidence",
    )?
    .iter()
    .map(|value| {
        value
            .as_f64()
            .filter(|value| value.is_finite() && *value >= 0.0)
            .map(|value| value as f32)
            .filter(|value| value.is_finite())
            .ok_or_else(|| {
                "validated Layer-1 route authority has an invalid normalized weight".to_owned()
            })
    })
    .collect::<Result<Vec<_>, _>>()?;
    let route_ids = ids.try_into().map_err(|_: Vec<u16>| {
        "validated Layer-1 route authority did not retain exactly ten route IDs".to_owned()
    })?;
    let route_weights = weights.try_into().map_err(|_: Vec<f32>| {
        "validated Layer-1 route authority did not retain exactly ten route weights".to_owned()
    })?;
    let fixed_l1_payloads = array_field(
        root,
        "fixed_l1_payloads",
        "validated Layer-1 route authority",
    )?
    .to_vec();
    let deterministic_waves = array_field(
        root,
        "deterministic_waves",
        "validated Layer-1 route authority",
    )?
    .to_vec();
    Ok(ValidatedSourceTokenL1RouteAuthority {
        joint_assessment_path: assessment.path,
        joint_assessment_raw_sha256: assessment.raw_sha256,
        joint_assessment_document_sha256: assessment.document_sha256,
        joint_assessment_document_seal_sha256: assessment.document_seal_sha256,
        route_authority_path: authority.path,
        route_authority_raw_sha256: authority.raw_sha256,
        route_authority_document_sha256: authority.document_sha256,
        route_authority_document_seal_sha256: authority.document_seal_sha256,
        route_ids,
        route_weights,
        fixed_l1_payloads,
        deterministic_waves,
    })
}

fn build_preflight(
    assessment: &BoundDocument,
    route_authority: Option<&BoundDocument>,
) -> Result<Value, String> {
    validate_joint_assessment(assessment)?;
    if let Some(route_authority) = route_authority {
        validate_route_authority(route_authority, assessment)?;
    }
    let route_authority_ready = route_authority.is_some();
    let mut output = json!({
        "schema": SCHEMA,
        "status": if route_authority_ready { PREFLIGHTED_STATUS } else { PREPARED_STATUS },
        "preflight_ready_for_future_outer_authority_only": route_authority_ready,
        "antecedent_l0_l1_component": binding_json(assessment),
        "antecedent_scope": {
            "source_token_id": SOURCE_TOKEN_ID,
            "fresh_l0_dispatches": L0_DISPATCHES,
            "fresh_l1_prefix_dispatches": L1_PREFIX_DISPATCHES,
            "antecedent_total_dispatches": L0_DISPATCHES + L1_PREFIX_DISPATCHES,
            "opaque_same_runtime_continuation_required": true,
            "prior_capture_is_provenance_only_not_transferable_execution_input": true,
        },
        "l1_source_token_route_authority": {
            "present_and_valid": route_authority_ready,
            "binding": route_authority.map(binding_json),
            "requirements": route_authority_requirements(),
        },
        "future_joint_command_graph": {
            "source_token_id": SOURCE_TOKEN_ID,
            "l0_layer": L0_LAYER,
            "l1_layer": L1_LAYER,
            "l1_linear_state_slot": L1_LINEAR_STATE_SLOT,
            "l0_reencode_dispatches": L0_DISPATCHES,
            "l1_prefix_dispatches": L1_PREFIX_DISPATCHES,
            "l1_moe_suffix_dispatches": L1_MOE_SUFFIX_DISPATCHES,
            "total_dispatches": TOTAL_DISPATCHES,
            "single_runtime_required": true,
            "single_token_command_buffer_required": true,
            "single_fence_after_all_dispatches_required": true,
            "non_timed_trace_required": true,
            "exact_kernel_trace": dispatch_trace(),
        },
        "future_receipt_contract": future_receipt_contract(),
        "remaining_prerequisites": if route_authority_ready {
            vec![
                "build a same-runtime L1 suffix host that consumes the opaque fresh L0-L1 continuation",
                "independent outer preflight/replay/lease/reaper lifecycle for the 46-dispatch component",
                "one fresh component-only lease and capture only after explicit authorization",
            ]
        } else {
            vec![
                "create and seal a current-admitted source-token L1 CPU router/all-ten authority with the required 36 direct-packed descriptors",
                "build a same-runtime L1 suffix host that consumes the opaque fresh L0-L1 continuation",
                "independent outer preflight/replay/lease/reaper lifecycle for the 46-dispatch component",
            ]
        },
        "claim_boundary": {
            "cpu_file_only_preflight": true,
            "artifact_scan_or_payload_open_performed": false,
            "metal_context_or_dispatch_performed": false,
            "lease_issued_or_consumed": false,
            "watcher_server_hcli_or_runtime_changed": false,
            "future_l1_moe_component_is_not_a_complete_token_or_decoder": true,
            "tps_tg_or_tournament_claim_earned": false,
        },
    });
    seal(&mut output)?;
    Ok(output)
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create {}: {error}", path.display()))?;
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("cannot write {}: {error}", path.display()))
}

fn run(args: Args) -> Result<Value, String> {
    let assessment = read_bound_document(&args.joint_assessment, "joint assessment")?;
    let route_authority = args
        .l1_route_authority
        .as_deref()
        .map(|path| read_bound_document(path, "Layer-1 route authority"))
        .transpose()?;
    let output = build_preflight(&assessment, route_authority.as_ref())?;
    let mut bytes = serde_json::to_vec_pretty(&output)
        .map_err(|error| format!("cannot render preflight: {error}"))?;
    bytes.push(b'\n');
    write_new(&args.out, &bytes)?;
    Ok(output)
}

fn main() {
    let args = match parse_args(env::args().skip(1)) {
        Ok(args) => args,
        Err(error) => {
            eprintln!("Qwen80 Layer-1 MoE completion preflight argument refusal: {error}");
            process::exit(2);
        }
    };
    match run(args) {
        Ok(output) => {
            println!(
                "{}",
                json!({
                    "schema": output["schema"],
                    "status": output["status"],
                    "seal_sha256": output["seal_sha256"],
                    "metal_or_gpu_activity_performed": false,
                })
            );
        }
        Err(error) => {
            eprintln!("Qwen80 Layer-1 MoE completion preflight refusal: {error}");
            process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn numbered_sha(value: u64) -> String {
        format!("{value:064x}")
    }

    fn seal_document(mut value: Value) -> Value {
        seal(&mut value).expect("fixture document must seal");
        value
    }

    fn assessment_document() -> Value {
        seal_document(json!({
            "schema": JOINT_ASSESSMENT_SCHEMA,
            "status": JOINT_ASSESSMENT_STATUS,
            "earned_component_only": true,
            "component_scope": {
                "source_token_id": SOURCE_TOKEN_ID,
                "fresh_l0_dispatches": L0_DISPATCHES,
                "fresh_l1_slot1_prefix_dispatches": L1_PREFIX_DISPATCHES,
                "fresh_total_dispatches": L0_DISPATCHES + L1_PREFIX_DISPATCHES,
                "opaque_same_runtime_continuation_required": true,
                "single_fence_required": true,
                "full_layer_or_token_decoder_earned": false,
            },
            "sealed_inputs": {
                "joint_inner_capture": {
                    "present": true, "valid": true,
                    "document_sha256": sha('a'), "document_seal_sha256": sha('b'),
                },
                "joint_outer_terminal": {
                    "present": true, "valid": true,
                    "document_sha256": sha('c'), "document_seal_sha256": sha('d'),
                },
                "joint_lease_release": {
                    "present": true, "valid": true,
                    "document_sha256": sha('e'), "document_seal_sha256": sha('f'),
                },
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
        }))
    }

    fn bound(value: Value) -> BoundDocument {
        let seal = verify_seal(&value, "fixture").expect("fixture must verify");
        BoundDocument {
            path: PathBuf::from("/tmp/fixture.json"),
            raw_sha256: sha('0'),
            document_sha256: document_sha256(&value, "fixture").unwrap(),
            document_seal_sha256: seal,
            value,
        }
    }

    fn descriptor(expected: &Value, nonce: u64) -> Value {
        let expected = object(expected, "expected descriptor").unwrap();
        let required_layout =
            object_field(expected, "required_layout", "expected descriptor").unwrap();
        json!({
            "tensor_name": expected["tensor_name"],
            "shape": expected["shape"],
            "group_size": expected["group_size"],
            "artifact_sha256": numbered_sha(nonce * 3 + 1),
            "direct_packed_payload_sha256": numbered_sha(nonce * 3 + 2),
            "header_sha256": numbered_sha(nonce * 3 + 3),
            "payload_bytes": 64,
            "layout": required_layout,
        })
    }

    fn route_authority_document(assessment: &BoundDocument) -> Value {
        let fixed = expected_fixed_payloads()
            .iter()
            .enumerate()
            .map(|(index, expected)| descriptor(expected, index as u64))
            .collect::<Vec<_>>();
        let ids = (10_u64..20).collect::<Vec<_>>();
        let weights = vec![0.1_f64; TOP_K];
        let waves = ids
            .iter()
            .enumerate()
            .map(|(index, expert)| {
                json!({
                    "wave_index": index,
                    "layer": L1_LAYER,
                    "expert_id": expert,
                    "normalized_weight": weights[index],
                    "normalized_weight_bits_hex": format!("0x{:016x}", weights[index].to_bits()),
                    "gate": descriptor(&expected_route_descriptor("gate", *expert), 100 + (index * 3) as u64),
                    "up": descriptor(&expected_route_descriptor("up", *expert), 101 + (index * 3) as u64),
                    "down": descriptor(&expected_route_descriptor("down", *expert), 102 + (index * 3) as u64),
                })
            })
            .collect::<Vec<_>>();
        seal_document(json!({
            "schema": ROUTE_AUTHORITY_SCHEMA,
            "status": ROUTE_AUTHORITY_STATUS,
            "fixture_or_synthetic": false,
            "metal_or_gpu_activity_performed": false,
            "source_binding": {
                "model_id": MODEL_ID,
                "model_key": MODEL_KEY,
                "source_repository": SOURCE_REPOSITORY,
                "source_revision": SOURCE_REVISION,
                "manifest_document_sha256": MANIFEST_DOCUMENT_SHA256,
                "manifest_seal_sha256": MANIFEST_SEAL_SHA256,
                "admission_receipt_seal_sha256": ADMISSION_RECEIPT_SEAL_SHA256,
                "joint_l0_l1_assessment": {
                    "document_sha256": assessment.document_sha256,
                    "document_seal_sha256": assessment.document_seal_sha256,
                },
                "prior_joint_assessment_is_provenance_only": true,
                "cross_process_pinned_buffer_import_allowed": false,
            },
            "source_token_l1_cpu_oracle": {
                "source_token_id": SOURCE_TOKEN_ID,
                "layer": L1_LAYER,
                "linear_state_slot": L1_LINEAR_STATE_SLOT,
                "fresh_l0_reencode_dispatches": L0_DISPATCHES,
                "fresh_l1_prefix_dispatches": L1_PREFIX_DISPATCHES,
                "cpu_oracle_reencodes_l0_then_l1_prefix": true,
                "zero_initial_l0_state": true,
                "zero_initial_l1_slot1_state": true,
                "source_input_f32le_sha256": sha('a'),
                "l0_second_residual_cpu_f32le_sha256": sha('b'),
                "l1_prefix_input_cpu_f32le_sha256": sha('c'),
                "l1_first_residual_cpu_f32le_sha256": sha('d'),
                "l1_post_attention_normalized_hidden_cpu_f32le_sha256": sha('e'),
                "l1_router_logits_cpu_f32le_sha256": sha('f'),
                "l1_post_conv_state_cpu_f32le_sha256": sha('1'),
                "l1_post_recurrent_state_cpu_f32le_sha256": sha('2'),
            },
            "source_token_router_evidence": {
                "logit_count": EXPERTS,
                "top_k": TOP_K,
                "selection": "source_qwen80_topk_router",
                "tie_break": "lowest_expert_id_within_route_tie_epsilon",
                "softmax": "subtract_max_exp_f32",
                "route_tie_epsilon_source": "HAWKING_DS_ROUTE_TIE_EPS",
                "route_tie_epsilon": 0.0_f32,
                "route_tie_epsilon_f32_bits_hex": "0x00000000",
                "selected_probabilities_renormalized": true,
                "source_stable_route_ids": ids,
                "source_stable_normalized_weights": weights,
                "weights_sum": 1.0_f64,
            },
            "fixed_l1_payloads": fixed,
            "deterministic_waves": waves,
            "rawls_real_all_ten_provenance_gate": {
                "all_ten_source_bindings_complete": true,
                "expected_layer": L1_LAYER,
                "execution_receipt_required_for_each_wave": true,
                "direct_packed_execution_required_for_each_wave": true,
                "source_bound_input_required_for_each_wave": true,
                "route_combine_receipt_required_separately": true,
                "shared_expert_receipt_required_separately": true,
                "first_and_second_residual_receipts_required_separately": true,
                "rejects_tensor_substitution": true,
                "rejects_route_reorder": true,
                "rejects_duplicate_experts": true,
                "rejects_missing_tensor_or_weight": true,
            },
            "route_execution_performed": false,
            "route_combine_performed": false,
            "shared_expert_performed": false,
            "residual_combine_performed": false,
            "metal_device_or_dispatch_performed": false,
            "model_execution_performed": false,
            "hcli_execution_performed": false,
            "tps_or_tg_measurement_performed": false,
            "complete_layer_or_decoder_claim_earned": false,
        }))
    }

    #[test]
    fn exact_future_trace_is_fresh_l0_23_plus_l1_9_plus_l1_moe_14() {
        let trace = dispatch_trace();
        assert_eq!(trace.len(), TOTAL_DISPATCHES as usize);
        assert_eq!(trace[0]["phase"], "fresh_l0_reencode");
        assert_eq!(
            trace[22]["kernel"],
            "qwen80_moe_wave_aggregate_second_residual_add_shared_residual"
        );
        assert_eq!(trace[23]["phase"], "fresh_l1_deltanet_prefix");
        assert_eq!(trace[31]["kernel"], "qwen_next_add_residual");
        assert_eq!(trace[32]["phase"], "fresh_l1_moe_suffix");
        assert_eq!(
            trace[45]["kernel"],
            "qwen80_moe_wave_aggregate_second_residual_add_shared_residual"
        );
    }

    #[test]
    fn prepared_contract_binds_earned_antecedent_but_requires_new_l1_route_authority() {
        let assessment = bound(assessment_document());
        let result =
            build_preflight(&assessment, None).expect("antecedent-only contract must build");
        assert_eq!(result["status"], PREPARED_STATUS);
        assert_eq!(
            result["l1_source_token_route_authority"]["present_and_valid"],
            false
        );
        assert_eq!(
            result["future_joint_command_graph"]["total_dispatches"],
            TOTAL_DISPATCHES
        );
        verify_seal(&result, "prepared contract").expect("output must be sealed");
    }

    #[test]
    fn route_authority_requires_exact_l1_routes_descriptors_and_provenance_only_antecedent() {
        let assessment = bound(assessment_document());
        let authority = bound(route_authority_document(&assessment));
        let result = build_preflight(&assessment, Some(&authority))
            .expect("exact current-admitted L1 authority must preflight");
        assert_eq!(result["status"], PREFLIGHTED_STATUS);
        assert_eq!(
            result["l1_source_token_route_authority"]["requirements"]["all_ten_waves_required"]
                ["count"],
            TOP_K
        );
    }

    #[test]
    fn route_authority_refuses_legacy_layer_or_reordered_route() {
        let assessment = bound(assessment_document());
        let mut invalid = route_authority_document(&assessment);
        invalid["deterministic_waves"][0]["layer"] = json!(0);
        invalid.as_object_mut().unwrap().remove("seal_sha256");
        seal(&mut invalid).unwrap();
        assert!(build_preflight(&assessment, Some(&bound(invalid))).is_err());

        let mut reordered = route_authority_document(&assessment);
        reordered["deterministic_waves"]
            .as_array_mut()
            .unwrap()
            .swap(0, 1);
        reordered.as_object_mut().unwrap().remove("seal_sha256");
        seal(&mut reordered).unwrap();
        assert!(build_preflight(&assessment, Some(&bound(reordered))).is_err());
    }

    #[test]
    fn route_authority_refuses_detached_or_duplicate_payloads() {
        let assessment = bound(assessment_document());
        let mut invalid = route_authority_document(&assessment);
        invalid["source_binding"]["cross_process_pinned_buffer_import_allowed"] = json!(true);
        invalid.as_object_mut().unwrap().remove("seal_sha256");
        seal(&mut invalid).unwrap();
        assert!(build_preflight(&assessment, Some(&bound(invalid))).is_err());

        let mut duplicate = route_authority_document(&assessment);
        let hash =
            duplicate["deterministic_waves"][0]["gate"]["direct_packed_payload_sha256"].clone();
        duplicate["deterministic_waves"][1]["gate"]["direct_packed_payload_sha256"] = hash;
        duplicate.as_object_mut().unwrap().remove("seal_sha256");
        seal(&mut duplicate).unwrap();
        assert!(build_preflight(&assessment, Some(&bound(duplicate))).is_err());

        let mut duplicate_artifact = route_authority_document(&assessment);
        let hash = duplicate_artifact["deterministic_waves"][0]["gate"]["artifact_sha256"].clone();
        duplicate_artifact["deterministic_waves"][1]["gate"]["artifact_sha256"] = hash;
        duplicate_artifact
            .as_object_mut()
            .unwrap()
            .remove("seal_sha256");
        seal(&mut duplicate_artifact).unwrap();
        assert!(build_preflight(&assessment, Some(&bound(duplicate_artifact))).is_err());
    }

    #[test]
    fn parser_refuses_relative_or_overwritten_output() {
        assert!(parse_args([
            "--joint-assessment".into(),
            "/tmp/assessment.json".into(),
            "--out".into(),
            "relative.json".into(),
        ])
        .is_err());
        assert!(parse_args([
            "--joint-assessment".into(),
            "/tmp/assessment.json".into(),
            "--out".into(),
            "/tmp/missing-parent/output.json".into(),
        ])
        .is_err());
    }
}
