//! CPU-only Qwen80 source-token L0-to-L1 handoff authority.
//!
//! The source-token L0 strict-Metal component earns a same-command-buffer
//! first-residual through second-residual parity record.  It deliberately
//! stops before the next layer.  In particular, an output SHA is not a
//! retained device buffer, and a prefix parity record is not proof that the
//! DeltaNet state bytes were committed or checkpointed.  This program joins
//! that sealed outer/inner pair to the static per-session state-layout
//! contract and makes the missing L0-to-L1 device handoff witness explicit.
//!
//! It is metadata-only: it never opens a complete artifact, constructs a
//! Metal context, allocates a device buffer, mutates a watcher, or executes a
//! model operation.  A current component record is expected to produce a
//! sealed *incomplete* assessment.  A future positive assessment remains
//! component-only; it cannot qualify a full layer, token, decoder, server,
//! HCLI, TPS, TG, or tournament result.

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const SCHEMA: &str = "hawking.ascension.qwen80_l0_to_layer1_handoff_authority.v1";
const INCOMPLETE_STATUS: &str =
    "ASSESSED_QWEN80_SOURCE_TOKEN_L0_TO_LAYER1_HANDOFF_INCOMPLETE_MISSING_RETAINED_DEVICE_OUTPUT_AND_POST_STATE_WITNESSES";
const READY_COMPONENT_ONLY_STATUS: &str =
    "VALIDATED_QWEN80_SOURCE_TOKEN_L0_TO_LAYER1_HANDOFF_WITNESS_COMPONENT_ONLY_NOT_COMPLETE_LAYER_OR_TOKEN";
const OUTER_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_true_input_all_ten_moe_graph_outer_launcher.v1";
const OUTER_STATUS: &str =
    "CAPTURED_QWEN80_SOURCE_TOKEN_TRUE_INPUT_ALL_TEN_MOE_OUTER_TERMINAL_COMPONENT_ONLY";
const INNER_SCHEMA: &str = "hawking.ascension.qwen80_source_token_all_ten_true_moe_graph_device.v1";
const INNER_STATUS: &str =
    "EARNED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_LAYER0_TRUE_INPUT_ALL_TEN_ROUTE_SHARED_SECOND_RESIDUAL_STRICT_MATH_METAL_COMPONENT_NOT_COMPLETE_LAYER_OR_TOKEN";
const STATE_LAYOUT_SCHEMA: &str = "hawking.ascension.qwen80_device_state_buffer_layout_contract.v1";
const STATE_LAYOUT_STATUS: &str =
    "NOT_READY_NO_DEVICE_ALLOCATION_NO_STATE_PARITY_NO_ROLLBACK_CAPTURE_QWEN80_PER_SESSION_BUFFER_LAYOUT_CONTRACT";
const HANDOFF_WITNESS_SCHEMA: &str =
    "hawking.ascension.qwen80_l0_to_layer1_device_handoff_witness.v1";
const HANDOFF_WITNESS_STATUS: &str =
    "CAPTURED_QWEN80_SOURCE_TOKEN_L0_TO_LAYER1_RETAINED_DEVICE_HANDOFF_COMPONENT_ONLY";
const MODEL_KEY: &str = "qwen80";
const SOURCE_REVISION: &str = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb";
const MANIFEST_DOCUMENT_SHA: &str =
    "a0fcac0401a7962402bb8cb87d5055c83667b39575f9e0f4c7470d080758aa10";
const MANIFEST_SEAL: &str = "14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b";
const ADMISSION_RECEIPT_SEAL: &str =
    "939b41322363da3db774a2530b207bf380ed641d23cae671fc6438c0eecbf628";
const HIDDEN: u64 = 2_048;
const HIDDEN_BYTES: u64 = HIDDEN * 4;
const SOURCE_TOKEN_ID: u64 = 1;
const PREFIX_DISPATCHES: u64 = 9;
const SUFFIX_DISPATCHES: u64 = 14;
const TOTAL_DISPATCHES: u64 = PREFIX_DISPATCHES + SUFFIX_DISPATCHES;

#[derive(Clone, Debug)]
struct BoundFile {
    path: PathBuf,
    bytes: u64,
    sha256: String,
}

impl BoundFile {
    fn json(&self) -> Value {
        json!({
            "path": self.path,
            "present": true,
            "bytes": self.bytes,
            "sha256": self.sha256,
        })
    }
}

#[derive(Debug)]
struct Args {
    outer_terminal: PathBuf,
    inner_receipt: PathBuf,
    state_layout: PathBuf,
    out: PathBuf,
}

#[derive(Clone, Debug)]
struct StateRange {
    allocation_id: String,
    slot: u64,
    offset_bytes: u64,
    capacity_bytes: u64,
}

impl StateRange {
    fn json(&self) -> Value {
        json!({
            "allocation_id": self.allocation_id,
            "slot": self.slot,
            "offset_bytes": self.offset_bytes,
            "capacity_bytes": self.capacity_bytes,
        })
    }
}

#[derive(Clone, Debug)]
struct L0L1StateLayout {
    session_id: String,
    l0_active_conv: StateRange,
    l0_active_recurrent: StateRange,
    l0_rollback_conv: StateRange,
    l0_rollback_recurrent: StateRange,
    l1_active_conv: StateRange,
    l1_active_recurrent: StateRange,
    l1_rollback_conv: StateRange,
    l1_rollback_recurrent: StateRange,
}

#[derive(Clone, Debug)]
struct ComponentEvidence {
    outer: BoundFile,
    outer_seal: String,
    inner: BoundFile,
    state_layout_file: BoundFile,
    state_layout: L0L1StateLayout,
    input_hidden_sha: String,
    initial_conv_state_sha: String,
    initial_recurrent_state_sha: String,
    device_first_residual_sha: String,
    second_residual_sha: String,
    route_ids: Vec<u64>,
    route_witness_count: u64,
}

enum HandoffAssessment {
    Incomplete { missing: Vec<&'static str> },
    Ready,
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| {
            byte.is_ascii_digit() || (byte.is_ascii_lowercase() && byte.is_ascii_hexdigit())
        })
}

fn canonical_regular(path: &Path, label: &str) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err(format!("{label} must be an absolute path"));
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    fs::canonicalize(path).map_err(|error| format!("cannot canonicalize {label}: {error}"))
}

fn read_json(path: &Path, label: &str) -> Result<(BoundFile, Value), String> {
    let path = canonical_regular(path, label)?;
    let raw = fs::read(&path).map_err(|error| format!("cannot read {label}: {error}"))?;
    let value: Value =
        serde_json::from_slice(&raw).map_err(|error| format!("cannot parse {label}: {error}"))?;
    if !value.is_object() {
        return Err(format!("{label} must be a JSON object"));
    }
    Ok((
        BoundFile {
            path,
            bytes: raw.len() as u64,
            sha256: sha256_hex(&raw),
        },
        value,
    ))
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

fn u64_field(object: &Map<String, Value>, field: &str, label: &str) -> Result<u64, String> {
    object
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{label}.{field} must be an unsigned integer"))
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
    let observed = string_field(object, field, label)?;
    if observed != expected {
        return Err(format!(
            "{label}.{field}={observed:?}, expected {expected:?}"
        ));
    }
    Ok(())
}

fn sha_field(object: &Map<String, Value>, field: &str, label: &str) -> Result<String, String> {
    let value = string_field(object, field, label)?;
    if !is_sha256(value) {
        return Err(format!("{label}.{field} must be a lowercase SHA-256"));
    }
    Ok(value.into())
}

fn canonical_json(value: &Value) -> Result<Value, String> {
    match value {
        Value::Array(values) => values
            .iter()
            .map(canonical_json)
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        Value::Object(values) => {
            let mut ordered = BTreeMap::new();
            for (key, value) in values {
                ordered.insert(key.clone(), canonical_json(value)?);
            }
            Ok(Value::Object(ordered.into_iter().collect()))
        }
        value => Ok(value.clone()),
    }
}

fn seal(value: &mut Value) -> Result<String, String> {
    let object = object(value, "authority output")?;
    if object.contains_key("seal_sha256") {
        return Err("authority output must be unsealed before sealing".into());
    }
    let canonical = canonical_json(value)?;
    let seal = sha256_hex(&serde_json::to_vec(&canonical).map_err(|error| error.to_string())?);
    value
        .as_object_mut()
        .expect("authority output object")
        .insert("seal_sha256".into(), Value::String(seal.clone()));
    Ok(seal)
}

fn verify_seal(value: &Value, label: &str) -> Result<String, String> {
    let object = object(value, label)?;
    let observed = sha_field(object, "seal_sha256", label)?;
    let mut unsigned = object.clone();
    unsigned.remove("seal_sha256");
    let canonical = canonical_json(&Value::Object(unsigned))?;
    let expected = sha256_hex(&serde_json::to_vec(&canonical).map_err(|error| error.to_string())?);
    if observed != expected {
        return Err(format!("{label} seal mismatch"));
    }
    Ok(observed)
}

fn validate_source_identity(object: &Map<String, Value>, label: &str) -> Result<(), String> {
    exact_string(
        object,
        "manifest_document_sha256",
        MANIFEST_DOCUMENT_SHA,
        label,
    )?;
    exact_string(object, "manifest_seal_sha256", MANIFEST_SEAL, label)?;
    exact_string(
        object,
        "admission_receipt_seal_sha256",
        ADMISSION_RECEIPT_SEAL,
        label,
    )?;
    Ok(())
}

fn state_range(
    binding: &Map<String, Value>,
    arena: &str,
    domain: &str,
    label: &str,
) -> Result<StateRange, String> {
    let field = match arena {
        "Active" => "active_ranges",
        "Rollback" => "rollback_ranges",
        _ => return Err("unsupported state arena".into()),
    };
    let ranges = array_field(binding, field, label)?;
    let range = ranges
        .iter()
        .filter_map(Value::as_object)
        .find(|range| {
            range.get("arena").and_then(Value::as_str) == Some(arena)
                && range.get("domain").and_then(Value::as_str) == Some(domain)
        })
        .ok_or_else(|| format!("{label}.{field} lacks {arena}/{domain}"))?;
    let allocation_id = string_field(range, "allocation_id", label)?.to_owned();
    let slot = u64_field(range, "slot", label)?;
    let offset_bytes = u64_field(range, "offset_bytes", label)?;
    let capacity_bytes = u64_field(range, "capacity_bytes", label)?;
    if allocation_id.trim().is_empty() || capacity_bytes <= offset_bytes {
        return Err(format!("{label} has invalid {arena}/{domain} range"));
    }
    Ok(StateRange {
        allocation_id,
        slot,
        offset_bytes,
        capacity_bytes,
    })
}

fn extract_state_layout(value: &Value) -> Result<L0L1StateLayout, String> {
    let root = object(value, "state layout")?;
    exact_string(root, "schema", STATE_LAYOUT_SCHEMA, "state layout")?;
    exact_string(root, "status", STATE_LAYOUT_STATUS, "state layout")?;
    bool_field(root, "ready_for_decoder_graph", false, "state layout")?;
    bool_field(
        root,
        "actual_device_allocation_performed",
        false,
        "state layout",
    )?;
    let source = object_field(root, "source_identity", "state layout")?;
    exact_string(
        source,
        "model_key",
        MODEL_KEY,
        "state layout.source_identity",
    )?;
    exact_string(
        source,
        "source_revision",
        SOURCE_REVISION,
        "state layout.source_identity",
    )?;
    exact_string(
        source,
        "manifest_seal_sha256",
        MANIFEST_SEAL,
        "state layout.source_identity",
    )?;
    exact_string(
        source,
        "admission_receipt_seal_sha256",
        ADMISSION_RECEIPT_SEAL,
        "state layout.source_identity",
    )?;
    let session = object_field(root, "session_layout", "state layout")?;
    let session_id = string_field(session, "session_id", "state layout.session_layout")?.to_owned();
    if u64_field(session, "max_seq_len", "state layout.session_layout")? == 0 {
        return Err("state layout.session_layout.max_seq_len must be positive".into());
    }
    let bindings = array_field(session, "layer_bindings", "state layout.session_layout")?;
    if bindings.len() != 48 {
        return Err("state layout must contain exactly 48 layer bindings".into());
    }
    let l0 = bindings
        .first()
        .and_then(Value::as_object)
        .ok_or("state layout layer 0 must be an object")?;
    let l1 = bindings
        .get(1)
        .and_then(Value::as_object)
        .ok_or("state layout layer 1 must be an object")?;
    for (binding, layer, slot, label) in [
        (l0, 0, 0, "state layout layer 0"),
        (l1, 1, 1, "state layout layer 1"),
    ] {
        if u64_field(binding, "layer", label)? != layer
            || u64_field(binding, "state_slot", label)? != slot
        {
            return Err(format!("{label} does not bind DeltaNet state slot {slot}"));
        }
        exact_string(binding, "mixer", "DeltaNet", label)?;
    }
    let layout = L0L1StateLayout {
        session_id,
        l0_active_conv: state_range(l0, "Active", "DeltaNetConv", "state layout layer 0")?,
        l0_active_recurrent: state_range(
            l0,
            "Active",
            "DeltaNetRecurrent",
            "state layout layer 0",
        )?,
        l0_rollback_conv: state_range(l0, "Rollback", "DeltaNetConv", "state layout layer 0")?,
        l0_rollback_recurrent: state_range(
            l0,
            "Rollback",
            "DeltaNetRecurrent",
            "state layout layer 0",
        )?,
        l1_active_conv: state_range(l1, "Active", "DeltaNetConv", "state layout layer 1")?,
        l1_active_recurrent: state_range(
            l1,
            "Active",
            "DeltaNetRecurrent",
            "state layout layer 1",
        )?,
        l1_rollback_conv: state_range(l1, "Rollback", "DeltaNetConv", "state layout layer 1")?,
        l1_rollback_recurrent: state_range(
            l1,
            "Rollback",
            "DeltaNetRecurrent",
            "state layout layer 1",
        )?,
    };
    validate_state_layout_separation(&layout)?;
    Ok(layout)
}

fn validate_state_layout_separation(layout: &L0L1StateLayout) -> Result<(), String> {
    for (left, right, label) in [
        (
            &layout.l0_active_conv,
            &layout.l1_active_conv,
            "active conv",
        ),
        (
            &layout.l0_active_recurrent,
            &layout.l1_active_recurrent,
            "active recurrent",
        ),
        (
            &layout.l0_rollback_conv,
            &layout.l1_rollback_conv,
            "rollback conv",
        ),
        (
            &layout.l0_rollback_recurrent,
            &layout.l1_rollback_recurrent,
            "rollback recurrent",
        ),
    ] {
        if left.allocation_id != right.allocation_id
            || left.slot != 0
            || right.slot != 1
            || left.capacity_bytes > right.offset_bytes
        {
            return Err(format!("state layout {label} L0/L1 ranges alias or drift"));
        }
    }
    for (active, rollback, label) in [
        (&layout.l0_active_conv, &layout.l0_rollback_conv, "L0 conv"),
        (
            &layout.l0_active_recurrent,
            &layout.l0_rollback_recurrent,
            "L0 recurrent",
        ),
        (&layout.l1_active_conv, &layout.l1_rollback_conv, "L1 conv"),
        (
            &layout.l1_active_recurrent,
            &layout.l1_rollback_recurrent,
            "L1 recurrent",
        ),
    ] {
        if active.allocation_id == rollback.allocation_id {
            return Err(format!("state layout {label} active/rollback ids alias"));
        }
    }
    Ok(())
}

fn u64_array(value: &[Value], label: &str) -> Result<Vec<u64>, String> {
    value
        .iter()
        .map(|value| {
            value
                .as_u64()
                .ok_or_else(|| format!("{label} must contain unsigned integers"))
        })
        .collect()
}

fn validate_component(
    outer: BoundFile,
    outer_value: &Value,
    inner: BoundFile,
    inner_value: &Value,
    state_layout_file: BoundFile,
    state_layout_value: &Value,
) -> Result<ComponentEvidence, String> {
    let outer_object = object(outer_value, "L0 outer terminal")?;
    exact_string(outer_object, "schema", OUTER_SCHEMA, "L0 outer terminal")?;
    exact_string(outer_object, "status", OUTER_STATUS, "L0 outer terminal")?;
    let outer_seal = verify_seal(outer_value, "L0 outer terminal")?;
    let inner_binding = object_field(outer_object, "inner_probe_capture", "L0 outer terminal")?;
    if inner_binding.get("present").and_then(Value::as_bool) != Some(true)
        || string_field(
            inner_binding,
            "path",
            "L0 outer terminal.inner_probe_capture",
        )? != inner.path.to_string_lossy()
        || u64_field(
            inner_binding,
            "bytes",
            "L0 outer terminal.inner_probe_capture",
        )? != inner.bytes
        || sha_field(
            inner_binding,
            "sha256",
            "L0 outer terminal.inner_probe_capture",
        )? != inner.sha256
    {
        return Err("L0 outer terminal does not bind supplied inner receipt bytes".into());
    }
    exact_string(
        inner_binding,
        "schema",
        INNER_SCHEMA,
        "L0 outer terminal.inner_probe_capture",
    )?;
    exact_string(
        inner_binding,
        "status",
        INNER_STATUS,
        "L0 outer terminal.inner_probe_capture",
    )?;
    let source = object_field(outer_object, "source_binding", "L0 outer terminal")?;
    let artifact = object_field(
        source,
        "artifact_identity",
        "L0 outer terminal.source_binding",
    )?;
    validate_source_identity(
        artifact,
        "L0 outer terminal.source_binding.artifact_identity",
    )?;
    let implementation = object_field(
        source,
        "implementation_binding",
        "L0 outer terminal.source_binding",
    )?;
    if u64_field(
        implementation,
        "source_token_id",
        "L0 outer terminal.source_binding.implementation_binding",
    )? != SOURCE_TOKEN_ID
        || u64_field(
            implementation,
            "prefix_dispatches",
            "L0 outer terminal.source_binding.implementation_binding",
        )? != PREFIX_DISPATCHES
        || u64_field(
            implementation,
            "suffix_dispatches",
            "L0 outer terminal.source_binding.implementation_binding",
        )? != SUFFIX_DISPATCHES
        || u64_field(
            implementation,
            "total_dispatches",
            "L0 outer terminal.source_binding.implementation_binding",
        )? != TOTAL_DISPATCHES
    {
        return Err("L0 outer terminal same-command-graph dispatch contract drifted".into());
    }
    bool_field(
        implementation,
        "same_command_buffer_fence_required",
        true,
        "L0 outer terminal.source_binding.implementation_binding",
    )?;

    let inner_object = object(inner_value, "L0 inner receipt")?;
    exact_string(inner_object, "schema", INNER_SCHEMA, "L0 inner receipt")?;
    exact_string(inner_object, "status", INNER_STATUS, "L0 inner receipt")?;
    exact_string(inner_object, "mode", "metal", "L0 inner receipt")?;
    bool_field(
        inner_object,
        "metal_device_or_dispatch_performed",
        true,
        "L0 inner receipt",
    )?;
    bool_field(inner_object, "component_only", true, "L0 inner receipt")?;
    bool_field(
        inner_object,
        "complete_layer_or_token_performed",
        false,
        "L0 inner receipt",
    )?;
    bool_field(
        inner_object,
        "raw_bf16_or_safetensors_opened",
        false,
        "L0 inner receipt",
    )?;
    let inner_artifact = object_field(inner_object, "artifact_binding", "L0 inner receipt")?;
    validate_source_identity(inner_artifact, "L0 inner receipt.artifact_binding")?;
    exact_string(
        inner_artifact,
        "source_revision",
        SOURCE_REVISION,
        "L0 inner receipt.artifact_binding",
    )?;
    if u64_field(inner_artifact, "layer", "L0 inner receipt.artifact_binding")? != 0
        || u64_field(
            inner_artifact,
            "linear_state_slot",
            "L0 inner receipt.artifact_binding",
        )? != 0
    {
        return Err("L0 inner receipt does not bind layer 0 / DeltaNet slot 0".into());
    }
    let graph = object_field(inner_object, "same_command_graph", "L0 inner receipt")?;
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
        "same_command_graph_required",
        "same_command_graph_retained",
        "command_buffer_fenced_once_after_prefix_and_suffix",
        "encoded_kernel_order_matches_expected",
    ] {
        bool_field(graph, field, true, "L0 inner receipt.same_command_graph")?;
    }
    let kernels = array_field(
        graph,
        "encoded_kernel_names",
        "L0 inner receipt.same_command_graph",
    )?;
    if kernels.len() as u64 != TOTAL_DISPATCHES
        || kernels.iter().any(|kernel| kernel.as_str().is_none())
    {
        return Err("L0 inner receipt must retain exactly 23 named kernel dispatches".into());
    }
    let phase = object_field(inner_object, "execution_phase", "L0 inner receipt")?;
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
        return Err("L0 inner receipt execution phase dispatch count drifted".into());
    }
    let prefix = object_field(inner_object, "prefix_parity", "L0 inner receipt")?;
    if u64_field(prefix, "source_token_id", "L0 inner receipt.prefix_parity")? != SOURCE_TOKEN_ID
        || u64_field(prefix, "elements", "L0 inner receipt.prefix_parity")? != HIDDEN
        || u64_field(prefix, "bytes", "L0 inner receipt.prefix_parity")? != HIDDEN_BYTES
    {
        return Err("L0 inner receipt prefix hidden geometry drifted".into());
    }
    let input_hidden_sha = sha_field(
        prefix,
        "input_hidden_f32le_sha256",
        "L0 inner receipt.prefix_parity",
    )?;
    let initial_conv_state_sha = sha_field(
        prefix,
        "initial_conv_state_f32le_sha256",
        "L0 inner receipt.prefix_parity",
    )?;
    let initial_recurrent_state_sha = sha_field(
        prefix,
        "initial_recurrent_state_f32le_sha256",
        "L0 inner receipt.prefix_parity",
    )?;
    let device_first_residual_sha = sha_field(
        prefix,
        "device_first_residual_f32le_sha256",
        "L0 inner receipt.prefix_parity",
    )?;
    let prefix_antecedent = object_field(
        source,
        "first_residual_receipt",
        "L0 outer terminal.source_binding",
    )?;
    if sha_field(
        prefix_antecedent,
        "output_sha256",
        "L0 outer terminal.source_binding.first_residual_receipt",
    )? != device_first_residual_sha
    {
        return Err("L0 outer prefix antecedent does not match inner device first residual".into());
    }
    let route_guard = object_field(inner_object, "route_guard_readback", "L0 inner receipt")?;
    bool_field(
        route_guard,
        "passed",
        true,
        "L0 inner receipt.route_guard_readback",
    )?;
    if u64_field(
        route_guard,
        "value",
        "L0 inner receipt.route_guard_readback",
    )? != 1
    {
        return Err("L0 inner receipt.route_guard_readback.value must be one".into());
    }
    let expected_ids = u64_array(
        array_field(
            route_guard,
            "expected_ids",
            "L0 inner receipt.route_guard_readback",
        )?,
        "L0 inner receipt.route_guard_readback.expected_ids",
    )?;
    let observed_ids = u64_array(
        array_field(
            route_guard,
            "observed_ids",
            "L0 inner receipt.route_guard_readback",
        )?,
        "L0 inner receipt.route_guard_readback.observed_ids",
    )?;
    if expected_ids.len() != 10
        || observed_ids != expected_ids
        || expected_ids.iter().copied().collect::<BTreeSet<_>>().len() != 10
    {
        return Err(
            "L0 inner receipt route guard must retain ten ordered unique source routes".into(),
        );
    }
    let parity = object_field(inner_object, "readback_parity", "L0 inner receipt")?;
    let route_witness_count = u64_field(
        parity,
        "all_ten_route_witness_count",
        "L0 inner receipt.readback_parity",
    )?;
    let witnesses = array_field(
        parity,
        "all_ten_route_witnesses",
        "L0 inner receipt.readback_parity",
    )?;
    if route_witness_count != 10 || witnesses.len() != 10 {
        return Err("L0 inner receipt must retain ten route witnesses".into());
    }
    for (index, witness) in witnesses.iter().enumerate() {
        let witness = object(witness, "L0 inner receipt route witness")?;
        if u64_field(witness, "wave_index", "L0 inner receipt route witness")? as usize != index
            || u64_field(witness, "expert_id", "L0 inner receipt route witness")?
                != expected_ids[index]
            || u64_field(witness, "elements", "L0 inner receipt route witness")? != HIDDEN
        {
            return Err(format!("L0 inner receipt route witness {index} drifted"));
        }
        sha_field(witness, "output_sha256", "L0 inner receipt route witness")?;
    }
    let second = object_field(
        parity,
        "second_residual",
        "L0 inner receipt.readback_parity",
    )?;
    if u64_field(
        second,
        "elements",
        "L0 inner receipt.readback_parity.second_residual",
    )? != HIDDEN
    {
        return Err("L0 inner receipt second residual must have 2048 elements".into());
    }
    let second_residual_sha = sha_field(
        second,
        "output_sha256",
        "L0 inner receipt.readback_parity.second_residual",
    )?;
    let state_layout = extract_state_layout(state_layout_value)?;
    Ok(ComponentEvidence {
        outer,
        outer_seal,
        inner,
        state_layout_file,
        state_layout,
        input_hidden_sha,
        initial_conv_state_sha,
        initial_recurrent_state_sha,
        device_first_residual_sha,
        second_residual_sha,
        route_ids: expected_ids,
        route_witness_count,
    })
}

fn validate_range_witness(
    object: &Map<String, Value>,
    expected: &StateRange,
    expected_hash_field: &str,
    label: &str,
) -> Result<(), String> {
    if string_field(object, "allocation_id", label)? != expected.allocation_id
        || u64_field(object, "slot", label)? != expected.slot
        || u64_field(object, "offset_bytes", label)? != expected.offset_bytes
        || u64_field(object, "capacity_bytes", label)? != expected.capacity_bytes
    {
        return Err(format!("{label} does not match the static state layout"));
    }
    if string_field(object, "device_buffer_id", label)?
        .trim()
        .is_empty()
    {
        return Err(format!("{label}.device_buffer_id must be non-empty"));
    }
    sha_field(object, expected_hash_field, label)?;
    Ok(())
}

fn assess_handoff(
    inner_value: &Value,
    evidence: &ComponentEvidence,
) -> Result<HandoffAssessment, String> {
    let inner = object(inner_value, "L0 inner receipt")?;
    let Some(witness_value) = inner.get("next_layer_handoff") else {
        return Ok(HandoffAssessment::Incomplete {
            missing: vec![
                "retained L0 second-residual device-buffer identity and 8192-byte buffer witness",
                "post-L0 DeltaNet conv and recurrent active-state byte hashes plus rollback checkpoint identities",
                "same-runtime/same-command-graph Layer-1 input binding to the retained L0 output and distinct active DeltaNet slot 1",
            ],
        });
    };
    let witness = object(witness_value, "L0 inner receipt.next_layer_handoff")?;
    exact_string(
        witness,
        "schema",
        HANDOFF_WITNESS_SCHEMA,
        "L0 inner receipt.next_layer_handoff",
    )?;
    exact_string(
        witness,
        "status",
        HANDOFF_WITNESS_STATUS,
        "L0 inner receipt.next_layer_handoff",
    )?;
    exact_string(
        witness,
        "session_id",
        &evidence.state_layout.session_id,
        "L0 inner receipt.next_layer_handoff",
    )?;
    if u64_field(
        witness,
        "source_token_id",
        "L0 inner receipt.next_layer_handoff",
    )? != SOURCE_TOKEN_ID
    {
        return Err("L0 inner receipt.next_layer_handoff source token drifted".into());
    }
    bool_field(
        witness,
        "same_command_graph_retained",
        true,
        "L0 inner receipt.next_layer_handoff",
    )?;
    let output = object_field(
        witness,
        "retained_l0_second_residual",
        "L0 inner receipt.next_layer_handoff",
    )?;
    if u64_field(
        output,
        "elements",
        "L0 inner receipt.next_layer_handoff.retained_l0_second_residual",
    )? != HIDDEN
        || u64_field(
            output,
            "bytes",
            "L0 inner receipt.next_layer_handoff.retained_l0_second_residual",
        )? != HIDDEN_BYTES
        || sha_field(
            output,
            "f32le_sha256",
            "L0 inner receipt.next_layer_handoff.retained_l0_second_residual",
        )? != evidence.second_residual_sha
    {
        return Err("L0 retained second-residual witness geometry/hash drifted".into());
    }
    if string_field(
        output,
        "device_buffer_id",
        "L0 inner receipt.next_layer_handoff.retained_l0_second_residual",
    )?
    .trim()
    .is_empty()
        || output
            .get("retained_through_layer1_encode")
            .and_then(Value::as_bool)
            != Some(true)
    {
        return Err(
            "L0 retained second-residual witness lacks a live Layer-1 buffer guarantee".into(),
        );
    }
    let state = object_field(
        witness,
        "l0_post_state_commit",
        "L0 inner receipt.next_layer_handoff",
    )?;
    if u64_field(
        state,
        "layer",
        "L0 inner receipt.next_layer_handoff.l0_post_state_commit",
    )? != 0
        || u64_field(
            state,
            "linear_state_slot",
            "L0 inner receipt.next_layer_handoff.l0_post_state_commit",
        )? != 0
    {
        return Err("L0 post-state witness layer/slot drifted".into());
    }
    bool_field(
        state,
        "checkpoint_before_mutation",
        true,
        "L0 inner receipt.next_layer_handoff.l0_post_state_commit",
    )?;
    for (field, expected, hash) in [
        (
            "active_conv",
            &evidence.state_layout.l0_active_conv,
            "post_state_f32le_sha256",
        ),
        (
            "active_recurrent",
            &evidence.state_layout.l0_active_recurrent,
            "post_state_f32le_sha256",
        ),
        (
            "rollback_conv",
            &evidence.state_layout.l0_rollback_conv,
            "checkpoint_f32le_sha256",
        ),
        (
            "rollback_recurrent",
            &evidence.state_layout.l0_rollback_recurrent,
            "checkpoint_f32le_sha256",
        ),
    ] {
        validate_range_witness(
            object_field(
                state,
                field,
                "L0 inner receipt.next_layer_handoff.l0_post_state_commit",
            )?,
            expected,
            hash,
            &format!("L0 inner receipt.next_layer_handoff.l0_post_state_commit.{field}"),
        )?;
    }
    let l1 = object_field(
        witness,
        "layer1_input_binding",
        "L0 inner receipt.next_layer_handoff",
    )?;
    if u64_field(
        l1,
        "layer",
        "L0 inner receipt.next_layer_handoff.layer1_input_binding",
    )? != 1
        || u64_field(
            l1,
            "linear_state_slot",
            "L0 inner receipt.next_layer_handoff.layer1_input_binding",
        )? != 1
        || string_field(
            l1,
            "session_id",
            "L0 inner receipt.next_layer_handoff.layer1_input_binding",
        )? != evidence.state_layout.session_id
        || string_field(
            l1,
            "input_device_buffer_id",
            "L0 inner receipt.next_layer_handoff.layer1_input_binding",
        )? != string_field(
            output,
            "device_buffer_id",
            "L0 inner receipt.next_layer_handoff.retained_l0_second_residual",
        )?
        || sha_field(
            l1,
            "input_f32le_sha256",
            "L0 inner receipt.next_layer_handoff.layer1_input_binding",
        )? != evidence.second_residual_sha
    {
        return Err("Layer-1 input does not retain the exact L0 second-residual buffer".into());
    }
    bool_field(
        l1,
        "same_command_graph_retained",
        true,
        "L0 inner receipt.next_layer_handoff.layer1_input_binding",
    )?;
    // Layer 0 has mutated its recurrent state, so its rollback bytes are a
    // required witness. Layer 1 has not executed yet; this boundary binds
    // only its distinct active slot. A Layer-1 rollback checkpoint belongs
    // to the later Layer-1 execution record and cannot be invented here.
    for (field, expected) in [
        ("active_conv", &evidence.state_layout.l1_active_conv),
        (
            "active_recurrent",
            &evidence.state_layout.l1_active_recurrent,
        ),
    ] {
        validate_range_witness(
            object_field(
                l1,
                field,
                "L0 inner receipt.next_layer_handoff.layer1_input_binding",
            )?,
            expected,
            "device_buffer_identity_sha256",
            &format!("L0 inner receipt.next_layer_handoff.layer1_input_binding.{field}"),
        )?;
    }
    Ok(HandoffAssessment::Ready)
}

fn report(evidence: &ComponentEvidence, assessment: HandoffAssessment) -> Value {
    let (status, ready_for_l1_device_handoff, missing) = match assessment {
        HandoffAssessment::Incomplete { missing } => (INCOMPLETE_STATUS, false, missing),
        HandoffAssessment::Ready => (READY_COMPONENT_ONLY_STATUS, true, Vec::new()),
    };
    json!({
        "schema": SCHEMA,
        "status": status,
        "ready_for_l1_device_handoff": ready_for_l1_device_handoff,
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
            "outer_terminal": evidence.outer.json(),
            "outer_terminal_seal_sha256": evidence.outer_seal,
            "inner_receipt": evidence.inner.json(),
            "layer": 0,
            "linear_state_slot": 0,
            "same_command_graph": {
                "prefix_dispatches": PREFIX_DISPATCHES,
                "suffix_dispatches": SUFFIX_DISPATCHES,
                "total_dispatches": TOTAL_DISPATCHES,
                "fenced_after_prefix_and_suffix": true,
            },
            "source_input_hidden_f32le_sha256": evidence.input_hidden_sha,
            "initial_conv_state_f32le_sha256": evidence.initial_conv_state_sha,
            "initial_recurrent_state_f32le_sha256": evidence.initial_recurrent_state_sha,
            "device_first_residual_f32le_sha256": evidence.device_first_residual_sha,
            "second_residual": {
                "elements": HIDDEN,
                "bytes": HIDDEN_BYTES,
                "f32le_sha256": evidence.second_residual_sha,
            },
            "route_guard": {
                "ordered_source_route_ids": evidence.route_ids,
                "route_witness_count": evidence.route_witness_count,
                "passed": true,
            },
        },
        "static_state_layout_authority": {
            "file": evidence.state_layout_file.json(),
            "schema": STATE_LAYOUT_SCHEMA,
            "session_id": evidence.state_layout.session_id,
            "l0": {
                "linear_state_slot": 0,
                "active_conv": evidence.state_layout.l0_active_conv.json(),
                "active_recurrent": evidence.state_layout.l0_active_recurrent.json(),
                "rollback_conv": evidence.state_layout.l0_rollback_conv.json(),
                "rollback_recurrent": evidence.state_layout.l0_rollback_recurrent.json(),
            },
            "l1": {
                "linear_state_slot": 1,
                "active_conv": evidence.state_layout.l1_active_conv.json(),
                "active_recurrent": evidence.state_layout.l1_active_recurrent.json(),
                "rollback_conv": evidence.state_layout.l1_rollback_conv.json(),
                "rollback_recurrent": evidence.state_layout.l1_rollback_recurrent.json(),
            },
            "l0_and_l1_slots_verified_disjoint": true,
            "actual_device_allocation_or_state_mutation_performed": false,
        },
        "next_required_real_decoder_dependency": {
            "schema": HANDOFF_WITNESS_SCHEMA,
            "required_status": HANDOFF_WITNESS_STATUS,
            "required_fields": [
                "retained_l0_second_residual: same-runtime live 2048-f32/8192-byte device buffer, hash-matched to this L0 capture and retained through Layer-1 encoding",
                "l0_post_state_commit: source-slot-0 active conv/recurrent post-state hashes plus distinct rollback checkpoint buffer identities and hashes",
                "layer1_input_binding: same session and command graph, exact retained L0 buffer/hash, and source-owned distinct active DeltaNet slot-1 identities (Layer-1 rollback remains a later execution obligation)",
            ],
        },
        "handoff_assessment": {
            "missing_real_evidence": missing,
            "historical_output_hash_is_not_a_retained_device_buffer": true,
            "historical_initial_state_hashes_are_not_post_state_commit_witnesses": true,
        },
        "claim_boundary": {
            "cpu_only_assessment": true,
            "no_new_artifact_scan_or_payload_open": true,
            "no_metal_context_or_device_dispatch": true,
            "not_a_complete_layer_token_decoder_generation_server_hcli_tps_tg_or_tournament_result": true,
            "watcher_transition_not_authorized": true,
        },
    })
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if !path.is_absolute() {
        return Err("--out must be absolute".into());
    }
    let parent = path.parent().ok_or("--out has no parent directory")?;
    let metadata = fs::symlink_metadata(parent)
        .map_err(|error| format!("cannot stat output parent {}: {error}", parent.display()))?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err("--out parent must be a real existing directory".into());
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("refusing to overwrite --out {}: {error}", path.display()))?;
    file.write_all(bytes)
        .map_err(|error| format!("cannot write --out: {error}"))?;
    file.sync_all()
        .map_err(|error| format!("cannot sync --out: {error}"))?;
    Ok(())
}

fn usage() -> &'static str {
    "usage: ascension_qwen80_l0_to_layer1_handoff_authority \\\n+--outer-terminal ABSOLUTE_PATH --inner-receipt ABSOLUTE_PATH \\\n+--state-layout ABSOLUTE_PATH --out ABSOLUTE_NEW_PATH"
}

fn parse_args<I>(arguments: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut values = BTreeMap::<String, String>::new();
    let mut arguments = arguments.into_iter();
    while let Some(flag) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value after {flag}; {}", usage()))?;
        if !matches!(
            flag.as_str(),
            "--outer-terminal" | "--inner-receipt" | "--state-layout" | "--out"
        ) {
            return Err(format!("unsupported option {flag:?}; {}", usage()));
        }
        if values.insert(flag.clone(), value).is_some() {
            return Err(format!("{flag} was repeated"));
        }
    }
    let take = |flag: &str| {
        values
            .get(flag)
            .map(PathBuf::from)
            .ok_or_else(|| format!("missing {flag}; {}", usage()))
    };
    let args = Args {
        outer_terminal: take("--outer-terminal")?,
        inner_receipt: take("--inner-receipt")?,
        state_layout: take("--state-layout")?,
        out: take("--out")?,
    };
    if !args.out.is_absolute() {
        return Err("--out must be absolute".into());
    }
    Ok(args)
}

fn run(args: Args) -> Result<(PathBuf, String, String), String> {
    let (outer, outer_value) = read_json(&args.outer_terminal, "--outer-terminal")?;
    let (inner, inner_value) = read_json(&args.inner_receipt, "--inner-receipt")?;
    let (state_layout, state_layout_value) = read_json(&args.state_layout, "--state-layout")?;
    let evidence = validate_component(
        outer,
        &outer_value,
        inner,
        &inner_value,
        state_layout,
        &state_layout_value,
    )?;
    let assessment = assess_handoff(&inner_value, &evidence)?;
    let mut output = report(&evidence, assessment);
    let status = string_field(
        object(&output, "authority output")?,
        "status",
        "authority output",
    )?
    .to_owned();
    let seal = seal(&mut output)?;
    let bytes = serde_json::to_vec_pretty(&output).map_err(|error| error.to_string())?;
    write_new(&args.out, &bytes)?;
    Ok((args.out, seal, status))
}

fn main() {
    match parse_args(env::args().skip(1)).and_then(run) {
        Ok((path, seal, status)) => println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "path": path,
                "seal_sha256": seal,
                "status": status,
                "cpu_only": true,
            }))
            .expect("result JSON")
        ),
        Err(error) => {
            eprintln!("ascension_qwen80_l0_to_layer1_handoff_authority: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha() -> String {
        "a".repeat(64)
    }

    fn range(arena: &str, domain: &str, slot: u64, offset: u64, capacity: u64) -> Value {
        json!({
            "arena": arena,
            "domain": domain,
            "allocation_id": format!("{arena}-{domain}"),
            "slot": slot,
            "offset_bytes": offset,
            "capacity_bytes": capacity,
        })
    }

    fn layout_fixture() -> Value {
        let binding = |layer: u64, slot: u64| {
            let conv_bytes = 98_304;
            let recurrent_bytes = 2_097_152;
            json!({
                "layer": layer,
                "mixer": "DeltaNet",
                "state_slot": slot,
                "active_ranges": [
                    range("Active", "DeltaNetConv", slot, slot * conv_bytes, (slot + 1) * conv_bytes),
                    range("Active", "DeltaNetRecurrent", slot, slot * recurrent_bytes, (slot + 1) * recurrent_bytes),
                ],
                "rollback_ranges": [
                    range("Rollback", "DeltaNetConv", slot, slot * conv_bytes, (slot + 1) * conv_bytes),
                    range("Rollback", "DeltaNetRecurrent", slot, slot * recurrent_bytes, (slot + 1) * recurrent_bytes),
                ],
            })
        };
        let mut bindings = vec![binding(0, 0), binding(1, 1)];
        for layer in 2..48 {
            bindings.push(json!({"layer":layer,"mixer":"DeltaNet","state_slot":layer}));
        }
        json!({
            "schema": STATE_LAYOUT_SCHEMA,
            "status": STATE_LAYOUT_STATUS,
            "ready_for_decoder_graph": false,
            "actual_device_allocation_performed": false,
            "source_identity": {
                "model_key": MODEL_KEY,
                "source_revision": SOURCE_REVISION,
                "manifest_seal_sha256": MANIFEST_SEAL,
                "admission_receipt_seal_sha256": ADMISSION_RECEIPT_SEAL,
            },
            "session_layout": {"session_id":"test-session","max_seq_len":1,"layer_bindings":bindings},
        })
    }

    fn sealed(mut value: Value) -> Value {
        seal(&mut value).unwrap();
        value
    }

    fn inner_fixture() -> Value {
        let routes = (0..10).map(|index| index as u64).collect::<Vec<_>>();
        let witnesses = routes
            .iter()
            .enumerate()
            .map(|(index, id)| {
                json!({
                    "wave_index":index,
                    "expert_id":id,
                    "elements":HIDDEN,
                    "output_sha256":sha(),
                })
            })
            .collect::<Vec<_>>();
        json!({
            "schema": INNER_SCHEMA,
            "status": INNER_STATUS,
            "mode":"metal",
            "metal_device_or_dispatch_performed":true,
            "component_only":true,
            "complete_layer_or_token_performed":false,
            "raw_bf16_or_safetensors_opened":false,
            "artifact_binding":{
                "manifest_document_sha256":MANIFEST_DOCUMENT_SHA,
                "manifest_seal_sha256":MANIFEST_SEAL,
                "admission_receipt_seal_sha256":ADMISSION_RECEIPT_SEAL,
                "source_revision":SOURCE_REVISION,
                "layer":0,
                "linear_state_slot":0,
            },
            "same_command_graph":{
                "source_token_id":SOURCE_TOKEN_ID,
                "prefix_dispatches":PREFIX_DISPATCHES,
                "suffix_dispatches":SUFFIX_DISPATCHES,
                "total_dispatches":TOTAL_DISPATCHES,
                "same_command_graph_required":true,
                "same_command_graph_retained":true,
                "command_buffer_fenced_once_after_prefix_and_suffix":true,
                "encoded_kernel_order_matches_expected":true,
                "encoded_kernel_names":vec!["kernel"; TOTAL_DISPATCHES as usize],
            },
            "execution_phase":{
                "strict_artifact_admission_started":true,
                "strict_artifact_admission_succeeded":true,
                "metal_context_construction_attempted":true,
                "metal_context_constructed":true,
                "structural_kernel_trace_enabled":true,
                "command_commit_attempted":true,
                "command_fence_succeeded":true,
                "readback_started":true,
                "device_dispatch_may_have_occurred":true,
                "dispatches_encoded":TOTAL_DISPATCHES,
            },
            "prefix_parity":{
                "source_token_id":SOURCE_TOKEN_ID,
                "elements":HIDDEN,
                "bytes":HIDDEN_BYTES,
                "input_hidden_f32le_sha256":sha(),
                "initial_conv_state_f32le_sha256":sha(),
                "initial_recurrent_state_f32le_sha256":sha(),
                "device_first_residual_f32le_sha256":sha(),
            },
            "route_guard_readback":{"passed":true,"value":1,"expected_ids":routes,"observed_ids":(0..10).collect::<Vec<_>>()},
            "readback_parity":{
                "all_ten_route_witness_count":10,
                "all_ten_route_witnesses":witnesses,
                "second_residual":{"elements":HIDDEN,"output_sha256":sha()},
            },
        })
    }

    fn outer_fixture(inner: &BoundFile) -> Value {
        sealed(json!({
            "schema": OUTER_SCHEMA,
            "status": OUTER_STATUS,
            "inner_probe_capture":{
                "present":true,
                "path":inner.path,
                "bytes":inner.bytes,
                "sha256":inner.sha256,
                "schema":INNER_SCHEMA,
                "status":INNER_STATUS,
            },
            "source_binding":{
                "artifact_identity":{
                    "manifest_document_sha256":MANIFEST_DOCUMENT_SHA,
                    "manifest_seal_sha256":MANIFEST_SEAL,
                    "admission_receipt_seal_sha256":ADMISSION_RECEIPT_SEAL,
                },
                "implementation_binding":{
                    "source_token_id":SOURCE_TOKEN_ID,
                    "prefix_dispatches":PREFIX_DISPATCHES,
                    "suffix_dispatches":SUFFIX_DISPATCHES,
                    "total_dispatches":TOTAL_DISPATCHES,
                    "same_command_buffer_fence_required":true,
                },
                "first_residual_receipt":{"output_sha256":sha()},
            },
        }))
    }

    fn evidence_fixture() -> ComponentEvidence {
        let state = extract_state_layout(&layout_fixture()).unwrap();
        ComponentEvidence {
            outer: BoundFile {
                path: "/tmp/outer.json".into(),
                bytes: 1,
                sha256: sha(),
            },
            outer_seal: sha(),
            inner: BoundFile {
                path: "/tmp/inner.json".into(),
                bytes: 1,
                sha256: sha(),
            },
            state_layout_file: BoundFile {
                path: "/tmp/layout.json".into(),
                bytes: 1,
                sha256: sha(),
            },
            state_layout: state,
            input_hidden_sha: sha(),
            initial_conv_state_sha: sha(),
            initial_recurrent_state_sha: sha(),
            device_first_residual_sha: sha(),
            second_residual_sha: sha(),
            route_ids: (0..10).collect(),
            route_witness_count: 10,
        }
    }

    #[test]
    fn static_layout_requires_l0_and_l1_to_use_distinct_deltanet_slots() {
        let layout = extract_state_layout(&layout_fixture()).unwrap();
        assert_eq!(layout.l0_active_conv.slot, 0);
        assert_eq!(layout.l1_active_conv.slot, 1);
        assert!(layout.l0_active_conv.capacity_bytes <= layout.l1_active_conv.offset_bytes);
        assert_ne!(
            layout.l0_active_conv.allocation_id,
            layout.l0_rollback_conv.allocation_id
        );
    }

    #[test]
    fn current_style_component_without_handoff_is_explicitly_incomplete() {
        let evidence = evidence_fixture();
        let assessment = assess_handoff(&inner_fixture(), &evidence).unwrap();
        match assessment {
            HandoffAssessment::Incomplete { missing } => assert_eq!(missing.len(), 3),
            HandoffAssessment::Ready => panic!("missing device handoff must not promote"),
        }
    }

    #[test]
    fn malformed_handoff_cannot_be_silently_treated_as_ready() {
        let evidence = evidence_fixture();
        let mut inner = inner_fixture();
        inner["next_layer_handoff"] =
            json!({"schema":HANDOFF_WITNESS_SCHEMA,"status":HANDOFF_WITNESS_STATUS});
        assert!(assess_handoff(&inner, &evidence).is_err());
    }

    #[test]
    fn outer_seal_and_inner_byte_binding_are_required() {
        let inner = BoundFile {
            path: "/tmp/inner.json".into(),
            bytes: 1,
            sha256: sha(),
        };
        let outer = outer_fixture(&inner);
        assert!(verify_seal(&outer, "outer").is_ok());
        let mut tampered = outer.clone();
        tampered["inner_probe_capture"]["sha256"] = json!("b".repeat(64));
        assert!(verify_seal(&tampered, "outer").is_err());
    }

    #[test]
    fn report_seal_is_stable_and_component_only() {
        let evidence = evidence_fixture();
        let mut report = report(
            &evidence,
            HandoffAssessment::Incomplete {
                missing: vec!["missing"],
            },
        );
        let seal_value = seal(&mut report).unwrap();
        assert_eq!(verify_seal(&report, "report").unwrap(), seal_value);
        assert_eq!(report["component_only"], true);
        assert_eq!(report["ready_for_l1_device_handoff"], false);
    }
}
