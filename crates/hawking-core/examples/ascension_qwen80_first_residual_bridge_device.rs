//! Build-only source-input L0 DeltaNet -> true all-ten MoE command-graph bridge.
//!
//! This future child intentionally performs no runtime action in `main`.
//! Its macOS-only encoder is type-checked against the existing direct-packed
//! DeltaNet kernels and the staged all-ten graph host: it keeps the actual
//! L0 `first_residual` `PinnedBuffer` alive through the *same*
//! `TokenCommandBuffer` that encodes the 14-dispatch MoE suffix.  A later
//! explicit lease/registry integration may call the encoder, fence once, and
//! produce the strict inner receipt validated by the outer launcher.
//!
//! There is deliberately no `MetalContext::new`, no shader registration, no
//! admission run, and no device invocation in this example today.

#[cfg(target_os = "macos")]
#[allow(dead_code)]
#[path = "ascension_qwen80_all_ten_true_moe_graph_device.rs"]
mod staged_all_ten_graph;

#[cfg(target_os = "macos")]
use hawking_core::metal::TokenCommandBuffer;
#[cfg(target_os = "macos")]
use hawking_core::model::qwen80_complete_runtime::{
    Qwen80AllTenTrueMoeDeviceBridge, Qwen80AllTenTrueMoeSourceBridge,
    Qwen80CanonicalSourceTokenL0TrueMoeContinuation, Qwen80CompleteArtifactCatalog,
    Qwen80CompleteNativeRuntime, Qwen80CompleteRuntimeOptions, Qwen80L0TrueMoeFixedDeviceBuffers,
    Qwen80SourceInputFirstResidualEncoder, Qwen80SourceInputFirstResidualParity,
};
#[cfg(target_os = "macos")]
use hawking_core::model::qwen_complete_binary::{CompleteBinaryAdmission, QwenCompleteBinaryModel};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

const CHILD_SCHEMA: &str = "hawking.ascension.qwen80_first_residual_bridge_device.v1";
const PREPARED_STATUS: &str =
    "PREPARED_QWEN80_SOURCE_INPUT_FIRST_RESIDUAL_SAME_COMMAND_GRAPH_ENCODER_NOT_EXECUTED";
const STRICT_METAL_MODE: &str = "metal";
const CPU_BASELINE_SCHEMA: &str = "hawking.ascension.qwen80_first_residual_bridge_inner.v1";
const CPU_BASELINE_STATUS: &str =
    "EARNED_CURRENT_ADMITTED_QWEN80_LAYER0_FIRST_RESIDUAL_CPU_ORACLE_BASELINE_METAL_LEASE_REQUIRED";
const LEASE_SCHEMA: &str = "hawking.ascension.qwen80_first_residual_quiet_metal_lease.v1";
const LEASE_STATUS: &str = "GRANTED_QWEN80_FIRST_RESIDUAL_NON_TIMED_DEVICE_PARITY_LEASE";
const MANIFEST_SCHEMA: &str = "hawking.ascension.qwen80_complete_binary_gravity.v1";
const ADMISSION_POINTER_SCHEMA: &str =
    "hawking.ascension.qwen_complete_binary_gravity_admission_current_pointer.v1";
const ADMISSION_POINTER_STATUS: &str = "CURRENT_COMPLETE_BINARY_ADMISSION_RECEIPT_SELECTED";
const ADMISSION_RECEIPT_SCHEMA: &str =
    "hawking.ascension.qwen_complete_binary_gravity_admission_receipt.v1";
const ADMISSION_RECEIPT_STATUS: &str =
    "EARNED_COMPLETE_BINARY_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED";
const HIDDEN: usize = 2_048;
const FIRST_RESIDUAL_BYTES: usize = HIDDEN * std::mem::size_of::<f32>();
const CONV_STATE_ELEMENTS: usize = 8_192 * 3;
const RECURRENT_STATE_ELEMENTS: usize = 32 * 128 * 128;
const PREFIX_DISPATCHES: usize = 9;
const TRUE_MOE_SUFFIX_DISPATCHES: usize = 14;

/// Explicit arguments required by the first-residual outer reaper.  Parsing
/// and validating this type only binds immutable input authority; it does not
/// admit an artifact, open a Metal context, create a command buffer, or
/// authorize a device dispatch.
#[derive(Clone, Debug, Eq, PartialEq)]
struct FirstResidualChildArgs {
    manifest: PathBuf,
    admission_current: PathBuf,
    cpu_baseline_receipt: PathBuf,
    lease_receipt: PathBuf,
    outer_capture_dir: PathBuf,
    capture_dir: PathBuf,
    workers: usize,
}

fn first_residual_child_usage() -> &'static str {
    "usage: ascension_qwen80_first_residual_bridge_device \\
--manifest ABSOLUTE_PATH --admission-current ABSOLUTE_PATH \\
--cpu-baseline-receipt ABSOLUTE_PATH \\
--lease-receipt ABSOLUTE_PATH --outer-capture-dir ABSOLUTE_DIRECTORY \\
--capture-dir NEW_ABSOLUTE_DIRECTORY --mode metal --workers 1..4"
}

fn parse_first_residual_child_args<I>(arguments: I) -> Result<FirstResidualChildArgs, String>
where
    I: IntoIterator<Item = String>,
{
    let mut values = std::collections::BTreeMap::<String, String>::new();
    let mut arguments = arguments.into_iter();
    while let Some(flag) = arguments.next() {
        match flag.as_str() {
            "--manifest"
            | "--admission-current"
            | "--cpu-baseline-receipt"
            | "--lease-receipt"
            | "--outer-capture-dir"
            | "--capture-dir"
            | "--mode"
            | "--workers" => {
                let value = arguments.next().ok_or_else(|| {
                    format!("{flag} requires a value; {}", first_residual_child_usage())
                })?;
                if values.insert(flag.clone(), value).is_some() {
                    return Err(format!("{flag} repeated; {}", first_residual_child_usage()));
                }
            }
            "--help" | "-h" => return Err(first_residual_child_usage().to_owned()),
            _ => {
                return Err(format!(
                    "unknown argument {flag:?}; {}",
                    first_residual_child_usage()
                ))
            }
        }
    }
    let mut required = |flag: &str| -> Result<PathBuf, String> {
        let value = values
            .remove(flag)
            .ok_or_else(|| format!("missing {flag}; {}", first_residual_child_usage()))?;
        let path = PathBuf::from(value);
        if !path.is_absolute() {
            return Err(format!("{flag} must be absolute"));
        }
        Ok(path)
    };
    let manifest = required("--manifest")?;
    let admission_current = required("--admission-current")?;
    let cpu_baseline_receipt = required("--cpu-baseline-receipt")?;
    let lease_receipt = required("--lease-receipt")?;
    let outer_capture_dir = required("--outer-capture-dir")?;
    let capture_dir = required("--capture-dir")?;
    let mode = values
        .remove("--mode")
        .ok_or_else(|| format!("missing --mode; {}", first_residual_child_usage()))?;
    if mode != STRICT_METAL_MODE {
        return Err(format!("--mode must be {STRICT_METAL_MODE:?}"));
    }
    let workers = values
        .remove("--workers")
        .ok_or_else(|| format!("missing --workers; {}", first_residual_child_usage()))?
        .parse::<usize>()
        .map_err(|_| "--workers must be an unsigned integer".to_owned())?;
    if !(1..=4).contains(&workers) {
        return Err("--workers must be 1..4".to_owned());
    }
    if !values.is_empty() {
        return Err(format!("unconsumed arguments: {values:?}"));
    }
    Ok(FirstResidualChildArgs {
        manifest,
        admission_current,
        cpu_baseline_receipt,
        lease_receipt,
        outer_capture_dir,
        capture_dir,
        workers,
    })
}

fn canonical_regular_first_residual_authority(path: &Path, label: &str) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err(format!("{label} must be absolute"));
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    fs::canonicalize(path)
        .map_err(|error| format!("cannot canonicalize {label} {}: {error}", path.display()))
}

fn validate_first_residual_child_paths(
    args: &FirstResidualChildArgs,
) -> Result<FirstResidualChildArgs, String> {
    let manifest = canonical_regular_first_residual_authority(&args.manifest, "--manifest")?;
    let admission_current =
        canonical_regular_first_residual_authority(&args.admission_current, "--admission-current")?;
    let cpu_baseline_receipt = canonical_regular_first_residual_authority(
        &args.cpu_baseline_receipt,
        "--cpu-baseline-receipt",
    )?;
    let lease_receipt =
        canonical_regular_first_residual_authority(&args.lease_receipt, "--lease-receipt")?;
    let outer_metadata = fs::symlink_metadata(&args.outer_capture_dir).map_err(|error| {
        format!(
            "cannot stat --outer-capture-dir {}: {error}",
            args.outer_capture_dir.display()
        )
    })?;
    if outer_metadata.file_type().is_symlink() || !outer_metadata.is_dir() {
        return Err("--outer-capture-dir must be an existing non-symlink directory".to_owned());
    }
    let outer_capture_dir = fs::canonicalize(&args.outer_capture_dir).map_err(|error| {
        format!(
            "cannot canonicalize --outer-capture-dir {}: {error}",
            args.outer_capture_dir.display()
        )
    })?;
    if args.capture_dir.exists() {
        return Err(
            "--capture-dir must not exist before the future child creates its inner capture"
                .to_owned(),
        );
    }
    let capture_parent = args
        .capture_dir
        .parent()
        .ok_or("--capture-dir has no parent")?;
    let capture_parent = fs::canonicalize(capture_parent).map_err(|error| {
        format!(
            "cannot canonicalize --capture-dir parent {}: {error}",
            capture_parent.display()
        )
    })?;
    if capture_parent != outer_capture_dir {
        return Err("--capture-dir must be a direct child of --outer-capture-dir".to_owned());
    }
    Ok(FirstResidualChildArgs {
        manifest,
        admission_current,
        cpu_baseline_receipt,
        lease_receipt,
        outer_capture_dir,
        capture_dir: args.capture_dir.clone(),
        workers: args.workers,
    })
}

/// Owns the source-input mixer buffers and the compact all-ten route bridge
/// until the caller commits/fences the one common command buffer.  The fixed
/// suffix buffers are intentionally borrowed by the encoder only; the caller
/// must retain them as well until the same fence.
#[cfg(target_os = "macos")]
pub struct Qwen80SourceInputL0TrueMoeGraph {
    first_residual: Qwen80SourceInputFirstResidualEncoder,
    _route_bridge: Qwen80AllTenTrueMoeDeviceBridge,
    pub prefix_dispatches: usize,
    pub suffix_dispatches: usize,
}

#[cfg(target_os = "macos")]
impl Qwen80SourceInputL0TrueMoeGraph {
    /// Verify the retained L0 state/output only after the caller has fenced
    /// the command buffer that contains both the nine DeltaNet and fourteen
    /// staged true-MoE dispatches.
    pub fn verify_first_residual_after_fence(
        &self,
        runtime: &Qwen80CompleteNativeRuntime,
    ) -> hawking_core::Result<Qwen80SourceInputFirstResidualParity> {
        self.first_residual.verify_after_fence(runtime)
    }

    pub fn source_token_id(&self) -> u32 {
        self.first_residual.source_token_id()
    }
}

/// Convert the runtime-owned six fixed compact uploads/scratch buffers into
/// the staged suffix's borrowed ABI.  The function deliberately has no
/// context, dispatch, fence, or readback path; it is a compile-checked
/// one-to-one resource map that keeps the fixed source contract separate from
/// the source-selected ten-route bridge.
#[cfg(target_os = "macos")]
pub fn staged_fixed_true_moe_buffers(
    fixed: &Qwen80L0TrueMoeFixedDeviceBuffers,
) -> staged_all_ten_graph::Qwen80AllTenTrueMoeGraphFixedBuffers<'_> {
    staged_all_ten_graph::Qwen80AllTenTrueMoeGraphFixedBuffers {
        postnorm_signs: &fixed.postnorm.signs,
        postnorm_scales: &fixed.postnorm.scales,
        postnorm_hidden: &fixed.postnorm_hidden,
        router_signs: &fixed.router.signs,
        router_scales: &fixed.router.scales,
        router_logits: &fixed.router_logits,
        router_probabilities: &fixed.router_probabilities,
        router_route_ids: &fixed.router_route_ids,
        router_route_weights: &fixed.router_route_weights,
        route_guard: &fixed.route_guard,
        route_gate: &fixed.route_gate,
        route_up: &fixed.route_up,
        route_activated: &fixed.route_activated,
        route_weighted: &fixed.route_weighted,
        shared_gate_signs: &fixed.shared_gate_proj.signs,
        shared_gate_scales: &fixed.shared_gate_proj.scales,
        shared_up_signs: &fixed.shared_up_proj.signs,
        shared_up_scales: &fixed.shared_up_proj.scales,
        shared_down_signs: &fixed.shared_down_proj.signs,
        shared_down_scales: &fixed.shared_down_proj.scales,
        shared_scalar_signs: &fixed.shared_expert_gate.signs,
        shared_scalar_scales: &fixed.shared_expert_gate.scales,
        shared_gate: &fixed.shared_gate,
        shared_up: &fixed.shared_up,
        shared_activated: &fixed.shared_activated,
        shared_output: &fixed.shared_output,
        shared_scalar_logit: &fixed.shared_scalar_logit,
        gated_shared: &fixed.gated_shared,
        routed_sum: &fixed.routed_sum,
        second_residual: &fixed.second_residual,
    }
}

/// Encode the non-synthetic source-token L0 DeltaNet prefix then the staged
/// all-ten source-selected MoE suffix into one already-open command buffer.
///
/// This function is intentionally not a device entrypoint: the caller must
/// supply a runtime created from one admitted catalog, a fresh lease-authorized
/// `TokenCommandBuffer` from `begin_component_token_command_buffer`, an
/// immutable all-ten route bridge, and every non-route suffix buffer.  It
/// neither commits nor reads back, so a later strict capture can retain and
/// verify the actual first-residual buffer instead of copying CPU reference
/// bytes between command buffers.
#[cfg(target_os = "macos")]
pub fn encode_source_input_l0_true_moe_graph(
    runtime: &Qwen80CompleteNativeRuntime,
    command: &mut TokenCommandBuffer<'_>,
    token_id: u32,
    source_bridge: &Qwen80AllTenTrueMoeSourceBridge,
    fixed: staged_all_ten_graph::Qwen80AllTenTrueMoeGraphFixedBuffers<'_>,
) -> Result<Qwen80SourceInputL0TrueMoeGraph, String> {
    let dispatches_before = command.dispatch_count();
    let first_residual = runtime
        .encode_source_token_first_linear_deltanet_into(command, token_id)
        .map_err(|error| error.to_string())?;
    let prefix_dispatches = command
        .dispatch_count()
        .checked_sub(dispatches_before)
        .ok_or("source-input L0 prefix dispatch count underflow")?;
    if prefix_dispatches != 9 {
        return Err(format!(
            "source-input L0 DeltaNet prefix encoded {prefix_dispatches} dispatches, expected 9"
        ));
    }
    // `metal::Buffer` is a retained Objective-C handle.  This clone denotes
    // the same allocation, not a CPU copy, and both owners remain live in the
    // returned graph until the caller fences the shared command buffer.
    let route_bridge = runtime
        .bind_source_input_first_residual_to_all_ten(source_bridge, &first_residual)
        .map_err(|error| error.to_string())?;
    let buffers = staged_all_ten_graph::Qwen80AllTenTrueMoeGraphBuffers::from_admitted_route_bridge(
        &route_bridge,
        fixed,
    );
    let suffix_dispatches =
        staged_all_ten_graph::encode_all_ten_true_moe_from_first_residual(command, &buffers)?;
    if suffix_dispatches != 14 {
        return Err(format!(
            "source-input L0 true-MoE suffix encoded {suffix_dispatches} dispatches, expected 14"
        ));
    }
    Ok(Qwen80SourceInputL0TrueMoeGraph {
        first_residual,
        _route_bridge: route_bridge,
        prefix_dispatches,
        suffix_dispatches,
    })
}

/// Owns every fixed compact upload/scratch allocation together with the
/// source-input prefix and source-selected route bridge until one shared
/// command-buffer fence.  This is the exact future strict-Metal child body
/// boundary: callers cannot discard the first-residual allocation after the
/// DeltaNet prefix and replace it with CPU bytes before encoding the suffix.
#[cfg(target_os = "macos")]
pub struct Qwen80SourceInputL0TrueMoeCaptureResources {
    pub graph: Qwen80SourceInputL0TrueMoeGraph,
    pub fixed: Qwen80L0TrueMoeFixedDeviceBuffers,
}

#[cfg(target_os = "macos")]
impl Qwen80SourceInputL0TrueMoeCaptureResources {
    /// Convert the exact resource owner returned by this module's canonical
    /// L0 encoder into the opaque core continuation capability.  This is the
    /// only bridge intended for a future L1 prefix: it consumes the private
    /// source-prefix/route/fixed holders rather than accepting a caller-made
    /// 8192-byte `PinnedBuffer` or a detached receipt vector.
    pub fn into_canonical_l0_true_moe_continuation(
        self,
        runtime: &Qwen80CompleteNativeRuntime,
        command: &TokenCommandBuffer<'_>,
    ) -> Result<Qwen80CanonicalSourceTokenL0TrueMoeContinuation, String> {
        if self.graph.prefix_dispatches != PREFIX_DISPATCHES
            || self.graph.suffix_dispatches != TRUE_MOE_SUFFIX_DISPATCHES
        {
            return Err(
                "source-input L0 resources do not retain the exact canonical 9+14 graph".to_owned(),
            );
        }
        runtime
            .certify_source_token_l0_true_moe_continuation(
                command,
                self.graph.first_residual,
                self.graph._route_bridge,
                self.fixed,
            )
            .map_err(|error| error.to_string())
    }
}

/// Allocate the exact admitted fixed suffix resources and encode the complete
/// source-input L0 prefix plus static-ABI true-MoE suffix into one already
/// open non-timed command buffer.  It deliberately does not commit or fence:
/// the eventual outer-reaped child owns that final action and must capture
/// first-residual/state and suffix readbacks before sealing an inner receipt.
#[cfg(target_os = "macos")]
pub fn encode_source_input_l0_true_moe_capture(
    runtime: &Qwen80CompleteNativeRuntime,
    command: &mut TokenCommandBuffer<'_>,
    token_id: u32,
    source_bridge: &Qwen80AllTenTrueMoeSourceBridge,
) -> Result<Qwen80SourceInputL0TrueMoeCaptureResources, String> {
    let fixed = runtime
        .upload_l0_true_moe_fixed_device_buffers()
        .map_err(|error| error.to_string())?;
    let graph = encode_source_input_l0_true_moe_graph(
        runtime,
        command,
        token_id,
        source_bridge,
        staged_fixed_true_moe_buffers(&fixed),
    )?;
    if graph.prefix_dispatches + graph.suffix_dispatches
        != PREFIX_DISPATCHES + TRUE_MOE_SUFFIX_DISPATCHES
    {
        return Err("source-input L0 true-MoE command graph dispatch count drifted".to_owned());
    }
    Ok(Qwen80SourceInputL0TrueMoeCaptureResources { graph, fixed })
}

fn prepared_document() -> serde_json::Value {
    json!({
        "schema": CHILD_SCHEMA,
        "status": PREPARED_STATUS,
        "device_execution_performed": false,
        "command_graph_contract": {
            "source_input": "direct-packed source embedding CPU/reference row uploaded as L0 input; not a native embedding-gather claim",
            "prefix_dispatches": PREFIX_DISPATCHES,
            "suffix_dispatches": 0,
            "prefix_only": true,
            "same_token_command_buffer_required": true,
            "first_residual_device_buffer_elements": 2048,
            "first_residual_device_buffer_bytes": 8192,
            "cpu_reference_copy_may_not_replace_the_retained_device_buffer": true,
            "commit_and_readback_are_owned_by_the_outer-reaped_prefix-child": true,
        },
        "future_capture_requirements": [
            "current admitted manifest/admission/source revalidation binding",
            "new non-synthetic source-token CPU baseline",
            "fresh component-only quiet lease",
            "existing registered Qwen-Next prefix kernels",
            "same-command-buffer fence and first-residual/state readback parity",
            "receipt-last inner plus sealed outer terminal receipt",
        ],
        "claim_boundary": {
            "no_metal_context_or_gpu_dispatch": true,
            "no_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim": true,
            "not_an_authorization_to_start_watcher_or_server": true,
        },
    })
}

fn sha256_file(path: &Path, label: &str) -> Result<String, String> {
    let bytes = fs::read(path)
        .map_err(|error| format!("cannot read {label} {}: {error}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn read_json_authority(path: &Path, label: &str) -> Result<Value, String> {
    let bytes = fs::read(path)
        .map_err(|error| format!("cannot read {label} {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("cannot parse {label} {}: {error}", path.display()))
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a serde_json::Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be a JSON object"))
}

fn field_object<'a>(
    value: &'a Value,
    field: &str,
    label: &str,
) -> Result<&'a serde_json::Map<String, Value>, String> {
    value
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{label}.{field} must be a JSON object"))
}

fn field_string<'a>(
    value: &'a serde_json::Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| format!("{label}.{field} must be a non-empty string"))
}

fn field_bool(
    value: &serde_json::Map<String, Value>,
    field: &str,
    expected: bool,
    label: &str,
) -> Result<(), String> {
    if value.get(field).and_then(Value::as_bool) != Some(expected) {
        return Err(format!("{label}.{field} must be {expected}"));
    }
    Ok(())
}

fn require_sha256(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || value
            .bytes()
            .any(|byte| !byte.is_ascii_hexdigit() || byte.is_ascii_uppercase())
    {
        return Err(format!("{label} must be a lowercase SHA-256"));
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct FirstResidualChildAuthority {
    args: FirstResidualChildArgs,
    manifest_document_sha256: String,
    manifest_seal_sha256: String,
    admission_pointer_seal_sha256: String,
    admission_receipt_seal_sha256: String,
    source_audit_seal_sha256: String,
    source_revision: String,
    cpu_baseline_sha256: String,
    cpu_input_f32le_sha256: String,
    cpu_reference_first_residual_sha256: String,
    initial_conv_state_f32le_sha256: String,
    initial_recurrent_state_f32le_sha256: String,
    source_token_id: u32,
    lease_sha256: String,
    lease_seal_sha256: String,
}

fn validate_first_residual_child_authority(
    args: &FirstResidualChildArgs,
) -> Result<FirstResidualChildAuthority, String> {
    let args = validate_first_residual_child_paths(args)?;
    let manifest = read_json_authority(&args.manifest, "manifest")?;
    let manifest_object = object(&manifest, "manifest")?;
    if field_string(manifest_object, "schema", "manifest")? != MANIFEST_SCHEMA {
        return Err("manifest schema drifted".to_owned());
    }
    let manifest_seal_sha256 = field_string(manifest_object, "seal_sha256", "manifest")?.to_owned();
    require_sha256(&manifest_seal_sha256, "manifest.seal_sha256")?;
    let manifest_document_sha256 = sha256_file(&args.manifest, "manifest")?;

    let admission = read_json_authority(&args.admission_current, "admission current")?;
    let admission_object = object(&admission, "admission current")?;
    if field_string(admission_object, "schema", "admission current")? != ADMISSION_POINTER_SCHEMA
        || field_string(admission_object, "status", "admission current")?
            != ADMISSION_POINTER_STATUS
    {
        return Err("admission current schema/status drifted".to_owned());
    }
    let admission_pointer_seal_sha256 =
        field_string(admission_object, "seal_sha256", "admission current")?.to_owned();
    require_sha256(&admission_pointer_seal_sha256, "admission current seal")?;
    let selected_manifest = field_object(&admission, "complete_manifest", "admission current")?;
    if field_string(
        selected_manifest,
        "document_sha256",
        "admission current complete_manifest",
    )? != manifest_document_sha256
        || field_string(
            selected_manifest,
            "seal_sha256",
            "admission current complete_manifest",
        )? != manifest_seal_sha256
    {
        return Err("admission current manifest identity drifted".to_owned());
    }
    let receipt_selection = field_object(&admission, "admission_receipt", "admission current")?;
    let receipt_path = canonical_regular_first_residual_authority(
        Path::new(field_string(
            receipt_selection,
            "path",
            "admission receipt",
        )?),
        "admission receipt",
    )?;
    let receipt = read_json_authority(&receipt_path, "admission receipt")?;
    let receipt_object = object(&receipt, "admission receipt")?;
    if field_string(receipt_object, "schema", "admission receipt")? != ADMISSION_RECEIPT_SCHEMA
        || field_string(receipt_object, "status", "admission receipt")? != ADMISSION_RECEIPT_STATUS
    {
        return Err("admission receipt schema/status drifted".to_owned());
    }
    let admission_receipt_seal_sha256 =
        field_string(receipt_object, "seal_sha256", "admission receipt")?.to_owned();
    require_sha256(&admission_receipt_seal_sha256, "admission receipt seal")?;
    if field_string(
        receipt_selection,
        "seal_sha256",
        "admission receipt selection",
    )? != admission_receipt_seal_sha256
    {
        return Err("admission receipt selection seal drifted".to_owned());
    }
    let revalidation = field_object(&receipt, "current_source_revalidation", "admission receipt")?;
    let source_audit_seal_sha256 = field_string(
        revalidation,
        "source_audit_seal_sha256",
        "admission revalidation",
    )?
    .to_owned();
    require_sha256(&source_audit_seal_sha256, "admission source audit seal")?;
    let source_revision =
        field_string(revalidation, "revision", "admission revalidation")?.to_owned();

    let baseline = read_json_authority(&args.cpu_baseline_receipt, "CPU baseline")?;
    let baseline_object = object(&baseline, "CPU baseline")?;
    if field_string(baseline_object, "schema", "CPU baseline")? != CPU_BASELINE_SCHEMA
        || field_string(baseline_object, "status", "CPU baseline")? != CPU_BASELINE_STATUS
        || field_string(baseline_object, "mode", "CPU baseline")? != "cpu-oracle"
    {
        return Err("CPU baseline schema/status/mode drifted".to_owned());
    }
    field_bool(
        baseline_object,
        "metal_device_or_dispatch_performed",
        false,
        "CPU baseline",
    )?;
    field_bool(baseline_object, "component_only", true, "CPU baseline")?;
    field_bool(
        baseline_object,
        "complete_layer_or_token_performed",
        false,
        "CPU baseline",
    )?;
    let baseline_artifact = field_object(&baseline, "artifact_binding", "CPU baseline")?;
    if field_string(
        baseline_artifact,
        "manifest_document_sha256",
        "CPU baseline artifact",
    )? != manifest_document_sha256
        || field_string(
            baseline_artifact,
            "manifest_seal_sha256",
            "CPU baseline artifact",
        )? != manifest_seal_sha256
        || field_string(
            baseline_artifact,
            "source_audit_seal_sha256",
            "CPU baseline artifact",
        )? != source_audit_seal_sha256
        || field_string(
            baseline_artifact,
            "source_revision",
            "CPU baseline artifact",
        )? != source_revision
        || baseline_artifact.get("layer").and_then(Value::as_u64) != Some(0)
        || baseline_artifact
            .get("linear_state_slot")
            .and_then(Value::as_u64)
            != Some(0)
    {
        return Err("CPU baseline artifact identity/geometry drifted".to_owned());
    }
    let provenance = field_object(&baseline, "same_input_provenance", "CPU baseline")?;
    if field_string(provenance, "kind", "CPU baseline provenance")?
        != "source_direct_packed_embedding_with_zeroed_layer0_deltanet_state"
        || field_string(provenance, "embedding_tensor", "CPU baseline provenance")?
            != "model.embed_tokens.weight"
    {
        return Err("CPU baseline source input provenance drifted".to_owned());
    }
    let source_token_id = provenance
        .get("token_id")
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| "CPU baseline token_id is invalid".to_owned())?;
    let cpu_input_f32le_sha256 = field_string(
        provenance,
        "input_hidden_f32le_sha256",
        "CPU baseline provenance",
    )?
    .to_owned();
    require_sha256(&cpu_input_f32le_sha256, "CPU baseline source input SHA")?;
    let input_file = provenance
        .get("input_hidden")
        .and_then(Value::as_object)
        .ok_or_else(|| "CPU baseline provenance.input_hidden must be an object".to_owned())?;
    let input_path = canonical_regular_first_residual_authority(
        Path::new(field_string(input_file, "path", "CPU baseline input file")?),
        "CPU baseline input file",
    )?;
    if input_file.get("bytes").and_then(Value::as_u64) != Some(FIRST_RESIDUAL_BYTES as u64)
        || field_string(input_file, "sha256", "CPU baseline input file")? != cpu_input_f32le_sha256
        || sha256_file(&input_path, "CPU baseline input file")? != cpu_input_f32le_sha256
    {
        return Err("CPU baseline input file identity drifted".to_owned());
    }
    let mut state_hashes = std::collections::BTreeMap::new();
    for (state_name, elements) in [
        ("initial_conv_state", CONV_STATE_ELEMENTS),
        ("initial_recurrent_state", RECURRENT_STATE_ELEMENTS),
    ] {
        let state = provenance
            .get(state_name)
            .and_then(Value::as_object)
            .ok_or_else(|| "CPU baseline state must be an object".to_owned())?;
        if state.get("elements").and_then(Value::as_u64) != Some(elements as u64) {
            return Err("CPU baseline state element geometry drifted".to_owned());
        }
        field_bool(state, "zero_initialized", true, "CPU baseline state")?;
        let hash = field_string(state, "f32le_sha256", "CPU baseline state")?.to_owned();
        require_sha256(&hash, "CPU baseline state SHA")?;
        state_hashes.insert(state_name, hash);
    }
    let initial_conv_state_f32le_sha256 = state_hashes
        .remove("initial_conv_state")
        .ok_or("CPU baseline conv state SHA missing")?;
    let initial_recurrent_state_f32le_sha256 = state_hashes
        .remove("initial_recurrent_state")
        .ok_or("CPU baseline recurrent state SHA missing")?;
    let output = field_object(&baseline, "first_residual_output", "CPU baseline")?;
    let cpu_reference_first_residual_sha256 =
        field_string(output, "sha256", "CPU baseline first residual")?.to_owned();
    require_sha256(
        &cpu_reference_first_residual_sha256,
        "CPU baseline first residual SHA",
    )?;
    if output.get("layer").and_then(Value::as_u64) != Some(0)
        || output.get("linear_state_slot").and_then(Value::as_u64) != Some(0)
        || output.get("elements").and_then(Value::as_u64) != Some(HIDDEN as u64)
        || output.get("bytes").and_then(Value::as_u64) != Some(FIRST_RESIDUAL_BYTES as u64)
        || field_string(output, "f32le_sha256", "CPU baseline first residual")?
            != cpu_reference_first_residual_sha256
    {
        return Err("CPU baseline first-residual geometry/hash drifted".to_owned());
    }
    let output_file = output
        .get("file")
        .and_then(Value::as_object)
        .ok_or_else(|| "CPU baseline first_residual_output.file must be an object".to_owned())?;
    let output_path = canonical_regular_first_residual_authority(
        Path::new(field_string(
            output_file,
            "path",
            "CPU baseline first-residual file",
        )?),
        "CPU baseline first-residual file",
    )?;
    if output_file.get("bytes").and_then(Value::as_u64) != Some(FIRST_RESIDUAL_BYTES as u64)
        || field_string(output_file, "sha256", "CPU baseline first-residual file")?
            != cpu_reference_first_residual_sha256
        || sha256_file(&output_path, "CPU baseline first-residual file")?
            != cpu_reference_first_residual_sha256
    {
        return Err("CPU baseline first-residual file identity drifted".to_owned());
    }
    let cpu_baseline_sha256 = sha256_file(&args.cpu_baseline_receipt, "CPU baseline")?;

    let lease = read_json_authority(&args.lease_receipt, "lease receipt")?;
    let lease_object = object(&lease, "lease receipt")?;
    if field_string(lease_object, "schema", "lease receipt")? != LEASE_SCHEMA
        || field_string(lease_object, "status", "lease receipt")? != LEASE_STATUS
    {
        return Err("lease schema/status drifted".to_owned());
    }
    let lease_seal_sha256 = field_string(lease_object, "seal_sha256", "lease receipt")?.to_owned();
    require_sha256(&lease_seal_sha256, "lease seal")?;
    let lease_artifact = field_object(&lease, "artifact_binding", "lease receipt")?;
    if field_string(lease_artifact, "manifest_document_sha256", "lease artifact")?
        != manifest_document_sha256
        || field_string(lease_artifact, "manifest_seal_sha256", "lease artifact")?
            != manifest_seal_sha256
        || field_string(
            lease_artifact,
            "admission_receipt_seal_sha256",
            "lease artifact",
        )? != admission_receipt_seal_sha256
    {
        return Err("lease artifact identity drifted".to_owned());
    }
    let lifecycle = field_object(&lease, "lifecycle", "lease receipt")?;
    if lifecycle
        .get("fresh_for_this_exact_launch")
        .and_then(Value::as_bool)
        != Some(true)
        || lifecycle
            .get("automatic_retry_prohibited")
            .and_then(Value::as_bool)
            != Some(true)
        || lifecycle
            .get("outer_reaped_capture_required")
            .and_then(Value::as_bool)
            != Some(true)
    {
        return Err("lease lifecycle is not fresh/one-shot/outer-reaped".to_owned());
    }
    let policy = field_object(&lease, "execution_policy", "lease receipt")?;
    if field_string(policy, "component", "lease execution policy")?
        != "qwen80_first_residual_bridge"
        || policy
            .get("quiet_qwen80_device_lease")
            .and_then(Value::as_bool)
            != Some(true)
        || policy.get("strict_math").and_then(Value::as_bool) != Some(true)
        || policy
            .get("timing_or_benchmarking_allowed")
            .and_then(Value::as_bool)
            != Some(false)
        || policy
            .get("complete_layer_or_token_allowed")
            .and_then(Value::as_bool)
            != Some(false)
        || policy
            .get("tps_or_tg_claim_allowed")
            .and_then(Value::as_bool)
            != Some(false)
    {
        return Err("lease policy is not prefix-only strict-Math component authority".to_owned());
    }
    let baseline_binding = field_object(&lease, "cpu_baseline_binding", "lease receipt")?;
    if field_string(
        baseline_binding,
        "receipt_document_sha256",
        "lease CPU baseline",
    )? != cpu_baseline_sha256
        || field_string(baseline_binding, "schema", "lease CPU baseline")? != CPU_BASELINE_SCHEMA
        || field_string(baseline_binding, "status", "lease CPU baseline")? != CPU_BASELINE_STATUS
    {
        return Err("lease CPU baseline identity drifted".to_owned());
    }
    let lease_sha256 = sha256_file(&args.lease_receipt, "lease receipt")?;
    Ok(FirstResidualChildAuthority {
        args,
        manifest_document_sha256,
        manifest_seal_sha256,
        admission_pointer_seal_sha256,
        admission_receipt_seal_sha256,
        source_audit_seal_sha256,
        source_revision,
        cpu_baseline_sha256,
        cpu_input_f32le_sha256,
        cpu_reference_first_residual_sha256,
        initial_conv_state_f32le_sha256,
        initial_recurrent_state_f32le_sha256,
        source_token_id,
        lease_sha256,
        lease_seal_sha256,
    })
}

fn write_new_atomic(capture_dir: &Path, name: &str, contents: &[u8]) -> Result<(), String> {
    let target = capture_dir.join(name);
    let temporary = capture_dir.join(format!(".{name}.{}.tmp", process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| format!("cannot create {}: {error}", temporary.display()))?;
    if let Err(error) = file.write_all(contents).and_then(|_| file.sync_all()) {
        let _ = fs::remove_file(&temporary);
        return Err(format!("cannot write {}: {error}", temporary.display()));
    }
    drop(file);
    fs::hard_link(&temporary, &target).map_err(|error| {
        let _ = fs::remove_file(&temporary);
        format!("cannot publish {}: {error}", target.display())
    })?;
    fs::remove_file(&temporary)
        .map_err(|error| format!("cannot retire {}: {error}", temporary.display()))
}

fn capture_started_unix_millis() -> Result<u128, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock before Unix epoch: {error}"))
        .map(|duration| duration.as_millis())
}

fn begin_prefix_capture(authority: &FirstResidualChildAuthority) -> Result<(), String> {
    fs::create_dir(&authority.args.capture_dir).map_err(|error| {
        format!(
            "refusing non-exclusive --capture-dir {}: {error}",
            authority.args.capture_dir.display()
        )
    })?;
    let invocation = json!({
        "schema": CHILD_SCHEMA,
        "status": "STARTED_QWEN80_SOURCE_INPUT_FIRST_RESIDUAL_STRICT_MATH_PREFIX_COMPONENT",
        "started_unix_millis": capture_started_unix_millis()?,
        "mode": STRICT_METAL_MODE,
        "manifest": authority.args.manifest,
        "admission_current": authority.args.admission_current,
        "cpu_baseline_receipt": authority.args.cpu_baseline_receipt,
        "lease_receipt": authority.args.lease_receipt,
        "workers": authority.args.workers,
        "execution_policy": {
            "strict_math": true,
            "timing_or_benchmarking_allowed": false,
            "prefix_dispatches_expected": PREFIX_DISPATCHES,
            "true_moe_suffix_dispatches_expected": 0,
            "complete_layer_or_token_allowed": false,
            "tps_or_tg_claim_allowed": false,
        },
    });
    write_new_atomic(
        &authority.args.capture_dir,
        "invocation.json",
        &serde_json::to_vec_pretty(&invocation).map_err(|error| error.to_string())?,
    )
}

#[cfg(target_os = "macos")]
fn run_prefix_component(
    authority: &FirstResidualChildAuthority,
) -> Result<serde_json::Value, String> {
    let admission = CompleteBinaryAdmission {
        model: QwenCompleteBinaryModel::Qwen80CoderNext,
        expected_manifest_seal_sha256: authority.manifest_seal_sha256.clone(),
        expected_source_audit_seal_sha256: authority.source_audit_seal_sha256.clone(),
        expected_source_revision: authority.source_revision.clone(),
    };
    // Exactly one strict all-artifact scan happens here.  The catalog is then
    // consumed in-process by the strict-Math native state body; no per-tensor
    // file reopen or BF16/MPS shadow route is used.
    let catalog = Qwen80CompleteArtifactCatalog::load(&authority.args.manifest, &admission)
        .map_err(|error| format!("strict Qwen80 artifact admission failed: {error}"))?;
    let runtime = Qwen80CompleteNativeRuntime::from_admitted_catalog_strict_math(
        catalog,
        Qwen80CompleteRuntimeOptions {
            max_seq_len: 1,
            trace_dispatch: false,
        },
    )
    .map_err(|error| format!("strict-Math Qwen80 native state construction failed: {error}"))?;
    let mut command = runtime.begin_component_token_command_buffer();
    let command_buffer_identity = format!(
        "qwen80-l0-prefix-pid{}-{}-{}",
        process::id(),
        capture_started_unix_millis()?,
        &authority.manifest_document_sha256[..12]
    );
    let encoder = runtime
        .encode_source_token_first_linear_deltanet_into(&mut command, authority.source_token_id)
        .map_err(|error| format!("source-token L0 DeltaNet prefix encode failed: {error}"))?;
    let dispatches = command.dispatch_count();
    if dispatches != PREFIX_DISPATCHES {
        return Err(format!(
            "source-token L0 prefix encoded {dispatches} dispatches, expected {PREFIX_DISPATCHES}"
        ));
    }
    command
        .commit_and_wait()
        .map_err(|error| format!("source-token L0 prefix fence failed: {error}"))?;
    let parity = encoder
        .verify_after_fence(&runtime)
        .map_err(|error| format!("source-token L0 prefix CPU/device parity failed: {error}"))?;
    if parity.input_f32le_sha256 != authority.cpu_input_f32le_sha256
        || parity.initial_conv_state_f32le_sha256 != authority.initial_conv_state_f32le_sha256
        || parity.initial_recurrent_state_f32le_sha256
            != authority.initial_recurrent_state_f32le_sha256
        || parity.cpu_first_residual_f32le_sha256 != authority.cpu_reference_first_residual_sha256
    {
        return Err(
            "source-token input or zero-state identity differs from the CPU baseline".to_owned(),
        );
    }
    if parity.dispatches_encoded_before_suffix != PREFIX_DISPATCHES
        || parity.first_residual_elements != HIDDEN
        || parity.first_residual_bytes != FIRST_RESIDUAL_BYTES
        || !parity.same_command_graph_required
    {
        return Err("source-token first-residual device geometry/order drifted".to_owned());
    }
    let tolerance = 1.0e-3f32;
    if parity.first_residual_max_abs_error > tolerance
        || parity.conv_state_max_abs_error > tolerance
        || parity.recurrent_state_max_abs_error > tolerance
    {
        return Err("source-token L0 prefix parity exceeds strict component tolerance".to_owned());
    }
    Ok(json!({
        "schema": CHILD_SCHEMA,
        "status": "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_MIXER_FIRST_RESIDUAL_STRICT_MATH_METAL_COMPONENT_NOT_COMPLETE_LAYER_OR_TOKEN",
        "mode": STRICT_METAL_MODE,
        "metal_device_or_dispatch_performed": true,
        "component_only": true,
        "complete_layer_or_token_performed": false,
        "synthetic_input": false,
        "fixture_only": false,
        "complete_artifact_scan_performed_once": true,
        "raw_bf16_or_safetensors_opened": false,
        "artifact_binding": {
            "manifest_path": authority.args.manifest,
            "manifest_document_sha256": authority.manifest_document_sha256,
            "manifest_seal_sha256": authority.manifest_seal_sha256,
            "admission_current_path": authority.args.admission_current,
            "admission_pointer_seal_sha256": authority.admission_pointer_seal_sha256,
            "admission_receipt_seal_sha256": authority.admission_receipt_seal_sha256,
            "source_audit_seal_sha256": authority.source_audit_seal_sha256,
            "source_revision": authority.source_revision,
            "layer": 0,
            "linear_state_slot": 0,
            "native_device": runtime.device_name(),
        },
        "cpu_baseline_binding": {
            "receipt_path": authority.args.cpu_baseline_receipt,
            "receipt_document_sha256": authority.cpu_baseline_sha256,
            "schema": CPU_BASELINE_SCHEMA,
            "status": CPU_BASELINE_STATUS,
        },
        "same_input_provenance": {
            "kind": "source_direct_packed_embedding_with_zeroed_layer0_deltanet_state",
            "token_id": authority.source_token_id,
            "embedding_tensor": "model.embed_tokens.weight",
            "input_hidden_f32le_sha256": parity.input_f32le_sha256,
            "initial_conv_state": {
                "elements": CONV_STATE_ELEMENTS,
                "f32le_sha256": parity.initial_conv_state_f32le_sha256,
                "zero_initialized": true,
            },
            "initial_recurrent_state": {
                "elements": RECURRENT_STATE_ELEMENTS,
                "f32le_sha256": parity.initial_recurrent_state_f32le_sha256,
                "zero_initialized": true,
            },
        },
        "first_residual_output": {
            "layer": parity.layer,
            "linear_state_slot": parity.linear_state_slot,
            "elements": parity.first_residual_elements,
            "bytes": parity.first_residual_bytes,
            "sha256": parity.device_first_residual_f32le_sha256,
            "cpu_reference_sha256": parity.cpu_first_residual_f32le_sha256,
            "all_finite": true,
        },
        "same_command_graph": {
            "same_command_graph_required": true,
            "same_command_graph_retained": true,
            "command_buffer_identity": command_buffer_identity,
            "device_first_residual_buffer_bytes": parity.first_residual_bytes,
            "input_then_deltanet_then_first_residual_then_fence_order": true,
            "prefix_only": true,
            "prefix_dispatches": dispatches,
            "suffix_dispatches": 0,
            "total_dispatches": dispatches,
            "no_true_moe_suffix_encoded": true,
        },
        "cpu_device_parity": {
            "checked_elements": parity.first_residual_elements,
            "passed": true,
            "max_abs_error": parity.first_residual_max_abs_error,
            "tolerance": tolerance,
        },
        "state_witness": {
            "linear_state_slot": parity.linear_state_slot,
            "conv_state_elements": CONV_STATE_ELEMENTS,
            "recurrent_state_elements": RECURRENT_STATE_ELEMENTS,
            "initial_conv_state_identity_matches_cpu_baseline": true,
            "initial_recurrent_state_identity_matches_cpu_baseline": true,
            "post_fence_conv_state_max_abs_error": parity.conv_state_max_abs_error,
            "post_fence_recurrent_state_max_abs_error": parity.recurrent_state_max_abs_error,
            "state_commit_after_parity_fence": true,
        },
        "metal_execution_policy": {
            "strict_math_required": true,
            "timing_or_benchmarking_allowed": false,
            "complete_layer_or_token_allowed": false,
            "tps_or_tg_claim_allowed": false,
            "lease_binding": {
                "receipt_path": authority.args.lease_receipt,
                "receipt_document_sha256": authority.lease_sha256,
                "seal_sha256": authority.lease_seal_sha256,
                "schema": LEASE_SCHEMA,
                "status": LEASE_STATUS,
            },
        },
        "claim_boundary": {
            "source_input_l0_deltanet_prefix_only": true,
            "postnorm_router_routed_experts_shared_expert_second_residual_not_executed": true,
            "not_a_complete_layer_or_token_or_decoder_generation_hcli_tps_tg_or_tournament_result": true,
        },
    }))
}

#[cfg(not(target_os = "macos"))]
fn run_prefix_component(
    _authority: &FirstResidualChildAuthority,
) -> Result<serde_json::Value, String> {
    Err("Qwen80 strict-Metal prefix component requires macOS".to_owned())
}

fn finalize_prefix_capture(
    authority: &FirstResidualChildAuthority,
    stage_result: Result<serde_json::Value, String>,
) -> Result<(serde_json::Value, Option<String>), String> {
    let (mut receipt, failure) = match stage_result {
        Ok(value) => (value, None),
        Err(error) => (
            json!({
                "schema": CHILD_SCHEMA,
                "status": "REFUSED_QWEN80_SOURCE_INPUT_FIRST_RESIDUAL_STRICT_MATH_COMPONENT_ERROR",
                "mode": STRICT_METAL_MODE,
                "metal_device_or_dispatch_performed": "unknown_after_authorized_attempt_error",
                "component_only": true,
                "complete_layer_or_token_performed": false,
                "error": error,
                "claim_boundary": {
                    "no_successful_device_parity_or_runtime_result_is_claimed": true,
                },
            }),
            Some(error),
        ),
    };
    let object = receipt
        .as_object_mut()
        .ok_or("prefix result must be a JSON object")?;
    object.insert(
        "durable_capture".to_owned(),
        json!({
            "directory": authority.args.capture_dir,
            "invocation_file": "invocation.json",
            "stdout_file": "stdout.jsonl",
            "stderr_file": "stderr.log",
            "receipt_file": "receipt.json",
            "receipt_written_last_is_completion_marker": true,
            "outer_reaped_capture_required": true,
            "replay_guarded": true,
        }),
    );
    let rendered = serde_json::to_vec_pretty(&receipt).map_err(|error| error.to_string())?;
    let mut stdout = rendered.clone();
    stdout.push(b'\n');
    write_new_atomic(&authority.args.capture_dir, "stdout.jsonl", &stdout)?;
    let stderr = failure
        .as_ref()
        .map_or_else(|| b"\n".to_vec(), |error| format!("{error}\n").into_bytes());
    write_new_atomic(&authority.args.capture_dir, "stderr.log", &stderr)?;
    write_new_atomic(&authority.args.capture_dir, "receipt.json", &rendered)?;
    Ok((receipt, failure))
}

fn main() {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    if arguments.is_empty() {
        println!("{}", prepared_document());
        return;
    }
    let args = match parse_first_residual_child_args(arguments) {
        Ok(args) => args,
        Err(error) => {
            eprintln!("Qwen80 first-residual strict-Metal child refused arguments: {error}");
            std::process::exit(2);
        }
    };
    let authority = match validate_first_residual_child_authority(&args) {
        Ok(authority) => authority,
        Err(error) => {
            eprintln!(
                "Qwen80 first-residual strict-Metal child refused authority binding: {error}"
            );
            std::process::exit(2);
        }
    };
    if let Err(error) = begin_prefix_capture(&authority) {
        eprintln!("Qwen80 first-residual strict-Metal child refused capture setup: {error}");
        std::process::exit(2);
    }
    match finalize_prefix_capture(&authority, run_prefix_component(&authority)) {
        Ok((receipt, None)) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&receipt)
                    .expect("prefix receipt JSON serialization must succeed")
            );
        }
        Ok((_receipt, Some(error))) => {
            eprintln!("Qwen80 first-residual strict-Metal component refused: {error}");
            std::process::exit(2);
        }
        Err(error) => {
            eprintln!("Qwen80 first-residual strict-Metal capture finalization failed: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        parse_first_residual_child_args, prepared_document, CHILD_SCHEMA, PREPARED_STATUS,
    };

    fn first_residual_child_args() -> Vec<String> {
        [
            "--manifest",
            "/tmp/q80/manifest.json",
            "--admission-current",
            "/tmp/q80/admission-current.json",
            "--cpu-baseline-receipt",
            "/tmp/q80/cpu-baseline.json",
            "--lease-receipt",
            "/tmp/q80/lease.json",
            "--outer-capture-dir",
            "/tmp/q80/outer",
            "--capture-dir",
            "/tmp/q80/outer/inner",
            "--mode",
            "metal",
            "--workers",
            "2",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    }

    #[test]
    fn prepared_contract_retains_the_exact_single_command_buffer_boundary() {
        let document = prepared_document();
        assert_eq!(document["schema"], CHILD_SCHEMA);
        assert_eq!(document["status"], PREPARED_STATUS);
        assert_eq!(document["device_execution_performed"], false);
        assert_eq!(document["command_graph_contract"]["prefix_dispatches"], 9);
        assert_eq!(document["command_graph_contract"]["suffix_dispatches"], 0);
        assert_eq!(
            document["command_graph_contract"]["same_token_command_buffer_required"],
            true
        );
        assert_eq!(
            document["command_graph_contract"]["first_residual_device_buffer_bytes"],
            8192
        );
    }

    #[test]
    fn prepared_contract_cannot_be_mistaken_for_a_device_or_token_result() {
        let document = prepared_document();
        assert_eq!(
            document["claim_boundary"]["no_metal_context_or_gpu_dispatch"],
            true
        );
        assert_eq!(
            document["claim_boundary"]
                ["no_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim"],
            true
        );
    }

    #[test]
    fn child_requires_cpu_baseline_lease_and_outer_capture_paths() {
        let parsed = parse_first_residual_child_args(first_residual_child_args()).unwrap();
        assert_eq!(parsed.workers, 2);
        assert!(parsed.cpu_baseline_receipt.is_absolute());
        assert_eq!(
            parsed.capture_dir.parent().unwrap(),
            parsed.outer_capture_dir.as_path()
        );
    }

    #[test]
    fn child_refuses_missing_or_duplicate_authority_arguments() {
        let missing = vec!["--manifest".to_owned(), "/tmp/q80/manifest.json".to_owned()];
        assert!(parse_first_residual_child_args(missing)
            .unwrap_err()
            .contains("missing --admission-current"));
        let mut duplicate = first_residual_child_args();
        duplicate.extend(["--workers".to_owned(), "3".to_owned()]);
        assert!(parse_first_residual_child_args(duplicate)
            .unwrap_err()
            .contains("--workers repeated"));
    }

    #[test]
    fn child_refuses_non_metal_or_out_of_range_worker_preflight() {
        let mut non_metal = first_residual_child_args();
        let mode_index = non_metal
            .iter()
            .position(|value| value == "--mode")
            .unwrap();
        non_metal[mode_index + 1] = "cpu-oracle".to_owned();
        assert!(parse_first_residual_child_args(non_metal)
            .unwrap_err()
            .contains("--mode must be \"metal\""));
        let mut oversized = first_residual_child_args();
        let worker_index = oversized
            .iter()
            .position(|value| value == "--workers")
            .unwrap();
        oversized[worker_index + 1] = "5".to_owned();
        assert!(parse_first_residual_child_args(oversized)
            .unwrap_err()
            .contains("--workers must be 1..4"));
    }
}
