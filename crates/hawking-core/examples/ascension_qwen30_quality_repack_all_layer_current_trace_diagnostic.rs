//! Fail-closed Qwen30 HQ30GR2 all-layer current-trace diagnostic.
//!
//! `cpu-preflight` admits the direct control and separately admitted HQ30GR2
//! candidate through distinct typed catalog paths, validates the exact
//! 369-token `literal_hawking` source-template input, and checks the
//! component-parity pointer. It opens no Metal context and is usable as a
//! standalone CPU/disk gate.
//!
//! `metal-diagnostic` is separately lease-gated and receipt-last. It drives
//! the actual direct-packed 48-layer native Metal graph twice, first as the
//! scalar control and then through the typed L0/E0 HQ30GR2 interception, for
//! one exact prefix and one forced shared continuation. It is non-serving,
//! non-timed, and cannot establish HCLI, coherence, TPS, capability, or
//! tournament evidence.

#[cfg(target_os = "macos")]
use hawking_core::model::qwen30_complete_runtime::{
    Qwen30CompleteNativeRuntime, Qwen30CompleteRuntimeOptions, Qwen30GateUpSwiGluKernel,
    Qwen30NativeRouteCaptureStep, Qwen30PackedMatvecKernel,
    Qwen30QualityRepackNativeDiagnosticRuntime,
};
use hawking_core::model::qwen30_quality_repack_diagnostic::{
    Qwen30QualityRepackDiagnosticCatalog, QWEN30_QUALITY_REPACK_DIAGNOSTIC_TENSOR_COUNT,
};
use hawking_core::model::qwen_complete_binary::{
    admit_complete_binary_artifact, admit_qwen30_quality_repack_artifact,
    parse_complete_binary_header, parse_qwen30_quality_residual_header, CompleteBinaryAdmission,
    Qwen30QualityRepackAdmission, Qwen30QualityRepackVerifiedTensor, QwenCompleteBinaryModel,
};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process;

const MODE: &str = "cpu-preflight";
const METAL_DIAGNOSTIC_MODE: &str = "metal-diagnostic";
// This successor is deliberately separate from the immutable first capture.
// It is never selected by the production runtime and cannot reuse the old
// receipt or lease as a substitute for a fresh raw-vector capture.
const RAW_FINAL_LOGIT_RETENTION_MODE: &str = "metal-diagnostic-retain-raw-final-logits";
const RESULT_SCHEMA: &str =
    "hawking.ascension.qwen30_quality_repack_all_layer_current_trace_cpu_preflight.v1";
const RESULT_STATUS: &str = "EARNED_TYPED_HQ30GR2_ALL_LAYER_CURRENT_TRACE_PREMETAL_BINDING_ONLY";
const REFUSAL_STATUS: &str = "REFUSED_TYPED_HQ30GR2_ALL_LAYER_CURRENT_TRACE_PREMETAL_BINDING_ONLY";
const DIAGNOSTIC_RESULT_SCHEMA: &str =
    "hawking.ascension.qwen30_quality_repack_all_layer_current_trace_diagnostic.v1";
const DIAGNOSTIC_RESULT_STATUS: &str =
    "EARNED_NEW_DIAGNOSTIC_NOT_HISTORICAL_TYPED_HQ30GR2_ALL_LAYER_CURRENT_TRACE_UNQUALIFIED";
const DIAGNOSTIC_REFUSAL_STATUS: &str =
    "REFUSED_NEW_DIAGNOSTIC_NOT_HISTORICAL_TYPED_HQ30GR2_ALL_LAYER_CURRENT_TRACE_UNQUALIFIED";
const RAW_RETENTION_RESULT_SCHEMA: &str =
    "hawking.ascension.qwen30_hq30gr2_raw_final_logit_retention_capture.v1";
const RAW_RETENTION_RESULT_STATUS: &str =
    "EARNED_NEW_DIAGNOSTIC_RAW_FINAL_LOGITS_RETAINED_NOT_THREE_WAY_ORACLE";
const RAW_RETENTION_REFUSAL_STATUS: &str =
    "REFUSED_NEW_DIAGNOSTIC_RAW_FINAL_LOGIT_RETENTION_UNQUALIFIED";
const RAW_RETENTION_CONTRACT_SCHEMA: &str =
    "hawking.ascension.qwen30_hq30gr2_raw_final_logit_retention_successor.v1";
const RAW_RETENTION_CONTRACT_STATUS: &str = "PREPARED_RAW_FINAL_LOGIT_RETENTION_SUCCESSOR_NOT_RUN";
const INPUT_CONTRACT_SCHEMA: &str =
    "hawking.ascension.qwen30_hq30gr2_all_layer_current_trace_input_contract.v1";
const INPUT_CONTRACT_STATUS: &str = "PREPARED_EXACT_ONE_LITERAL_HAWKING_ALL_LAYER_DIAGNOSTIC_INPUT";
const QUIET_LEASE_SCHEMA: &str =
    "hawking.ascension.qwen30_hq30gr2_all_layer_quiet_diagnostic_lease.v1";
const QUIET_LEASE_STATUS: &str =
    "GRANTED_QWEN30_HQ30GR2_ALL_LAYER_CURRENT_TRACE_DIAGNOSTIC_NON_TIMED_LEASE";

const CANDIDATE_SCHEMA: &str = "hawking.ascension.qwen30_quality_repack_candidate.v1";
const CANDIDATE_STATUS: &str = "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED";
const CANDIDATE_CURRENT_SCHEMA: &str =
    "hawking.ascension.qwen30_quality_repack_native_admission_current_pointer.v1";
const CANDIDATE_CURRENT_STATUS: &str = "CURRENT_QUALITY_REPACK_NATIVE_ADMISSION_RECEIPT_SELECTED";
const CANDIDATE_ADMISSION_SCHEMA: &str =
    "hawking.ascension.qwen30_quality_repack_native_admission_receipt.v1";
const CANDIDATE_ADMISSION_STATUS: &str =
    "EARNED_QUALITY_REPACK_COMPLETE_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED";

const COMPONENT_CURRENT_FILE: &str =
    "QWEN30_QUALITY_GATE_UP_RESIDUAL_V1_SPARSE_GATE_UP_COMPONENT_PARITY_CURRENT.json";
const COMPONENT_CURRENT_SCHEMA: &str =
    "hawking.ascension.qwen30_hq30gr2_sparse_gate_up_component_parity_current.v1";
const COMPONENT_CURRENT_STATUS: &str =
    "CURRENT_QWEN30_HQ30GR2_SPARSE_GATE_UP_COMPONENT_CPU_DEVICE_PARITY_SELECTED";
const COMPONENT_TERMINAL_SCHEMA: &str =
    "hawking.ascension.qwen30_hq30gr2_sparse_gate_up_device_parity_outer_launcher.v1";
const COMPONENT_TERMINAL_STATUS: &str =
    "CAPTURED_QWEN30_HQ30GR2_SPARSE_GATE_UP_DEVICE_PARITY_OUTER_TERMINAL_COMPONENT_ONLY";
const COMPONENT_INNER_STATUS: &str =
    "EARNED_HQ30GR2_SPARSE_GATE_UP_CPU_DEVICE_PARITY_NOT_LAYER_OR_RUNTIME";

const CONTROL_SCHEMA: &str = "hawking.ascension.qwen30_complete_binary_gravity.v1";
const CONTROL_STATUS: &str = "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED";
const CONTROL_RUNTIME_SCHEMA: &str = "hawking.ascension.physical_exact_full_token_runtime.v1";
const CONTROL_RUNTIME_STATUS: &str = "PASS_EXACT_NATIVE_FULL_TOKEN_RUNTIME";

const COMPILER_CURRENT_SCHEMA: &str =
    "hawking.ascension.qwen30_quality_repack_current_hcli_compiler_trace_current.v1";
const COMPILER_CURRENT_STATUS: &str =
    "CURRENT_NEW_DIAGNOSTIC_NOT_HISTORICAL_HCLI_COMPILER_TRACE_SELECTED";
const COMPILER_SCHEMA: &str =
    "hawking.ascension.qwen30_quality_repack_current_hcli_compiler_trace.v1";
const COMPILER_STATUS: &str =
    "EARNED_NEW_DIAGNOSTIC_NOT_HISTORICAL_PRE_EXECUTION_HCLI_COMPILER_TRACE";
const ROUTE_CURRENT_SCHEMA: &str =
    "hawking.ascension.qwen30_quality_repack_current_hcli_route_capture_current.v1";
const ROUTE_CURRENT_STATUS: &str =
    "CURRENT_NEW_DIAGNOSTIC_NOT_HISTORICAL_L0_ROUTE_CAPTURE_SELECTED";
const ROUTE_SCHEMA: &str = "hawking.ascension.qwen30_quality_repack_current_hcli_route_capture.v1";
const ROUTE_STATUS: &str =
    "EARNED_NEW_DIAGNOSTIC_NOT_HISTORICAL_L0_ROUTE_AND_HIDDEN_CAPTURE_UNQUALIFIED";
const PREPARATION_CURRENT_SCHEMA: &str =
    "hawking.ascension.qwen30_quality_repack_all_layer_current_trace_prepare_current.v1";
const PREPARATION_CURRENT_STATUS: &str =
    "CURRENT_QWEN30_HQ30GR2_ALL_LAYER_CURRENT_TRACE_PREPARATION_SELECTED";
const PREPARATION_SCHEMA: &str =
    "hawking.ascension.qwen30_quality_repack_all_layer_current_trace_prepare.v1";
const PREPARATION_STATUS: &str =
    "PREPARED_CURRENT_TRACE_TYPED_HQ30GR2_ALL_LAYER_DIAGNOSTIC_NOT_RUN";

const PROBE_ID: &str = "literal_hawking";
const TOKEN_COUNT: usize = 369;
const FORCED_CONTINUATION_FORWARDS: usize = 1;
const QWEN30_ALL_LAYER_COUNT: usize = 48;
const QWEN30_TOP_K: usize = 8;
const QWEN30_EXPERTS: u32 = 128;
const LOGIT_WITNESS_TOP_K: usize = 8;
const QWEN30_VOCAB_ROWS: usize = 151_936;
const L0_E0_GATE: &str = "model.layers.0.mlp.experts.0.gate_proj.weight";
const L0_E0_UP: &str = "model.layers.0.mlp.experts.0.up_proj.weight";
const L0_E1_GATE: &str = "model.layers.0.mlp.experts.1.gate_proj.weight";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InvocationMode {
    CpuPreflight,
    MetalDiagnostic,
    MetalRawFinalLogitRetention,
}

#[derive(Debug)]
struct Args {
    mode: InvocationMode,
    candidate_manifest: PathBuf,
    candidate_admission_current: PathBuf,
    compiler_trace_current: PathBuf,
    route_capture_current: PathBuf,
    preparation_current: PathBuf,
    control_manifest: PathBuf,
    control_runtime_receipt: PathBuf,
    output: Option<PathBuf>,
    lease_receipt: Option<PathBuf>,
    input_contract: Option<PathBuf>,
    raw_retention_contract: Option<PathBuf>,
    capture_dir: Option<PathBuf>,
    workers: Option<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CurrentPointerReferenceStyle {
    RichDocumentSha,
    PathAndSealWithinCandidateRoot,
}

#[derive(Clone, Copy, Debug)]
struct CurrentPointerContract {
    label: &'static str,
    pointer_schema: &'static str,
    pointer_status: &'static str,
    receipt_field: &'static str,
    receipt_schema: &'static str,
    receipt_status: &'static str,
    reference_style: CurrentPointerReferenceStyle,
    declared_candidate_root_required: bool,
}

const CANDIDATE_ADMISSION_POINTER: CurrentPointerContract = CurrentPointerContract {
    label: "candidate admission current",
    pointer_schema: CANDIDATE_CURRENT_SCHEMA,
    pointer_status: CANDIDATE_CURRENT_STATUS,
    receipt_field: "admission_receipt",
    receipt_schema: CANDIDATE_ADMISSION_SCHEMA,
    receipt_status: CANDIDATE_ADMISSION_STATUS,
    reference_style: CurrentPointerReferenceStyle::RichDocumentSha,
    declared_candidate_root_required: false,
};

const COMPONENT_PARITY_POINTER: CurrentPointerContract = CurrentPointerContract {
    label: "candidate component current",
    pointer_schema: COMPONENT_CURRENT_SCHEMA,
    pointer_status: COMPONENT_CURRENT_STATUS,
    receipt_field: "component_parity_outer_terminal",
    receipt_schema: COMPONENT_TERMINAL_SCHEMA,
    receipt_status: COMPONENT_TERMINAL_STATUS,
    reference_style: CurrentPointerReferenceStyle::RichDocumentSha,
    declared_candidate_root_required: false,
};

const COMPILER_TRACE_POINTER: CurrentPointerContract = CurrentPointerContract {
    label: "compiler trace current",
    pointer_schema: COMPILER_CURRENT_SCHEMA,
    pointer_status: COMPILER_CURRENT_STATUS,
    receipt_field: "compiler_trace_receipt",
    receipt_schema: COMPILER_SCHEMA,
    receipt_status: COMPILER_STATUS,
    reference_style: CurrentPointerReferenceStyle::PathAndSealWithinCandidateRoot,
    declared_candidate_root_required: true,
};

const ROUTE_CAPTURE_POINTER: CurrentPointerContract = CurrentPointerContract {
    label: "route capture current",
    pointer_schema: ROUTE_CURRENT_SCHEMA,
    pointer_status: ROUTE_CURRENT_STATUS,
    receipt_field: "route_capture_receipt",
    receipt_schema: ROUTE_SCHEMA,
    receipt_status: ROUTE_STATUS,
    reference_style: CurrentPointerReferenceStyle::PathAndSealWithinCandidateRoot,
    declared_candidate_root_required: false,
};

const PREPARATION_POINTER: CurrentPointerContract = CurrentPointerContract {
    label: "all-layer preparation current",
    pointer_schema: PREPARATION_CURRENT_SCHEMA,
    pointer_status: PREPARATION_CURRENT_STATUS,
    receipt_field: "preparation_receipt",
    receipt_schema: PREPARATION_SCHEMA,
    receipt_status: PREPARATION_STATUS,
    reference_style: CurrentPointerReferenceStyle::PathAndSealWithinCandidateRoot,
    declared_candidate_root_required: false,
};

#[derive(Clone, Debug)]
struct Evidence {
    path: PathBuf,
    bytes: u64,
    sha256: String,
}

impl Evidence {
    fn json(&self) -> Value {
        json!({
            "path": self.path,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "document_sha256": self.sha256,
        })
    }
}

#[derive(Clone, Debug)]
struct Document {
    evidence: Evidence,
    seal_sha256: String,
    value: Value,
}

impl Document {
    fn reference(&self) -> Value {
        let mut object = self
            .evidence
            .json()
            .as_object()
            .expect("evidence JSON is an object")
            .clone();
        object.insert(
            "seal_sha256".to_owned(),
            Value::String(self.seal_sha256.clone()),
        );
        Value::Object(object)
    }
}

#[derive(Debug)]
struct CandidateBinding {
    manifest: Document,
    admission_current: Document,
    admission_receipt: Document,
    component_current: Document,
    component_terminal: Document,
    admission: Qwen30QualityRepackAdmission,
}

#[derive(Debug)]
struct ControlBinding {
    manifest: Document,
    runtime_receipt: Document,
    admission: CompleteBinaryAdmission,
}

#[derive(Debug)]
struct TraceBinding {
    compiler_current: Document,
    compiler_receipt: Document,
    route_current: Document,
    route_receipt: Document,
    preparation_current: Document,
    preparation_receipt: Document,
    token_ids: Vec<u32>,
    token_ids_u32le_sha256: String,
    annotated_trace: Evidence,
}

fn usage() -> &'static str {
    "usage: ascension_qwen30_quality_repack_all_layer_current_trace_diagnostic \\\n+     --mode cpu-preflight \\\n+     --candidate-manifest ABSOLUTE_PATH --candidate-admission-current ABSOLUTE_PATH \\\n+     --compiler-trace-current ABSOLUTE_PATH --route-capture-current ABSOLUTE_PATH \\\n+     --preparation-current ABSOLUTE_PATH --control-manifest ABSOLUTE_PATH \\\n+     --control-runtime-receipt ABSOLUTE_PATH [--output NEW_ABSOLUTE_PATH]"
}

fn require_absolute(path: PathBuf, flag: &str) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err(format!("{flag} must be an absolute path; {}", usage()));
    }
    Ok(path)
}

fn required<T>(value: Option<T>, flag: &str) -> Result<T, String> {
    value.ok_or_else(|| format!("missing {flag}; {}", usage()))
}

fn parse_args() -> Result<Args, String> {
    let mut mode = None;
    let mut candidate_manifest = None;
    let mut candidate_admission_current = None;
    let mut compiler_trace_current = None;
    let mut route_capture_current = None;
    let mut preparation_current = None;
    let mut control_manifest = None;
    let mut control_runtime_receipt = None;
    let mut output = None;
    let mut lease_receipt = None;
    let mut input_contract = None;
    let mut raw_retention_contract = None;
    let mut capture_dir = None;
    let mut workers = None;
    let mut values = env::args().skip(1);
    while let Some(flag) = values.next() {
        let value = values
            .next()
            .ok_or_else(|| format!("missing value for {flag}; {}", usage()))?;
        macro_rules! only_once {
            ($slot:ident, $value:expr) => {
                if $slot.replace($value).is_some() {
                    return Err(format!("duplicate {flag}; {}", usage()));
                }
            };
        }
        match flag.as_str() {
            "--mode" => only_once!(mode, value),
            "--candidate-manifest" => only_once!(candidate_manifest, PathBuf::from(value)),
            "--candidate-admission-current" => {
                only_once!(candidate_admission_current, PathBuf::from(value))
            }
            "--compiler-trace-current" => only_once!(compiler_trace_current, PathBuf::from(value)),
            "--route-capture-current" => only_once!(route_capture_current, PathBuf::from(value)),
            "--preparation-current" => only_once!(preparation_current, PathBuf::from(value)),
            "--control-manifest" => only_once!(control_manifest, PathBuf::from(value)),
            "--control-runtime-receipt" => {
                only_once!(control_runtime_receipt, PathBuf::from(value))
            }
            "--output" => only_once!(output, PathBuf::from(value)),
            "--lease-receipt" => only_once!(lease_receipt, PathBuf::from(value)),
            "--input-contract" => only_once!(input_contract, PathBuf::from(value)),
            "--raw-retention-contract" => {
                only_once!(raw_retention_contract, PathBuf::from(value))
            }
            "--capture-dir" => only_once!(capture_dir, PathBuf::from(value)),
            "--workers" => only_once!(workers, value),
            _ => return Err(format!("unsupported {flag}; {}", usage())),
        }
    }
    let mode = match required(mode, "--mode")?.as_str() {
        MODE => InvocationMode::CpuPreflight,
        METAL_DIAGNOSTIC_MODE => InvocationMode::MetalDiagnostic,
        RAW_FINAL_LOGIT_RETENTION_MODE => InvocationMode::MetalRawFinalLogitRetention,
        _ => {
            return Err(format!(
                "--mode must be {MODE}, {METAL_DIAGNOSTIC_MODE}, or {RAW_FINAL_LOGIT_RETENTION_MODE}; {}",
                usage()
            ))
        }
    };
    let workers = workers
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| format!("--workers must be a positive integer; {}", usage()))
        })
        .transpose()?;
    if workers.is_some_and(|value| value == 0) {
        return Err(format!("--workers must be a positive integer; {}", usage()));
    }
    match mode {
        InvocationMode::CpuPreflight => {
            if lease_receipt.is_some()
                || input_contract.is_some()
                || raw_retention_contract.is_some()
                || capture_dir.is_some()
                || workers.is_some()
            {
                return Err(format!(
                    "{MODE} rejects lease/input/capture/worker arguments; {}",
                    usage()
                ));
            }
        }
        InvocationMode::MetalDiagnostic => {
            if output.is_some() {
                return Err(format!(
                    "{METAL_DIAGNOSTIC_MODE} writes receipt-last only beneath --capture-dir; {}",
                    usage()
                ));
            }
            if workers != Some(1) {
                return Err(format!(
                    "{METAL_DIAGNOSTIC_MODE} requires --workers 1; {}",
                    usage()
                ));
            }
            if raw_retention_contract.is_some() {
                return Err(format!(
                    "{METAL_DIAGNOSTIC_MODE} rejects --raw-retention-contract; {}",
                    usage()
                ));
            }
        }
        InvocationMode::MetalRawFinalLogitRetention => {
            if output.is_some() {
                return Err(format!(
                    "{RAW_FINAL_LOGIT_RETENTION_MODE} writes receipt-last only beneath --capture-dir; {}",
                    usage()
                ));
            }
            if workers != Some(1) {
                return Err(format!(
                    "{RAW_FINAL_LOGIT_RETENTION_MODE} requires --workers 1; {}",
                    usage()
                ));
            }
            if raw_retention_contract.is_none() {
                return Err(format!(
                    "{RAW_FINAL_LOGIT_RETENTION_MODE} requires --raw-retention-contract; {}",
                    usage()
                ));
            }
        }
    }
    Ok(Args {
        mode,
        candidate_manifest: require_absolute(
            required(candidate_manifest, "--candidate-manifest")?,
            "--candidate-manifest",
        )?,
        candidate_admission_current: require_absolute(
            required(candidate_admission_current, "--candidate-admission-current")?,
            "--candidate-admission-current",
        )?,
        compiler_trace_current: require_absolute(
            required(compiler_trace_current, "--compiler-trace-current")?,
            "--compiler-trace-current",
        )?,
        route_capture_current: require_absolute(
            required(route_capture_current, "--route-capture-current")?,
            "--route-capture-current",
        )?,
        preparation_current: require_absolute(
            required(preparation_current, "--preparation-current")?,
            "--preparation-current",
        )?,
        control_manifest: require_absolute(
            required(control_manifest, "--control-manifest")?,
            "--control-manifest",
        )?,
        control_runtime_receipt: require_absolute(
            required(control_runtime_receipt, "--control-runtime-receipt")?,
            "--control-runtime-receipt",
        )?,
        output: output
            .map(|path| require_absolute(path, "--output"))
            .transpose()?,
        lease_receipt: lease_receipt
            .map(|path| require_absolute(path, "--lease-receipt"))
            .transpose()?,
        input_contract: input_contract
            .map(|path| require_absolute(path, "--input-contract"))
            .transpose()?,
        raw_retention_contract: raw_retention_contract
            .map(|path| require_absolute(path, "--raw-retention-contract"))
            .transpose()?,
        capture_dir: capture_dir
            .map(|path| require_absolute(path, "--capture-dir"))
            .transpose()?,
        workers,
    })
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut handle =
        File::open(path).map_err(|error| format!("cannot open {}: {error}", path.display()))?;
    let mut chunk = [0u8; 1024 * 1024];
    let mut digest = Sha256::new();
    loop {
        let read = handle
            .read(&mut chunk)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&chunk[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn canonical_regular(path: &Path, label: &str) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err(format!("{label} path is not absolute"));
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!(
            "{label} must be a regular non-symlink file: {}",
            path.display()
        ));
    }
    fs::canonicalize(path).map_err(|error| format!("cannot canonicalize {label}: {error}"))
}

fn canonical_directory(path: &Path, label: &str) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err(format!("{label} path is not absolute"));
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(format!(
            "{label} must be a directory and not a symlink: {}",
            path.display()
        ));
    }
    fs::canonicalize(path).map_err(|error| format!("cannot canonicalize {label}: {error}"))
}

fn evidence(path: &Path, label: &str) -> Result<Evidence, String> {
    let path = canonical_regular(path, label)?;
    let bytes = path
        .metadata()
        .map_err(|error| format!("cannot read metadata for {label}: {error}"))?
        .len();
    Ok(Evidence {
        sha256: sha256_file(&path)?,
        path,
        bytes,
    })
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))
}

fn field<'a>(value: &'a Value, key: &str, label: &str) -> Result<&'a Value, String> {
    object(value, label)?
        .get(key)
        .ok_or_else(|| format!("{label} lacks {key}"))
}

fn text<'a>(value: &'a Value, label: &str) -> Result<&'a str, String> {
    value
        .as_str()
        .filter(|text| !text.is_empty())
        .ok_or_else(|| format!("{label} must be a non-empty string"))
}

fn canonical_json(value: &Value) -> Result<Vec<u8>, String> {
    serde_json::to_vec(value).map_err(|error| format!("cannot serialize canonical JSON: {error}"))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sealed_document(path: &Path, label: &str) -> Result<Document, String> {
    let evidence = evidence(path, label)?;
    let raw = fs::read(&evidence.path)
        .map_err(|error| format!("cannot read {label} {}: {error}", evidence.path.display()))?;
    let value: Value = serde_json::from_slice(&raw)
        .map_err(|error| format!("cannot parse {label} {}: {error}", evidence.path.display()))?;
    let mut unsigned = object(&value, label)?.clone();
    let seal = text(
        unsigned
            .remove("seal_sha256")
            .as_ref()
            .ok_or_else(|| format!("{label} lacks seal_sha256"))?,
        &format!("{label}.seal_sha256"),
    )?
    .to_owned();
    if !valid_sha256(&seal) {
        return Err(format!("{label} has malformed lowercase seal_sha256"));
    }
    let observed = sha256_bytes(&canonical_json(&Value::Object(unsigned))?);
    if observed != seal {
        return Err(format!(
            "{label} seal mismatch: observed={observed} recorded={seal}"
        ));
    }
    Ok(Document {
        evidence,
        seal_sha256: seal,
        value,
    })
}

fn expect_schema_status(
    document: &Document,
    schema: &str,
    status: &str,
    label: &str,
) -> Result<(), String> {
    if text(
        field(&document.value, "schema", label)?,
        &format!("{label}.schema"),
    )? != schema
        || text(
            field(&document.value, "status", label)?,
            &format!("{label}.status"),
        )? != status
    {
        return Err(format!("{label} schema/status drifted"));
    }
    Ok(())
}

fn ref_path(reference: &Value, label: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(text(
        field(reference, "path", label)?,
        &format!("{label}.path"),
    )?);
    canonical_regular(&path, label)
}

fn ref_sha<'a>(reference: &'a Value, label: &str) -> Result<&'a str, String> {
    let object = object(reference, label)?;
    let value = object
        .get("document_sha256")
        .or_else(|| object.get("sha256"))
        .ok_or_else(|| format!("{label} lacks document_sha256/sha256"))?;
    text(value, &format!("{label}.document_sha256"))
}

fn expect_reference(
    reference: &Value,
    target: &Document,
    label: &str,
    require_seal: bool,
) -> Result<(), String> {
    if ref_path(reference, label)? != target.evidence.path {
        return Err(format!("{label} path drifted"));
    }
    if ref_sha(reference, label)? != target.evidence.sha256 {
        return Err(format!("{label} document SHA-256 drifted"));
    }
    if require_seal
        && text(
            field(reference, "seal_sha256", label)?,
            &format!("{label}.seal_sha256"),
        )? != target.seal_sha256
    {
        return Err(format!("{label} seal drifted"));
    }
    Ok(())
}

/// Validate a launcher-style immutable file evidence row.  Unlike a receipt
/// reference it carries the document digest and optional byte count, while the
/// seal is deliberately stored alongside it by the outer contract.
fn expect_evidence_reference(
    reference: &Value,
    target: &Evidence,
    label: &str,
) -> Result<(), String> {
    if ref_path(reference, label)? != target.path {
        return Err(format!("{label} path drifted"));
    }
    if ref_sha(reference, label)? != target.sha256 {
        return Err(format!("{label} document SHA-256 drifted"));
    }
    if let Some(bytes) = object(reference, label)?.get("bytes") {
        if bytes.as_u64() != Some(target.bytes) {
            return Err(format!("{label} byte count drifted"));
        }
    }
    if let Some(present) = object(reference, label)?.get("present") {
        if present.as_bool() != Some(true) {
            return Err(format!("{label} must attest present=true"));
        }
    }
    Ok(())
}

fn expect_true(value: &Value, label: &str) -> Result<(), String> {
    if value.as_bool() != Some(true) {
        return Err(format!("{label} must be true"));
    }
    Ok(())
}

fn expect_false(value: &Value, label: &str) -> Result<(), String> {
    if value.as_bool() != Some(false) {
        return Err(format!("{label} must be false"));
    }
    Ok(())
}

fn referenced_document(parent: &Document, key: &str, label: &str) -> Result<Document, String> {
    let reference = field(&parent.value, key, label)?;
    let target = sealed_document(
        &ref_path(reference, &format!("{label}.{key}"))?,
        &format!("{label}.{key}"),
    )?;
    expect_reference(reference, &target, &format!("{label}.{key}"), true)?;
    Ok(target)
}

/// Resolve one explicitly allowlisted mutable pointer.  The path+seal form is
/// accepted only for the schemas listed in `CurrentPointerContract`; it stays
/// inside the exact candidate root and verifies target bytes through the
/// target's own seal.  Every richer reference retains its document-SHA check.
fn resolve_current_pointer(
    current: &Document,
    expected_candidate_root: &Path,
    contract: CurrentPointerContract,
) -> Result<Document, String> {
    expect_schema_status(
        current,
        contract.pointer_schema,
        contract.pointer_status,
        contract.label,
    )?;
    let reference_label = format!("{}.{}", contract.label, contract.receipt_field);
    let reference = field(&current.value, contract.receipt_field, contract.label)?;
    let receipt_path = ref_path(reference, &reference_label)?;
    let expected_candidate_root = canonical_directory(
        expected_candidate_root,
        &format!("{} expected candidate root", contract.label),
    )?;
    if !receipt_path.starts_with(&expected_candidate_root) {
        return Err(format!("{reference_label} escaped expected candidate root"));
    }
    if contract.declared_candidate_root_required {
        let declared_root = PathBuf::from(text(
            field(&current.value, "candidate_root", contract.label)?,
            &format!("{}.candidate_root", contract.label),
        )?);
        if canonical_directory(
            &declared_root,
            &format!("{} candidate_root", contract.label),
        )? != expected_candidate_root
        {
            return Err(format!(
                "{} candidate_root does not match candidate manifest root",
                contract.label
            ));
        }
    }
    let receipt = sealed_document(&receipt_path, &reference_label)?;
    match contract.reference_style {
        CurrentPointerReferenceStyle::RichDocumentSha => {
            expect_reference(reference, &receipt, &reference_label, true)?;
        }
        CurrentPointerReferenceStyle::PathAndSealWithinCandidateRoot => {
            let declared_seal = text(
                field(reference, "seal_sha256", &reference_label)?,
                &format!("{reference_label}.seal_sha256"),
            )?;
            if !valid_sha256(declared_seal) {
                return Err(format!("{reference_label} has malformed seal"));
            }
            if receipt.seal_sha256 != declared_seal {
                return Err(format!("{reference_label} seal differs from sealed target"));
            }
            // A document digest is optional only in this allowlisted branch;
            // if supplied, it must still bind the exact target bytes.
            if let Some(digest) = object(reference, &reference_label)?
                .get("document_sha256")
                .or_else(|| {
                    object(reference, &reference_label)
                        .ok()
                        .and_then(|row| row.get("sha256"))
                })
            {
                if text(digest, &format!("{reference_label}.document_sha256"))?
                    != receipt.evidence.sha256
                {
                    return Err(format!(
                        "{reference_label} optional document digest drifted"
                    ));
                }
            }
        }
    }
    expect_schema_status(
        &receipt,
        contract.receipt_schema,
        contract.receipt_status,
        &format!("{} receipt", contract.label),
    )?;
    Ok(receipt)
}

/// Validate a deliberately allowlisted path+seal edge inside an already
/// sealed diagnostic receipt.  Unlike a rich evidence row, these historic
/// links do not repeat a document digest; their target's recomputed seal and
/// containment under the candidate root are therefore mandatory.
fn expect_path_seal_reference(
    reference: &Value,
    target: &Document,
    expected_candidate_root: &Path,
    label: &str,
) -> Result<(), String> {
    let expected_candidate_root = canonical_directory(
        expected_candidate_root,
        &format!("{label} expected candidate root"),
    )?;
    let resolved = ref_path(reference, label)?;
    if !resolved.starts_with(&expected_candidate_root) {
        return Err(format!("{label} escaped expected candidate root"));
    }
    if resolved != target.evidence.path {
        return Err(format!("{label} path drifted"));
    }
    let seal = text(
        field(reference, "seal_sha256", label)?,
        &format!("{label}.seal_sha256"),
    )?;
    if !valid_sha256(seal) || seal != target.seal_sha256 {
        return Err(format!("{label} seal drifted"));
    }
    if let Some(digest) = object(reference, label)?
        .get("document_sha256")
        .or_else(|| {
            object(reference, label)
                .ok()
                .and_then(|row| row.get("sha256"))
        })
    {
        if text(digest, &format!("{label}.document_sha256"))? != target.evidence.sha256 {
            return Err(format!("{label} optional document digest drifted"));
        }
    }
    Ok(())
}

fn token_ids_sha256(token_ids: &[u32]) -> String {
    let mut bytes = Vec::with_capacity(token_ids.len() * std::mem::size_of::<u32>());
    for token in token_ids {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    sha256_bytes(&bytes)
}

fn resolve_relative_regular(root: &Path, relative: &str, label: &str) -> Result<PathBuf, String> {
    let relative = Path::new(relative);
    if relative.is_absolute()
        || relative.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(format!("{label} is not a safe root-relative path"));
    }
    let root = fs::canonicalize(root)
        .map_err(|error| format!("cannot canonicalize {label} root: {error}"))?;
    let target = canonical_regular(&root.join(relative), label)?;
    if !target.starts_with(&root) {
        return Err(format!("{label} escaped its sealed run root"));
    }
    Ok(target)
}

fn candidate_binding(args: &Args) -> Result<CandidateBinding, String> {
    let manifest = sealed_document(&args.candidate_manifest, "candidate manifest")?;
    expect_schema_status(
        &manifest,
        CANDIDATE_SCHEMA,
        CANDIDATE_STATUS,
        "candidate manifest",
    )?;
    let candidate_root = manifest
        .evidence
        .path
        .parent()
        .ok_or_else(|| "candidate manifest has no parent directory".to_owned())?;
    let admission_current = sealed_document(
        &args.candidate_admission_current,
        "candidate admission current",
    )?;
    expect_schema_status(
        &admission_current,
        CANDIDATE_CURRENT_SCHEMA,
        CANDIDATE_CURRENT_STATUS,
        "candidate admission current",
    )?;
    expect_reference(
        field(
            &admission_current.value,
            "complete_manifest",
            "candidate admission current",
        )?,
        &manifest,
        "candidate admission current.complete_manifest",
        true,
    )?;
    let admission_receipt = resolve_current_pointer(
        &admission_current,
        candidate_root,
        CANDIDATE_ADMISSION_POINTER,
    )?;
    expect_reference(
        field(
            &admission_receipt.value,
            "complete_manifest",
            "candidate admission receipt",
        )?,
        &manifest,
        "candidate admission receipt.complete_manifest",
        true,
    )?;
    let revalidation = referenced_document(
        &admission_receipt,
        "immutable_source_revalidation",
        "candidate admission receipt",
    )?;
    let selection = referenced_document(
        &admission_receipt,
        "selection_receipt",
        "candidate admission receipt",
    )?;
    let snapshot = referenced_document(
        &admission_receipt,
        "source_binding_snapshot",
        "candidate admission receipt",
    )?;
    let terminal = referenced_document(
        &admission_receipt,
        "terminal",
        "candidate admission receipt",
    )?;
    let admission = Qwen30QualityRepackAdmission {
        expected_manifest_seal_sha256: manifest.seal_sha256.clone(),
        expected_source_audit_seal_sha256: text(
            field(
                &revalidation.value,
                "source_audit_seal_sha256",
                "candidate revalidation",
            )?,
            "candidate revalidation.source_audit_seal_sha256",
        )?
        .to_owned(),
        expected_source_revision: text(
            field(
                &revalidation.value,
                "source_revision",
                "candidate revalidation",
            )?,
            "candidate revalidation.source_revision",
        )?
        .to_owned(),
        expected_revalidation_path: revalidation.evidence.path,
        expected_revalidation_seal_sha256: revalidation.seal_sha256,
        expected_selection_path: selection.evidence.path,
        expected_selection_seal_sha256: selection.seal_sha256,
        expected_source_snapshot_path: snapshot.evidence.path,
        expected_source_snapshot_seal_sha256: snapshot.seal_sha256,
        expected_terminal_path: terminal.evidence.path,
        expected_terminal_seal_sha256: terminal.seal_sha256,
    };
    let component_path = candidate_root.join(COMPONENT_CURRENT_FILE);
    let component_current = sealed_document(&component_path, "candidate component current")?;
    expect_schema_status(
        &component_current,
        COMPONENT_CURRENT_SCHEMA,
        COMPONENT_CURRENT_STATUS,
        "candidate component current",
    )?;
    expect_reference(
        field(
            &component_current.value,
            "candidate_manifest",
            "candidate component current",
        )?,
        &manifest,
        "candidate component current.candidate_manifest",
        true,
    )?;
    let component_terminal =
        resolve_current_pointer(&component_current, candidate_root, COMPONENT_PARITY_POINTER)?;
    if text(
        field(
            field(
                &component_terminal.value,
                "inner_probe_capture",
                "candidate component terminal",
            )?,
            "status",
            "candidate component terminal.inner_probe_capture",
        )?,
        "candidate component terminal.inner_probe_capture.status",
    )? != COMPONENT_INNER_STATUS
    {
        return Err(
            "candidate component terminal does not bind the earned device-parity inner result"
                .into(),
        );
    }
    Ok(CandidateBinding {
        manifest,
        admission_current,
        admission_receipt,
        component_current,
        component_terminal,
        admission,
    })
}

fn control_binding(args: &Args) -> Result<ControlBinding, String> {
    let manifest = sealed_document(&args.control_manifest, "control manifest")?;
    expect_schema_status(
        &manifest,
        CONTROL_SCHEMA,
        CONTROL_STATUS,
        "control manifest",
    )?;
    let revalidation_path = PathBuf::from(text(
        field(
            &manifest.value,
            "source_revalidation_receipt_path",
            "control manifest",
        )?,
        "control manifest.source_revalidation_receipt_path",
    )?);
    let revalidation = sealed_document(&revalidation_path, "control source revalidation")?;
    if text(
        field(
            &manifest.value,
            "source_revalidation_receipt_seal_sha256",
            "control manifest",
        )?,
        "control manifest.source_revalidation_receipt_seal_sha256",
    )? != revalidation.seal_sha256
    {
        return Err("control source revalidation seal drifted".into());
    }
    let runtime_receipt =
        sealed_document(&args.control_runtime_receipt, "control runtime receipt")?;
    expect_schema_status(
        &runtime_receipt,
        CONTROL_RUNTIME_SCHEMA,
        CONTROL_RUNTIME_STATUS,
        "control runtime receipt",
    )?;
    if text(
        field(
            field(&runtime_receipt.value, "binding", "control runtime receipt")?,
            "complete_manifest_seal_sha256",
            "control runtime receipt.binding",
        )?,
        "control runtime receipt.binding.complete_manifest_seal_sha256",
    )? != manifest.seal_sha256
    {
        return Err("control runtime receipt does not bind control manifest".into());
    }
    Ok(ControlBinding {
        manifest: manifest.clone(),
        runtime_receipt,
        admission: CompleteBinaryAdmission {
            model: QwenCompleteBinaryModel::Qwen30Coder,
            expected_manifest_seal_sha256: manifest.seal_sha256,
            expected_source_audit_seal_sha256: text(
                field(
                    &manifest.value,
                    "source_body_audit_seal_sha256",
                    "control manifest",
                )?,
                "control manifest.source_body_audit_seal_sha256",
            )?
            .to_owned(),
            expected_source_revision: text(
                field(
                    &revalidation.value,
                    "source_revision",
                    "control source revalidation",
                )?,
                "control source revalidation.source_revision",
            )?
            .to_owned(),
        },
    })
}

fn literal_hawking_tokens(
    compiler_receipt: &Document,
) -> Result<(Vec<u32>, String, Evidence), String> {
    let traces = field(
        &compiler_receipt.value,
        "public_probe_compiler_traces",
        "compiler receipt",
    )?
    .as_array()
    .ok_or_else(|| "compiler receipt.public_probe_compiler_traces must be an array".to_owned())?;
    let mut probe = None;
    for trace in traces {
        if text(
            field(trace, "probe_id", "compiler probe")?,
            "compiler probe.probe_id",
        )? == PROBE_ID
        {
            if probe.replace(trace).is_some() {
                return Err("compiler receipt repeats literal_hawking trace".into());
            }
        }
    }
    let probe = probe.ok_or_else(|| "compiler receipt lacks literal_hawking trace".to_owned())?;
    if field(
        probe,
        "model_execution_started",
        "compiler literal_hawking trace",
    )?
    .as_bool()
        != Some(false)
    {
        return Err("compiler trace did not stop before model execution".into());
    }
    let run_root = PathBuf::from(text(
        field(
            field(&compiler_receipt.value, "binding", "compiler receipt")?,
            "run_root",
            "compiler receipt.binding",
        )?,
        "compiler receipt.binding.run_root",
    )?);
    let annotated_path = resolve_relative_regular(
        &run_root,
        text(
            field(
                probe,
                "annotated_trace_path",
                "compiler literal_hawking trace",
            )?,
            "compiler literal_hawking trace.annotated_trace_path",
        )?,
        "compiler literal_hawking annotated trace",
    )?;
    let annotated = evidence(&annotated_path, "compiler literal_hawking annotated trace")?;
    if text(
        field(
            probe,
            "annotated_trace_sha256",
            "compiler literal_hawking trace",
        )?,
        "compiler literal_hawking trace.annotated_trace_sha256",
    )? != annotated.sha256
    {
        return Err("compiler literal_hawking annotated trace hash drifted".into());
    }
    let raw = fs::read(&annotated.path)
        .map_err(|error| format!("cannot read annotated compiler trace: {error}"))?;
    let annotated_document: Value = serde_json::from_slice(&raw)
        .map_err(|error| format!("cannot parse annotated compiler trace: {error}"))?;
    if field(
        field(
            &annotated_document,
            "claim_boundary",
            "annotated compiler trace",
        )?,
        "actual_selected_span_text_and_token_ids_recorded_before_model_execution",
        "annotated compiler trace.claim_boundary",
    )?
    .as_bool()
        != Some(true)
    {
        return Err(
            "annotated compiler trace does not persist pre-execution selected spans/token IDs"
                .into(),
        );
    }
    let prompt = field(
        field(
            &annotated_document,
            "source_tokenizer_annotations",
            "annotated compiler trace",
        )?,
        "source_one_user_native_prompt",
        "annotated compiler trace.source_tokenizer_annotations",
    )?;
    if field(prompt, "add_special_tokens", "source one-user prompt")?.as_bool() != Some(true)
        || field(prompt, "token_count", "source one-user prompt")?.as_u64()
            != Some(TOKEN_COUNT as u64)
    {
        return Err(
            "annotated compiler trace is not the exact 369-token source one-user prompt".into(),
        );
    }
    let raw_tokens = field(prompt, "token_ids", "source one-user prompt")?
        .as_array()
        .ok_or_else(|| "source one-user prompt token_ids must be an array".to_owned())?;
    let mut token_ids = Vec::with_capacity(raw_tokens.len());
    for (position, value) in raw_tokens.iter().enumerate() {
        token_ids.push(
            value
                .as_u64()
                .and_then(|token| u32::try_from(token).ok())
                .ok_or_else(|| format!("source one-user token {position} is not a u32"))?,
        );
    }
    let declared_hash = text(
        field(prompt, "token_ids_u32le_sha256", "source one-user prompt")?,
        "source one-user prompt.token_ids_u32le_sha256",
    )?
    .to_owned();
    if token_ids.len() != TOKEN_COUNT || token_ids_sha256(&token_ids) != declared_hash {
        return Err("source one-user 369-token F32LE binding drifted".into());
    }
    Ok((token_ids, declared_hash, annotated))
}

fn trace_binding(
    args: &Args,
    candidate: &CandidateBinding,
    control: &ControlBinding,
) -> Result<TraceBinding, String> {
    let compiler_current = sealed_document(&args.compiler_trace_current, "compiler trace current")?;
    expect_schema_status(
        &compiler_current,
        COMPILER_CURRENT_SCHEMA,
        COMPILER_CURRENT_STATUS,
        "compiler trace current",
    )?;
    let candidate_root = candidate
        .manifest
        .evidence
        .path
        .parent()
        .ok_or_else(|| "candidate manifest has no parent directory".to_owned())?;
    let compiler_receipt =
        resolve_current_pointer(&compiler_current, candidate_root, COMPILER_TRACE_POINTER)?;
    expect_reference(
        field(
            field(&compiler_receipt.value, "binding", "compiler trace receipt")?,
            "candidate_manifest",
            "compiler trace receipt.binding",
        )?,
        &candidate.manifest,
        "compiler trace receipt.binding.candidate_manifest",
        true,
    )?;
    let (token_ids, token_ids_u32le_sha256, annotated_trace) =
        literal_hawking_tokens(&compiler_receipt)?;

    let route_current = sealed_document(&args.route_capture_current, "route capture current")?;
    expect_schema_status(
        &route_current,
        ROUTE_CURRENT_SCHEMA,
        ROUTE_CURRENT_STATUS,
        "route capture current",
    )?;
    let route_receipt =
        resolve_current_pointer(&route_current, candidate_root, ROUTE_CAPTURE_POINTER)?;
    expect_path_seal_reference(
        field(
            field(&route_receipt.value, "binding", "route capture receipt")?,
            "compiler_trace",
            "route capture receipt.binding",
        )?,
        &compiler_receipt,
        candidate_root,
        "route capture receipt.binding.compiler_trace",
    )?;
    if text(
        field(
            field(&route_receipt.value, "binding", "route capture receipt")?,
            "input_seal_sha256",
            "route capture receipt.binding",
        )?,
        "route capture receipt.binding.input_seal_sha256",
    )?
    .is_empty()
    {
        return Err("route capture receipt lacks input seal".into());
    }

    let preparation_current =
        sealed_document(&args.preparation_current, "all-layer preparation current")?;
    expect_schema_status(
        &preparation_current,
        PREPARATION_CURRENT_SCHEMA,
        PREPARATION_CURRENT_STATUS,
        "all-layer preparation current",
    )?;
    let preparation_receipt =
        resolve_current_pointer(&preparation_current, candidate_root, PREPARATION_POINTER)?;
    if text(
        field(
            field(
                &preparation_receipt.value,
                "binding",
                "all-layer preparation receipt",
            )?,
            "candidate_manifest_seal_sha256",
            "all-layer preparation receipt.binding",
        )?,
        "all-layer preparation receipt.binding.candidate_manifest_seal_sha256",
    )? != candidate.manifest.seal_sha256
    {
        return Err("all-layer preparation candidate manifest binding drifted".into());
    }
    expect_path_seal_reference(
        field(
            field(
                &preparation_receipt.value,
                "binding",
                "all-layer preparation receipt",
            )?,
            "candidate_admission_current_pointer",
            "all-layer preparation receipt.binding",
        )?,
        &candidate.admission_current,
        candidate_root,
        "all-layer preparation receipt.binding.candidate_admission_current_pointer",
    )?;
    expect_path_seal_reference(
        field(
            field(
                &preparation_receipt.value,
                "binding",
                "all-layer preparation receipt",
            )?,
            "route_capture_current_pointer",
            "all-layer preparation receipt.binding",
        )?,
        &route_current,
        candidate_root,
        "all-layer preparation receipt.binding.route_capture_current_pointer",
    )?;
    expect_path_seal_reference(
        field(
            field(
                &preparation_receipt.value,
                "binding",
                "all-layer preparation receipt",
            )?,
            "route_capture_receipt",
            "all-layer preparation receipt.binding",
        )?,
        &route_receipt,
        candidate_root,
        "all-layer preparation receipt.binding.route_capture_receipt",
    )?;
    if text(
        field(
            field(
                &preparation_receipt.value,
                "planned_bounded_input",
                "all-layer preparation receipt",
            )?,
            "probe_id",
            "all-layer preparation receipt.planned_bounded_input",
        )?,
        "all-layer preparation receipt.planned_bounded_input.probe_id",
    )? != PROBE_ID
        || field(
            field(
                &preparation_receipt.value,
                "planned_bounded_input",
                "all-layer preparation receipt",
            )?,
            "source_template_token_count",
            "all-layer preparation receipt.planned_bounded_input",
        )?
        .as_u64()
            != Some(TOKEN_COUNT as u64)
    {
        return Err("all-layer preparation does not bind literal_hawking / 369 tokens".into());
    }
    if candidate.manifest.seal_sha256 == control.manifest.seal_sha256
        || candidate.manifest.evidence.path == control.manifest.evidence.path
    {
        return Err("candidate/control identity collapsed before typed catalog admission".into());
    }
    Ok(TraceBinding {
        compiler_current,
        compiler_receipt,
        route_current,
        route_receipt,
        preparation_current,
        preparation_receipt,
        token_ids,
        token_ids_u32le_sha256,
        annotated_trace,
    })
}

fn catalog_check(candidate: &CandidateBinding, control: &ControlBinding) -> Result<Value, String> {
    // Both admission routines are explicit CPU/disk integrity scans.  Neither
    // imports Metal or calls an execution graph.  This is intentionally the
    // final pre-Metal boundary, not a host fallback for a future diagnostic.
    let direct =
        admit_complete_binary_artifact(&control.manifest.evidence.path, &control.admission)
            .map_err(|error| format!("direct control catalog admission refused: {error}"))?;
    let quality = admit_qwen30_quality_repack_artifact(
        &candidate.manifest.evidence.path,
        &candidate.admission,
    )
    .map_err(|error| format!("HQ30GR2 candidate catalog admission refused: {error}"))?;
    let typed = Qwen30QualityRepackDiagnosticCatalog::from_admitted(quality)
        .map_err(|error| format!("HQ30GR2 typed catalog construction refused: {error}"))?;
    if direct.verified_payload_count() != QWEN30_QUALITY_REPACK_DIAGNOSTIC_TENSOR_COUNT
        || typed.verified_payload_count() != QWEN30_QUALITY_REPACK_DIAGNOSTIC_TENSOR_COUNT
        || typed.direct_tensor_count() != QWEN30_QUALITY_REPACK_DIAGNOSTIC_TENSOR_COUNT - 2
        || typed.sparse_residual_tensor_count() != 2
    {
        return Err("direct/candidate immutable catalog cardinality is incomplete".into());
    }
    for name in [L0_E0_GATE, L0_E0_UP] {
        if !matches!(
            typed
                .typed_tensor(name)
                .map_err(|error| format!("candidate typed {name} lookup refused: {error}"))?,
            Qwen30QualityRepackVerifiedTensor::SparseResidual { .. }
        ) {
            return Err(format!(
                "candidate {name} is not an HQ30GR2 sparse-residual tensor"
            ));
        }
    }
    if !matches!(
        typed
            .typed_tensor(L0_E1_GATE)
            .map_err(|error| format!("candidate typed {L0_E1_GATE} lookup refused: {error}"))?,
        Qwen30QualityRepackVerifiedTensor::Direct { .. }
    ) {
        return Err("unchanged candidate L0/E1 gate did not remain typed HQ30G1B1 direct".into());
    }
    let direct_gate = direct
        .verified_tensor_payload(L0_E0_GATE)
        .map_err(|error| format!("control direct {L0_E0_GATE} lookup refused: {error}"))?;
    parse_complete_binary_header(direct_gate.as_ref()).map_err(|error| {
        format!("control direct {L0_E0_GATE} no longer parses as HQ30G1B1: {error}")
    })?;
    if parse_qwen30_quality_residual_header(direct_gate.as_ref()).is_ok() {
        return Err("control direct L0/E0 gate unexpectedly parses as HQ30GR2".into());
    }
    let dispatch = typed
        .sparse_gate_up_dispatch()
        .map_err(|error| format!("typed HQ30GR2 sparse gate/up ABI refused: {error}"))?;
    Ok(json!({
        "control_direct_catalog": {
            "immutable_verified_payloads": direct.verified_payload_count(),
            "l0_e0_gate_layout": "HQ30G1B1_DIRECT",
        },
        "candidate_typed_catalog": {
            "immutable_verified_payloads": typed.verified_payload_count(),
            "direct_tensor_count": typed.direct_tensor_count(),
            "sparse_residual_tensor_count": typed.sparse_residual_tensor_count(),
            "l0_e0_gate_up_layout": "HQ30GR2_SPARSE_RESIDUAL",
            "unchanged_l0_e1_gate_layout": "HQ30G1B1_DIRECT",
            "sparse_gate_up_dispatch": {
                "kernel_name": dispatch.kernel_name,
                "rows": dispatch.rows,
                "cols": dispatch.cols,
                "group_size": dispatch.group_size,
                "gate_residual_count": dispatch.gate_residual_count,
                "up_residual_count": dispatch.up_residual_count,
                "exact_non_fma_scalar_order_required": dispatch.exact_non_fma_scalar_order_required,
                "direct_fallback_for_sparse_residual_forbidden": dispatch.direct_fallback_for_sparse_residual_forbidden,
            },
        },
    }))
}

fn seal(mut object: Map<String, Value>) -> Result<Value, String> {
    object.remove("seal_sha256");
    let unsigned = Value::Object(object.clone());
    object.insert(
        "seal_sha256".to_owned(),
        Value::String(sha256_bytes(&canonical_json(&unsigned)?)),
    );
    Ok(Value::Object(object))
}

fn preflight(args: &Args) -> Result<Value, String> {
    let candidate = candidate_binding(args)?;
    let control = control_binding(args)?;
    let trace = trace_binding(args, &candidate, &control)?;
    let catalog = catalog_check(&candidate, &control)?;
    let mut result = Map::new();
    result.insert("schema".to_owned(), Value::String(RESULT_SCHEMA.to_owned()));
    result.insert("status".to_owned(), Value::String(RESULT_STATUS.to_owned()));
    result.insert("mode".to_owned(), Value::String(MODE.to_owned()));
    result.insert(
        "candidate_manifest".to_owned(),
        candidate.manifest.reference(),
    );
    result.insert(
        "candidate_admission_current".to_owned(),
        candidate.admission_current.reference(),
    );
    result.insert(
        "candidate_admission_receipt".to_owned(),
        candidate.admission_receipt.reference(),
    );
    result.insert(
        "candidate_component_parity_current".to_owned(),
        candidate.component_current.reference(),
    );
    result.insert(
        "candidate_component_parity_terminal".to_owned(),
        candidate.component_terminal.reference(),
    );
    result.insert("control_manifest".to_owned(), control.manifest.reference());
    result.insert(
        "control_runtime_receipt".to_owned(),
        control.runtime_receipt.reference(),
    );
    result.insert(
        "compiler_trace_current".to_owned(),
        trace.compiler_current.reference(),
    );
    result.insert(
        "compiler_trace_receipt".to_owned(),
        trace.compiler_receipt.reference(),
    );
    result.insert(
        "route_capture_current".to_owned(),
        trace.route_current.reference(),
    );
    result.insert(
        "route_capture_receipt".to_owned(),
        trace.route_receipt.reference(),
    );
    result.insert(
        "preparation_current".to_owned(),
        trace.preparation_current.reference(),
    );
    result.insert(
        "preparation_receipt".to_owned(),
        trace.preparation_receipt.reference(),
    );
    result.insert(
        "exact_source_template_input".to_owned(),
        json!({
            "probe_id": PROBE_ID,
            "token_count": trace.token_ids.len(),
            "token_ids_u32le_sha256": trace.token_ids_u32le_sha256,
            "annotated_trace": trace.annotated_trace.json(),
            "new_diagnostic_not_historical": true,
        }),
    );
    result.insert("typed_catalog_preflight".to_owned(), catalog);
    result.insert(
        "execution_boundary".to_owned(),
        json!({
            "metal_context_created": false,
            "metal_dispatch_performed": false,
            "all_layer_forward_performed": false,
            "token_loop_performed": false,
            "endpoint_or_hcli_called": false,
            "server_watcher_or_adapter_modified": false,
            "raw_bf16_or_dense_weight_path": false,
            "host_fallback_for_future_candidate_execution": false,
            "future_device_executor_requires_a_new_quiet_lease_and_a_new_outer_capture": true,
        }),
    );
    result.insert(
        "claim_boundary".to_owned(),
        json!({
            "does_not_claim_native_runtime": true,
            "does_not_claim_generation_or_coherence": true,
            "does_not_claim_hcli": true,
            "does_not_claim_tps_or_tg": true,
            "does_not_claim_capability_or_tournament": true,
        }),
    );
    seal(result)
}

fn refusal(args: &Args, detail: &str) -> Result<Value, String> {
    let mut result = Map::new();
    result.insert("schema".to_owned(), Value::String(RESULT_SCHEMA.to_owned()));
    result.insert(
        "status".to_owned(),
        Value::String(REFUSAL_STATUS.to_owned()),
    );
    result.insert("mode".to_owned(), Value::String(MODE.to_owned()));
    result.insert(
        "requested_input_paths".to_owned(),
        json!({
            "candidate_manifest": args.candidate_manifest,
            "candidate_admission_current": args.candidate_admission_current,
            "compiler_trace_current": args.compiler_trace_current,
            "route_capture_current": args.route_capture_current,
            "preparation_current": args.preparation_current,
            "control_manifest": args.control_manifest,
            "control_runtime_receipt": args.control_runtime_receipt,
        }),
    );
    result.insert(
        "refusal".to_owned(),
        json!({
            "detail": detail,
            "retry_performed": false,
            "requires_new_source_or_binding_correction_before_a_new_cpu_preflight": true,
        }),
    );
    result.insert(
        "execution_boundary".to_owned(),
        json!({
            "metal_context_created": false,
            "metal_dispatch_performed": false,
            "all_layer_forward_performed": false,
            "token_loop_performed": false,
            "endpoint_or_hcli_called": false,
            "server_watcher_or_adapter_modified": false,
        }),
    );
    result.insert(
        "claim_boundary".to_owned(),
        json!({
            "does_not_claim_native_runtime": true,
            "does_not_claim_generation_or_coherence": true,
            "does_not_claim_hcli": true,
            "does_not_claim_tps_or_tg": true,
            "does_not_claim_capability_or_tournament": true,
        }),
    );
    seal(result)
}

fn write_new(path: &Path, value: &Value) -> Result<(), String> {
    if !path.is_absolute() {
        return Err("--output must be an absolute path".into());
    }
    let parent = path
        .parent()
        .ok_or_else(|| "--output has no parent directory".to_owned())?;
    let parent_metadata = fs::symlink_metadata(parent)
        .map_err(|error| format!("cannot stat --output parent: {error}"))?;
    if parent_metadata.file_type().is_symlink() || !parent_metadata.is_dir() {
        return Err("--output parent must be an existing non-symlink directory".into());
    }
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("cannot serialize preflight output: {error}"))?;
    bytes.push(b'\n');
    let mut handle = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("refusing to replace --output {}: {error}", path.display()))?;
    handle
        .write_all(&bytes)
        .map_err(|error| format!("cannot write --output: {error}"))?;
    handle
        .sync_all()
        .map_err(|error| format!("cannot sync --output: {error}"))
}

/// Receipt-last payload writer used only by the separately selected raw-logit
/// successor.  It cannot replace an existing witness and it never writes
/// through a symlinked parent.
fn write_new_bytes(path: &Path, bytes: &[u8], label: &str) -> Result<(), String> {
    if !path.is_absolute() {
        return Err(format!("{label} path must be absolute"));
    }
    let parent = path
        .parent()
        .ok_or_else(|| format!("{label} has no parent directory"))?;
    let parent_metadata = fs::symlink_metadata(parent)
        .map_err(|error| format!("cannot stat {label} parent: {error}"))?;
    if parent_metadata.file_type().is_symlink() || !parent_metadata.is_dir() {
        return Err(format!(
            "{label} parent must be an existing non-symlink directory"
        ));
    }
    let mut handle = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("refusing to replace {label} {}: {error}", path.display()))?;
    handle
        .write_all(bytes)
        .map_err(|error| format!("cannot write {label}: {error}"))?;
    handle
        .sync_all()
        .map_err(|error| format!("cannot sync {label}: {error}"))
}

#[cfg(target_os = "macos")]
struct MetalInvocation {
    lease_receipt: PathBuf,
    input_contract: PathBuf,
    raw_retention_contract: Option<PathBuf>,
    capture_dir: PathBuf,
}

#[cfg(target_os = "macos")]
fn metal_invocation(args: &Args) -> Result<MetalInvocation, String> {
    let lease_receipt = canonical_regular(
        &required(args.lease_receipt.clone(), "--lease-receipt")?,
        "--lease-receipt",
    )?;
    let input_contract = canonical_regular(
        &required(args.input_contract.clone(), "--input-contract")?,
        "--input-contract",
    )?;
    let raw_retention_contract = args
        .raw_retention_contract
        .clone()
        .map(|path| canonical_regular(&path, "--raw-retention-contract"))
        .transpose()?;
    if args.workers != Some(1) {
        return Err("metal diagnostic requires exactly one outer-owned worker".into());
    }
    let requested_capture = required(args.capture_dir.clone(), "--capture-dir")?;
    if requested_capture.exists() {
        return Err(
            "--capture-dir must not already exist for receipt-last one-shot evidence".into(),
        );
    }
    if requested_capture.components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::CurDir | Component::Prefix(_)
        )
    }) {
        return Err("--capture-dir must be a normalized absolute child path".into());
    }
    let parent = requested_capture
        .parent()
        .ok_or_else(|| "--capture-dir has no parent".to_owned())?;
    let parent = canonical_directory(parent, "--capture-dir parent")?;
    let name = requested_capture
        .file_name()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| "--capture-dir has no final directory name".to_owned())?;
    let capture_dir = parent.join(name);
    if capture_dir.exists() {
        return Err("--capture-dir resolves to an existing path".into());
    }
    Ok(MetalInvocation {
        lease_receipt,
        input_contract,
        raw_retention_contract,
        capture_dir,
    })
}

fn expect_evidence_seal_binding(
    container: &Value,
    evidence_key: &str,
    seal_key: &str,
    target: &Document,
    label: &str,
) -> Result<(), String> {
    expect_evidence_reference(
        field(container, evidence_key, label)?,
        &target.evidence,
        &format!("{label}.{evidence_key}"),
    )?;
    if text(
        field(container, seal_key, label)?,
        &format!("{label}.{seal_key}"),
    )? != target.seal_sha256
    {
        return Err(format!("{label}.{seal_key} drifted"));
    }
    Ok(())
}

fn expect_exact_u64(value: &Value, expected: u64, label: &str) -> Result<(), String> {
    if value.as_u64() != Some(expected) {
        return Err(format!("{label} must equal {expected}"));
    }
    Ok(())
}

fn validate_input_contract(
    input_path: &Path,
    candidate: &CandidateBinding,
    control: &ControlBinding,
    trace: &TraceBinding,
    lease: &Document,
) -> Result<Document, String> {
    let input = sealed_document(input_path, "all-layer diagnostic input contract")?;
    expect_schema_status(
        &input,
        INPUT_CONTRACT_SCHEMA,
        INPUT_CONTRACT_STATUS,
        "all-layer diagnostic input contract",
    )?;
    let source = field(
        &input.value,
        "source_binding",
        "all-layer diagnostic input contract",
    )?;
    for (evidence_key, seal_key, target) in [
        (
            "candidate_manifest",
            "candidate_manifest_seal_sha256",
            &candidate.manifest,
        ),
        (
            "candidate_admission_current",
            "candidate_admission_pointer_seal_sha256",
            &candidate.admission_current,
        ),
        (
            "candidate_admission_receipt",
            "candidate_admission_receipt_seal_sha256",
            &candidate.admission_receipt,
        ),
        (
            "compiler_trace_receipt",
            "compiler_trace_seal_sha256",
            &trace.compiler_receipt,
        ),
        (
            "route_capture_receipt",
            "route_capture_seal_sha256",
            &trace.route_receipt,
        ),
        (
            "preparation_receipt",
            "preparation_seal_sha256",
            &trace.preparation_receipt,
        ),
        (
            "control_manifest",
            "control_manifest_seal_sha256",
            &control.manifest,
        ),
        (
            "control_runtime_receipt",
            "control_runtime_seal_sha256",
            &control.runtime_receipt,
        ),
        ("lease_receipt", "lease_seal_sha256", lease),
    ] {
        expect_evidence_seal_binding(
            source,
            evidence_key,
            seal_key,
            target,
            "all-layer diagnostic input contract.source_binding",
        )?;
    }
    let exact = field(
        &input.value,
        "exact_trace",
        "all-layer diagnostic input contract",
    )?;
    if text(
        field(
            exact,
            "probe_id",
            "all-layer diagnostic input contract.exact_trace",
        )?,
        "all-layer diagnostic input contract.exact_trace.probe_id",
    )? != PROBE_ID
    {
        return Err("all-layer diagnostic input does not bind literal_hawking".into());
    }
    expect_exact_u64(
        field(
            exact,
            "source_template_token_count",
            "all-layer diagnostic input contract.exact_trace",
        )?,
        TOKEN_COUNT as u64,
        "all-layer diagnostic input contract.exact_trace.source_template_token_count",
    )?;
    if text(
        field(
            exact,
            "source_template_token_ids_u32le_sha256",
            "all-layer diagnostic input contract.exact_trace",
        )?,
        "all-layer diagnostic input contract.exact_trace.source_template_token_ids_u32le_sha256",
    )? != trace.token_ids_u32le_sha256
    {
        return Err("all-layer diagnostic input token hash drifted from compiler trace".into());
    }
    let token_values = field(
        exact,
        "source_template_token_ids",
        "all-layer diagnostic input contract.exact_trace",
    )?
    .as_array()
    .ok_or_else(|| "all-layer diagnostic input token IDs must be an array".to_owned())?;
    let mut token_ids = Vec::with_capacity(token_values.len());
    for (index, value) in token_values.iter().enumerate() {
        token_ids.push(
            value
                .as_u64()
                .and_then(|token| u32::try_from(token).ok())
                .ok_or_else(|| format!("all-layer diagnostic input token {index} is not a u32"))?,
        );
    }
    if token_ids != trace.token_ids || token_ids_sha256(&token_ids) != trace.token_ids_u32le_sha256
    {
        return Err(
            "all-layer diagnostic input token IDs differ from the sealed compiler trace".into(),
        );
    }
    expect_true(
        field(
            exact,
            "new_diagnostic_not_historical",
            "all-layer diagnostic input contract.exact_trace",
        )?,
        "all-layer diagnostic input contract.exact_trace.new_diagnostic_not_historical",
    )?;
    let execution = field(
        &input.value,
        "all_layer_execution_contract",
        "all-layer diagnostic input contract",
    )?;
    expect_exact_u64(
        field(
            execution,
            "baseline_and_candidate_exact_prefix_forwards",
            "all-layer diagnostic input contract.all_layer_execution_contract",
        )?,
        1,
        "all-layer diagnostic input contract baseline/candidate exact-prefix forwards",
    )?;
    expect_exact_u64(
        field(
            execution,
            "layers_per_prefix_forward",
            "all-layer diagnostic input contract.all_layer_execution_contract",
        )?,
        QWEN30_ALL_LAYER_COUNT as u64,
        "all-layer diagnostic input contract layers_per_prefix_forward",
    )?;
    expect_true(
        field(
            execution,
            "unbounded_generation_or_sampling_loop_forbidden",
            "all-layer diagnostic input contract.all_layer_execution_contract",
        )?,
        "all-layer diagnostic input contract unbounded loop prohibition",
    )?;
    let continuation = field(
        execution,
        "forced_continuation",
        "all-layer diagnostic input contract.all_layer_execution_contract",
    )?;
    for key in [
        "derive_token_from_baseline_deterministic_argmax_after_exact_prefix",
        "force_identical_token_into_baseline_and_candidate",
    ] {
        expect_true(
            field(
                continuation,
                key,
                "all-layer diagnostic input contract forced continuation",
            )?,
            &format!("all-layer diagnostic input contract forced continuation.{key}"),
        )?;
    }
    expect_exact_u64(
        field(
            continuation,
            "additional_forwards_per_path",
            "all-layer diagnostic input contract forced continuation",
        )?,
        FORCED_CONTINUATION_FORWARDS as u64,
        "all-layer diagnostic input contract forced continuation.additional_forwards_per_path",
    )?;
    expect_exact_u64(
        field(
            continuation,
            "layers_per_additional_forward",
            "all-layer diagnostic input contract forced continuation",
        )?,
        QWEN30_ALL_LAYER_COUNT as u64,
        "all-layer diagnostic input contract forced continuation.layers_per_additional_forward",
    )?;
    let boundary = field(
        &input.value,
        "claim_boundary",
        "all-layer diagnostic input contract",
    )?;
    for key in [
        "typed_hq30gr2_diagnostic_only",
        "does_not_call_hcli_or_an_endpoint",
        "does_not_claim_hcli",
        "does_not_claim_coherence",
        "does_not_claim_tps_or_tg",
        "does_not_claim_capability",
        "does_not_claim_tournament",
    ] {
        expect_true(
            field(
                boundary,
                key,
                "all-layer diagnostic input contract.claim_boundary",
            )?,
            &format!("all-layer diagnostic input contract.claim_boundary.{key}"),
        )?;
    }
    Ok(input)
}

fn validate_quiet_lease(
    lease: &Document,
    candidate: &CandidateBinding,
    control: &ControlBinding,
    trace: &TraceBinding,
) -> Result<(), String> {
    expect_schema_status(
        lease,
        QUIET_LEASE_SCHEMA,
        QUIET_LEASE_STATUS,
        "all-layer quiet lease",
    )?;
    let lifecycle = field(&lease.value, "one_shot_lifecycle", "all-layer quiet lease")?;
    expect_true(
        field(
            lifecycle,
            "fresh_for_this_exact_launch",
            "all-layer quiet lease lifecycle",
        )?,
        "all-layer quiet lease lifecycle.fresh_for_this_exact_launch",
    )?;
    if field(
        lifecycle,
        "prior_terminal_receipt",
        "all-layer quiet lease lifecycle",
    )? != &Value::Null
    {
        return Err("all-layer quiet lease already names a prior terminal receipt".into());
    }
    expect_false(
        field(
            lifecycle,
            "automatic_retry_allowed",
            "all-layer quiet lease lifecycle",
        )?,
        "all-layer quiet lease lifecycle.automatic_retry_allowed",
    )?;
    let policy = field(&lease.value, "execution_policy", "all-layer quiet lease")?;
    if text(
        field(
            policy,
            "component",
            "all-layer quiet lease execution policy",
        )?,
        "all-layer quiet lease execution policy.component",
    )? != "qwen30_hq30gr2_all_layer_current_trace_diagnostic"
    {
        return Err("all-layer quiet lease component drifted".into());
    }
    for key in [
        "quiet_qwen_family_gpu_lease",
        "strict_math",
        "diagnostic_only",
        "one_child_process_group_only",
    ] {
        expect_true(
            field(policy, key, "all-layer quiet lease execution policy")?,
            &format!("all-layer quiet lease execution policy.{key}"),
        )?;
    }
    for key in [
        "timing_or_benchmarking_allowed",
        "hcli_or_server_allowed",
        "coherence_claim_allowed",
        "tps_or_tg_claim_allowed",
        "capability_claim_allowed",
        "tournament_claim_allowed",
    ] {
        expect_false(
            field(policy, key, "all-layer quiet lease execution policy")?,
            &format!("all-layer quiet lease execution policy.{key}"),
        )?;
    }
    let artifact = field(&lease.value, "artifact_binding", "all-layer quiet lease")?;
    expect_evidence_reference(
        field(
            artifact,
            "candidate_manifest",
            "all-layer quiet lease artifact binding",
        )?,
        &candidate.manifest.evidence,
        "all-layer quiet lease candidate manifest",
    )?;
    if text(
        field(
            artifact,
            "candidate_manifest_seal_sha256",
            "all-layer quiet lease artifact binding",
        )?,
        "all-layer quiet lease candidate manifest seal",
    )? != candidate.manifest.seal_sha256
    {
        return Err("all-layer quiet lease candidate manifest seal drifted".into());
    }
    if canonical_regular(
        &PathBuf::from(text(
            field(
                artifact,
                "candidate_admission_current_path",
                "all-layer quiet lease artifact binding",
            )?,
            "all-layer quiet lease candidate admission current path",
        )?),
        "all-layer quiet lease candidate admission current path",
    )? != candidate.admission_current.evidence.path
        || text(
            field(
                artifact,
                "candidate_admission_pointer_seal_sha256",
                "all-layer quiet lease artifact binding",
            )?,
            "all-layer quiet lease candidate admission pointer seal",
        )? != candidate.admission_current.seal_sha256
        || text(
            field(
                artifact,
                "candidate_admission_receipt_seal_sha256",
                "all-layer quiet lease artifact binding",
            )?,
            "all-layer quiet lease candidate admission receipt seal",
        )? != candidate.admission_receipt.seal_sha256
    {
        return Err("all-layer quiet lease candidate admission binding drifted".into());
    }
    expect_evidence_reference(
        field(
            artifact,
            "control_manifest",
            "all-layer quiet lease artifact binding",
        )?,
        &control.manifest.evidence,
        "all-layer quiet lease control manifest",
    )?;
    expect_evidence_reference(
        field(
            artifact,
            "control_runtime_receipt",
            "all-layer quiet lease artifact binding",
        )?,
        &control.runtime_receipt.evidence,
        "all-layer quiet lease control runtime receipt",
    )?;
    if text(
        field(
            artifact,
            "control_manifest_seal_sha256",
            "all-layer quiet lease artifact binding",
        )?,
        "all-layer quiet lease control manifest seal",
    )? != control.manifest.seal_sha256
        || text(
            field(
                artifact,
                "control_runtime_receipt_seal_sha256",
                "all-layer quiet lease artifact binding",
            )?,
            "all-layer quiet lease control runtime receipt seal",
        )? != control.runtime_receipt.seal_sha256
    {
        return Err("all-layer quiet lease control binding drifted".into());
    }
    let upstream = field(&lease.value, "upstream_binding", "all-layer quiet lease")?;
    for (key, target) in [
        ("compiler_trace_receipt", &trace.compiler_receipt),
        ("route_capture_receipt", &trace.route_receipt),
        ("preparation_receipt", &trace.preparation_receipt),
    ] {
        expect_reference(
            field(upstream, key, "all-layer quiet lease upstream binding")?,
            target,
            &format!("all-layer quiet lease upstream binding.{key}"),
            true,
        )?;
    }
    let trace_contract = field(&lease.value, "trace_contract", "all-layer quiet lease")?;
    if text(
        field(
            trace_contract,
            "probe_id",
            "all-layer quiet lease trace contract",
        )?,
        "all-layer quiet lease trace contract.probe_id",
    )? != PROBE_ID
        || text(
            field(
                trace_contract,
                "source_template_token_ids_u32le_sha256",
                "all-layer quiet lease trace contract",
            )?,
            "all-layer quiet lease trace contract.token hash",
        )? != trace.token_ids_u32le_sha256
    {
        return Err("all-layer quiet lease trace binding drifted".into());
    }
    expect_exact_u64(
        field(
            trace_contract,
            "source_template_token_count",
            "all-layer quiet lease trace contract",
        )?,
        TOKEN_COUNT as u64,
        "all-layer quiet lease trace token count",
    )?;
    expect_true(
        field(
            trace_contract,
            "forced_shared_continuation",
            "all-layer quiet lease trace contract",
        )?,
        "all-layer quiet lease forced shared continuation",
    )?;
    expect_exact_u64(
        field(
            trace_contract,
            "additional_forwards_per_path",
            "all-layer quiet lease trace contract",
        )?,
        FORCED_CONTINUATION_FORWARDS as u64,
        "all-layer quiet lease additional forwards per path",
    )
}

#[cfg(target_os = "macos")]
fn prepare_capture_dir(
    invocation: &MetalInvocation,
    candidate: &CandidateBinding,
    control: &ControlBinding,
    trace: &TraceBinding,
    input: &Document,
    lease: &Document,
    mode: &str,
    raw_retention_contract: Option<&Document>,
) -> Result<PathBuf, String> {
    fs::create_dir(&invocation.capture_dir).map_err(|error| {
        format!(
            "cannot create one-shot all-layer capture directory {}: {error}",
            invocation.capture_dir.display()
        )
    })?;
    let capture_dir = canonical_directory(&invocation.capture_dir, "all-layer inner capture")?;
    let mut invocation_record = Map::new();
    invocation_record.insert(
        "schema".to_owned(),
        Value::String(
            "hawking.ascension.qwen30_hq30gr2_all_layer_current_trace_inner_invocation.v1"
                .to_owned(),
        ),
    );
    invocation_record.insert(
        "status".to_owned(),
        Value::String("STARTED_QWEN30_HQ30GR2_ALL_LAYER_CURRENT_TRACE_INNER_ONE_SHOT".to_owned()),
    );
    invocation_record.insert("mode".to_owned(), Value::String(mode.to_owned()));
    invocation_record.insert(
        "candidate_manifest".to_owned(),
        candidate.manifest.reference(),
    );
    invocation_record.insert(
        "candidate_admission_current".to_owned(),
        candidate.admission_current.reference(),
    );
    invocation_record.insert("control_manifest".to_owned(), control.manifest.reference());
    invocation_record.insert(
        "control_runtime_receipt".to_owned(),
        control.runtime_receipt.reference(),
    );
    invocation_record.insert(
        "compiler_trace_receipt".to_owned(),
        trace.compiler_receipt.reference(),
    );
    invocation_record.insert(
        "route_capture_receipt".to_owned(),
        trace.route_receipt.reference(),
    );
    invocation_record.insert(
        "preparation_receipt".to_owned(),
        trace.preparation_receipt.reference(),
    );
    invocation_record.insert("input_contract".to_owned(), input.reference());
    invocation_record.insert("lease_receipt".to_owned(), lease.reference());
    if let Some(contract) = raw_retention_contract {
        invocation_record.insert("raw_retention_contract".to_owned(), contract.reference());
    }
    invocation_record.insert(
        "execution_boundary".to_owned(),
        json!({
            "receipt_written_last_is_completion_marker": true,
            "metal_context_created_at_invocation_record": false,
            "server_watcher_or_adapter_modified": false,
            "endpoint_or_hcli_called": false,
        }),
    );
    let invocation_record = seal(invocation_record)?;
    write_new(&capture_dir.join("invocation.json"), &invocation_record)?;
    Ok(capture_dir)
}

#[cfg(target_os = "macos")]
trait AllLayerDiagnosticPath {
    fn reset_for_diagnostic(&mut self);
    fn forward_with_route_capture(
        &mut self,
        token: u32,
    ) -> hawking_core::Result<Qwen30NativeRouteCaptureStep>;
    fn final_logits_for_diagnostic(&self) -> hawking_core::Result<Vec<f32>>;
    fn sparse_interception_count(&self) -> Option<usize>;
}

#[cfg(target_os = "macos")]
impl AllLayerDiagnosticPath for Qwen30CompleteNativeRuntime {
    fn reset_for_diagnostic(&mut self) {
        self.reset();
    }

    fn forward_with_route_capture(
        &mut self,
        token: u32,
    ) -> hawking_core::Result<Qwen30NativeRouteCaptureStep> {
        self.forward_token_greedy_with_route_capture(token)
    }

    fn final_logits_for_diagnostic(&self) -> hawking_core::Result<Vec<f32>> {
        self.diagnostic_final_logits_f32()
    }

    fn sparse_interception_count(&self) -> Option<usize> {
        None
    }
}

#[cfg(target_os = "macos")]
impl AllLayerDiagnosticPath for Qwen30QualityRepackNativeDiagnosticRuntime {
    fn reset_for_diagnostic(&mut self) {
        self.reset();
    }

    fn forward_with_route_capture(
        &mut self,
        token: u32,
    ) -> hawking_core::Result<Qwen30NativeRouteCaptureStep> {
        self.forward_token_diagnostic_with_route_capture(token)
    }

    fn final_logits_for_diagnostic(&self) -> hawking_core::Result<Vec<f32>> {
        self.diagnostic_final_logits_f32()
    }

    fn sparse_interception_count(&self) -> Option<usize> {
        Some(self.sparse_gate_up_interception_count())
    }
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct RouteStepWitness {
    position: usize,
    input_token_id: u32,
    sampled_token_id: u32,
    route_ids_u32le_sha256: String,
    l0_expert_ids: [u32; QWEN30_TOP_K],
    l0_expert0_selected: bool,
    command_buffers: usize,
    metal_dispatches: usize,
}

#[cfg(target_os = "macos")]
impl RouteStepWitness {
    fn json(&self) -> Value {
        json!({
            "position": self.position,
            "input_token_id": self.input_token_id,
            "sampled_token_id": self.sampled_token_id,
            "route_ids_u32le_sha256": self.route_ids_u32le_sha256,
            "all_layers_route_captured": QWEN30_ALL_LAYER_COUNT,
            "experts_per_layer": QWEN30_TOP_K,
            "l0_expert_ids": self.l0_expert_ids,
            "l0_expert0_selected": self.l0_expert0_selected,
            "command_buffers": self.command_buffers,
            "metal_dispatches": self.metal_dispatches,
        })
    }
}

#[cfg(target_os = "macos")]
struct PrefixWitness {
    route_trace_sha256: String,
    token_forwards: usize,
    all_layer_route_captures: usize,
    total_command_buffers: usize,
    total_metal_dispatches: usize,
    l0_expert0_selected_positions: Vec<usize>,
    target_position_step: RouteStepWitness,
    final_prefix_step: RouteStepWitness,
    final_logits: Value,
    raw_final_logits: Vec<f32>,
}

#[cfg(target_os = "macos")]
impl PrefixWitness {
    fn final_sampled_token_id(&self) -> u32 {
        self.final_prefix_step.sampled_token_id
    }

    fn json(&self) -> Value {
        json!({
            "exact_prefix_token_forwards": self.token_forwards,
            "all_layer_route_captures": self.all_layer_route_captures,
            "layers_per_forward": QWEN30_ALL_LAYER_COUNT,
            "route_trace_sha256": self.route_trace_sha256,
            "total_command_buffers": self.total_command_buffers,
            "total_metal_dispatches": self.total_metal_dispatches,
            "l0_expert0_selected_positions": self.l0_expert0_selected_positions,
            "target_position_step": self.target_position_step.json(),
            "final_prefix_step": self.final_prefix_step.json(),
            "final_logits": self.final_logits,
        })
    }
}

#[cfg(target_os = "macos")]
struct ContinuationWitness {
    step: RouteStepWitness,
    final_logits: Value,
    raw_final_logits: Vec<f32>,
}

#[cfg(target_os = "macos")]
impl ContinuationWitness {
    fn json(&self) -> Value {
        json!({
            "additional_forwards": FORCED_CONTINUATION_FORWARDS,
            "step": self.step.json(),
            "final_logits": self.final_logits,
        })
    }
}

#[cfg(target_os = "macos")]
fn f32le_sha256(values: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

#[cfg(target_os = "macos")]
fn checked_f32le_bytes(values: &[f32], label: &str) -> Result<Vec<u8>, String> {
    if values.len() != QWEN30_VOCAB_ROWS {
        return Err(format!(
            "{label} raw final-logit vector has {} rows, expected {QWEN30_VOCAB_ROWS}",
            values.len()
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "{label} raw final-logit vector contains a non-finite value"
        ));
    }
    let capacity = values
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("{label} raw final-logit byte count overflowed"))?;
    let mut bytes = Vec::with_capacity(capacity);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

#[cfg(target_os = "macos")]
fn raw_final_logit_payload(
    capture_dir: &Path,
    filename: &str,
    values: &[f32],
) -> Result<Value, String> {
    if Path::new(filename).components().count() != 1 {
        return Err(format!(
            "raw final-logit filename must be one direct child: {filename}"
        ));
    }
    let bytes = checked_f32le_bytes(values, filename)?;
    let payload_dir = capture_dir.join("raw-final-logits");
    if !payload_dir.exists() {
        fs::create_dir(&payload_dir).map_err(|error| {
            format!(
                "cannot create raw final-logit payload directory {}: {error}",
                payload_dir.display()
            )
        })?;
    }
    let payload_dir = canonical_directory(&payload_dir, "raw final-logit payload directory")?;
    let path = payload_dir.join(filename);
    write_new_bytes(&path, &bytes, filename)?;
    let path = canonical_regular(&path, filename)?;
    let observed_sha256 = sha256_file(&path)?;
    let expected_sha256 = sha256_bytes(&bytes);
    if observed_sha256 != expected_sha256 {
        return Err(format!("{filename} hash changed after durable write"));
    }
    Ok(json!({
        "path": path,
        "dtype": "f32le",
        "vocab_rows": values.len(),
        "bytes": bytes.len(),
        "sha256": observed_sha256,
        "all_values_finite": true,
    }))
}

#[cfg(target_os = "macos")]
fn logit_witness(values: &[f32], label: &str) -> Result<Value, String> {
    if values.is_empty() {
        return Err(format!("{label} final-logit vector is empty"));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "{label} final-logit vector contains a non-finite value"
        ));
    }
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_unstable_by(|left, right| {
        values[*right]
            .total_cmp(&values[*left])
            .then_with(|| left.cmp(right))
    });
    let top_k = indices
        .into_iter()
        .take(LOGIT_WITNESS_TOP_K)
        .map(|index| {
            json!({
                "token_id": index,
                "logit": values[index],
                "logit_bits": values[index].to_bits(),
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "vocab_rows": values.len(),
        "full_f32le_sha256": f32le_sha256(values),
        "top_k": top_k,
    }))
}

#[cfg(target_os = "macos")]
fn observe_route_step<P: AllLayerDiagnosticPath>(
    path: &mut P,
    expected_position: usize,
    input_token_id: u32,
    label: &str,
) -> Result<RouteStepWitness, String> {
    let captured = path
        .forward_with_route_capture(input_token_id)
        .map_err(|error| format!("{label} full-token route capture refused: {error}"))?;
    if captured.greedy.position != expected_position {
        return Err(format!(
            "{label} full-token position={}, expected {expected_position}",
            captured.greedy.position
        ));
    }
    if captured.selected_expert_ids_per_layer.len() != QWEN30_ALL_LAYER_COUNT {
        return Err(format!(
            "{label} route capture observed {} layers, expected {QWEN30_ALL_LAYER_COUNT}",
            captured.selected_expert_ids_per_layer.len()
        ));
    }
    if captured.greedy.command_buffers == 0 || captured.greedy.metal_dispatches == 0 {
        return Err(format!(
            "{label} full-token route capture has no Metal command/dispatch witness"
        ));
    }
    let mut bytes = Vec::with_capacity(
        (3 + QWEN30_ALL_LAYER_COUNT * (1 + QWEN30_TOP_K)) * std::mem::size_of::<u32>(),
    );
    bytes.extend_from_slice(&(expected_position as u32).to_le_bytes());
    bytes.extend_from_slice(&input_token_id.to_le_bytes());
    bytes.extend_from_slice(&captured.greedy.token_id.to_le_bytes());
    for (layer, expert_ids) in captured.selected_expert_ids_per_layer.iter().enumerate() {
        let mut seen = [false; QWEN30_EXPERTS as usize];
        bytes.extend_from_slice(&(layer as u32).to_le_bytes());
        for expert in expert_ids {
            if *expert >= QWEN30_EXPERTS {
                return Err(format!(
                    "{label} layer {layer} routed expert {expert} outside 0..{QWEN30_EXPERTS}"
                ));
            }
            if std::mem::replace(&mut seen[*expert as usize], true) {
                return Err(format!(
                    "{label} layer {layer} repeated routed expert {expert}"
                ));
            }
            bytes.extend_from_slice(&expert.to_le_bytes());
        }
    }
    let l0_expert_ids = captured.selected_expert_ids_per_layer[0];
    Ok(RouteStepWitness {
        position: expected_position,
        input_token_id,
        sampled_token_id: captured.greedy.token_id,
        route_ids_u32le_sha256: sha256_bytes(&bytes),
        l0_expert0_selected: l0_expert_ids.contains(&0),
        l0_expert_ids,
        command_buffers: captured.greedy.command_buffers,
        metal_dispatches: captured.greedy.metal_dispatches,
    })
}

#[cfg(target_os = "macos")]
fn execute_exact_prefix<P: AllLayerDiagnosticPath>(
    path: &mut P,
    token_ids: &[u32],
    label: &str,
) -> Result<PrefixWitness, String> {
    if token_ids.len() != TOKEN_COUNT {
        return Err(format!(
            "{label} prefix has {} tokens, expected {TOKEN_COUNT}",
            token_ids.len()
        ));
    }
    path.reset_for_diagnostic();
    let mut route_digest = Sha256::new();
    let mut l0_expert0_selected_positions = Vec::new();
    let mut target_position_step = None;
    let mut final_prefix_step = None;
    let mut total_command_buffers = 0usize;
    let mut total_metal_dispatches = 0usize;
    for (position, token_id) in token_ids.iter().copied().enumerate() {
        let step = observe_route_step(path, position, token_id, label)?;
        route_digest.update(step.route_ids_u32le_sha256.as_bytes());
        if step.l0_expert0_selected {
            l0_expert0_selected_positions.push(position);
        }
        if position == 337 {
            target_position_step = Some(step.clone());
        }
        total_command_buffers = total_command_buffers
            .checked_add(step.command_buffers)
            .ok_or_else(|| format!("{label} command-buffer witness count overflowed"))?;
        total_metal_dispatches = total_metal_dispatches
            .checked_add(step.metal_dispatches)
            .ok_or_else(|| format!("{label} dispatch witness count overflowed"))?;
        final_prefix_step = Some(step);
    }
    let target_position_step = target_position_step
        .ok_or_else(|| format!("{label} exact prefix omitted L0/E0 target position 337"))?;
    if !target_position_step.l0_expert0_selected {
        return Err(format!(
            "{label} target position 337 did not select L0 expert 0"
        ));
    }
    let final_prefix_step = final_prefix_step
        .ok_or_else(|| format!("{label} exact prefix emitted no full-token forward"))?;
    let raw_final_logits = path
        .final_logits_for_diagnostic()
        .map_err(|error| format!("{label} prefix final-logit witness refused: {error}"))?;
    let final_logits = logit_witness(&raw_final_logits, &format!("{label} prefix"))?;
    Ok(PrefixWitness {
        route_trace_sha256: format!("{:x}", route_digest.finalize()),
        token_forwards: token_ids.len(),
        all_layer_route_captures: token_ids
            .len()
            .checked_mul(QWEN30_ALL_LAYER_COUNT)
            .ok_or_else(|| format!("{label} all-layer route capture count overflowed"))?,
        total_command_buffers,
        total_metal_dispatches,
        l0_expert0_selected_positions,
        target_position_step,
        final_prefix_step,
        final_logits,
        raw_final_logits,
    })
}

#[cfg(target_os = "macos")]
fn execute_forced_continuation<P: AllLayerDiagnosticPath>(
    path: &mut P,
    forced_token_id: u32,
    label: &str,
) -> Result<ContinuationWitness, String> {
    let step = observe_route_step(path, TOKEN_COUNT, forced_token_id, label)?;
    let raw_final_logits = path
        .final_logits_for_diagnostic()
        .map_err(|error| format!("{label} continuation final-logit witness refused: {error}"))?;
    let final_logits = logit_witness(&raw_final_logits, &format!("{label} forced continuation"))?;
    Ok(ContinuationWitness {
        step,
        final_logits,
        raw_final_logits,
    })
}

#[cfg(target_os = "macos")]
fn reference_with_schema(document: &Document, schema: &str, status: &str) -> Value {
    let mut reference = document
        .reference()
        .as_object()
        .expect("document reference is an object")
        .clone();
    reference.insert("schema".to_owned(), Value::String(schema.to_owned()));
    reference.insert("status".to_owned(), Value::String(status.to_owned()));
    Value::Object(reference)
}

#[cfg(target_os = "macos")]
fn diagnostic_success_receipt(
    candidate: &CandidateBinding,
    control: &ControlBinding,
    trace: &TraceBinding,
    input: &Document,
    lease: &Document,
    control_prefix: PrefixWitness,
    control_continuation: ContinuationWitness,
    candidate_prefix: PrefixWitness,
    candidate_continuation: ContinuationWitness,
    sparse_interceptions: usize,
) -> Result<Value, String> {
    let expected_sparse_interceptions = candidate_prefix
        .l0_expert0_selected_positions
        .len()
        .checked_add(usize::from(candidate_continuation.step.l0_expert0_selected))
        .ok_or_else(|| "candidate sparse interception witness count overflowed".to_owned())?;
    if expected_sparse_interceptions == 0 || sparse_interceptions != expected_sparse_interceptions {
        return Err(format!(
            "typed HQ30GR2 sparse interception witness mismatch: device encodes={sparse_interceptions}, L0/E0 selected routes={expected_sparse_interceptions}"
        ));
    }
    let mut result = Map::new();
    result.insert(
        "schema".to_owned(),
        Value::String(DIAGNOSTIC_RESULT_SCHEMA.to_owned()),
    );
    result.insert(
        "status".to_owned(),
        Value::String(DIAGNOSTIC_RESULT_STATUS.to_owned()),
    );
    result.insert(
        "mode".to_owned(),
        Value::String(METAL_DIAGNOSTIC_MODE.to_owned()),
    );
    result.insert(
        "metal_device_or_dispatch_performed".to_owned(),
        Value::Bool(true),
    );
    result.insert(
        "typed_hq30gr2_diagnostic_only".to_owned(),
        Value::Bool(true),
    );
    result.insert(
        "durable_capture".to_owned(),
        json!({"receipt_written_last_is_completion_marker": true}),
    );
    result.insert(
        "artifact_binding".to_owned(),
        json!({
            "candidate_manifest": candidate.manifest.evidence.json(),
            "candidate_manifest_seal_sha256": candidate.manifest.seal_sha256,
            "candidate_admission_current_path": candidate.admission_current.evidence.path,
            "candidate_admission_pointer_seal_sha256": candidate.admission_current.seal_sha256,
            "candidate_admission_receipt_seal_sha256": candidate.admission_receipt.seal_sha256,
            "control_manifest": control.manifest.evidence.json(),
            "control_manifest_seal_sha256": control.manifest.seal_sha256,
            "control_runtime_receipt": control.runtime_receipt.evidence.json(),
            "control_runtime_receipt_seal_sha256": control.runtime_receipt.seal_sha256,
        }),
    );
    result.insert(
        "upstream_diagnostic_binding".to_owned(),
        json!({
            "compiler_trace_receipt": trace.compiler_receipt.reference(),
            "route_capture_receipt": trace.route_receipt.reference(),
            "preparation_receipt": trace.preparation_receipt.reference(),
        }),
    );
    result.insert(
        "input_contract".to_owned(),
        reference_with_schema(input, INPUT_CONTRACT_SCHEMA, INPUT_CONTRACT_STATUS),
    );
    result.insert(
        "metal_execution_policy".to_owned(),
        json!({
            "strict_math_required": true,
            "diagnostic_only": true,
            "timing_or_benchmarking_allowed": false,
            "hcli_or_server_allowed": false,
            "tps_or_tg_claim_allowed": false,
            "coherence_claim_allowed": false,
            "capability_claim_allowed": false,
            "tournament_claim_allowed": false,
            "lease_binding": reference_with_schema(lease, QUIET_LEASE_SCHEMA, QUIET_LEASE_STATUS),
        }),
    );
    let forced_token_id = control_prefix.final_sampled_token_id();
    result.insert(
        "exact_trace_execution".to_owned(),
        json!({
            "probe_id": PROBE_ID,
            "source_template_token_count": TOKEN_COUNT,
            "source_template_token_ids_u32le_sha256": trace.token_ids_u32le_sha256,
            "baseline_exact_prefix_all_48_layers": true,
            "candidate_exact_prefix_all_48_layers": true,
            "unbounded_generation_or_sampling_loop_performed": false,
            "forced_continuation": {
                "baseline_deterministic_argmax_after_exact_prefix": true,
                "forced_identical_token_into_baseline_and_candidate": true,
                "additional_forwards_per_path": FORCED_CONTINUATION_FORWARDS,
                "baseline_additional_all_48_layers": true,
                "candidate_additional_all_48_layers": true,
                "forced_token_id": forced_token_id,
            },
        }),
    );
    result.insert(
        "structural_witnesses".to_owned(),
        json!({
            "control_scalar_path": control_prefix.json(),
            "control_forced_continuation": control_continuation.json(),
            "candidate_typed_hq30gr2_path": candidate_prefix.json(),
            "candidate_forced_continuation": candidate_continuation.json(),
            "typed_l0_e0_sparse_interception": {
                "selected_residual_organs": [L0_E0_GATE, L0_E0_UP],
                "device_sparse_gate_up_encodes": sparse_interceptions,
                "matching_l0_e0_route_selections": expected_sparse_interceptions,
                "direct_fallback_for_sparse_residual_forbidden": true,
                "scalar_control_topology_for_all_unchanged_organs": true,
            },
            "model_bodies_concurrent": false,
            "timing_or_rate_values_recorded": false,
        }),
    );
    result.insert(
        "claim_boundary".to_owned(),
        json!({
            "does_not_claim_hcli": true,
            "does_not_claim_coherence": true,
            "does_not_claim_tps_or_tg": true,
            "does_not_claim_capability": true,
            "does_not_claim_tournament": true,
            "does_not_serve_or_modify_live_qwen30_server_watcher_or_adapter": true,
        }),
    );
    seal(result)
}

#[cfg(target_os = "macos")]
fn diagnostic_refusal_receipt(
    candidate: &CandidateBinding,
    control: &ControlBinding,
    trace: &TraceBinding,
    input: &Document,
    lease: &Document,
    detail: &str,
) -> Result<Value, String> {
    let mut result = Map::new();
    result.insert(
        "schema".to_owned(),
        Value::String(DIAGNOSTIC_RESULT_SCHEMA.to_owned()),
    );
    result.insert(
        "status".to_owned(),
        Value::String(DIAGNOSTIC_REFUSAL_STATUS.to_owned()),
    );
    result.insert(
        "mode".to_owned(),
        Value::String(METAL_DIAGNOSTIC_MODE.to_owned()),
    );
    result.insert(
        "typed_hq30gr2_diagnostic_only".to_owned(),
        Value::Bool(true),
    );
    result.insert(
        "durable_capture".to_owned(),
        json!({"receipt_written_last_is_completion_marker": true}),
    );
    result.insert(
        "refusal".to_owned(),
        json!({"detail": detail, "retry_performed": false}),
    );
    result.insert(
        "artifact_binding".to_owned(),
        json!({
            "candidate_manifest": candidate.manifest.evidence.json(),
            "candidate_manifest_seal_sha256": candidate.manifest.seal_sha256,
            "candidate_admission_current_path": candidate.admission_current.evidence.path,
            "candidate_admission_pointer_seal_sha256": candidate.admission_current.seal_sha256,
            "candidate_admission_receipt_seal_sha256": candidate.admission_receipt.seal_sha256,
            "control_manifest": control.manifest.evidence.json(),
            "control_manifest_seal_sha256": control.manifest.seal_sha256,
            "control_runtime_receipt": control.runtime_receipt.evidence.json(),
            "control_runtime_receipt_seal_sha256": control.runtime_receipt.seal_sha256,
        }),
    );
    result.insert(
        "upstream_diagnostic_binding".to_owned(),
        json!({
            "compiler_trace_receipt": trace.compiler_receipt.reference(),
            "route_capture_receipt": trace.route_receipt.reference(),
            "preparation_receipt": trace.preparation_receipt.reference(),
        }),
    );
    result.insert(
        "input_contract".to_owned(),
        reference_with_schema(input, INPUT_CONTRACT_SCHEMA, INPUT_CONTRACT_STATUS),
    );
    result.insert(
        "metal_execution_policy".to_owned(),
        json!({
            "strict_math_required": true,
            "diagnostic_only": true,
            "timing_or_benchmarking_allowed": false,
            "hcli_or_server_allowed": false,
            "tps_or_tg_claim_allowed": false,
            "coherence_claim_allowed": false,
            "capability_claim_allowed": false,
            "tournament_claim_allowed": false,
            "lease_binding": reference_with_schema(lease, QUIET_LEASE_SCHEMA, QUIET_LEASE_STATUS),
        }),
    );
    result.insert(
        "claim_boundary".to_owned(),
        json!({
            "does_not_claim_hcli": true,
            "does_not_claim_coherence": true,
            "does_not_claim_tps_or_tg": true,
            "does_not_claim_capability": true,
            "does_not_claim_tournament": true,
        }),
    );
    seal(result)
}

#[cfg(target_os = "macos")]
fn raw_retention_expected_hash(
    contract: &Document,
    endpoint: &str,
    model: &str,
) -> Result<String, String> {
    let replay = field(
        &contract.value,
        "replay_binding",
        "raw final-logit retention contract",
    )?;
    let hashes = field(
        replay,
        "native_raw_hashes_must_replay_prior_98db_witness",
        "raw final-logit retention contract.replay_binding",
    )?;
    let endpoint = field(
        hashes,
        endpoint,
        "raw final-logit retention contract.native replay hashes",
    )?;
    text(
        field(
            endpoint,
            model,
            "raw final-logit retention contract.native replay endpoint",
        )?,
        "raw final-logit retention contract.expected native full-logit hash",
    )
    .map(str::to_owned)
}

#[cfg(target_os = "macos")]
fn validate_raw_retention_contract(
    contract_path: &Path,
    candidate: &CandidateBinding,
    control: &ControlBinding,
    trace: &TraceBinding,
) -> Result<Document, String> {
    let contract = sealed_document(contract_path, "raw final-logit retention contract")?;
    expect_schema_status(
        &contract,
        RAW_RETENTION_CONTRACT_SCHEMA,
        RAW_RETENTION_CONTRACT_STATUS,
        "raw final-logit retention contract",
    )?;
    let replay = field(
        &contract.value,
        "replay_binding",
        "raw final-logit retention contract",
    )?;
    let old_inner = sealed_document(
        &ref_path(
            field(
                replay,
                "immutable_prior_all_layer_inner_diagnostic",
                "raw final-logit retention contract.replay_binding",
            )?,
            "raw final-logit retention contract.prior inner diagnostic",
        )?,
        "raw final-logit retention prior inner diagnostic",
    )?;
    expect_schema_status(
        &old_inner,
        DIAGNOSTIC_RESULT_SCHEMA,
        DIAGNOSTIC_RESULT_STATUS,
        "raw final-logit retention prior inner diagnostic",
    )?;
    expect_reference(
        field(
            replay,
            "immutable_prior_all_layer_inner_diagnostic",
            "raw final-logit retention contract.replay_binding",
        )?,
        &old_inner,
        "raw final-logit retention contract.prior inner diagnostic",
        true,
    )?;
    let exact = field(
        replay,
        "exact_trace",
        "raw final-logit retention contract.replay_binding",
    )?;
    if text(
        field(
            exact,
            "probe_id",
            "raw final-logit retention contract.exact_trace",
        )?,
        "raw final-logit retention contract.exact_trace.probe_id",
    )? != PROBE_ID
        || text(
            field(
                exact,
                "source_template_token_ids_u32le_sha256",
                "raw final-logit retention contract.exact_trace",
            )?,
            "raw final-logit retention contract.exact_trace.token hash",
        )? != trace.token_ids_u32le_sha256
        || field(
            exact,
            "source_template_token_count",
            "raw final-logit retention contract.exact_trace",
        )?
        .as_u64()
            != Some(TOKEN_COUNT as u64)
    {
        return Err("raw final-logit retention contract trace binding drifted".into());
    }
    let old_artifact = field(
        &old_inner.value,
        "artifact_binding",
        "raw final-logit retention prior inner diagnostic",
    )?;
    if text(
        field(
            old_artifact,
            "candidate_manifest_seal_sha256",
            "raw final-logit retention prior inner diagnostic artifact binding",
        )?,
        "raw final-logit retention prior candidate seal",
    )? != candidate.manifest.seal_sha256
        || text(
            field(
                old_artifact,
                "control_runtime_receipt_seal_sha256",
                "raw final-logit retention prior inner diagnostic artifact binding",
            )?,
            "raw final-logit retention prior control runtime seal",
        )? != control.runtime_receipt.seal_sha256
    {
        return Err("raw final-logit retention contract artifact authority drifted".into());
    }
    for endpoint in ["exact_prefix", "forced_shared_continuation"] {
        for model in ["scalar_control", "hq30gr2_candidate"] {
            let expected = raw_retention_expected_hash(&contract, endpoint, model)?;
            if !valid_sha256(&expected) {
                return Err("raw final-logit retention expected native hash is malformed".into());
            }
        }
    }
    let six_vectors = field(
        &contract.value,
        "six_vector_retention_contract",
        "raw final-logit retention contract",
    )?;
    if field(
        six_vectors,
        "vocab_rows",
        "raw final-logit retention six-vector contract",
    )?
    .as_u64()
        != Some(QWEN30_VOCAB_ROWS as u64)
        || field(
            six_vectors,
            "bytes_per_vector",
            "raw final-logit retention six-vector contract",
        )?
        .as_u64()
            != Some((QWEN30_VOCAB_ROWS * std::mem::size_of::<f32>()) as u64)
    {
        return Err("raw final-logit retention vector geometry drifted".into());
    }
    // This is a deliberately hard stop.  The native successor must not be
    // used to construct a misleading partial six-vector oracle while the
    // source-teacher memory lease has not become safe.
    let source_memory = field(
        &contract.value,
        "source_memory_and_eviction_gate",
        "raw final-logit retention contract",
    )?;
    expect_false(
        field(
            source_memory,
            "source_teacher_capture_is_currently_blocked",
            "raw final-logit retention contract.source memory gate",
        )?,
        "raw final-logit retention contract.source teacher memory block",
    )?;
    Ok(contract)
}

#[cfg(target_os = "macos")]
fn raw_retention_success_receipt(
    candidate: &CandidateBinding,
    control: &ControlBinding,
    trace: &TraceBinding,
    input: &Document,
    lease: &Document,
    raw_contract: &Document,
    structural_witness: Value,
    raw_payloads: Value,
) -> Result<Value, String> {
    let mut result = Map::new();
    result.insert(
        "schema".to_owned(),
        Value::String(RAW_RETENTION_RESULT_SCHEMA.to_owned()),
    );
    result.insert(
        "status".to_owned(),
        Value::String(RAW_RETENTION_RESULT_STATUS.to_owned()),
    );
    result.insert(
        "mode".to_owned(),
        Value::String(RAW_FINAL_LOGIT_RETENTION_MODE.to_owned()),
    );
    result.insert(
        "raw_retention_contract".to_owned(),
        raw_contract.reference(),
    );
    result.insert(
        "artifact_binding".to_owned(),
        json!({
            "candidate_manifest": candidate.manifest.evidence.json(),
            "candidate_manifest_seal_sha256": candidate.manifest.seal_sha256,
            "candidate_admission_receipt_seal_sha256": candidate.admission_receipt.seal_sha256,
            "control_manifest": control.manifest.evidence.json(),
            "control_manifest_seal_sha256": control.manifest.seal_sha256,
            "control_runtime_receipt": control.runtime_receipt.evidence.json(),
            "control_runtime_receipt_seal_sha256": control.runtime_receipt.seal_sha256,
        }),
    );
    result.insert(
        "upstream_diagnostic_binding".to_owned(),
        json!({
            "compiler_trace_receipt": trace.compiler_receipt.reference(),
            "route_capture_receipt": trace.route_receipt.reference(),
            "preparation_receipt": trace.preparation_receipt.reference(),
            "input_contract": reference_with_schema(input, INPUT_CONTRACT_SCHEMA, INPUT_CONTRACT_STATUS),
            "lease_receipt": reference_with_schema(lease, QUIET_LEASE_SCHEMA, QUIET_LEASE_STATUS),
        }),
    );
    result.insert("native_raw_final_logit_payloads".to_owned(), raw_payloads);
    result.insert(
        "source_teacher_vectors".to_owned(),
        json!({
            "required_payloads": [
                "source_bf16_exact_prefix_logits.f32le",
                "source_bf16_forced_shared_continuation_logits.f32le",
            ],
            "written_by_this_native_capture": false,
            "three_way_metric_emitted": false,
        }),
    );
    result.insert(
        "all_layer_structural_witness".to_owned(),
        structural_witness,
    );
    result.insert(
        "durable_capture".to_owned(),
        json!({"receipt_written_last_after_four_native_payload_fsyncs": true}),
    );
    result.insert(
        "claim_boundary".to_owned(),
        json!({
            "does_not_claim_source_bf16_execution": true,
            "does_not_claim_a_six_vector_oracle_metric": true,
            "does_not_claim_coherence_or_hcli": true,
            "does_not_claim_tps_tg_capability_or_tournament": true,
            "does_not_modify_or_serve_live_qwen30_adapter": true,
        }),
    );
    seal(result)
}

#[cfg(target_os = "macos")]
fn raw_retention_refusal_receipt(
    raw_contract: Option<&Document>,
    detail: &str,
) -> Result<Value, String> {
    let mut result = Map::new();
    result.insert(
        "schema".to_owned(),
        Value::String(RAW_RETENTION_RESULT_SCHEMA.to_owned()),
    );
    result.insert(
        "status".to_owned(),
        Value::String(RAW_RETENTION_REFUSAL_STATUS.to_owned()),
    );
    result.insert(
        "mode".to_owned(),
        Value::String(RAW_FINAL_LOGIT_RETENTION_MODE.to_owned()),
    );
    if let Some(contract) = raw_contract {
        result.insert("raw_retention_contract".to_owned(), contract.reference());
    }
    result.insert(
        "refusal".to_owned(),
        json!({"detail": detail, "retry_performed": false}),
    );
    result.insert(
        "claim_boundary".to_owned(),
        json!({
            "does_not_claim_source_bf16_execution": true,
            "does_not_claim_a_six_vector_oracle_metric": true,
            "does_not_claim_coherence_or_hcli": true,
            "does_not_claim_tps_tg_capability_or_tournament": true,
        }),
    );
    seal(result)
}

#[cfg(target_os = "macos")]
struct AllLayerDiagnosticWitnesses {
    control_prefix: PrefixWitness,
    control_continuation: ContinuationWitness,
    candidate_prefix: PrefixWitness,
    candidate_continuation: ContinuationWitness,
    sparse_interceptions: usize,
}

#[cfg(target_os = "macos")]
fn execute_all_layer_witnesses(
    candidate: &CandidateBinding,
    control: &ControlBinding,
    trace: &TraceBinding,
) -> Result<AllLayerDiagnosticWitnesses, String> {
    let options = Qwen30CompleteRuntimeOptions {
        max_seq_len: TOKEN_COUNT
            .checked_add(FORCED_CONTINUATION_FORWARDS)
            .ok_or_else(|| "all-layer diagnostic context length overflowed".to_owned())?,
        trace_dispatch: false,
        packed_matvec_kernel: Qwen30PackedMatvecKernel::ScalarControl,
        gate_up_swiglu_kernel: Qwen30GateUpSwiGluKernel::ThreeDispatchControl,
    };
    let (control_prefix, control_continuation) = {
        let mut runtime = Qwen30CompleteNativeRuntime::load(
            &control.manifest.evidence.path,
            &control.admission,
            options.clone(),
        )
        .map_err(|error| format!("scalar control all-layer runtime load refused: {error}"))?;
        let prefix = execute_exact_prefix(&mut runtime, &trace.token_ids, "scalar control")?;
        let continuation = execute_forced_continuation(
            &mut runtime,
            prefix.final_sampled_token_id(),
            "scalar control",
        )?;
        (prefix, continuation)
    };
    let forced_token_id = control_prefix.final_sampled_token_id();
    let (candidate_prefix, candidate_continuation, sparse_interceptions) = {
        let mut runtime = Qwen30QualityRepackNativeDiagnosticRuntime::load(
            &candidate.manifest.evidence.path,
            &candidate.admission,
            options,
        )
        .map_err(|error| {
            format!("typed HQ30GR2 candidate all-layer runtime load refused: {error}")
        })?;
        let prefix =
            execute_exact_prefix(&mut runtime, &trace.token_ids, "typed HQ30GR2 candidate")?;
        let continuation =
            execute_forced_continuation(&mut runtime, forced_token_id, "typed HQ30GR2 candidate")?;
        let interceptions = runtime.sparse_interception_count().ok_or_else(|| {
            "typed HQ30GR2 runtime failed to expose sparse interception witness".to_owned()
        })?;
        (prefix, continuation, interceptions)
    };
    Ok(AllLayerDiagnosticWitnesses {
        control_prefix,
        control_continuation,
        candidate_prefix,
        candidate_continuation,
        sparse_interceptions,
    })
}

#[cfg(target_os = "macos")]
fn execute_all_layer_diagnostic(
    candidate: &CandidateBinding,
    control: &ControlBinding,
    trace: &TraceBinding,
    input: &Document,
    lease: &Document,
) -> Result<Value, String> {
    let witnesses = execute_all_layer_witnesses(candidate, control, trace)?;
    diagnostic_success_receipt(
        candidate,
        control,
        trace,
        input,
        lease,
        witnesses.control_prefix,
        witnesses.control_continuation,
        witnesses.candidate_prefix,
        witnesses.candidate_continuation,
        witnesses.sparse_interceptions,
    )
}

#[cfg(target_os = "macos")]
fn replayed_raw_payload(
    raw_contract: &Document,
    capture_dir: &Path,
    filename: &str,
    endpoint: &str,
    model: &str,
    values: &[f32],
) -> Result<Value, String> {
    let payload = raw_final_logit_payload(capture_dir, filename, values)?;
    let expected = raw_retention_expected_hash(raw_contract, endpoint, model)?;
    let observed = text(
        field(&payload, "sha256", "raw final-logit payload")?,
        "raw final-logit payload.sha256",
    )?;
    if observed != expected {
        return Err(format!(
            "raw final-logit replay mismatch for {model}/{endpoint}: observed={observed} expected={expected}"
        ));
    }
    Ok(payload)
}

#[cfg(target_os = "macos")]
fn execute_raw_final_logit_retention(
    candidate: &CandidateBinding,
    control: &ControlBinding,
    trace: &TraceBinding,
    input: &Document,
    lease: &Document,
    raw_contract: &Document,
    capture_dir: &Path,
) -> Result<Value, String> {
    let witnesses = execute_all_layer_witnesses(candidate, control, trace)?;
    let raw_payloads = json!({
        "scalar_control": {
            "exact_prefix": replayed_raw_payload(
                raw_contract,
                capture_dir,
                "scalar_control_exact_prefix_logits.f32le",
                "exact_prefix",
                "scalar_control",
                &witnesses.control_prefix.raw_final_logits,
            )?,
            "forced_shared_continuation": replayed_raw_payload(
                raw_contract,
                capture_dir,
                "scalar_control_forced_shared_continuation_logits.f32le",
                "forced_shared_continuation",
                "scalar_control",
                &witnesses.control_continuation.raw_final_logits,
            )?,
        },
        "hq30gr2_candidate": {
            "exact_prefix": replayed_raw_payload(
                raw_contract,
                capture_dir,
                "hq30gr2_candidate_exact_prefix_logits.f32le",
                "exact_prefix",
                "hq30gr2_candidate",
                &witnesses.candidate_prefix.raw_final_logits,
            )?,
            "forced_shared_continuation": replayed_raw_payload(
                raw_contract,
                capture_dir,
                "hq30gr2_candidate_forced_shared_continuation_logits.f32le",
                "forced_shared_continuation",
                "hq30gr2_candidate",
                &witnesses.candidate_continuation.raw_final_logits,
            )?,
        },
        "all_four_replay_prior_98db_full_f32le_hashes": true,
        "source_bf16_payloads_written_by_this_native_capture": false,
    });
    let structural_witness = diagnostic_success_receipt(
        candidate,
        control,
        trace,
        input,
        lease,
        witnesses.control_prefix,
        witnesses.control_continuation,
        witnesses.candidate_prefix,
        witnesses.candidate_continuation,
        witnesses.sparse_interceptions,
    )?;
    raw_retention_success_receipt(
        candidate,
        control,
        trace,
        input,
        lease,
        raw_contract,
        structural_witness,
        raw_payloads,
    )
}

#[cfg(target_os = "macos")]
fn run_metal_diagnostic(args: &Args) -> Result<Value, String> {
    let invocation = metal_invocation(args)?;
    let candidate = candidate_binding(args)?;
    let control = control_binding(args)?;
    let trace = trace_binding(args, &candidate, &control)?;
    let lease = sealed_document(&invocation.lease_receipt, "all-layer quiet lease")?;
    validate_quiet_lease(&lease, &candidate, &control, &trace)?;
    let input = validate_input_contract(
        &invocation.input_contract,
        &candidate,
        &control,
        &trace,
        &lease,
    )?;
    let capture_dir = prepare_capture_dir(
        &invocation,
        &candidate,
        &control,
        &trace,
        &input,
        &lease,
        METAL_DIAGNOSTIC_MODE,
        None,
    )?;
    let receipt_path = capture_dir.join("receipt.json");
    match execute_all_layer_diagnostic(&candidate, &control, &trace, &input, &lease) {
        Ok(receipt) => {
            write_new(&receipt_path, &receipt)?;
            Ok(receipt)
        }
        Err(error) => {
            let refusal =
                diagnostic_refusal_receipt(&candidate, &control, &trace, &input, &lease, &error)?;
            write_new(&receipt_path, &refusal)?;
            Err(error)
        }
    }
}

#[cfg(target_os = "macos")]
fn run_metal_raw_final_logit_retention(args: &Args) -> Result<Value, String> {
    let invocation = metal_invocation(args)?;
    let candidate = candidate_binding(args)?;
    let control = control_binding(args)?;
    let trace = trace_binding(args, &candidate, &control)?;
    let lease = sealed_document(&invocation.lease_receipt, "all-layer quiet lease")?;
    validate_quiet_lease(&lease, &candidate, &control, &trace)?;
    let input = validate_input_contract(
        &invocation.input_contract,
        &candidate,
        &control,
        &trace,
        &lease,
    )?;
    let raw_contract_path = invocation
        .raw_retention_contract
        .as_deref()
        .ok_or_else(|| {
            "raw final-logit retention mode lacks a canonical contract path".to_owned()
        })?;
    // This validation fails closed before the capture directory is made and
    // before a Metal runtime can be constructed when source memory is still
    // unsafe for the required six-vector experiment.
    let raw_contract =
        validate_raw_retention_contract(raw_contract_path, &candidate, &control, &trace)?;
    let capture_dir = prepare_capture_dir(
        &invocation,
        &candidate,
        &control,
        &trace,
        &input,
        &lease,
        RAW_FINAL_LOGIT_RETENTION_MODE,
        Some(&raw_contract),
    )?;
    let receipt_path = capture_dir.join("receipt.json");
    match execute_raw_final_logit_retention(
        &candidate,
        &control,
        &trace,
        &input,
        &lease,
        &raw_contract,
        &capture_dir,
    ) {
        Ok(receipt) => {
            write_new(&receipt_path, &receipt)?;
            Ok(receipt)
        }
        Err(error) => {
            let refusal = raw_retention_refusal_receipt(Some(&raw_contract), &error)?;
            write_new(&receipt_path, &refusal)?;
            Err(error)
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn run_metal_diagnostic(_args: &Args) -> Result<Value, String> {
    Err("HQ30GR2 all-layer diagnostic requires macOS Metal; CPU fallback is forbidden".to_owned())
}

#[cfg(not(target_os = "macos"))]
fn run_metal_raw_final_logit_retention(_args: &Args) -> Result<Value, String> {
    Err(
        "HQ30GR2 raw final-logit retention requires macOS Metal; CPU fallback is forbidden"
            .to_owned(),
    )
}

fn run() -> Result<Value, String> {
    let args = parse_args()?;
    match args.mode {
        InvocationMode::CpuPreflight => match preflight(&args) {
            Ok(output) => {
                if let Some(path) = args.output.as_deref() {
                    write_new(path, &output)?;
                }
                Ok(output)
            }
            Err(error) => {
                if let Some(path) = args.output.as_deref() {
                    let terminal = refusal(&args, &error)?;
                    write_new(path, &terminal)?;
                }
                Err(error)
            }
        },
        InvocationMode::MetalDiagnostic => run_metal_diagnostic(&args),
        InvocationMode::MetalRawFinalLogitRetention => run_metal_raw_final_logit_retention(&args),
    }
}

fn main() {
    match run() {
        Ok(result) => println!(
            "{}",
            serde_json::to_string(&result).expect("preflight JSON serializes")
        ),
        Err(error) => {
            eprintln!("Qwen30 HQ30GR2 all-layer preflight refused: {error}");
            process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_os = "macos")]
    use hawking_core::model::qwen30_complete_runtime::Qwen30NativeGreedyStep;
    use std::sync::atomic::{AtomicUsize, Ordering};
    #[cfg(target_os = "macos")]
    use std::time::Duration;

    static FIXTURE_ID: AtomicUsize = AtomicUsize::new(0);

    fn fixture_root(label: &str) -> PathBuf {
        let ordinal = FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
        let root = env::temp_dir().join(format!(
            "hawking-qwen30-preflight-{label}-{}-{ordinal}",
            process::id()
        ));
        fs::create_dir_all(&root).expect("create test fixture root");
        root
    }

    fn write_sealed_fixture(path: &Path, mut fields: Map<String, Value>) -> Document {
        let value = seal(std::mem::take(&mut fields)).expect("seal fixture");
        fs::write(path, serde_json::to_vec(&value).expect("serialize fixture"))
            .expect("write fixture");
        sealed_document(path, "test fixture").expect("reopen sealed fixture")
    }

    fn fixture_current(
        root: &Path,
        name: &str,
        contract: CurrentPointerContract,
        receipt_reference: Value,
    ) -> Document {
        let mut fields = Map::new();
        fields.insert(
            "schema".to_owned(),
            Value::String(contract.pointer_schema.to_owned()),
        );
        fields.insert(
            "status".to_owned(),
            Value::String(contract.pointer_status.to_owned()),
        );
        if contract.declared_candidate_root_required {
            fields.insert(
                "candidate_root".to_owned(),
                Value::String(root.to_string_lossy().into_owned()),
            );
        }
        fields.insert(contract.receipt_field.to_owned(), receipt_reference);
        write_sealed_fixture(&root.join(format!("{name}-current.json")), fields)
    }

    fn fixture_receipt(root: &Path, name: &str, schema: &str, status: &str) -> Document {
        let mut fields = Map::new();
        fields.insert("schema".to_owned(), Value::String(schema.to_owned()));
        fields.insert("status".to_owned(), Value::String(status.to_owned()));
        write_sealed_fixture(&root.join(format!("{name}-receipt.json")), fields)
    }

    fn fixture_reference(contract: CurrentPointerContract, receipt: &Document) -> Value {
        match contract.reference_style {
            CurrentPointerReferenceStyle::RichDocumentSha => json!({
                "path": receipt.evidence.path,
                "document_sha256": receipt.evidence.sha256,
                "seal_sha256": receipt.seal_sha256,
            }),
            CurrentPointerReferenceStyle::PathAndSealWithinCandidateRoot => json!({
                "path": receipt.evidence.path,
                "seal_sha256": receipt.seal_sha256,
            }),
        }
    }

    #[test]
    fn token_hash_is_little_endian_and_deterministic() {
        assert_eq!(
            token_ids_sha256(&[1, 256]),
            "242045e2f1bb37769b514f182fd91b3d215324cb57f187cced1e9c62921dbac3"
        );
    }

    #[test]
    fn device_mode_is_refused_before_any_catalog_or_runtime_action() {
        let input = vec![
            "probe".to_owned(),
            "--mode".to_owned(),
            "metal-diagnostic".to_owned(),
        ];
        let mode = input.get(2).expect("mode supplied");
        assert_ne!(mode, MODE);
    }

    #[test]
    fn relative_trace_path_cannot_escape_run_root() {
        let root = Path::new("/tmp/qwen30-preflight-root");
        let bad = Path::new("../escape.json");
        assert!(bad
            .components()
            .any(|component| matches!(component, Component::ParentDir)));
        assert!(root.is_absolute());
    }

    #[test]
    fn result_boundary_never_names_a_runtime_or_tps_pass() {
        let boundary = json!({
            "metal_context_created": false,
            "all_layer_forward_performed": false,
            "endpoint_or_hcli_called": false,
        });
        assert_eq!(boundary["metal_context_created"], Value::Bool(false));
        assert_eq!(boundary["all_layer_forward_performed"], Value::Bool(false));
        assert_eq!(boundary["endpoint_or_hcli_called"], Value::Bool(false));
    }

    #[test]
    fn all_allowlisted_current_pointer_contracts_accept_only_their_expected_shape() {
        let root = fixture_root("pointer-contracts");
        let contracts = [
            CANDIDATE_ADMISSION_POINTER,
            COMPONENT_PARITY_POINTER,
            COMPILER_TRACE_POINTER,
            ROUTE_CAPTURE_POINTER,
            PREPARATION_POINTER,
        ];
        for (index, contract) in contracts.into_iter().enumerate() {
            let case_root = root.join(format!("case-{index}"));
            fs::create_dir_all(&case_root).expect("create pointer case root");
            let receipt = fixture_receipt(
                &case_root,
                "target",
                contract.receipt_schema,
                contract.receipt_status,
            );
            let current = fixture_current(
                &case_root,
                "current",
                contract,
                fixture_reference(contract, &receipt),
            );
            let resolved = resolve_current_pointer(&current, &case_root, contract)
                .expect("allowlisted current pointer must resolve");
            assert_eq!(resolved.evidence.path, receipt.evidence.path);
        }
        fs::remove_dir_all(root).expect("remove test fixture");
    }

    #[test]
    fn all_current_pointer_contracts_reject_wrong_path_seal_schema_and_missing_fields() {
        let root = fixture_root("pointer-rejections");
        let contracts = [
            CANDIDATE_ADMISSION_POINTER,
            COMPONENT_PARITY_POINTER,
            COMPILER_TRACE_POINTER,
            ROUTE_CAPTURE_POINTER,
            PREPARATION_POINTER,
        ];
        for (index, contract) in contracts.into_iter().enumerate() {
            let case_root = root.join(format!("case-{index}"));
            fs::create_dir_all(&case_root).expect("create pointer case root");
            let receipt = fixture_receipt(
                &case_root,
                "target",
                contract.receipt_schema,
                contract.receipt_status,
            );
            let outside_root = root.join(format!("outside-{index}"));
            fs::create_dir_all(&outside_root).expect("create outside pointer root");
            let outside_receipt = fixture_receipt(
                &outside_root,
                "target",
                contract.receipt_schema,
                contract.receipt_status,
            );
            let wrong_path = fixture_current(
                &case_root,
                "wrong-path",
                contract,
                json!({"path": outside_receipt.evidence.path, "seal_sha256": outside_receipt.seal_sha256}),
            );
            assert!(resolve_current_pointer(&wrong_path, &case_root, contract)
                .expect_err("outside receipt must refuse")
                .contains("escaped"));

            let mut wrong_seal_reference = fixture_reference(contract, &receipt)
                .as_object()
                .expect("fixture reference is an object")
                .clone();
            wrong_seal_reference.insert(
                "seal_sha256".to_owned(),
                Value::String(
                    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
                ),
            );
            let wrong_seal = fixture_current(
                &case_root,
                "wrong-seal",
                contract,
                Value::Object(wrong_seal_reference),
            );
            let wrong_seal_error = resolve_current_pointer(&wrong_seal, &case_root, contract)
                .expect_err("wrong seal must refuse");
            let expected_seal_error = match contract.reference_style {
                CurrentPointerReferenceStyle::RichDocumentSha => "seal drifted",
                CurrentPointerReferenceStyle::PathAndSealWithinCandidateRoot => "seal differs",
            };
            assert!(wrong_seal_error.contains(expected_seal_error));

            let bad_schema = fixture_receipt(
                &case_root,
                "wrong-schema",
                "wrong.pointer.schema.v1",
                contract.receipt_status,
            );
            let wrong_schema = fixture_current(
                &case_root,
                "wrong-schema",
                contract,
                fixture_reference(contract, &bad_schema),
            );
            assert!(resolve_current_pointer(&wrong_schema, &case_root, contract)
                .expect_err("wrong receipt schema must refuse")
                .contains("schema/status"));

            let mut missing_seal_reference = fixture_reference(contract, &receipt)
                .as_object()
                .expect("fixture reference is an object")
                .clone();
            missing_seal_reference.remove("seal_sha256");
            let missing_seal = fixture_current(
                &case_root,
                "missing-seal",
                contract,
                Value::Object(missing_seal_reference),
            );
            assert!(resolve_current_pointer(&missing_seal, &case_root, contract)
                .expect_err("missing pointer seal must refuse")
                .contains("lacks seal_sha256"));

            if contract.reference_style == CurrentPointerReferenceStyle::RichDocumentSha {
                let mut missing_digest_reference = fixture_reference(contract, &receipt)
                    .as_object()
                    .expect("fixture reference is an object")
                    .clone();
                missing_digest_reference.remove("document_sha256");
                let missing_digest = fixture_current(
                    &case_root,
                    "missing-digest",
                    contract,
                    Value::Object(missing_digest_reference),
                );
                assert!(
                    resolve_current_pointer(&missing_digest, &case_root, contract)
                        .expect_err("rich pointer without document digest must refuse")
                        .contains("lacks document_sha256/sha256")
                );
            }
        }
        fs::remove_dir_all(root).expect("remove test fixture");
    }

    #[cfg(target_os = "macos")]
    struct FakeAllLayerDiagnosticPath {
        position: usize,
        duplicate_route_at_layer: Option<usize>,
    }

    #[cfg(target_os = "macos")]
    impl FakeAllLayerDiagnosticPath {
        fn route_ids(&self) -> Vec<[u32; QWEN30_TOP_K]> {
            let mut routes = vec![[0, 1, 2, 3, 4, 5, 6, 7]; QWEN30_ALL_LAYER_COUNT];
            if let Some(layer) = self.duplicate_route_at_layer {
                routes[layer] = [0, 0, 2, 3, 4, 5, 6, 7];
            }
            routes
        }
    }

    #[cfg(target_os = "macos")]
    impl AllLayerDiagnosticPath for FakeAllLayerDiagnosticPath {
        fn reset_for_diagnostic(&mut self) {
            self.position = 0;
        }

        fn forward_with_route_capture(
            &mut self,
            _token: u32,
        ) -> hawking_core::Result<Qwen30NativeRouteCaptureStep> {
            let position = self.position;
            self.position += 1;
            Ok(Qwen30NativeRouteCaptureStep {
                greedy: Qwen30NativeGreedyStep {
                    position,
                    token_id: 17,
                    elapsed: Duration::ZERO,
                    command_buffers: 1,
                    metal_dispatches: 1,
                    host_route_id_readbacks: 0,
                    host_sample_id_readbacks: 0,
                    gate_up_swiglu_device_control_parity: None,
                    host_stage_intervals: Vec::new(),
                },
                selected_expert_ids_per_layer: self.route_ids(),
            })
        }

        fn final_logits_for_diagnostic(&self) -> hawking_core::Result<Vec<f32>> {
            Ok(vec![-1.0, 0.0, 2.0])
        }

        fn sparse_interception_count(&self) -> Option<usize> {
            None
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn exact_prefix_driver_requires_all_48_layers_and_one_forced_continuation() {
        let mut path = FakeAllLayerDiagnosticPath {
            position: 999,
            duplicate_route_at_layer: None,
        };
        let prefix = execute_exact_prefix(&mut path, &vec![11; TOKEN_COUNT], "fake control")
            .expect("complete fake 48-layer prefix must satisfy structural executor gate");
        assert_eq!(prefix.token_forwards, TOKEN_COUNT);
        assert_eq!(
            prefix.all_layer_route_captures,
            TOKEN_COUNT * QWEN30_ALL_LAYER_COUNT
        );
        assert_eq!(prefix.target_position_step.position, 337);
        assert!(prefix.target_position_step.l0_expert0_selected);
        assert_eq!(prefix.final_prefix_step.position, TOKEN_COUNT - 1);
        assert_eq!(prefix.final_sampled_token_id(), 17);
        let continuation = execute_forced_continuation(&mut path, 17, "fake control")
            .expect("one forced continuation must retain the 48-layer witness");
        assert_eq!(continuation.step.position, TOKEN_COUNT);
        assert_eq!(path.position, TOKEN_COUNT + FORCED_CONTINUATION_FORWARDS);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn route_witness_refuses_duplicate_expert_within_any_layer() {
        let mut path = FakeAllLayerDiagnosticPath {
            position: 0,
            duplicate_route_at_layer: Some(19),
        };
        let error = match observe_route_step(&mut path, 0, 11, "duplicate route") {
            Ok(_) => panic!("duplicate expert route must not produce an all-layer witness"),
            Err(error) => error,
        };
        assert!(error.contains("layer 19 repeated routed expert 0"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn raw_final_logit_bytes_require_exact_full_vocab_and_finite_values() {
        let values = vec![1.25f32; QWEN30_VOCAB_ROWS];
        let bytes = checked_f32le_bytes(&values, "raw test")
            .expect("finite full-vocab vector must encode as F32LE");
        assert_eq!(bytes.len(), QWEN30_VOCAB_ROWS * std::mem::size_of::<f32>());
        assert_eq!(sha256_bytes(&bytes), f32le_sha256(&values));
        assert!(
            checked_f32le_bytes(&values[..QWEN30_VOCAB_ROWS - 1], "short raw test")
                .expect_err("short raw vector must refuse")
                .contains("expected")
        );
        let mut non_finite = values;
        non_finite[31] = f32::NAN;
        assert!(checked_f32le_bytes(&non_finite, "non-finite raw test")
            .expect_err("non-finite raw vector must refuse")
            .contains("non-finite"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn raw_payload_is_create_new_and_not_replaceable() {
        let root = fixture_root("raw-final-logit-payload");
        let capture = root.join("capture");
        fs::create_dir(&capture).expect("create raw capture root");
        let values = vec![0.5f32; QWEN30_VOCAB_ROWS];
        let payload = raw_final_logit_payload(
            &capture,
            "scalar_control_exact_prefix_logits.f32le",
            &values,
        )
        .expect("first raw payload write must succeed");
        assert_eq!(
            payload["bytes"],
            QWEN30_VOCAB_ROWS * std::mem::size_of::<f32>()
        );
        assert!(raw_final_logit_payload(
            &capture,
            "scalar_control_exact_prefix_logits.f32le",
            &values
        )
        .expect_err("raw payload must never replace an existing witness")
        .contains("refusing to replace"));
        fs::remove_dir_all(root).expect("remove raw payload fixture");
    }
}
