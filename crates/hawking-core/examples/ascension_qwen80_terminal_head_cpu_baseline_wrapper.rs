//! Seal the exact current Qwen80 terminal-head + tokenizer/sampler CPU component
//! receipts into one cryptographically sealed CPU-baseline wrapper.
//!
//! This is ordered host-dispatch step 1 only:
//!   "verify a cryptographically sealed CPU-baseline wrapper binds both exact
//!    raw component receipt documents and preimages"
//!
//! It does not authorize device dispatch. Raw component receipts remain
//! unsealed provenance; this wrapper binds their document SHA-256 (raw file
//! bytes) and their unsealed-preimage SHA-256 values. Step 2 (actual all-48-layer
//! hidden [2048] with sealed device-parity) stays separately blocked.
//!
//! CPU/file-only: no capture, no lease, no Metal context, no model load.
//!
//! ```text
//! cargo run -p hawking-core --example ascension_qwen80_terminal_head_cpu_baseline_wrapper -- \
//!   --terminal-head-cpu-receipt /abs/QWEN80_DIRECT_PACKED_TERMINAL_HEAD_CPU_COMPONENT_RECEIPT.json \
//!   --tokenizer-sampler-receipt /abs/QWEN80_TOKENIZER_TEMPLATE_SAMPLER_HANDOFF_COMPONENT_RECEIPT.json \
//!   --out /abs/new/QWEN80_TERMINAL_HEAD_CPU_BASELINE_WRAPPER.json
//! ```

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const SCHEMA: &str = "hawking.ascension.qwen80_terminal_head_cpu_baseline_wrapper.v1";
const STATUS: &str = "SEALED_CURRENT_ADMITTED_QWEN80_TERMINAL_HEAD_AND_SAMPLER_CPU_BASELINE";
const TERMINAL_RECEIPT_SCHEMA: &str = "hawking.ascension.qwen80_direct_packed_terminal_head_cpu.v1";
const TERMINAL_RECEIPT_STATUS: &str =
    "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_TERMINAL_COMPONENT_CPU_ONLY_NOT_RUNTIME_OR_TOKEN";
/// Raw file-bytes SHA-256 of the pinned terminal-head CPU component receipt.
const TERMINAL_RECEIPT_DOCUMENT_SHA256: &str =
    "1ebe19139833491ec06cc7515f6844fad0a122de15fb74c978dfda3524a38d04";
/// Document field `unsealed_preimage_sha256` of that same receipt.
const TERMINAL_RECEIPT_UNSEALED_PREIMAGE_SHA256: &str =
    "d815c6bfff615a1c238ed56863b14ba61349f1c04824195448722a6a3e81372b";
const TOKENIZER_RECEIPT_SCHEMA: &str =
    "hawking.ascension.qwen80_tokenizer_sampler_handoff_contract.v1";
const TOKENIZER_RECEIPT_STATUS: &str =
    "EARNED_SOURCE_BOUND_TOKENIZER_TEMPLATE_SAMPLER_HANDOFF_COMPONENT_NOT_RUNTIME_OR_TOKEN";
/// Raw file-bytes SHA-256 of the pinned (unsealed) tokenizer/sampler receipt.
const TOKENIZER_RECEIPT_DOCUMENT_SHA256: &str =
    "e152b21d9eae43e7039f9d646412b2806b8f07d3d3c7ea932ab281dc6c9a0792";
/// Document field `unsealed_preimage_sha256` of that same receipt.
const TOKENIZER_RECEIPT_UNSEALED_PREIMAGE_SHA256: &str =
    "5c2f66487c7a4fb387806bb9439259eb62c86f33b1e30ca4dac701ee38ac164c";
const MODEL_ID: &str = "Qwen3-Coder-Next-80B";
const MODEL_KEY: &str = "qwen80";
const SOURCE_REPOSITORY: &str = "Qwen/Qwen3-Coder-Next";
const SOURCE_REVISION: &str = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb";
const MANIFEST_SCHEMA: &str = "hawking.ascension.qwen80_complete_binary_gravity.v1";
const MANIFEST_SEAL: &str = "14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b";
const ADMISSION_RECEIPT_SEAL: &str =
    "939b41322363da3db774a2530b207bf380ed641d23cae671fc6438c0eecbf628";
const SOURCE_CONFIG_SHA256: &str =
    "a7b8098d3b05777f12bb5677a26bf1240a1bb09def1b06b29e6be86cae2e84f8";
const SOURCE_TOKENIZER_SHA256: &str =
    "19564a48c4f71a2a1b937cce34c737a1e662b171c5f5d7edf641a15cd896f07d";
const SOURCE_TOKENIZER_CONFIG_SHA256: &str =
    "fc76878832c668e3f0f8be66e6239a475b9093d2fe5cef97c242369779e6c6e6";
const SOURCE_CHAT_TEMPLATE_SHA256: &str =
    "c79a833039a43602150cce0902403d6e376c50930c1b2a139b2964e1f0c322a0";
const SOURCE_GENERATION_CONFIG_SHA256: &str =
    "37a3c1ef63516ca489c575f0db1c0405ddc0c3dbaca9ed987344c966c37aeef5";
const RMS_EPSILON_BITS: u32 = 897_988_541;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct SourceAdmissionBinding {
    model_id: String,
    model_key: String,
    source_repository: String,
    source_revision: String,
    manifest_schema: String,
    manifest_seal_sha256: String,
    admission_receipt_seal_sha256: String,
    source_config_sha256: String,
    source_tokenizer_sha256: String,
    source_tokenizer_config_sha256: String,
    source_chat_template_sha256: String,
    source_generation_config_sha256: String,
    rms_norm_epsilon_bits: u32,
}

impl SourceAdmissionBinding {
    fn exact() -> Self {
        Self {
            model_id: MODEL_ID.into(),
            model_key: MODEL_KEY.into(),
            source_repository: SOURCE_REPOSITORY.into(),
            source_revision: SOURCE_REVISION.into(),
            manifest_schema: MANIFEST_SCHEMA.into(),
            manifest_seal_sha256: MANIFEST_SEAL.into(),
            admission_receipt_seal_sha256: ADMISSION_RECEIPT_SEAL.into(),
            source_config_sha256: SOURCE_CONFIG_SHA256.into(),
            source_tokenizer_sha256: SOURCE_TOKENIZER_SHA256.into(),
            source_tokenizer_config_sha256: SOURCE_TOKENIZER_CONFIG_SHA256.into(),
            source_chat_template_sha256: SOURCE_CHAT_TEMPLATE_SHA256.into(),
            source_generation_config_sha256: SOURCE_GENERATION_CONFIG_SHA256.into(),
            rms_norm_epsilon_bits: RMS_EPSILON_BITS,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct BoundReceiptEvidence {
    absolute_path: String,
    bytes: usize,
    /// Raw file-bytes SHA-256 (identity convention used by the metal preflight
    /// and device-contract validators for `*_document_sha256`).
    document_sha256: String,
    /// Receipt field `unsealed_preimage_sha256` (content preimage, not seal).
    unsealed_preimage_sha256: String,
    schema: String,
    status: String,
}

#[derive(Clone, Debug, Serialize)]
struct SealedCpuBaselineWrapper {
    schema: &'static str,
    status: &'static str,
    seal_sha256: String,
    integrity_verified: bool,
    source_admission: SourceAdmissionBinding,
    terminal_receipt_schema: String,
    terminal_receipt_status: String,
    terminal_receipt_document_sha256: String,
    terminal_receipt_unsealed_preimage_sha256: String,
    tokenizer_receipt_schema: String,
    tokenizer_receipt_status: String,
    tokenizer_receipt_document_sha256: String,
    tokenizer_receipt_unsealed_preimage_sha256: String,
    /// Explicit step-1-only bound: raw receipts still cannot authorize dispatch.
    raw_component_receipts_can_authorize_device_dispatch: bool,
    device_dispatch_authorized: bool,
    ordered_host_dispatch_step_satisfied: u32,
    terminal_receipt_evidence: BoundReceiptEvidence,
    tokenizer_receipt_evidence: BoundReceiptEvidence,
    claim_boundary: Vec<&'static str>,
}

struct Args {
    terminal_head_cpu_receipt: PathBuf,
    tokenizer_sampler_receipt: PathBuf,
    out: PathBuf,
}

fn usage() -> &'static str {
    "usage: ascension_qwen80_terminal_head_cpu_baseline_wrapper \
--terminal-head-cpu-receipt ABSOLUTE_JSON \
--tokenizer-sampler-receipt ABSOLUTE_JSON \
--out ABSOLUTE_NEW_JSON"
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && value.bytes().any(|byte| byte != b'0')
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
            for (key, entry) in values {
                ordered.insert(key.clone(), canonical_json(entry)?);
            }
            Ok(Value::Object(ordered.into_iter().collect()))
        }
        other => Ok(other.clone()),
    }
}

fn json_sha(value: &Value) -> Result<String, String> {
    let canonical = canonical_json(value)?;
    let bytes = serde_json::to_vec(&canonical).map_err(|error| error.to_string())?;
    Ok(sha256_hex(&bytes))
}

fn seal_document(document: &mut Value) -> Result<String, String> {
    let root = document
        .as_object()
        .ok_or_else(|| "wrapper document must be a JSON object".to_string())?;
    if root.contains_key("seal_sha256") {
        return Err("wrapper document already contains seal_sha256".into());
    }
    let seal = json_sha(document)?;
    document
        .as_object_mut()
        .expect("checked object")
        .insert("seal_sha256".into(), Value::String(seal.clone()));
    Ok(seal)
}

fn verify_seal(document: &Value) -> Result<String, String> {
    let root = document
        .as_object()
        .ok_or_else(|| "wrapper document must be a JSON object".to_string())?;
    let seal = root
        .get("seal_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| "wrapper document missing seal_sha256".to_string())?;
    if !is_lower_sha256(seal) {
        return Err("wrapper seal_sha256 is not a valid non-zero lowercase SHA-256".into());
    }
    let mut unsigned = root.clone();
    unsigned.remove("seal_sha256");
    let expected = json_sha(&Value::Object(unsigned))?;
    if expected != seal {
        return Err("wrapper seal mismatch: seal_sha256 does not bind document content".into());
    }
    Ok(seal.to_owned())
}

fn require_regular_absolute(path: &Path, label: &str) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err(format!("{label} must be an absolute path"));
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("{label} metadata failed at {}: {error}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(format!(
            "{label} must be a regular non-symlink file: {}",
            path.display()
        ));
    }
    fs::canonicalize(path)
        .map_err(|error| format!("{label} canonicalize failed at {}: {error}", path.display()))
}

fn load_receipt(
    path: &Path,
    label: &str,
    expected_schema: &str,
    expected_status: &str,
    expected_document_sha256: &str,
    expected_preimage_sha256: &str,
) -> Result<BoundReceiptEvidence, String> {
    let absolute = require_regular_absolute(path, label)?;
    let bytes = fs::read(&absolute)
        .map_err(|error| format!("{label} read failed at {}: {error}", absolute.display()))?;
    let document_sha256 = sha256_hex(&bytes);
    if document_sha256 != expected_document_sha256 {
        return Err(format!(
            "{label} raw file-bytes SHA-256 drifted: expected {expected_document_sha256}, observed {document_sha256}"
        ));
    }
    let document: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("{label} invalid JSON at {}: {error}", absolute.display()))?;
    let root = document
        .as_object()
        .ok_or_else(|| format!("{label} root must be a JSON object"))?;
    let schema = root
        .get("schema")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label} missing schema"))?;
    if schema != expected_schema {
        return Err(format!("{label} schema drifted: {schema}"));
    }
    let status = root
        .get("status")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label} missing status"))?;
    if status != expected_status {
        return Err(format!("{label} status drifted: {status}"));
    }
    if root.contains_key("seal_sha256") {
        return Err(format!(
            "{label} is expected to be raw/unsealed component evidence (no seal_sha256)"
        ));
    }
    let preimage = root
        .get("unsealed_preimage_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label} missing unsealed_preimage_sha256"))?;
    if preimage != expected_preimage_sha256 || !is_lower_sha256(preimage) {
        return Err(format!(
            "{label} unsealed_preimage_sha256 drifted: expected {expected_preimage_sha256}, observed {preimage}"
        ));
    }
    Ok(BoundReceiptEvidence {
        absolute_path: absolute.display().to_string(),
        bytes: bytes.len(),
        document_sha256,
        unsealed_preimage_sha256: preimage.to_owned(),
        schema: schema.to_owned(),
        status: status.to_owned(),
    })
}

fn write_new(path: &Path, document: &Value) -> Result<(), String> {
    if !path.is_absolute() {
        return Err("--out must be an absolute path".into());
    }
    if path.exists() {
        return Err(format!(
            "--out must be create-new; refusing overwrite of {}",
            path.display()
        ));
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() && !parent.is_dir() {
            return Err(format!(
                "--out parent directory is missing: {}",
                parent.display()
            ));
        }
    }
    let mut bytes =
        serde_json::to_vec_pretty(document).map_err(|error| format!("serialize wrapper: {error}"))?;
    bytes.push(b'\n');
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("create --out {}: {error}", path.display()))?;
    file.write_all(&bytes)
        .map_err(|error| format!("write --out {}: {error}", path.display()))?;
    file.sync_all()
        .map_err(|error| format!("sync --out {}: {error}", path.display()))?;
    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut terminal = None;
    let mut tokenizer = None;
    let mut out = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--terminal-head-cpu-receipt" => {
                let value = args
                    .next()
                    .ok_or("missing absolute path after --terminal-head-cpu-receipt")?;
                if terminal.replace(PathBuf::from(value)).is_some() {
                    return Err("--terminal-head-cpu-receipt supplied more than once".into());
                }
            }
            "--tokenizer-sampler-receipt" => {
                let value = args
                    .next()
                    .ok_or("missing absolute path after --tokenizer-sampler-receipt")?;
                if tokenizer.replace(PathBuf::from(value)).is_some() {
                    return Err("--tokenizer-sampler-receipt supplied more than once".into());
                }
            }
            "--out" => {
                let value = args.next().ok_or("missing absolute path after --out")?;
                if out.replace(PathBuf::from(value)).is_some() {
                    return Err("--out supplied more than once".into());
                }
            }
            "--help" | "-h" => return Err(usage().into()),
            other => return Err(format!("unsupported option {other:?}; {}", usage()).into()),
        }
    }
    let terminal_head_cpu_receipt = terminal.ok_or_else(|| format!("missing --terminal-head-cpu-receipt; {}", usage()))?;
    let tokenizer_sampler_receipt = tokenizer.ok_or_else(|| format!("missing --tokenizer-sampler-receipt; {}", usage()))?;
    let out = out.ok_or_else(|| format!("missing --out; {}", usage()))?;
    if !out.is_absolute() {
        return Err("--out must be an absolute path".into());
    }
    Ok(Args {
        terminal_head_cpu_receipt,
        tokenizer_sampler_receipt,
        out,
    })
}

/// Fields required by the metal-preflight `SealedCpuBaseline` validator.
/// Kept as a separate view so producer extras (evidence, claim_boundary) cannot
/// hide a missing required pin.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct MetalPreflightBaselineView {
    schema: String,
    status: String,
    seal_sha256: String,
    integrity_verified: bool,
    source_admission: SourceAdmissionBinding,
    terminal_receipt_schema: String,
    terminal_receipt_status: String,
    terminal_receipt_document_sha256: String,
    terminal_receipt_unsealed_preimage_sha256: String,
    tokenizer_receipt_schema: String,
    tokenizer_receipt_status: String,
    tokenizer_receipt_document_sha256: String,
    tokenizer_receipt_unsealed_preimage_sha256: String,
}

fn metal_preflight_baseline_validation_errors(view: &MetalPreflightBaselineView) -> Vec<String> {
    let mut errors = Vec::new();
    if view.schema != SCHEMA
        || view.status != STATUS
        || !is_lower_sha256(&view.seal_sha256)
        || !view.integrity_verified
    {
        errors.push("terminal CPU baseline must be sealed and integrity-verified".into());
    }
    if view.source_admission != SourceAdmissionBinding::exact() {
        errors.push("terminal CPU baseline source/admission/tokenizer binding drifted".into());
    }
    if view.terminal_receipt_schema != TERMINAL_RECEIPT_SCHEMA
        || view.terminal_receipt_status != TERMINAL_RECEIPT_STATUS
        || view.terminal_receipt_document_sha256 != TERMINAL_RECEIPT_DOCUMENT_SHA256
        || view.terminal_receipt_unsealed_preimage_sha256
            != TERMINAL_RECEIPT_UNSEALED_PREIMAGE_SHA256
        || view.tokenizer_receipt_schema != TOKENIZER_RECEIPT_SCHEMA
        || view.tokenizer_receipt_status != TOKENIZER_RECEIPT_STATUS
        || view.tokenizer_receipt_document_sha256 != TOKENIZER_RECEIPT_DOCUMENT_SHA256
        || view.tokenizer_receipt_unsealed_preimage_sha256
            != TOKENIZER_RECEIPT_UNSEALED_PREIMAGE_SHA256
    {
        errors.push(
            "terminal CPU baseline does not bind the exact current terminal/tokenizer component receipts"
                .into(),
        );
    }
    errors
}

/// Project the full wrapper onto the metal-preflight required field set.
fn metal_preflight_view(document: &Value) -> Result<MetalPreflightBaselineView, String> {
    let root = document
        .as_object()
        .ok_or_else(|| "wrapper document must be a JSON object".to_string())?;
    let mut projected = Map::new();
    for key in [
        "schema",
        "status",
        "seal_sha256",
        "integrity_verified",
        "source_admission",
        "terminal_receipt_schema",
        "terminal_receipt_status",
        "terminal_receipt_document_sha256",
        "terminal_receipt_unsealed_preimage_sha256",
        "tokenizer_receipt_schema",
        "tokenizer_receipt_status",
        "tokenizer_receipt_document_sha256",
        "tokenizer_receipt_unsealed_preimage_sha256",
    ] {
        let value = root
            .get(key)
            .ok_or_else(|| format!("wrapper missing metal-preflight field {key:?}"))?
            .clone();
        projected.insert(key.into(), value);
    }
    serde_json::from_value(Value::Object(projected))
        .map_err(|error| format!("wrapper failed metal-preflight field decode: {error}"))
}

fn build_wrapper(
    terminal: BoundReceiptEvidence,
    tokenizer: BoundReceiptEvidence,
) -> Result<(SealedCpuBaselineWrapper, Value), String> {
    // integrity_verified asserts the producer verified receipt bindings before seal.
    // It is sealed with the document; seal crypto is verified separately after seal.
    let mut unsigned = SealedCpuBaselineWrapper {
        schema: SCHEMA,
        status: STATUS,
        seal_sha256: String::new(),
        integrity_verified: true,
        source_admission: SourceAdmissionBinding::exact(),
        terminal_receipt_schema: terminal.schema.clone(),
        terminal_receipt_status: terminal.status.clone(),
        terminal_receipt_document_sha256: terminal.document_sha256.clone(),
        terminal_receipt_unsealed_preimage_sha256: terminal.unsealed_preimage_sha256.clone(),
        tokenizer_receipt_schema: tokenizer.schema.clone(),
        tokenizer_receipt_status: tokenizer.status.clone(),
        tokenizer_receipt_document_sha256: tokenizer.document_sha256.clone(),
        tokenizer_receipt_unsealed_preimage_sha256: tokenizer.unsealed_preimage_sha256.clone(),
        raw_component_receipts_can_authorize_device_dispatch: false,
        device_dispatch_authorized: false,
        ordered_host_dispatch_step_satisfied: 1,
        terminal_receipt_evidence: terminal,
        tokenizer_receipt_evidence: tokenizer,
        claim_boundary: vec![
            "This sealed CPU-baseline wrapper binds only the exact raw terminal-head and tokenizer/sampler component receipt document SHA-256 values (raw file bytes) plus their unsealed-preimage SHA-256 values.",
            "raw_component_receipts_can_authorize_device_dispatch remains false. This wrapper satisfies ordered host-dispatch step 1 only.",
            "It does not authorize Metal compile/dispatch, grant a lease, bind a post-48-layer hidden buffer, load a model, start a runtime/server, invoke HCLI, or measure TPS/TG.",
            "Step 2 remains blocked until an actual source-bound, non-synthetic all-48-layer hidden vector [2048] is bound with a sealed device-parity receipt.",
        ],
    };

    let mut document = serde_json::to_value(&unsigned).map_err(|error| error.to_string())?;
    // Seal binds every field except seal_sha256 itself.
    document
        .as_object_mut()
        .ok_or("wrapper serialize lost object root")?
        .remove("seal_sha256");
    let seal = seal_document(&mut document)?;
    verify_seal(&document)?;
    unsigned.seal_sha256 = seal;

    // Hard identity pins for consumers that do not re-parse nested evidence.
    let view = metal_preflight_view(&document)?;
    let errors = metal_preflight_baseline_validation_errors(&view);
    if !errors.is_empty() {
        return Err(format!(
            "produced wrapper fails metal-preflight baseline validation: {}",
            errors.join("; ")
        ));
    }
    if document
        .get("raw_component_receipts_can_authorize_device_dispatch")
        .and_then(Value::as_bool)
        != Some(false)
        || document
            .get("device_dispatch_authorized")
            .and_then(Value::as_bool)
            != Some(false)
    {
        return Err(
            "wrapper must keep raw_component_receipts_can_authorize_device_dispatch and device_dispatch_authorized false"
                .into(),
        );
    }
    Ok((unsigned, document))
}

fn run(args: Args) -> Result<(PathBuf, String), Box<dyn Error>> {
    let terminal = load_receipt(
        &args.terminal_head_cpu_receipt,
        "terminal-head CPU receipt",
        TERMINAL_RECEIPT_SCHEMA,
        TERMINAL_RECEIPT_STATUS,
        TERMINAL_RECEIPT_DOCUMENT_SHA256,
        TERMINAL_RECEIPT_UNSEALED_PREIMAGE_SHA256,
    )?;
    let tokenizer = load_receipt(
        &args.tokenizer_sampler_receipt,
        "tokenizer/sampler receipt",
        TOKENIZER_RECEIPT_SCHEMA,
        TOKENIZER_RECEIPT_STATUS,
        TOKENIZER_RECEIPT_DOCUMENT_SHA256,
        TOKENIZER_RECEIPT_UNSEALED_PREIMAGE_SHA256,
    )?;
    let (wrapper, document) = build_wrapper(terminal, tokenizer)?;
    write_new(&args.out, &document)?;
    Ok((args.out, wrapper.seal_sha256))
}

fn main() {
    match parse_args().and_then(run) {
        Ok((out, seal)) => {
            println!(
                "{{\"schema\":\"{SCHEMA}\",\"status\":\"{STATUS}\",\"seal_sha256\":\"{seal}\",\"out\":\"{}\",\"raw_component_receipts_can_authorize_device_dispatch\":false,\"device_dispatch_authorized\":false,\"ordered_host_dispatch_step_satisfied\":1}}",
                out.display()
            );
        }
        Err(error) => {
            eprintln!("ascension_qwen80_terminal_head_cpu_baseline_wrapper refused: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_receipt(
        schema: &str,
        status: &str,
        preimage: &str,
        document_sha: &str,
    ) -> BoundReceiptEvidence {
        BoundReceiptEvidence {
            absolute_path: "/fixture/receipt.json".into(),
            bytes: 1,
            document_sha256: document_sha.into(),
            unsealed_preimage_sha256: preimage.into(),
            schema: schema.into(),
            status: status.into(),
        }
    }

    #[test]
    fn sealed_wrapper_binds_exact_pinned_receipt_identities() {
        let terminal = fixture_receipt(
            TERMINAL_RECEIPT_SCHEMA,
            TERMINAL_RECEIPT_STATUS,
            TERMINAL_RECEIPT_UNSEALED_PREIMAGE_SHA256,
            TERMINAL_RECEIPT_DOCUMENT_SHA256,
        );
        let tokenizer = fixture_receipt(
            TOKENIZER_RECEIPT_SCHEMA,
            TOKENIZER_RECEIPT_STATUS,
            TOKENIZER_RECEIPT_UNSEALED_PREIMAGE_SHA256,
            TOKENIZER_RECEIPT_DOCUMENT_SHA256,
        );
        let (wrapper, document) = build_wrapper(terminal, tokenizer).expect("build");
        assert_eq!(wrapper.schema, SCHEMA);
        assert_eq!(wrapper.status, STATUS);
        assert!(wrapper.integrity_verified);
        assert!(!wrapper.raw_component_receipts_can_authorize_device_dispatch);
        assert!(!wrapper.device_dispatch_authorized);
        assert_eq!(wrapper.ordered_host_dispatch_step_satisfied, 1);
        assert_eq!(
            wrapper.terminal_receipt_document_sha256,
            TERMINAL_RECEIPT_DOCUMENT_SHA256
        );
        assert_eq!(
            wrapper.tokenizer_receipt_document_sha256,
            TOKENIZER_RECEIPT_DOCUMENT_SHA256
        );
        assert_eq!(
            wrapper.terminal_receipt_unsealed_preimage_sha256,
            TERMINAL_RECEIPT_UNSEALED_PREIMAGE_SHA256
        );
        assert_eq!(
            wrapper.tokenizer_receipt_unsealed_preimage_sha256,
            TOKENIZER_RECEIPT_UNSEALED_PREIMAGE_SHA256
        );
        verify_seal(&document).expect("seal verifies");
        let view = metal_preflight_view(&document).expect("project");
        assert!(metal_preflight_baseline_validation_errors(&view).is_empty());
        assert!(wrapper.claim_boundary.iter().any(|line| line.contains("step 1 only")));
    }

    #[test]
    fn write_new_refuses_overwrite() {
        let dir = env::temp_dir().join(format!(
            "qwen80-cpu-baseline-wrapper-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(&dir).expect("temp dir");
        let absolute = dir.join("wrapper.json");
        assert!(absolute.is_absolute());
        write_new(&absolute, &serde_json::json!({"ok": true})).expect("first write");
        let error = write_new(&absolute, &serde_json::json!({"ok": false})).expect_err("overwrite");
        assert!(error.contains("create-new") || error.contains("refusing overwrite"));
        let _ = fs::remove_dir_all(&dir);
    }
}
