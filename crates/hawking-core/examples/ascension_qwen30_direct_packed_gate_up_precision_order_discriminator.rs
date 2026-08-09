//! CPU-only arithmetic discriminator for the Qwen30 direct-packed gate/up trial.
//!
//! This intentionally narrow diagnostic distinguishes the scalar control
//! shader's source-level `sum += weight * input` accumulation from the fused
//! candidate's explicit `fma(weight, input, sum)` accumulation.  It opens only
//! admission-verified `HQ30G1B1` payload snapshots and never creates a Metal
//! device, decodes a dense weight body, loads BF16, executes a Qwen layer or
//! token, starts an endpoint, or measures time/TPS.  It is evidence for the
//! next command-topology experiment, not an integration decision.

use half::f16;
use hawking_core::model::qwen_complete_binary::{
    admit_complete_binary_artifact, parse_complete_binary_header, CompleteBinaryAdmission,
    CompleteBinaryArtifact, CompleteBinaryHeader, QwenCompleteBinaryModel,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const SCHEMA: &str =
    "hawking.ascension.qwen30_direct_packed_gate_up_precision_order_discriminator.v1";
const STATUS: &str = "EARNED_CPU_DIRECT_PACKED_GATE_UP_ORDER_PRECISION_DISCRIMINATOR";
const MODEL_ID: &str = "Qwen3-Coder-30B-A3B-Instruct";
const GROUP_SIZE: usize = 128;
const EXPERT_ROWS: usize = 768;
const HIDDEN_COLS: usize = 2048;
const SAMPLES: &[(usize, usize)] = &[(0, 0), (47, 127)];

type Probe<T> = Result<T, Box<dyn Error>>;

struct Args {
    manifest: PathBuf,
    expected_manifest_seal_sha256: String,
    expected_source_audit_seal_sha256: String,
    expected_source_revision: String,
    runtime_receipt: PathBuf,
    expected_runtime_receipt_seal_sha256: String,
    expected_runtime_executable_sha256: String,
    out: PathBuf,
}

struct PackedProjection {
    name: String,
    payload: Arc<[u8]>,
    header: CompleteBinaryHeader,
    artifact_path: String,
    artifact_sha256: String,
}

#[derive(Default)]
struct DifferenceSummary {
    values_compared: u64,
    differing_f32_bits: u64,
    max_abs: f32,
    first_mismatch: Option<Value>,
}

fn failure(message: impl Into<String>) -> Box<dyn Error> {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
}

fn usage() -> &'static str {
    "usage: ascension_qwen30_direct_packed_gate_up_precision_order_discriminator \\
        --manifest ABSOLUTE_PATH \\
        --expected-manifest-seal-sha256 SHA256 \\
        --expected-source-audit-seal-sha256 SHA256 \\
        --expected-source-revision REVISION \\
        --runtime-receipt ABSOLUTE_PATH \\
        --expected-runtime-receipt-seal-sha256 SHA256 \\
        --expected-runtime-executable-sha256 SHA256 \\
        --out ABSOLUTE_PATH"
}

fn required<T>(value: Option<T>, flag: &str) -> Probe<T> {
    value.ok_or_else(|| failure(format!("missing {flag}; {}", usage())))
}

fn parse_args() -> Probe<Args> {
    let mut manifest = None;
    let mut expected_manifest_seal_sha256 = None;
    let mut expected_source_audit_seal_sha256 = None;
    let mut expected_source_revision = None;
    let mut runtime_receipt = None;
    let mut expected_runtime_receipt_seal_sha256 = None;
    let mut expected_runtime_executable_sha256 = None;
    let mut out = None;
    let mut values = env::args().skip(1);
    while let Some(flag) = values.next() {
        let value = values
            .next()
            .ok_or_else(|| failure(format!("missing value for {flag:?}; {}", usage())))?;
        match flag.as_str() {
            "--manifest" => {
                if manifest.replace(PathBuf::from(value)).is_some() {
                    return Err(failure("--manifest supplied more than once"));
                }
            }
            "--expected-manifest-seal-sha256" => {
                if expected_manifest_seal_sha256.replace(value).is_some() {
                    return Err(failure(
                        "--expected-manifest-seal-sha256 supplied more than once",
                    ));
                }
            }
            "--expected-source-audit-seal-sha256" => {
                if expected_source_audit_seal_sha256.replace(value).is_some() {
                    return Err(failure(
                        "--expected-source-audit-seal-sha256 supplied more than once",
                    ));
                }
            }
            "--expected-source-revision" => {
                if expected_source_revision.replace(value).is_some() {
                    return Err(failure(
                        "--expected-source-revision supplied more than once",
                    ));
                }
            }
            "--runtime-receipt" => {
                if runtime_receipt.replace(PathBuf::from(value)).is_some() {
                    return Err(failure("--runtime-receipt supplied more than once"));
                }
            }
            "--expected-runtime-receipt-seal-sha256" => {
                if expected_runtime_receipt_seal_sha256
                    .replace(value)
                    .is_some()
                {
                    return Err(failure(
                        "--expected-runtime-receipt-seal-sha256 supplied more than once",
                    ));
                }
            }
            "--expected-runtime-executable-sha256" => {
                if expected_runtime_executable_sha256.replace(value).is_some() {
                    return Err(failure(
                        "--expected-runtime-executable-sha256 supplied more than once",
                    ));
                }
            }
            "--out" => {
                if out.replace(PathBuf::from(value)).is_some() {
                    return Err(failure("--out supplied more than once"));
                }
            }
            _ => return Err(failure(format!("unsupported option {flag:?}; {}", usage()))),
        }
    }
    let args = Args {
        manifest: required(manifest, "--manifest")?,
        expected_manifest_seal_sha256: required(
            expected_manifest_seal_sha256,
            "--expected-manifest-seal-sha256",
        )?,
        expected_source_audit_seal_sha256: required(
            expected_source_audit_seal_sha256,
            "--expected-source-audit-seal-sha256",
        )?,
        expected_source_revision: required(expected_source_revision, "--expected-source-revision")?,
        runtime_receipt: required(runtime_receipt, "--runtime-receipt")?,
        expected_runtime_receipt_seal_sha256: required(
            expected_runtime_receipt_seal_sha256,
            "--expected-runtime-receipt-seal-sha256",
        )?,
        expected_runtime_executable_sha256: required(
            expected_runtime_executable_sha256,
            "--expected-runtime-executable-sha256",
        )?,
        out: required(out, "--out")?,
    };
    for (flag, path) in [
        ("--manifest", &args.manifest),
        ("--runtime-receipt", &args.runtime_receipt),
        ("--out", &args.out),
    ] {
        if !path.is_absolute() {
            return Err(failure(format!("{flag} must be an absolute path")));
        }
    }
    Ok(args)
}

fn file_sha256(path: &Path) -> Probe<String> {
    let mut digest = Sha256::new();
    let bytes = fs::read(path)?;
    digest.update(bytes);
    Ok(format!("{:x}", digest.finalize()))
}

fn source_sha256(text: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(text.as_bytes());
    format!("{:x}", digest.finalize())
}

fn read_runtime_binding(args: &Args) -> Probe<Value> {
    let raw = fs::read(&args.runtime_receipt)?;
    let receipt: Value = serde_json::from_slice(&raw)?;
    let object = receipt
        .as_object()
        .ok_or_else(|| failure("runtime receipt root is not an object"))?;
    let expected_schema = "hawking.ascension.physical_exact_full_token_runtime.v1";
    let expected_status = "PASS_EXACT_NATIVE_FULL_TOKEN_RUNTIME";
    if object.get("schema").and_then(Value::as_str) != Some(expected_schema)
        || object.get("status").and_then(Value::as_str) != Some(expected_status)
    {
        return Err(failure(
            "runtime receipt is not the current exact native full-token runtime PASS schema/status",
        ));
    }
    if object.get("seal_sha256").and_then(Value::as_str)
        != Some(args.expected_runtime_receipt_seal_sha256.as_str())
    {
        return Err(failure(
            "runtime receipt seal does not match --expected-runtime-receipt-seal-sha256",
        ));
    }
    let binding = object
        .get("binding")
        .and_then(Value::as_object)
        .ok_or_else(|| failure("runtime receipt has no binding object"))?;
    if binding
        .get("runtime_executable_sha256")
        .and_then(Value::as_str)
        != Some(args.expected_runtime_executable_sha256.as_str())
    {
        return Err(failure(
            "runtime receipt executable SHA-256 does not match --expected-runtime-executable-sha256",
        ));
    }
    Ok(json!({
        "path": args.runtime_receipt,
        "file_sha256": sha256_bytes(&raw),
        "schema": expected_schema,
        "status": expected_status,
        "seal_sha256": args.expected_runtime_receipt_seal_sha256,
        "runtime_executable_sha256": args.expected_runtime_executable_sha256,
        "input_receipt_is_previously_sealed_campaign_authority": true,
        "this_cpu_diagnostic_does_not_reissue_or_promote_the_runtime_receipt": true,
    }))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn projection(artifact: &CompleteBinaryArtifact, name: String) -> Probe<PackedProjection> {
    let tensor = artifact
        .tensor(&name)
        .map_err(|error| failure(error.to_string()))?;
    let payload = artifact
        .verified_tensor_payload(&name)
        .map_err(|error| failure(error.to_string()))?;
    let header =
        parse_complete_binary_header(&payload).map_err(|error| failure(error.to_string()))?;
    if header.group_size != GROUP_SIZE
        || header.shape.as_slice() != [EXPERT_ROWS, HIDDEN_COLS]
        || header.groups != EXPERT_ROWS * (HIDDEN_COLS / GROUP_SIZE)
    {
        return Err(failure(format!(
            "{name} is not the exact Qwen30 expert geometry [768, 2048] at group_size=128"
        )));
    }
    Ok(PackedProjection {
        name,
        payload,
        header,
        artifact_path: tensor.artifact_path.display().to_string(),
        artifact_sha256: tensor.artifact_sha256.clone(),
    })
}

fn direct_value(projection: &PackedProjection, index: usize) -> f32 {
    let group = index / GROUP_SIZE;
    let bit = index % GROUP_SIZE;
    let scale_at = projection.header.scale_offset + group * std::mem::size_of::<u16>();
    let scale = f16::from_bits(u16::from_le_bytes([
        projection.payload[scale_at],
        projection.payload[scale_at + 1],
    ]))
    .to_f32();
    let sign_at = projection.header.sign_offset + group * (GROUP_SIZE / 8) + bit / 8;
    let positive = ((projection.payload[sign_at] >> (bit % 8)) & 1) != 0;
    if positive {
        scale
    } else {
        -scale
    }
}

// Keep the product observable before the addition. This models the scalar
// control source expression rather than silently allowing an explicit FMA.
#[inline(never)]
fn control_add_after_product(sum: f32, weight: f32, input: f32) -> f32 {
    let product = std::hint::black_box(weight * input);
    std::hint::black_box(sum) + product
}

fn control_grouped_nonfused(projection: &PackedProjection, input: &[f32]) -> Vec<f32> {
    (0..projection.header.shape[0])
        .map(|row| {
            let mut sum = 0.0f32;
            let row_base = row * projection.header.shape[1];
            let groups_per_row = projection.header.shape[1] / GROUP_SIZE;
            for group in 0..groups_per_row {
                let start = group * GROUP_SIZE;
                for column in start..start + GROUP_SIZE {
                    sum = control_add_after_product(
                        sum,
                        direct_value(projection, row_base + column),
                        input[column],
                    );
                }
            }
            sum
        })
        .collect()
}

fn flat_nonfused(projection: &PackedProjection, input: &[f32]) -> Vec<f32> {
    (0..projection.header.shape[0])
        .map(|row| {
            let mut sum = 0.0f32;
            let row_base = row * projection.header.shape[1];
            for column in 0..projection.header.shape[1] {
                sum = control_add_after_product(
                    sum,
                    direct_value(projection, row_base + column),
                    input[column],
                );
            }
            sum
        })
        .collect()
}

fn separate_explicit_fma(projection: &PackedProjection, input: &[f32]) -> Vec<f32> {
    (0..projection.header.shape[0])
        .map(|row| {
            let mut sum = 0.0f32;
            let row_base = row * projection.header.shape[1];
            for column in 0..projection.header.shape[1] {
                sum = direct_value(projection, row_base + column).mul_add(input[column], sum);
            }
            sum
        })
        .collect()
}

fn paired_explicit_fma(
    gate: &PackedProjection,
    up: &PackedProjection,
    input: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    (0..gate.header.shape[0])
        .map(|row| {
            let row_base = row * gate.header.shape[1];
            let mut gate_sum = 0.0f32;
            let mut up_sum = 0.0f32;
            for column in 0..gate.header.shape[1] {
                let x = input[column];
                gate_sum = direct_value(gate, row_base + column).mul_add(x, gate_sum);
                up_sum = direct_value(up, row_base + column).mul_add(x, up_sum);
            }
            (gate_sum, up_sum)
        })
        .unzip()
}

fn paired_scalar_order_nonfused(
    gate: &PackedProjection,
    up: &PackedProjection,
    input: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    (0..gate.header.shape[0])
        .map(|row| {
            let row_base = row * gate.header.shape[1];
            let mut gate_sum = 0.0f32;
            let mut up_sum = 0.0f32;
            for column in 0..gate.header.shape[1] {
                let x = input[column];
                gate_sum =
                    control_add_after_product(gate_sum, direct_value(gate, row_base + column), x);
                up_sum = control_add_after_product(up_sum, direct_value(up, row_base + column), x);
            }
            (gate_sum, up_sum)
        })
        .unzip()
}

fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up)
        .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
        .collect()
}

fn compare(left: &[f32], right: &[f32], label: &str) -> DifferenceSummary {
    let mut summary = DifferenceSummary::default();
    for (index, (&expected, &actual)) in left.iter().zip(right).enumerate() {
        summary.values_compared += 1;
        let abs = (expected - actual).abs();
        summary.max_abs = summary.max_abs.max(abs);
        if expected.to_bits() != actual.to_bits() {
            summary.differing_f32_bits += 1;
            if summary.first_mismatch.is_none() {
                summary.first_mismatch = Some(json!({
                    "label": label,
                    "row": index,
                    "left_f32_bits_hex": format!("0x{:08x}", expected.to_bits()),
                    "right_f32_bits_hex": format!("0x{:08x}", actual.to_bits()),
                    "left_decimal": f32_text(expected),
                    "right_decimal": f32_text(actual),
                    "abs_decimal": f32_text(abs),
                }));
            }
        }
    }
    summary
}

fn f32_text(value: f32) -> String {
    format!("{value:.9e}")
}

fn summary_json(summary: DifferenceSummary) -> Value {
    json!({
        "values_compared": summary.values_compared,
        "differing_f32_bits": summary.differing_f32_bits,
        "max_abs_decimal": f32_text(summary.max_abs),
        "max_abs_f32_bits_hex": format!("0x{:08x}", summary.max_abs.to_bits()),
        "first_mismatch": summary.first_mismatch,
    })
}

fn input(profile: &str) -> Vec<f32> {
    match profile {
        "balanced_modulo" => (0..HIDDEN_COLS)
            .map(|index| ((index * 71 % 509) as f32 - 254.0) / 509.0)
            .collect(),
        "alternating_signed" => (0..HIDDEN_COLS)
            .map(|index| {
                let magnitude = 1.0 + ((index * 37 % 97) as f32 / 97.0);
                if index % 2 == 0 {
                    magnitude
                } else {
                    -magnitude
                }
            })
            .collect(),
        "dynamic_cancellation" => (0..HIDDEN_COLS)
            .map(|index| {
                let exponent = ((index * 29 % 17) as i32) - 8;
                let magnitude = 2.0f32.powi(exponent) * (1.0 + (index % 31) as f32 / 31.0);
                if (index / 7) % 2 == 0 {
                    magnitude
                } else {
                    -magnitude
                }
            })
            .collect(),
        _ => unreachable!("fixed profile names"),
    }
}

fn input_sha256(input: &[f32]) -> String {
    let mut raw = Vec::with_capacity(input.len() * std::mem::size_of::<u32>());
    for value in input {
        raw.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    sha256_bytes(&raw)
}

fn atomic_json(path: &Path, document: &Value) -> Probe<()> {
    if path.exists() {
        return Err(failure(format!(
            "refusing to overwrite existing discriminator receipt {}",
            path.display()
        )));
    }
    let parent = path
        .parent()
        .ok_or_else(|| failure("--out has no parent directory"))?;
    fs::create_dir_all(parent)?;
    let temporary = path.with_file_name(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| failure("--out name is not UTF-8"))?,
        std::process::id()
    ));
    fs::write(
        &temporary,
        format!("{}\n", serde_json::to_string_pretty(document)?),
    )?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn sealed(document: Value) -> Probe<Value> {
    let raw = serde_json::to_vec(&document)?;
    let seal = sha256_bytes(&raw);
    let mut object = document
        .as_object()
        .cloned()
        .ok_or_else(|| failure("receipt root is not an object"))?;
    object.insert("seal_sha256".into(), Value::String(seal));
    Ok(Value::Object(object))
}

fn tensor_json(projection: &PackedProjection) -> Value {
    json!({
        "name": projection.name,
        "artifact_path": projection.artifact_path,
        "artifact_sha256": projection.artifact_sha256,
        "payload_bytes": projection.header.payload_bytes,
        "shape": projection.header.shape,
        "groups": projection.header.groups,
        "group_size": projection.header.group_size,
        "payload_is_admission_verified_immutable_snapshot": true,
    })
}

fn run() -> Probe<()> {
    let args = parse_args()?;
    let runtime_binding = read_runtime_binding(&args)?;
    let admission = CompleteBinaryAdmission {
        model: QwenCompleteBinaryModel::Qwen30Coder,
        expected_manifest_seal_sha256: args.expected_manifest_seal_sha256.clone(),
        expected_source_audit_seal_sha256: args.expected_source_audit_seal_sha256.clone(),
        expected_source_revision: args.expected_source_revision.clone(),
    };
    let artifact = admit_complete_binary_artifact(&args.manifest, &admission)
        .map_err(|error| failure(error.to_string()))?;
    if artifact.verified_payload_count() != artifact.tensors.len()
        || !artifact.has_complete_verified_payload_cache()
    {
        return Err(failure(
            "strict full admission did not retain every direct payload as an immutable verified snapshot",
        ));
    }

    let mut sample_documents = Vec::new();
    let mut any_control_vs_fma_difference = false;
    let mut any_fma_pairing_difference = false;
    let mut any_group_vs_flat_nonfused_difference = false;
    let mut any_paired_scalar_order_difference = false;
    for &(layer, expert) in SAMPLES {
        let prefix = format!("model.layers.{layer}.mlp.experts.{expert}");
        let gate = projection(&artifact, format!("{prefix}.gate_proj.weight"))?;
        let up = projection(&artifact, format!("{prefix}.up_proj.weight"))?;
        let mut profile_documents = Vec::new();
        for profile in [
            "balanced_modulo",
            "alternating_signed",
            "dynamic_cancellation",
        ] {
            let values = input(profile);
            let control_gate = control_grouped_nonfused(&gate, &values);
            let control_up = control_grouped_nonfused(&up, &values);
            let flat_gate = flat_nonfused(&gate, &values);
            let flat_up = flat_nonfused(&up, &values);
            let fma_gate = separate_explicit_fma(&gate, &values);
            let fma_up = separate_explicit_fma(&up, &values);
            let (paired_gate, paired_up) = paired_explicit_fma(&gate, &up, &values);
            let (paired_scalar_gate, paired_scalar_up) =
                paired_scalar_order_nonfused(&gate, &up, &values);

            let control_activation = swiglu(&control_gate, &control_up);
            let fma_activation = swiglu(&fma_gate, &fma_up);
            let paired_activation = swiglu(&paired_gate, &paired_up);
            let paired_scalar_activation = swiglu(&paired_scalar_gate, &paired_scalar_up);
            let group_flat_gate = compare(
                &control_gate,
                &flat_gate,
                "grouped_nonfused_gate_vs_flat_nonfused_gate",
            );
            let group_flat_up = compare(
                &control_up,
                &flat_up,
                "grouped_nonfused_up_vs_flat_nonfused_up",
            );
            let precision_gate = compare(
                &control_gate,
                &fma_gate,
                "control_nonfused_gate_vs_explicit_fma_gate",
            );
            let precision_up = compare(
                &control_up,
                &fma_up,
                "control_nonfused_up_vs_explicit_fma_up",
            );
            let precision_activation = compare(
                &control_activation,
                &fma_activation,
                "control_nonfused_swiglu_vs_explicit_fma_swiglu",
            );
            let pairing_gate = compare(
                &fma_gate,
                &paired_gate,
                "separate_explicit_fma_gate_vs_paired_explicit_fma_gate",
            );
            let pairing_up = compare(
                &fma_up,
                &paired_up,
                "separate_explicit_fma_up_vs_paired_explicit_fma_up",
            );
            let pairing_activation = compare(
                &fma_activation,
                &paired_activation,
                "separate_explicit_fma_swiglu_vs_paired_explicit_fma_swiglu",
            );
            let scalar_pairing_gate = compare(
                &control_gate,
                &paired_scalar_gate,
                "scalar_control_gate_vs_paired_scalar_order_gate",
            );
            let scalar_pairing_up = compare(
                &control_up,
                &paired_scalar_up,
                "scalar_control_up_vs_paired_scalar_order_up",
            );
            let scalar_pairing_activation = compare(
                &control_activation,
                &paired_scalar_activation,
                "scalar_control_swiglu_vs_paired_scalar_order_swiglu",
            );
            any_group_vs_flat_nonfused_difference |=
                group_flat_gate.differing_f32_bits > 0 || group_flat_up.differing_f32_bits > 0;
            any_control_vs_fma_difference |= precision_gate.differing_f32_bits > 0
                || precision_up.differing_f32_bits > 0
                || precision_activation.differing_f32_bits > 0;
            any_fma_pairing_difference |= pairing_gate.differing_f32_bits > 0
                || pairing_up.differing_f32_bits > 0
                || pairing_activation.differing_f32_bits > 0;
            any_paired_scalar_order_difference |= scalar_pairing_gate.differing_f32_bits > 0
                || scalar_pairing_up.differing_f32_bits > 0
                || scalar_pairing_activation.differing_f32_bits > 0;
            profile_documents.push(json!({
                "profile": profile,
                "input_f32_count": values.len(),
                "input_f32_bits_sha256": input_sha256(&values),
                "comparisons": {
                    "source_group_loop_vs_flat_same_nonfused_arithmetic": {
                        "gate": summary_json(group_flat_gate),
                        "up": summary_json(group_flat_up),
                    },
                    "scalar_control_nonfused_vs_explicit_fma": {
                        "gate": summary_json(precision_gate),
                        "up": summary_json(precision_up),
                        "swiglu": summary_json(precision_activation),
                    },
                    "separate_explicit_fma_vs_paired_gate_up_explicit_fma": {
                        "gate": summary_json(pairing_gate),
                        "up": summary_json(pairing_up),
                        "swiglu": summary_json(pairing_activation),
                    },
                    "scalar_control_vs_paired_scalar_order_nonfused": {
                        "gate": summary_json(scalar_pairing_gate),
                        "up": summary_json(scalar_pairing_up),
                        "swiglu": summary_json(scalar_pairing_activation),
                    },
                },
            }));
        }
        sample_documents.push(json!({
            "layer": layer,
            "expert": expert,
            "gate": tensor_json(&gate),
            "up": tensor_json(&up),
            "profiles": profile_documents,
        }));
    }
    let outcome = if any_control_vs_fma_difference
        && !any_fma_pairing_difference
        && !any_paired_scalar_order_difference
    {
        "PRECISION_CONTRACTION_DIFFERENCE_OBSERVED_PAIRED_SCALAR_ORDER_CPU_EXACT"
    } else if any_control_vs_fma_difference && any_fma_pairing_difference {
        "PRECISION_AND_PAIRED_LOOP_ARITHMETIC_DIFFERENCE_OBSERVED"
    } else if !any_control_vs_fma_difference && !any_fma_pairing_difference {
        "NO_BOUNDED_CPU_PRECISION_OR_PAIRED_LOOP_ARITHMETIC_DIFFERENCE_OBSERVED"
    } else {
        "PAIRED_LOOP_ARITHMETIC_DIFFERENCE_OBSERVED_WITHOUT_CONTROL_FMA_DIFFERENCE"
    };
    let source_control_shader = include_str!("../shaders/qwen_binary.metal");
    let source_fused_shader =
        include_str!("../shaders/qwen_direct_packed_gate_up_swiglu_fused.metal");
    let source_paired_scalar_order_shader =
        include_str!("../shaders/qwen_direct_packed_gate_up_swiglu_paired_scalar_order.metal");
    let executable = env::current_exe()?;
    let document = json!({
        "schema": SCHEMA,
        "status": STATUS,
        "outcome": outcome,
        "binding": {
            "model_id": MODEL_ID,
            "manifest_path": args.manifest,
            "manifest_seal_sha256": artifact.manifest_seal_sha256,
            "source_audit_seal_sha256": artifact.source_audit_seal_sha256,
            "source_revision": artifact.source_revision,
            "admitted_tensor_count": artifact.tensors.len(),
            "verified_payload_count": artifact.verified_payload_count(),
            "complete_verified_payload_cache_at_admission": artifact.has_complete_verified_payload_cache(),
            "runtime": runtime_binding,
            "discriminator_executable_path": executable,
            "discriminator_executable_sha256": file_sha256(&executable)?,
            "source_hashes": {
                "scalar_control_shader_qwen_binary_metal": source_sha256(source_control_shader),
                "fused_gate_up_shader": source_sha256(source_fused_shader),
                "paired_scalar_order_gate_up_shader": source_sha256(source_paired_scalar_order_shader),
                "discriminator_source_embedded_at_build": source_sha256(include_str!("ascension_qwen30_direct_packed_gate_up_precision_order_discriminator.rs")),
            },
        },
        "method": {
            "exact_direct_payload_layout": "HQ30G1B1 group_size=128; LSB-first signs plus FP16 group scales",
            "selected_actual_artifact_tensors": "early and late actual Qwen30 routed expert gate/up pairs; all 768 rows each",
            "control_arithmetic": "group-major scalar source ordering with product made observable before f32 addition; models qwen_binary.metal source-level sum += weight * input",
            "flat_same_arithmetic": "flat column loop with the same non-fused operations; distinguishes loop grouping from arithmetic contraction",
            "explicit_fma_arithmetic": "f32::mul_add for every packed direct weight/input contribution; models fused shader explicit fma",
            "paired_explicit_fma": "gate and up fma reductions interleaved in one row loop; isolates paired reduction scheduling from fma arithmetic",
            "paired_scalar_order_nonfused": "gate and up reductions interleaved in one row loop while each contribution retains the control's observable product then f32 addition recurrence",
            "input_profiles": ["balanced_modulo", "alternating_signed", "dynamic_cancellation"],
            "no_dense_weight_materialization": true,
            "no_metal_device_or_command_buffer": true,
            "no_mps_or_raw_bf16_model_open": true,
            "no_qwen_layer_token_generation_endpoint_or_timing": true,
        },
        "observations": {
            "group_loop_vs_flat_nonfused_difference_observed": any_group_vs_flat_nonfused_difference,
            "control_nonfused_vs_explicit_fma_difference_observed": any_control_vs_fma_difference,
            "separate_vs_paired_explicit_fma_difference_observed": any_fma_pairing_difference,
            "scalar_control_vs_paired_scalar_order_nonfused_difference_observed": any_paired_scalar_order_difference,
            "interpretation": "A CPU difference isolates source-level arithmetic/order plausibility only. It cannot prove a Metal command-topology cause, promote the fused candidate, or replace a current-binding full-token device parity decision.",
        },
        "samples": sample_documents,
        "next_gate": {
            "required_before_any_qwen30_gpu_candidate": "Q80 bounded layer-0 native Metal parity must finish and explicitly release the shared lease",
            "if_paired_scalar_order_cpu_parity_is_exact": "a separately named Metal paired-topology candidate may be compiled, but it must preserve scalar arithmetic with its precise contract and prove fresh current-binding all-layer device/template parity",
            "if_precision_difference_observed": "next device trial must preserve scalar arithmetic or make its contraction/precision policy explicit and prove fresh current-binding all-layer template parity",
            "if_no_cpu_difference_observed": "next device trial may focus on command topology/state synchronization while retaining scalar control and fresh current-binding all-layer parity",
        },
        "claim_boundary": {
            "cpu_only_component_diagnostic": true,
            "not_a_complete_layer_token_or_generation_result": true,
            "does_not_select_or_serve_a_kernel": true,
            "does_not_claim_hcli_tps_tg_capability_coherence_or_tournament": true,
        },
    });
    let document = sealed(document)?;
    atomic_json(&args.out, &document)?;
    println!("{}", serde_json::to_string(&document)?);
    Ok(())
}

fn main() -> Probe<()> {
    run()
}

#[cfg(test)]
mod tests {
    use super::{compare, control_add_after_product};

    #[test]
    fn comparison_keeps_f32_bitwise_equality_distinct_from_decimal_equality() {
        let equal = compare(&[1.0], &[1.0], "equal");
        assert_eq!(equal.differing_f32_bits, 0);
        let negative_zero = compare(&[0.0], &[-0.0], "signed-zero");
        assert_eq!(negative_zero.differing_f32_bits, 1);
    }

    #[test]
    fn control_helper_returns_finite_for_finite_operands() {
        assert!(control_add_after_product(0.25, -0.5, 0.75).is_finite());
    }
}
