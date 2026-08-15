//! Source/artifact-bound Qwen80 L0 DeltaNet first-residual CPU witness.
//!
//! This binary is intentionally a CPU/reference capture only.  It admits the
//! complete compact artifact once, derives one source-token hidden vector
//! directly from `model.embed_tokens.weight`, runs the existing canonical L0
//! Gated DeltaNet CPU oracle, and durably retains the exact `[2048]`
//! first-residual bytes and hashes.  A later strict-Metal child must consume
//! this input/state provenance and prove device parity before an outer
//! component capture may use the result as the first-residual input for the
//! all-ten routed-MoE bridge.
//!
//! No Metal context, shader, watcher, server, HCLI, token loop, benchmark, or
//! TPS claim exists here.

use hawking_core::model::qwen80_complete_runtime::{
    Qwen80CanonicalLinearLayerCpuInput, Qwen80CompleteArtifactCatalog,
};
use hawking_core::model::qwen_complete_binary::{CompleteBinaryAdmission, QwenCompleteBinaryModel};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process;

const RESULT_SCHEMA: &str = "hawking.ascension.qwen80_first_residual_bridge_inner.v1";
const RESULT_STATUS: &str =
    "EARNED_CURRENT_ADMITTED_QWEN80_LAYER0_FIRST_RESIDUAL_CPU_ORACLE_BASELINE_METAL_LEASE_REQUIRED";
const HIDDEN: usize = 2_048;
const FIRST_RESIDUAL_FILENAME: &str = "first-residual.f32le";
const INPUT_HIDDEN_FILENAME: &str = "input-hidden.f32le";

#[derive(Debug)]
struct Args {
    manifest: PathBuf,
    expected_manifest_seal_sha256: String,
    expected_source_audit_seal_sha256: String,
    expected_source_revision: String,
    token_id: u32,
    capture_dir: PathBuf,
}

struct Witness {
    manifest_document_sha256: String,
    token_id: u32,
    embedding_tensor: String,
    input_hidden: Vec<f32>,
    initial_conv_state: Vec<f32>,
    initial_recurrent_state: Vec<f32>,
    first_residual: Vec<f32>,
    mixer_tensor_names: Value,
    source_algorithm_boundary: String,
}

fn usage() -> &'static str {
    "usage: ascension_qwen80_first_residual_bridge_witness \\
        --manifest ABSOLUTE_PATH \\
        --expected-manifest-seal-sha256 SHA256 \\
        --expected-source-audit-seal-sha256 SHA256 \\
        --expected-source-revision REVISION \\
        --token-id NON_RESERVED_TOKEN_ID \\
        --capture-dir NEW_ABSOLUTE_DIRECTORY"
}

fn parse_args() -> Result<Args, String> {
    let mut manifest = None;
    let mut expected_manifest_seal_sha256 = None;
    let mut expected_source_audit_seal_sha256 = None;
    let mut expected_source_revision = None;
    let mut token_id = None;
    let mut capture_dir = None;
    let mut values = env::args().skip(1);
    while let Some(flag) = values.next() {
        let value = values
            .next()
            .ok_or_else(|| format!("missing value for {flag:?}; {}", usage()))?;
        match flag.as_str() {
            "--manifest" => {
                if manifest.replace(PathBuf::from(value)).is_some() {
                    return Err(format!(
                        "--manifest was supplied more than once; {}",
                        usage()
                    ));
                }
            }
            "--expected-manifest-seal-sha256" => {
                if expected_manifest_seal_sha256.replace(value).is_some() {
                    return Err(format!(
                        "--expected-manifest-seal-sha256 was supplied more than once; {}",
                        usage()
                    ));
                }
            }
            "--expected-source-audit-seal-sha256" => {
                if expected_source_audit_seal_sha256.replace(value).is_some() {
                    return Err(format!(
                        "--expected-source-audit-seal-sha256 was supplied more than once; {}",
                        usage()
                    ));
                }
            }
            "--expected-source-revision" => {
                if expected_source_revision.replace(value).is_some() {
                    return Err(format!(
                        "--expected-source-revision was supplied more than once; {}",
                        usage()
                    ));
                }
            }
            "--token-id" => {
                if token_id.replace(value).is_some() {
                    return Err(format!(
                        "--token-id was supplied more than once; {}",
                        usage()
                    ));
                }
            }
            "--capture-dir" => {
                if capture_dir.replace(PathBuf::from(value)).is_some() {
                    return Err(format!(
                        "--capture-dir was supplied more than once; {}",
                        usage()
                    ));
                }
            }
            _ => return Err(format!("unsupported option {flag:?}; {}", usage())),
        }
    }
    let absolute = |path: Option<PathBuf>, label: &str| -> Result<PathBuf, String> {
        let path = path.ok_or_else(|| format!("missing {label}; {}", usage()))?;
        if !path.is_absolute() {
            return Err(format!("{label} must be absolute"));
        }
        Ok(path)
    };
    let capture_dir = absolute(capture_dir, "--capture-dir")?;
    if !capture_dir.parent().is_some_and(Path::is_dir) {
        return Err("--capture-dir parent must already exist".into());
    }
    let token_id = token_id
        .ok_or_else(|| format!("missing --token-id; {}", usage()))?
        .parse::<u32>()
        .map_err(|_| format!("--token-id must be an unsigned integer; {}", usage()))?;
    Ok(Args {
        manifest: absolute(manifest, "--manifest")?,
        expected_manifest_seal_sha256: expected_manifest_seal_sha256
            .ok_or_else(|| format!("missing --expected-manifest-seal-sha256; {}", usage()))?,
        expected_source_audit_seal_sha256: expected_source_audit_seal_sha256
            .ok_or_else(|| format!("missing --expected-source-audit-seal-sha256; {}", usage()))?,
        expected_source_revision: expected_source_revision
            .ok_or_else(|| format!("missing --expected-source-revision; {}", usage()))?,
        token_id,
        capture_dir,
    })
}

fn regular_bytes(path: &Path, label: &str) -> Result<Vec<u8>, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    fs::read(path).map_err(|error| format!("cannot read {label} {}: {error}", path.display()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn f32le_bytes(values: &[f32]) -> Result<Vec<u8>, String> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err("f32 witness contains no values or a non-finite value".into());
    }
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    Ok(bytes)
}

fn require_vector(values: &[f32], elements: usize, label: &str) -> Result<(), String> {
    if values.len() != elements {
        return Err(format!(
            "{label} has {} elements; expected {elements}",
            values.len()
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{label} contains a non-finite value"));
    }
    Ok(())
}

fn file_evidence(path: &Path, bytes: &[u8]) -> Value {
    json!({
        "path": path,
        "present": true,
        "bytes": bytes.len(),
        "sha256": sha256_hex(bytes),
    })
}

fn write_new(capture_dir: &Path, name: &str, bytes: &[u8]) -> Result<PathBuf, String> {
    let path = capture_dir.join(name);
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&path)
        .map_err(|error| format!("cannot create capture file {}: {error}", path.display()))?;
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .map_err(|error| {
            format!(
                "cannot durably write capture file {}: {error}",
                path.display()
            )
        })?;
    Ok(path)
}

fn run(args: &Args) -> Result<Witness, String> {
    let manifest_bytes = regular_bytes(&args.manifest, "complete manifest")?;
    let admission = CompleteBinaryAdmission {
        model: QwenCompleteBinaryModel::Qwen80CoderNext,
        expected_manifest_seal_sha256: args.expected_manifest_seal_sha256.clone(),
        expected_source_audit_seal_sha256: args.expected_source_audit_seal_sha256.clone(),
        expected_source_revision: args.expected_source_revision.clone(),
    };
    // Exactly one strict admission precedes both source embedding lookup and
    // the DeltaNet oracle.  This is not a raw/BF16/MPS fallback.
    let catalog = Qwen80CompleteArtifactCatalog::load(&args.manifest, &admission)
        .map_err(|error| error.to_string())?;
    let embedding = catalog
        .execute_embedding_lookup_cpu_oracle(args.token_id)
        .map_err(|error| error.to_string())?;
    require_vector(&embedding.hidden, HIDDEN, "source embedding hidden")?;
    let input = Qwen80CanonicalLinearLayerCpuInput::with_zero_state(embedding.hidden.clone());
    let oracle = catalog
        .execute_first_linear_layer_cpu_oracle(&input)
        .map_err(|error| error.to_string())?;
    if oracle.layer != 0 || oracle.linear_state_slot != 0 {
        return Err("canonical DeltaNet oracle did not bind layer/state slot zero".into());
    }
    require_vector(&oracle.mixer_residual_output, HIDDEN, "first residual")?;
    Ok(Witness {
        manifest_document_sha256: sha256_hex(&manifest_bytes),
        token_id: args.token_id,
        embedding_tensor: embedding.direct_packed_embedding_tensor,
        input_hidden: input.hidden,
        initial_conv_state: input.state.conv_state,
        initial_recurrent_state: input.state.recurrent_state,
        first_residual: oracle.mixer_residual_output,
        mixer_tensor_names: json!({
            "input_layernorm": oracle.direct_packed_input_layernorm_tensor,
            "qkvz": oracle.direct_packed_qkvz_tensor,
            "ba": oracle.direct_packed_ba_tensor,
            "conv": oracle.direct_packed_conv_tensor,
            "gated_norm": oracle.direct_packed_gated_norm_tensor,
            "out_proj": oracle.direct_packed_out_proj_tensor,
        }),
        source_algorithm_boundary: oracle.source_algorithm_boundary,
    })
}

fn result_document(
    args: &Args,
    witness: &Witness,
    input_path: &Path,
    residual_path: &Path,
) -> Result<Value, String> {
    let input_bytes = f32le_bytes(&witness.input_hidden)?;
    let residual_bytes = f32le_bytes(&witness.first_residual)?;
    let conv_bytes = f32le_bytes(&witness.initial_conv_state)?;
    let recurrent_bytes = f32le_bytes(&witness.initial_recurrent_state)?;
    if residual_bytes.len() != HIDDEN * std::mem::size_of::<f32>() {
        return Err("first-residual byte geometry drifted".into());
    }
    Ok(json!({
        "schema": RESULT_SCHEMA,
        "status": RESULT_STATUS,
        "mode": "cpu-oracle",
        "metal_device_or_dispatch_performed": false,
        "component_only": true,
        "complete_layer_or_token_performed": false,
        "artifact_binding": {
            "manifest_path": args.manifest,
            "manifest_document_sha256": witness.manifest_document_sha256,
            "manifest_seal_sha256": args.expected_manifest_seal_sha256,
            "source_audit_seal_sha256": args.expected_source_audit_seal_sha256,
            "source_revision": args.expected_source_revision,
            "layer": 0,
            "linear_state_slot": 0,
            "admission_scan_performed_once_before_catalog_reuse": true,
            "direct_packed_payloads_only": true,
        },
        "same_input_provenance": {
            "kind": "source_direct_packed_embedding_with_zeroed_layer0_deltanet_state",
            "token_id": witness.token_id,
            "embedding_tensor": witness.embedding_tensor,
            "input_hidden": file_evidence(input_path, &input_bytes),
            "input_hidden_f32le_sha256": sha256_hex(&input_bytes),
            "initial_conv_state": {
                "elements": witness.initial_conv_state.len(),
                "f32le_sha256": sha256_hex(&conv_bytes),
                "zero_initialized": true,
            },
            "initial_recurrent_state": {
                "elements": witness.initial_recurrent_state.len(),
                "f32le_sha256": sha256_hex(&recurrent_bytes),
                "zero_initialized": true,
            },
            "future_strict_metal_child_must_retain_exact_input_and_state_identity": true,
        },
        "first_residual_output": {
            "layer": 0,
            "linear_state_slot": 0,
            "elements": HIDDEN,
            "bytes": residual_bytes.len(),
            "f32le_sha256": sha256_hex(&residual_bytes),
            "sha256": sha256_hex(&residual_bytes),
            "file": file_evidence(residual_path, &residual_bytes),
            "producer": "source_direct_packed_layer0_deltanet_cpu_oracle_after_out_projection_plus_input_residual",
            "same_command_graph_required_for_future_strict_metal_bridge": true,
        },
        "mixer_direct_packed_tensors": witness.mixer_tensor_names,
        "source_algorithm_boundary": witness.source_algorithm_boundary,
        "durable_capture": {
            "input_hidden_written_before_receipt": true,
            "first_residual_written_before_receipt": true,
            "receipt_written_last_is_completion_marker": true,
            "outer_reaped_strict_metal_capture_required_before_any_device_or_layer_promotion": true,
        },
        "future_device_bridge": {
            "prepared_not_executed": true,
            "requires_same_command_graph_retention_of_device_first_residual_buffer": true,
            "requires_device_readback_parity_against_this_exact_f32le_reference": true,
            "requires_fresh_component_only_quiet_lease_and_outer_reaping": true,
        },
        "claim_boundary": {
            "cpu_reference_only_not_a_production_fallback": true,
            "no_metal_or_gpu_device_execution": true,
            "no_postnorm_router_routed_experts_shared_expert_or_second_residual": true,
            "not_a_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_result": true,
        },
    }))
}

fn failure_document(error: &str) -> Value {
    json!({
        "schema": RESULT_SCHEMA,
        "status": "REFUSED_QWEN80_LAYER0_FIRST_RESIDUAL_CPU_ORACLE_BASELINE",
        "mode": "cpu-oracle",
        "error": error,
        "claim_boundary": "no component, layer, token, or device result is earned on refusal",
    })
}

fn main() {
    let args = parse_args().unwrap_or_else(|error| {
        eprintln!("Qwen80 first-residual bridge witness refused: {error}");
        process::exit(2);
    });
    fs::create_dir(&args.capture_dir).unwrap_or_else(|error| {
        eprintln!(
            "Qwen80 first-residual bridge witness refused to create exclusive capture {}: {error}",
            args.capture_dir.display()
        );
        process::exit(2);
    });
    let invocation = json!({
        "schema": RESULT_SCHEMA,
        "status": "STARTED_QWEN80_LAYER0_FIRST_RESIDUAL_CPU_ORACLE_BASELINE",
        "pid": process::id(),
        "mode": "cpu-oracle",
        "metal_or_gpu_allowed": false,
        "manifest": args.manifest,
        "token_id": args.token_id,
        "claim_boundary": "CPU/source-reference capture only",
    });
    write_new(
        &args.capture_dir,
        "invocation.json",
        &serde_json::to_vec_pretty(&invocation).expect("invocation serializes"),
    )
    .unwrap_or_else(|error| {
        eprintln!("Qwen80 first-residual bridge witness refused: {error}");
        process::exit(2);
    });

    match run(&args) {
        Ok(witness) => {
            let input_bytes =
                f32le_bytes(&witness.input_hidden).expect("validated input remains finite");
            let residual_bytes =
                f32le_bytes(&witness.first_residual).expect("validated residual remains finite");
            let input_path = write_new(&args.capture_dir, INPUT_HIDDEN_FILENAME, &input_bytes)
                .unwrap_or_else(|error| {
                    eprintln!("Qwen80 first-residual bridge witness refused: {error}");
                    process::exit(2);
                });
            let residual_path =
                write_new(&args.capture_dir, FIRST_RESIDUAL_FILENAME, &residual_bytes)
                    .unwrap_or_else(|error| {
                        eprintln!("Qwen80 first-residual bridge witness refused: {error}");
                        process::exit(2);
                    });
            let result = result_document(&args, &witness, &input_path, &residual_path)
                .unwrap_or_else(|error| {
                    eprintln!("Qwen80 first-residual bridge witness refused: {error}");
                    process::exit(2);
                });
            let stdout = serde_json::to_vec_pretty(&result).expect("result serializes");
            write_new(&args.capture_dir, "stdout.jsonl", &stdout).unwrap_or_else(|error| {
                eprintln!("Qwen80 first-residual bridge witness refused: {error}");
                process::exit(2);
            });
            write_new(&args.capture_dir, "stderr.log", b"").unwrap_or_else(|error| {
                eprintln!("Qwen80 first-residual bridge witness refused: {error}");
                process::exit(2);
            });
            // The receipt is intentionally written last.  No future wrapper
            // may accept an incomplete capture without this completion marker.
            write_new(&args.capture_dir, "receipt.json", &stdout).unwrap_or_else(|error| {
                eprintln!("Qwen80 first-residual bridge witness refused: {error}");
                process::exit(2);
            });
            println!("{}", String::from_utf8_lossy(&stdout));
        }
        Err(error) => {
            let failure = failure_document(&error);
            let bytes = serde_json::to_vec_pretty(&failure).expect("failure serializes");
            let _ = write_new(&args.capture_dir, "stderr.log", &bytes);
            let _ = write_new(&args.capture_dir, "receipt.json", &bytes);
            eprintln!("Qwen80 first-residual bridge witness refused: {error}");
            process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{f32le_bytes, require_vector, HIDDEN};

    #[test]
    fn f32le_encoding_is_bit_exact_and_little_endian() {
        assert_eq!(
            f32le_bytes(&[1.0, -2.5]).unwrap(),
            vec![0, 0, 128, 63, 0, 0, 32, 192]
        );
    }

    #[test]
    fn first_residual_requires_exact_hidden_geometry() {
        assert!(require_vector(&vec![0.0; HIDDEN], HIDDEN, "first residual").is_ok());
        assert!(require_vector(&vec![0.0; HIDDEN - 1], HIDDEN, "first residual").is_err());
    }

    #[test]
    fn first_residual_rejects_nonfinite_values() {
        let mut values = vec![0.0; HIDDEN];
        values[127] = f32::NAN;
        assert!(require_vector(&values, HIDDEN, "first residual").is_err());
        assert!(f32le_bytes(&values).is_err());
    }
}
