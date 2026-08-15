//! Native, fail-closed admission entrypoint for Qwen complete-binary artifacts.
//!
//! This binary deliberately does one job: it calls Hawking Core's strict
//! complete-binary admission reader and emits a small machine-readable result
//! only after every tensor catalog entry, payload hash, direct-layout header,
//! source-index binding, and current source-file identity has passed.  It is
//! not a Qwen decoder, runtime, HCLI server, capability test, or TPS probe.

use hawking_core::model::qwen_complete_binary::{
    admit_complete_binary_artifact, CompleteBinaryAdmission, QwenCompleteBinaryModel,
};
use serde_json::json;
use std::env;
use std::path::PathBuf;
use std::process;

const RESULT_SCHEMA: &str = "hawking.ascension.qwen_complete_binary_native_admission_result.v1";
const RESULT_STATUS: &str =
    "EARNED_COMPLETE_BINARY_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED";

struct Arguments {
    model: QwenCompleteBinaryModel,
    model_key: &'static str,
    manifest: PathBuf,
    expected_manifest_seal_sha256: String,
    expected_source_audit_seal_sha256: String,
    expected_source_revision: String,
}

fn usage() -> &'static str {
    "usage: ascension_qwen_complete_binary_admission \\
        --model qwen30|qwen80 \\
        --manifest ABSOLUTE_PATH \\
        --expected-manifest-seal-sha256 SHA256 \\
        --expected-source-audit-seal-sha256 SHA256 \\
        --expected-source-revision REVISION"
}

fn parse_arguments() -> Result<Arguments, String> {
    let mut model = None;
    let mut manifest = None;
    let mut expected_manifest_seal_sha256 = None;
    let mut expected_source_audit_seal_sha256 = None;
    let mut expected_source_revision = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag:?}; {}", usage()))?;
        match flag.as_str() {
            "--model" => {
                let value = match value.as_str() {
                    "qwen30" => (QwenCompleteBinaryModel::Qwen30Coder, "qwen30"),
                    "qwen80" => (QwenCompleteBinaryModel::Qwen80CoderNext, "qwen80"),
                    _ => return Err(format!("unsupported --model {value:?}; {}", usage())),
                };
                if model.replace(value).is_some() {
                    return Err(format!("--model was supplied more than once; {}", usage()));
                }
            }
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
            _ => return Err(format!("unsupported option {flag:?}; {}", usage())),
        }
    }
    let (model, model_key) = model.ok_or_else(|| format!("missing --model; {}", usage()))?;
    let manifest = manifest.ok_or_else(|| format!("missing --manifest; {}", usage()))?;
    if !manifest.is_absolute() {
        return Err("--manifest must be an absolute path".into());
    }
    Ok(Arguments {
        model,
        model_key,
        manifest,
        expected_manifest_seal_sha256: expected_manifest_seal_sha256
            .ok_or_else(|| format!("missing --expected-manifest-seal-sha256; {}", usage()))?,
        expected_source_audit_seal_sha256: expected_source_audit_seal_sha256
            .ok_or_else(|| format!("missing --expected-source-audit-seal-sha256; {}", usage()))?,
        expected_source_revision: expected_source_revision
            .ok_or_else(|| format!("missing --expected-source-revision; {}", usage()))?,
    })
}

fn fail(detail: impl AsRef<str>) -> ! {
    eprintln!("complete-binary admission refused: {}", detail.as_ref());
    process::exit(2);
}

fn main() {
    let arguments = parse_arguments().unwrap_or_else(|error| fail(error));
    let admission = CompleteBinaryAdmission {
        model: arguments.model,
        expected_manifest_seal_sha256: arguments.expected_manifest_seal_sha256.clone(),
        expected_source_audit_seal_sha256: arguments.expected_source_audit_seal_sha256.clone(),
        expected_source_revision: arguments.expected_source_revision.clone(),
    };
    let artifact = admit_complete_binary_artifact(&arguments.manifest, &admission)
        .unwrap_or_else(|error| fail(error.to_string()));
    let result = json!({
        "schema": RESULT_SCHEMA,
        "status": RESULT_STATUS,
        "model": arguments.model_key,
        "manifest_path": artifact.manifest_path,
        "manifest_seal_sha256": artifact.manifest_seal_sha256,
        "source_audit_path": artifact.source_audit_path,
        "source_audit_seal_sha256": artifact.source_audit_seal_sha256,
        "source_revision": artifact.source_revision,
        "source_index_path": artifact.source_index_path,
        "tensor_count": artifact.tensors.len(),
        "source_weight_elements": artifact.source_weight_elements,
        "tensor_payload_bytes": artifact.tensor_payload_bytes,
        "claim_boundary": {
            "strict_complete_artifact_and_source_chain_admission_only": true,
            "not_native_qwen_decoder_or_token_runtime": true,
            "not_capability_hcli_tps_tg_or_tournament_qualification": true,
        },
    });
    println!(
        "{}",
        serde_json::to_string(&result).expect("admission result must serialize")
    );
}
