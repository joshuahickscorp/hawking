//! Strict native admission entrypoint for the isolated Qwen30 gate/up quality
//! candidate.  This is an artifact/catalog scanner only: it never opens a
//! decoder, server, benchmark, or tournament surface.

use hawking_core::model::qwen_complete_binary::{
    admit_qwen30_quality_repack_artifact, Qwen30QualityRepackAdmission,
    QWEN30_QUALITY_REPACK_PAYLOAD_VERIFY_MODE,
};
use serde_json::json;
use std::env;
use std::path::PathBuf;
use std::process;

const RESULT_SCHEMA: &str = "hawking.ascension.qwen30_quality_repack_native_admission_result.v1";
const RESULT_STATUS: &str =
    "EARNED_QUALITY_REPACK_COMPLETE_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED";

struct Arguments {
    manifest: PathBuf,
    expected_manifest_seal_sha256: String,
    expected_source_audit_seal_sha256: String,
    expected_source_revision: String,
    expected_revalidation_path: PathBuf,
    expected_revalidation_seal_sha256: String,
    expected_selection_path: PathBuf,
    expected_selection_seal_sha256: String,
    expected_source_snapshot_path: PathBuf,
    expected_source_snapshot_seal_sha256: String,
    expected_terminal_path: PathBuf,
    expected_terminal_seal_sha256: String,
}

fn usage() -> &'static str {
    "usage: ascension_qwen30_quality_repack_admission \\
        --manifest ABSOLUTE_PATH \\
        --expected-manifest-seal-sha256 SHA256 \\
        --expected-source-audit-seal-sha256 SHA256 \\
        --expected-source-revision REVISION \\
        --expected-revalidation-path ABSOLUTE_PATH \\
        --expected-revalidation-seal-sha256 SHA256 \\
        --expected-selection-path ABSOLUTE_PATH \\
        --expected-selection-seal-sha256 SHA256 \\
        --expected-source-snapshot-path ABSOLUTE_PATH \\
        --expected-source-snapshot-seal-sha256 SHA256 \\
        --expected-terminal-path ABSOLUTE_PATH \\
        --expected-terminal-seal-sha256 SHA256"
}

fn required_absolute_path(value: Option<PathBuf>, flag: &str) -> Result<PathBuf, String> {
    let value = value.ok_or_else(|| format!("missing {flag}; {}", usage()))?;
    if !value.is_absolute() {
        return Err(format!("{flag} must be an absolute path"));
    }
    Ok(value)
}

fn required(value: Option<String>, flag: &str) -> Result<String, String> {
    value.ok_or_else(|| format!("missing {flag}; {}", usage()))
}

fn parse_arguments() -> Result<Arguments, String> {
    let mut manifest = None;
    let mut expected_manifest_seal_sha256 = None;
    let mut expected_source_audit_seal_sha256 = None;
    let mut expected_source_revision = None;
    let mut expected_revalidation_path = None;
    let mut expected_revalidation_seal_sha256 = None;
    let mut expected_selection_path = None;
    let mut expected_selection_seal_sha256 = None;
    let mut expected_source_snapshot_path = None;
    let mut expected_source_snapshot_seal_sha256 = None;
    let mut expected_terminal_path = None;
    let mut expected_terminal_seal_sha256 = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args
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
            "--expected-revalidation-path" => {
                if expected_revalidation_path
                    .replace(PathBuf::from(value))
                    .is_some()
                {
                    return Err(format!(
                        "--expected-revalidation-path was supplied more than once; {}",
                        usage()
                    ));
                }
            }
            "--expected-revalidation-seal-sha256" => {
                if expected_revalidation_seal_sha256.replace(value).is_some() {
                    return Err(format!(
                        "--expected-revalidation-seal-sha256 was supplied more than once; {}",
                        usage()
                    ));
                }
            }
            "--expected-selection-path" => {
                if expected_selection_path
                    .replace(PathBuf::from(value))
                    .is_some()
                {
                    return Err(format!(
                        "--expected-selection-path was supplied more than once; {}",
                        usage()
                    ));
                }
            }
            "--expected-selection-seal-sha256" => {
                if expected_selection_seal_sha256.replace(value).is_some() {
                    return Err(format!(
                        "--expected-selection-seal-sha256 was supplied more than once; {}",
                        usage()
                    ));
                }
            }
            "--expected-source-snapshot-path" => {
                if expected_source_snapshot_path
                    .replace(PathBuf::from(value))
                    .is_some()
                {
                    return Err(format!(
                        "--expected-source-snapshot-path was supplied more than once; {}",
                        usage()
                    ));
                }
            }
            "--expected-source-snapshot-seal-sha256" => {
                if expected_source_snapshot_seal_sha256
                    .replace(value)
                    .is_some()
                {
                    return Err(format!(
                        "--expected-source-snapshot-seal-sha256 was supplied more than once; {}",
                        usage()
                    ));
                }
            }
            "--expected-terminal-path" => {
                if expected_terminal_path
                    .replace(PathBuf::from(value))
                    .is_some()
                {
                    return Err(format!(
                        "--expected-terminal-path was supplied more than once; {}",
                        usage()
                    ));
                }
            }
            "--expected-terminal-seal-sha256" => {
                if expected_terminal_seal_sha256.replace(value).is_some() {
                    return Err(format!(
                        "--expected-terminal-seal-sha256 was supplied more than once; {}",
                        usage()
                    ));
                }
            }
            _ => return Err(format!("unsupported option {flag:?}; {}", usage())),
        }
    }
    Ok(Arguments {
        manifest: required_absolute_path(manifest, "--manifest")?,
        expected_manifest_seal_sha256: required(
            expected_manifest_seal_sha256,
            "--expected-manifest-seal-sha256",
        )?,
        expected_source_audit_seal_sha256: required(
            expected_source_audit_seal_sha256,
            "--expected-source-audit-seal-sha256",
        )?,
        expected_source_revision: required(expected_source_revision, "--expected-source-revision")?,
        expected_revalidation_path: required_absolute_path(
            expected_revalidation_path,
            "--expected-revalidation-path",
        )?,
        expected_revalidation_seal_sha256: required(
            expected_revalidation_seal_sha256,
            "--expected-revalidation-seal-sha256",
        )?,
        expected_selection_path: required_absolute_path(
            expected_selection_path,
            "--expected-selection-path",
        )?,
        expected_selection_seal_sha256: required(
            expected_selection_seal_sha256,
            "--expected-selection-seal-sha256",
        )?,
        expected_source_snapshot_path: required_absolute_path(
            expected_source_snapshot_path,
            "--expected-source-snapshot-path",
        )?,
        expected_source_snapshot_seal_sha256: required(
            expected_source_snapshot_seal_sha256,
            "--expected-source-snapshot-seal-sha256",
        )?,
        expected_terminal_path: required_absolute_path(
            expected_terminal_path,
            "--expected-terminal-path",
        )?,
        expected_terminal_seal_sha256: required(
            expected_terminal_seal_sha256,
            "--expected-terminal-seal-sha256",
        )?,
    })
}

fn fail(detail: impl AsRef<str>) -> ! {
    eprintln!(
        "quality-repack native admission refused: {}",
        detail.as_ref()
    );
    process::exit(2);
}

fn main() {
    let arguments = parse_arguments().unwrap_or_else(|error| fail(error));
    let admission = Qwen30QualityRepackAdmission {
        expected_manifest_seal_sha256: arguments.expected_manifest_seal_sha256,
        expected_source_audit_seal_sha256: arguments.expected_source_audit_seal_sha256,
        expected_source_revision: arguments.expected_source_revision,
        expected_revalidation_path: arguments.expected_revalidation_path,
        expected_revalidation_seal_sha256: arguments.expected_revalidation_seal_sha256,
        expected_selection_path: arguments.expected_selection_path,
        expected_selection_seal_sha256: arguments.expected_selection_seal_sha256,
        expected_source_snapshot_path: arguments.expected_source_snapshot_path,
        expected_source_snapshot_seal_sha256: arguments.expected_source_snapshot_seal_sha256,
        expected_terminal_path: arguments.expected_terminal_path,
        expected_terminal_seal_sha256: arguments.expected_terminal_seal_sha256,
    };
    let artifact = admit_qwen30_quality_repack_artifact(&arguments.manifest, &admission)
        .unwrap_or_else(|error| fail(error.to_string()));
    let result = json!({
        "schema": RESULT_SCHEMA,
        "status": RESULT_STATUS,
        "model": "qwen30-quality-repack",
        "manifest_path": artifact.manifest_path,
        "manifest_seal_sha256": artifact.manifest_seal_sha256,
        "source_audit_path": artifact.source_audit_path,
        "source_audit_seal_sha256": artifact.source_audit_seal_sha256,
        "source_revision": artifact.source_revision,
        "source_index_path": artifact.source_index_path,
        "tensor_count": artifact.tensors.len(),
        "source_weight_elements": artifact.source_weight_elements,
        "tensor_payload_bytes": artifact.tensor_payload_bytes,
        "selected_residual_organs": artifact.selected_residual_organs,
        "selected_residual_discriminators_verified": true,
        "payload_verification": {
            "mode": QWEN30_QUALITY_REPACK_PAYLOAD_VERIFY_MODE,
            "workers_used": artifact.payload_verification_workers,
            "workers_cap": 4,
            "manifest_rows": artifact.tensors.len(),
            "result_order": "manifest_ordinal_ascending_before_catalog_and_receipt",
            "candidate_only_read_path": true,
        },
        "claim_boundary": {
            "strict_complete_artifact_source_chain_and_selected_residual_discriminator_admission_only": true,
            "not_a_native_qwen_decoder_or_token_runtime": true,
            "not_capability_hcli_tps_tg_or_tournament_qualification": true,
        },
    });
    println!(
        "{}",
        serde_json::to_string(&result).expect("result must serialize")
    );
}
