//! Strict Qwen3-Coder-Next complete-artifact/runtime handoff.
//!
//! This binary proves only that an independently admitted Qwen80 artifact has
//! the exact hybrid catalog, source config/tokenizer, and native state layout.
//! Its optional state mode additionally allocates that exact Metal state and
//! uploads one direct packed tensor. Neither mode runs a complete layer,
//! produces logits, generates text, exposes HCLI, or reports TPS.

use hawking_core::model::qwen80_complete_runtime::{
    preflight_from_admitted_catalog, Qwen80CompleteArtifactCatalog, Qwen80CompleteNativeRuntime,
    Qwen80CompleteRuntimeOptions,
};
use hawking_core::model::qwen_complete_binary::{CompleteBinaryAdmission, QwenCompleteBinaryModel};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

const RESULT_SCHEMA: &str = "hawking.ascension.qwen80_complete_runtime_preflight_result.v1";
const CATALOG_STATUS: &str =
    "EARNED_QWEN80_COMPLETE_ARTIFACT_CATALOG_BOUND_NATIVE_HYBRID_DECODER_PENDING";
const STATE_STATUS: &str =
    "EARNED_QWEN80_COMPLETE_ARTIFACT_NATIVE_STATE_BOUND_HYBRID_DECODER_PENDING";
const DIRECT_PACKED_LINEAR_STAGE_STATUS: &str =
    "EARNED_QWEN80_ADMITTED_DIRECT_PACKED_FIRST_LINEAR_DELTANET_ROUTER_EXPERT_STAGE_NOT_FULL_LAYER_OR_TOKEN";
const DIRECT_PACKED_MIXER_STAGE_STATUS: &str =
    "EARNED_QWEN80_ADMITTED_DIRECT_PACKED_LAYER0_DELTANET_MIXER_THROUGH_FIRST_RESIDUAL_NOT_COMPLETE_LAYER_OR_TOKEN";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Catalog,
    State,
    DirectPackedLinearStage,
    DirectPackedMixerStage,
}

struct Arguments {
    manifest: PathBuf,
    expected_manifest_seal_sha256: String,
    expected_source_audit_seal_sha256: String,
    expected_source_revision: String,
    mode: Mode,
    max_seq_len: usize,
    capture_dir: Option<PathBuf>,
}

fn usage() -> &'static str {
    "usage: ascension_qwen80_complete_runtime_preflight \\
        --manifest ABSOLUTE_PATH \\
        --expected-manifest-seal-sha256 SHA256 \\
        --expected-source-audit-seal-sha256 SHA256 \\
        --expected-source-revision REVISION \\
        [--mode catalog|state|direct-packed-linear-stage|direct-packed-mixer-stage] \\
        [--max-seq-len POSITIVE] [--capture-dir NEW_ABSOLUTE_DIRECTORY]"
}

fn parse_arguments() -> Result<Arguments, String> {
    let mut manifest = None;
    let mut expected_manifest_seal_sha256 = None;
    let mut expected_source_audit_seal_sha256 = None;
    let mut expected_source_revision = None;
    let mut mode = Mode::Catalog;
    let mut max_seq_len = 256usize;
    let mut capture_dir = None;
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
            "--mode" => {
                mode = match value.as_str() {
                    "catalog" => Mode::Catalog,
                    "state" => Mode::State,
                    "direct-packed-linear-stage" => Mode::DirectPackedLinearStage,
                    "direct-packed-mixer-stage" => Mode::DirectPackedMixerStage,
                    _ => return Err(format!("unsupported --mode {value:?}; {}", usage())),
                }
            }
            "--max-seq-len" => {
                max_seq_len = value
                    .parse::<usize>()
                    .ok()
                    .filter(|value| *value > 0)
                    .ok_or_else(|| format!("--max-seq-len must be positive; {}", usage()))?;
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
    let manifest = manifest.ok_or_else(|| format!("missing --manifest; {}", usage()))?;
    if !manifest.is_absolute() {
        return Err("--manifest must be an absolute path".into());
    }
    if let Some(capture_dir) = &capture_dir {
        if !capture_dir.is_absolute() {
            return Err("--capture-dir must be an absolute path".into());
        }
        if !capture_dir.parent().is_some_and(|parent| parent.is_dir()) {
            return Err("--capture-dir parent must already exist as a directory".into());
        }
    }
    Ok(Arguments {
        manifest,
        expected_manifest_seal_sha256: expected_manifest_seal_sha256
            .ok_or_else(|| format!("missing --expected-manifest-seal-sha256; {}", usage()))?,
        expected_source_audit_seal_sha256: expected_source_audit_seal_sha256
            .ok_or_else(|| format!("missing --expected-source-audit-seal-sha256; {}", usage()))?,
        expected_source_revision: expected_source_revision
            .ok_or_else(|| format!("missing --expected-source-revision; {}", usage()))?,
        mode,
        max_seq_len,
        capture_dir,
    })
}

fn fail(detail: impl AsRef<str>) -> ! {
    eprintln!(
        "Qwen80 complete-runtime preflight refused: {}",
        detail.as_ref()
    );
    process::exit(2);
}

fn state_report(
    arguments: &Arguments,
    catalog: Qwen80CompleteArtifactCatalog,
) -> Result<serde_json::Value, String> {
    #[cfg(target_os = "macos")]
    {
        let mut runtime = Qwen80CompleteNativeRuntime::from_admitted_catalog(
            catalog,
            Qwen80CompleteRuntimeOptions {
                max_seq_len: arguments.max_seq_len,
                trace_dispatch: false,
            },
        )
        .map_err(|error| error.to_string())?;
        let uploaded = runtime
            .upload_direct_tensor("model.layers.0.linear_attn.in_proj_qkvz.weight")
            .map_err(|error| error.to_string())?;
        let deltanet_component = runtime
            .verify_first_linear_deltanet_component()
            .map_err(|error| error.to_string())?;
        return Ok(json!({
            "status": STATE_STATUS,
            "device": runtime.device_name(),
            "state": runtime.state_geometry(),
            "direct_packed_upload": {
                "tensor": "model.layers.0.linear_attn.in_proj_qkvz.weight",
                "shape": uploaded.header.shape,
                "group_size": uploaded.header.group_size,
                "payload_bytes": uploaded.header.payload_bytes,
                "uses_admitted_direct_binary_sign_and_scale_buffers": true,
            },
            "native_deltanet_component": {
                "max_abs_state_error": deltanet_component.max_abs_state_error,
                "max_abs_output_error": deltanet_component.max_abs_output_error,
                "metal_dispatches": deltanet_component.metal_dispatches,
                "uses_exact_32x128x128_first_linear_layer_state_slice": true,
                "cpu_oracle_is_component_parity_only_not_a_model_fallback": true,
            },
            "claim_boundary": {
                "native_state_and_single_direct_tensor_upload_only": true,
                "runs_only_a_fixture_input_deltanet_recurrence_not_a_complete_qwen80_layer": true,
                "does_not_execute_qwen80_projections_attention_moe_or_lm_head": true,
                "does_not_generate_or_measure_tps": true,
            },
        }));
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (arguments, catalog);
        Err("Qwen80 native state mode requires macOS Metal".into())
    }
}

fn direct_packed_linear_stage_report(
    arguments: &Arguments,
    catalog: Qwen80CompleteArtifactCatalog,
) -> Result<serde_json::Value, String> {
    #[cfg(target_os = "macos")]
    {
        let mut runtime = Qwen80CompleteNativeRuntime::from_admitted_catalog(
            catalog,
            Qwen80CompleteRuntimeOptions {
                max_seq_len: arguments.max_seq_len,
                trace_dispatch: true,
            },
        )
        .map_err(|error| error.to_string())?;
        let stage = runtime
            .execute_first_linear_deltanet_router_expert_stage()
            .map_err(|error| error.to_string())?;
        return Ok(json!({
            "status": DIRECT_PACKED_LINEAR_STAGE_STATUS,
            "device": runtime.device_name(),
            "stage": stage,
            "claim_boundary": {
                "admitted_direct_packed_native_metal_component_stage_only": true,
                "cpu_is_used_only_as_a_direct_compact_representation_parity_oracle": true,
                "does_not_execute_source_input_norm_causal_convolution_qkvz_rearrangement_gated_rmsnorm_out_projection_or_residual": true,
                "does_not_execute_all_top10_expert_waves_shared_expert_or_complete_moe": true,
                "does_not_execute_a_complete_layer_or_48_layer_decoder": true,
                "does_not_generate_tokens_expose_hcli_or_measure_tps": true,
            },
        }));
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (arguments, catalog);
        Err("Qwen80 direct-packed linear stage requires macOS Metal".into())
    }
}

/// Run the first source-scheduled Qwen3-Next DeltaNet mixer through direct
/// packed Metal kernels and check every covered CPU/device boundary.  This
/// ends at the first residual by design: post-attention norm, router, all
/// routed/shared experts, the second residual, remaining layers, head and
/// token loop have no result here.
fn direct_packed_mixer_stage_report(
    arguments: &Arguments,
    catalog: Qwen80CompleteArtifactCatalog,
) -> Result<serde_json::Value, String> {
    #[cfg(target_os = "macos")]
    {
        let mut runtime = Qwen80CompleteNativeRuntime::from_admitted_catalog(
            catalog,
            Qwen80CompleteRuntimeOptions {
                max_seq_len: arguments.max_seq_len,
                trace_dispatch: true,
            },
        )
        .map_err(|error| error.to_string())?;
        let stage = runtime
            .execute_first_linear_deltanet_mixer_stage()
            .map_err(|error| error.to_string())?;
        return Ok(json!({
            "status": DIRECT_PACKED_MIXER_STAGE_STATUS,
            "device": runtime.device_name(),
            "stage": stage,
            "claim_boundary": {
                "admitted_direct_packed_native_metal_layer0_deltanet_mixer_only": true,
                "cpu_is_only_the_same_compact_representation_parity_oracle": true,
                "covers_input_norm_qkvz_ba_causal_convolution_qk_repetition_and_l2_recurrence_gated_norm_out_projection_and_first_residual": true,
                "does_not_execute_post_attention_norm_router_top10_routed_expert_waves_shared_expert_or_second_residual": true,
                "does_not_execute_a_complete_layer_or_48_layer_decoder": true,
                "does_not_generate_tokens_expose_hcli_or_measure_tps": true,
            },
        }));
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (arguments, catalog);
        Err("Qwen80 direct-packed mixer stage requires macOS Metal".into())
    }
}

fn mode_name(mode: Mode) -> &'static str {
    match mode {
        Mode::Catalog => "catalog",
        Mode::State => "state",
        Mode::DirectPackedLinearStage => "direct-packed-linear-stage",
        Mode::DirectPackedMixerStage => "direct-packed-mixer-stage",
    }
}

fn write_capture_file(
    capture_dir: &std::path::Path,
    name: &str,
    bytes: &[u8],
) -> Result<(), String> {
    let target = capture_dir.join(name);
    let temporary = capture_dir.join(format!(".{name}.{}.tmp", process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| format!("cannot create capture temporary {:?}: {error}", temporary))?;
    if let Err(error) = file.write_all(bytes).and_then(|_| file.sync_all()) {
        let _ = fs::remove_file(&temporary);
        return Err(format!(
            "cannot durably write capture temporary {:?}: {error}",
            temporary
        ));
    }
    drop(file);
    fs::rename(&temporary, &target).map_err(|error| {
        let _ = fs::remove_file(&temporary);
        format!(
            "cannot atomically publish new capture {:?} from {:?}: {error}",
            target, temporary
        )
    })
}

fn begin_capture(arguments: &Arguments) -> Result<(), String> {
    let Some(capture_dir) = &arguments.capture_dir else {
        return Ok(());
    };
    fs::create_dir(capture_dir).map_err(|error| {
        format!(
            "refusing to reuse or create non-exclusive capture directory {:?}: {error}",
            capture_dir
        )
    })?;
    let started_unix_millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock is before Unix epoch: {error}"))?
        .as_millis();
    let invocation = json!({
        "schema": "hawking.ascension.qwen80_layer0_deltanet_mixer_capture.v1",
        "status": "STARTED_QWEN80_DIRECT_PACKED_LAYER0_DELTANET_MIXER_PARITY_ATTEMPT",
        "started_unix_millis": started_unix_millis,
        "mode": mode_name(arguments.mode),
        "manifest": arguments.manifest,
        "expected_manifest_seal_sha256": arguments.expected_manifest_seal_sha256,
        "expected_source_audit_seal_sha256": arguments.expected_source_audit_seal_sha256,
        "expected_source_revision": arguments.expected_source_revision,
        "claim_boundary": {
            "first_residual_stage_only": true,
            "not_a_complete_layer_decoder_generation_hcli_or_tps_result": true,
        },
    });
    write_capture_file(
        capture_dir,
        "invocation.json",
        &serde_json::to_vec_pretty(&invocation).map_err(|error| error.to_string())?,
    )
}

fn finish_capture(
    arguments: &Arguments,
    result: &serde_json::Value,
    stderr: &str,
    exit_code: i32,
) -> Result<(), String> {
    let Some(capture_dir) = &arguments.capture_dir else {
        return Ok(());
    };
    let stdout = serde_json::to_vec(result).map_err(|error| error.to_string())?;
    write_capture_file(capture_dir, "stdout.jsonl", &stdout)?;
    write_capture_file(capture_dir, "stderr.log", stderr.as_bytes())?;
    let receipt = json!({
        "schema": "hawking.ascension.qwen80_layer0_deltanet_mixer_capture_receipt.v1",
        "status": result.get("status").and_then(serde_json::Value::as_str).unwrap_or("UNKNOWN"),
        "exit_code": exit_code,
        "mode": mode_name(arguments.mode),
        "manifest": arguments.manifest,
        "expected_manifest_seal_sha256": arguments.expected_manifest_seal_sha256,
        "expected_source_audit_seal_sha256": arguments.expected_source_audit_seal_sha256,
        "expected_source_revision": arguments.expected_source_revision,
        "stdout_json_sha256": format!("{:x}", Sha256::digest(&stdout)),
        "stderr_sha256": format!("{:x}", Sha256::digest(stderr.as_bytes())),
        "files": {
            "invocation": "invocation.json",
            "stdout": "stdout.jsonl",
            "stderr": "stderr.log",
            "receipt": "receipt.json",
            "receipt_is_completion_marker": true,
        },
        "claim_boundary": {
            "first_residual_stage_only": true,
            "no_full_layer_decoder_generation_hcli_or_tps_claim": true,
        },
    });
    write_capture_file(
        capture_dir,
        "receipt.json",
        &serde_json::to_vec_pretty(&receipt).map_err(|error| error.to_string())?,
    )
}

fn run(arguments: &Arguments) -> Result<serde_json::Value, String> {
    let admission = CompleteBinaryAdmission {
        model: QwenCompleteBinaryModel::Qwen80CoderNext,
        expected_manifest_seal_sha256: arguments.expected_manifest_seal_sha256.clone(),
        expected_source_audit_seal_sha256: arguments.expected_source_audit_seal_sha256.clone(),
        expected_source_revision: arguments.expected_source_revision.clone(),
    };
    // Exactly one strict direct-artifact admission occurs per invocation.
    // Structural and native state/stage work borrow or consume this one
    // catalog rather than independently rehashing the 74,391 payloads.
    let catalog = Qwen80CompleteArtifactCatalog::load(&arguments.manifest, &admission)
        .map_err(|error| error.to_string())?;
    let preflight = preflight_from_admitted_catalog(&catalog).map_err(|error| error.to_string())?;
    let catalog_reused_for_native_mode = arguments.mode != Mode::Catalog;
    let state = match arguments.mode {
        Mode::Catalog => None,
        Mode::State => Some(state_report(arguments, catalog)?),
        Mode::DirectPackedLinearStage => {
            Some(direct_packed_linear_stage_report(arguments, catalog)?)
        }
        Mode::DirectPackedMixerStage => Some(direct_packed_mixer_stage_report(arguments, catalog)?),
    };
    let status = state
        .as_ref()
        .and_then(|value| value.get("status"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or(CATALOG_STATUS);
    Ok(json!({
        "schema": RESULT_SCHEMA,
        "status": status,
        "model": {
            "key": "qwen80",
            "id": "Qwen3-Coder-Next-80B",
            "repository": "Qwen/Qwen3-Coder-Next",
            "revision": preflight.source_revision,
        },
        "manifest_path": preflight.manifest_path,
        "manifest_seal_sha256": preflight.manifest_seal_sha256,
        "strict_complete_artifact_admission": {
            "full_payload_scan_count_this_process": 1,
            "all_source_payload_header_and_ledger_checks_precede_catalog_reuse": true,
            "catalog_reused_for_requested_native_mode": catalog_reused_for_native_mode,
        },
        "preflight": {
            "config_path": preflight.config_path,
            "config_sha256": preflight.config_sha256,
            "tokenizer_path": preflight.tokenizer_path,
            "tokenizer_sha256": preflight.tokenizer_sha256,
            "tokenizer_vocab_size": preflight.tokenizer_vocab_size,
            "lm_head_vocab_size": 151936,
            "reserved_lm_head_tail_rows": preflight.reserved_lm_head_tail_rows,
            "tensor_count": preflight.tensor_count,
            "tensor_payload_bytes": preflight.tensor_payload_bytes,
            "source_weight_elements": preflight.source_weight_elements,
            "direct_layout_group_size": preflight.direct_layout_group_size,
            "default_native_state": preflight.default_native_state,
            "exact_hybrid_schedule": {
                "linear_attention_layers": 36,
                "full_attention_layers": 12,
                "source_pattern": "linear_attention,linear_attention,linear_attention,full_attention x12",
            },
        },
        "native_state": state,
        "claim_boundary": {
            "strict_complete_artifact_catalog_and_hybrid_geometry_only": true,
            "raw_bf16_source_is_not_opened_as_a_runtime_fallback": true,
            "complete_all_layer_decoder_generation_hcli_and_tps_remain_unimplemented": true,
            "not_capability_tg10_tg3_or_tournament_qualification": true,
        },
    }))
}

fn failure_result(arguments: &Arguments, error: &str) -> serde_json::Value {
    json!({
        "schema": RESULT_SCHEMA,
        "status": "REFUSED_QWEN80_DIRECT_PACKED_LAYER0_DELTANET_MIXER_PARITY_ATTEMPT",
        "error": error,
        "mode": mode_name(arguments.mode),
        "source_binding": {
            "manifest": arguments.manifest,
            "expected_manifest_seal_sha256": arguments.expected_manifest_seal_sha256,
            "expected_source_audit_seal_sha256": arguments.expected_source_audit_seal_sha256,
            "expected_source_revision": arguments.expected_source_revision,
        },
        "claim_boundary": {
            "no_cpu_or_metal_parity_is_claimed": true,
            "not_a_complete_layer_decoder_generation_hcli_or_tps_result": true,
        },
    })
}

fn main() {
    let arguments = parse_arguments().unwrap_or_else(|error| fail(error));
    begin_capture(&arguments).unwrap_or_else(|error| fail(error));
    match run(&arguments) {
        Ok(result) => {
            finish_capture(&arguments, &result, "", 0).unwrap_or_else(|error| fail(error));
            println!(
                "{}",
                serde_json::to_string(&result).expect("preflight result must serialize")
            );
        }
        Err(error) => {
            let result = failure_result(&arguments, &error);
            finish_capture(&arguments, &result, &format!("{error}\n"), 2)
                .unwrap_or_else(|capture_error| fail(capture_error));
            eprintln!("Qwen80 complete-runtime preflight refused: {error}");
            println!(
                "{}",
                serde_json::to_string(&result).expect("refusal result must serialize")
            );
            process::exit(2);
        }
    }
}
