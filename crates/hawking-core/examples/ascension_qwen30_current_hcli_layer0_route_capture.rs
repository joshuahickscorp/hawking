//! Isolated, source-bound L0 route/hidden capture for current HCLI compiler
//! traces.  This is not a serving path: it consumes already-sealed source
//! template token IDs, executes only the direct-packed embedding + layer-0
//! attention/router portion on Metal, and writes route membership plus the
//! device-produced router input hidden vector.  It deliberately has no
//! generation, lm_head, sampler, HCLI, or TPS mode.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("qwen30 current HCLI L0 route capture requires macOS Metal");
    std::process::exit(2);
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::model::qwen30_complete_runtime::{
        Qwen30CompleteNativeRuntime, Qwen30CompleteRuntimeOptions, Qwen30GateUpSwiGluKernel,
        Qwen30Layer0RouterCapture, Qwen30PackedMatvecKernel,
    };
    use hawking_core::model::qwen_complete_binary::{
        CompleteBinaryAdmission, QwenCompleteBinaryModel,
    };
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::collections::HashSet;
    use std::env;
    use std::fs::{self, File, OpenOptions};
    use std::io::{Read, Write};
    use std::path::{Path, PathBuf};
    use std::process;

    const INPUT_SCHEMA: &str =
        "hawking.ascension.qwen30_current_hcli_layer0_route_capture_input.v1";
    const RESULT_SCHEMA: &str =
        "hawking.ascension.qwen30_current_hcli_layer0_route_capture_result.v1";
    const TRACE_STATUS: &str = "NEW_DIAGNOSTIC_NOT_HISTORICAL";
    const CAPTURE_PROTOCOL_REVISION: &str = "l0-route-hidden-capture-output-parent-v2";

    struct Arguments {
        manifest: PathBuf,
        expected_manifest_seal_sha256: String,
        expected_source_audit_seal_sha256: String,
        expected_source_revision: String,
        input_json: PathBuf,
        output_dir: PathBuf,
        max_seq_len: usize,
    }

    fn usage() -> &'static str {
        "usage: ascension_qwen30_current_hcli_layer0_route_capture \\
            --manifest ABSOLUTE_PATH \\
            --expected-manifest-seal-sha256 SHA256 \\
            --expected-source-audit-seal-sha256 SHA256 \\
            --expected-source-revision REVISION \\
            --input-json ABSOLUTE_PATH --output-dir ABSOLUTE_PATH [--max-seq-len N]"
    }

    fn required<T>(value: Option<T>, flag: &str) -> Result<T, String> {
        value.ok_or_else(|| format!("missing {flag}; {}", usage()))
    }

    fn parse_usize(value: &str, flag: &str) -> Result<usize, String> {
        value
            .parse::<usize>()
            .map_err(|_| format!("{flag} must be an unsigned decimal integer; {}", usage()))
    }

    fn absolute(path: PathBuf, flag: &str) -> Result<PathBuf, String> {
        if !path.is_absolute() {
            return Err(format!("{flag} must be an absolute path; {}", usage()));
        }
        Ok(path)
    }

    fn parse_arguments() -> Result<Arguments, String> {
        let mut manifest = None;
        let mut expected_manifest_seal_sha256 = None;
        let mut expected_source_audit_seal_sha256 = None;
        let mut expected_source_revision = None;
        let mut input_json = None;
        let mut output_dir = None;
        let mut max_seq_len = 4096usize;
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
                "--input-json" => {
                    if input_json.replace(PathBuf::from(value)).is_some() {
                        return Err(format!(
                            "--input-json was supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--output-dir" => {
                    if output_dir.replace(PathBuf::from(value)).is_some() {
                        return Err(format!(
                            "--output-dir was supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--max-seq-len" => max_seq_len = parse_usize(&value, "--max-seq-len")?,
                _ => return Err(format!("unsupported option {flag:?}; {}", usage())),
            }
        }
        if max_seq_len == 0 {
            return Err("--max-seq-len must be positive".into());
        }
        Ok(Arguments {
            manifest: absolute(required(manifest, "--manifest")?, "--manifest")?,
            expected_manifest_seal_sha256: required(
                expected_manifest_seal_sha256,
                "--expected-manifest-seal-sha256",
            )?,
            expected_source_audit_seal_sha256: required(
                expected_source_audit_seal_sha256,
                "--expected-source-audit-seal-sha256",
            )?,
            expected_source_revision: required(
                expected_source_revision,
                "--expected-source-revision",
            )?,
            input_json: absolute(required(input_json, "--input-json")?, "--input-json")?,
            output_dir: absolute(required(output_dir, "--output-dir")?, "--output-dir")?,
            max_seq_len,
        })
    }

    fn admission(arguments: &Arguments) -> CompleteBinaryAdmission {
        CompleteBinaryAdmission {
            model: QwenCompleteBinaryModel::Qwen30Coder,
            expected_manifest_seal_sha256: arguments.expected_manifest_seal_sha256.clone(),
            expected_source_audit_seal_sha256: arguments.expected_source_audit_seal_sha256.clone(),
            expected_source_revision: arguments.expected_source_revision.clone(),
        }
    }

    fn sha256_file(path: &Path) -> Result<String, String> {
        let mut file =
            File::open(path).map_err(|error| format!("cannot open {}: {error}", path.display()))?;
        let mut digest = Sha256::new();
        let mut chunk = [0u8; 1024 * 1024];
        loop {
            let read = file
                .read(&mut chunk)
                .map_err(|error| format!("cannot hash {}: {error}", path.display()))?;
            if read == 0 {
                break;
            }
            digest.update(&chunk[..read]);
        }
        Ok(format!("{:x}", digest.finalize()))
    }

    fn current_executable_sha256() -> Result<String, String> {
        let path = env::current_exe()
            .map_err(|error| format!("cannot resolve capture executable: {error}"))?;
        sha256_file(&path)
    }

    fn parse_token_ids(value: &Value, probe_id: &str) -> Result<Vec<u32>, String> {
        let ids = value
            .as_array()
            .ok_or_else(|| format!("{probe_id} source template token IDs are not an array"))?;
        if ids.is_empty() {
            return Err(format!("{probe_id} source template token IDs are empty"));
        }
        ids.iter()
            .map(|id| {
                id.as_u64()
                    .and_then(|value| u32::try_from(value).ok())
                    .ok_or_else(|| format!("{probe_id} contains an invalid token ID"))
            })
            .collect()
    }

    fn parse_input(path: &Path) -> Result<(Value, Vec<(String, Vec<u32>)>), String> {
        let bytes = fs::read(path)
            .map_err(|error| format!("cannot read capture input {}: {error}", path.display()))?;
        let document: Value = serde_json::from_slice(&bytes)
            .map_err(|error| format!("capture input is not JSON: {error}"))?;
        if document.get("schema").and_then(Value::as_str) != Some(INPUT_SCHEMA) {
            return Err(
                "capture input schema is not the isolated current-HCLI route schema".into(),
            );
        }
        if document.get("status").and_then(Value::as_str) != Some(TRACE_STATUS) {
            return Err("capture input is not marked NEW_DIAGNOSTIC_NOT_HISTORICAL".into());
        }
        if document
            .pointer("/claim_boundary/model_execution_started")
            .and_then(Value::as_bool)
            != Some(false)
        {
            return Err(
                "capture input does not prove its compiler traces stopped before model execution"
                    .into(),
            );
        }
        let probes = document
            .get("probes")
            .and_then(Value::as_array)
            .ok_or_else(|| "capture input lacks probes".to_string())?;
        if probes.len() != 3 {
            return Err(
                "capture input must contain exactly the three protected public probes".into(),
            );
        }
        let mut seen = HashSet::new();
        let mut result = Vec::with_capacity(probes.len());
        for probe in probes {
            let probe_id = probe
                .get("probe_id")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| "capture input probe lacks a non-empty probe_id".to_string())?
                .to_string();
            if !seen.insert(probe_id.clone()) {
                return Err(format!("capture input repeats probe {probe_id:?}"));
            }
            let token_ids = parse_token_ids(
                probe
                    .pointer("/source_one_user_native_prompt/token_ids")
                    .ok_or_else(|| format!("{probe_id} lacks source one-user native token IDs"))?,
                &probe_id,
            )?;
            result.push((probe_id, token_ids));
        }
        Ok((document, result))
    }

    fn write_hidden(path: &Path, values: &[f32]) -> Result<(String, usize), String> {
        let parent = path
            .parent()
            .ok_or_else(|| format!("hidden capture path has no parent: {}", path.display()))?;
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "cannot create hidden capture directory {}: {error}",
                parent.display()
            )
        })?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|error| format!("cannot create hidden capture {}: {error}", path.display()))?;
        let mut digest = Sha256::new();
        for value in values {
            let bytes = value.to_le_bytes();
            file.write_all(&bytes).map_err(|error| {
                format!("cannot write hidden capture {}: {error}", path.display())
            })?;
            digest.update(bytes);
        }
        file.flush()
            .map_err(|error| format!("cannot flush hidden capture {}: {error}", path.display()))?;
        file.sync_all()
            .map_err(|error| format!("cannot sync hidden capture {}: {error}", path.display()))?;
        Ok((
            format!("{:x}", digest.finalize()),
            values.len() * std::mem::size_of::<f32>(),
        ))
    }

    fn capture_row(
        output_dir: &Path,
        probe_id: &str,
        capture: Qwen30Layer0RouterCapture,
    ) -> Result<Value, String> {
        let hidden_relative = format!("hidden/{probe_id}/{:06}.f32le", capture.position);
        let hidden_path = output_dir.join(&hidden_relative);
        let (hidden_sha256, hidden_bytes) =
            write_hidden(&hidden_path, &capture.router_input_hidden)?;
        Ok(json!({
            "position": capture.position,
            "input_token_id": capture.input_token_id,
            "selected_expert_ids": capture.selected_expert_ids,
            "normalized_route_weights": capture.normalized_route_weights,
            "router_input_hidden_f32le": {
                "relative_path": hidden_relative,
                "sha256": hidden_sha256,
                "bytes": hidden_bytes,
                "elements": capture.router_input_hidden.len(),
                "source": "device-produced L0 post-attention RMSNorm buffer copied after router command completion",
            },
            "all_48_layers_executed": false,
            "final_norm_lm_head_sampler_or_feedback_executed": false,
        }))
    }

    fn write_json_new(path: &Path, value: &Value) -> Result<(), String> {
        let text = serde_json::to_string_pretty(value)
            .map_err(|error| format!("cannot serialize capture result: {error}"))?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|error| format!("cannot create result {}: {error}", path.display()))?;
        file.write_all(text.as_bytes())
            .map_err(|error| format!("cannot write result {}: {error}", path.display()))?;
        file.write_all(b"\n")
            .map_err(|error| format!("cannot finish result {}: {error}", path.display()))?;
        file.flush()
            .map_err(|error| format!("cannot flush result {}: {error}", path.display()))?;
        file.sync_all()
            .map_err(|error| format!("cannot sync result {}: {error}", path.display()))?;
        Ok(())
    }

    fn fail(detail: impl AsRef<str>) -> ! {
        eprintln!(
            "qwen30 current HCLI L0 route capture refused: {}",
            detail.as_ref()
        );
        process::exit(2);
    }

    pub fn run() {
        let arguments = parse_arguments().unwrap_or_else(|error| fail(error));
        if arguments.output_dir.exists() {
            fail(format!(
                "refusing to reuse or overwrite route capture output directory {}",
                arguments.output_dir.display()
            ));
        }
        if !arguments
            .output_dir
            .parent()
            .is_some_and(|parent| parent.is_dir())
        {
            fail("route capture output parent must already exist");
        }
        let (input, probes) =
            parse_input(&arguments.input_json).unwrap_or_else(|error| fail(error));
        let input_sha256 = sha256_file(&arguments.input_json).unwrap_or_else(|error| fail(error));
        fs::create_dir(&arguments.output_dir).unwrap_or_else(|error| {
            fail(format!(
                "cannot create route capture output directory {}: {error}",
                arguments.output_dir.display()
            ))
        });
        let executable_sha256 = current_executable_sha256().unwrap_or_else(|error| fail(error));
        let mut runtime = Qwen30CompleteNativeRuntime::load(
            &arguments.manifest,
            &admission(&arguments),
            Qwen30CompleteRuntimeOptions {
                max_seq_len: arguments.max_seq_len,
                trace_dispatch: false,
                packed_matvec_kernel: Qwen30PackedMatvecKernel::ScalarControl,
                gate_up_swiglu_kernel: Qwen30GateUpSwiGluKernel::ThreeDispatchControl,
            },
        )
        .unwrap_or_else(|error| fail(error.to_string()));
        let mut probe_rows = Vec::with_capacity(probes.len());
        for (probe_id, token_ids) in probes {
            if token_ids.len() > arguments.max_seq_len {
                fail(format!(
                    "{probe_id} token length {} exceeds capture max sequence {}",
                    token_ids.len(),
                    arguments.max_seq_len
                ));
            }
            runtime.reset();
            let mut steps = Vec::with_capacity(token_ids.len());
            for token_id in token_ids {
                let capture = runtime
                    .capture_layer0_router_for_token(token_id)
                    .unwrap_or_else(|error| fail(error.to_string()));
                steps.push(
                    capture_row(&arguments.output_dir, &probe_id, capture)
                        .unwrap_or_else(|error| fail(error)),
                );
            }
            probe_rows.push(json!({
                "probe_id": probe_id,
                "source_one_user_native_prompt_token_count": steps.len(),
                "steps": steps,
            }));
        }
        let result = json!({
            "schema": RESULT_SCHEMA,
            "status": "EARNED_NEW_DIAGNOSTIC_NOT_HISTORICAL_L0_ROUTE_AND_HIDDEN_CAPTURE_UNQUALIFIED",
            "capture_protocol_revision": CAPTURE_PROTOCOL_REVISION,
            "input": {
                "path": arguments.input_json,
                "sha256": input_sha256,
                "schema": input.get("schema"),
                "status": input.get("status"),
            },
            "runtime_binding": {
                "manifest_path": arguments.manifest,
                "manifest_seal_sha256": runtime.artifact_manifest_seal(),
                "source_revision": runtime.config.source_revision,
                "runtime_executable_sha256": executable_sha256,
                "architecture": "Qwen3MoeForCausalLM",
                "metal_only": true,
                "raw_bf16_loader_not_opened": true,
                "immutable_complete_payload_catalog": {
                    "validated_during_process_admission": true,
                    "verified_payload_count": runtime.verified_payload_count(),
                    "expected_complete_tensor_count": 18867,
                    "complete_verified_payload_cache": runtime.has_complete_verified_payload_cache(),
                },
                "packed_matvec_kernel": runtime.packed_matvec_kernel().receipt_name(),
                "gate_up_swiglu_kernel": runtime.gate_up_swiglu_kernel().receipt_name(),
            },
            "probes": probe_rows,
            "logit_provenance": {
                "status": "NOT_EXECUTED_FAIL_CLOSED",
                "reason": "HQ30GR2 typed residual gate/up dispatch is not integrated into a candidate-only all-layer diagnostic runtime; baseline logits would not measure candidate-minus-control divergence",
            },
            "claim_boundary": {
                "new_diagnostic_not_historical": true,
                "only_embedding_layer0_attention_postnorm_router_executed": true,
                "all_48_layers_lm_head_sampler_autoregressive_feedback_and_generation_not_executed": true,
                "does_not_claim_coherence_hcli_tps_tg_capability_or_tournament": true,
                "qwen30_server_watcher_and_hcli_adapter_are_not_used": true,
            },
        });
        let result_path = arguments.output_dir.join("capture-result.json");
        write_json_new(&result_path, &result).unwrap_or_else(|error| fail(error));
        println!(
            "{}",
            serde_json::to_string(&result).expect("capture result must serialize")
        );
    }
}

#[cfg(target_os = "macos")]
fn main() {
    macos::run();
}
