//! Run the reusable artifact-backed P3A Metal sink through scheduler leases.
//!
//! This executes only the bounded layer-0 BOS mHC/norm/Q chain. It deliberately
//! stops before WKV, attention, KV, router, MoE, a causal loop, Engine, HCLI,
//! parity promotion, and TPS measurement.

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("gravity_deepseek_v4_p3a_scheduler_sink requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
mod macos {
    use std::error::Error;
    use std::fs::{self, File, OpenOptions};
    use std::io::Write;
    use std::path::{Path, PathBuf};

    use hawking_core::gravity_deepseek_v4::{FULL_STREAM_SCHEMA, FULL_STREAM_STATUS};
    use hawking_core::gravity_deepseek_v4_execution_context::{
        DeepSeekV4ExecutionContext, DeepSeekV4ExecutionContextConfig, DeepSeekV4SelectedRouteSet,
    };
    use hawking_core::gravity_deepseek_v4_layer_scheduler::{
        DeepSeekV4LayerPreparationScheduler, DeepSeekV4LayerPreparationStage,
    };
    use hawking_core::gravity_deepseek_v4_p3a_stage_sink::{
        DeepSeekV4P3aMetalStageSink, DeepSeekV4P3aStageSinkReport, DSV4F_P3A_Q_CHAIN_DISPATCHES,
    };
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};

    const RECEIPT_SCHEMA: &str = "hawking.gravity.deepseek_v4.p3a_scheduler_sink.v1";
    const RECEIPT_STATUS: &str = "PASS_REAL_METAL_P3A_SCHEDULER_SINK_NOT_RUNTIME";

    type ExampleResult<T> = Result<T, Box<dyn Error>>;

    struct Args {
        artifact: PathBuf,
        out: PathBuf,
    }

    pub fn run() -> ExampleResult<()> {
        let args = parse_args()?;
        let mut context = DeepSeekV4ExecutionContext::open(
            &args.artifact,
            DeepSeekV4ExecutionContextConfig::default(),
        )?;
        let prepared = context.prepare_decode_input(0)?;
        let mut sink = DeepSeekV4P3aMetalStageSink::new(&context, &prepared)?;
        let routes = DeepSeekV4SelectedRouteSet::new([0, 1, 2, 3, 4, 5])?;
        let mut scheduler = DeepSeekV4LayerPreparationScheduler::new(&context, 0, routes)?;

        let mut scheduled = Vec::new();
        for expected in [
            DeepSeekV4LayerPreparationStage::MhcAttentionControl,
            DeepSeekV4LayerPreparationStage::AttentionControl(
                hawking_core::gravity_deepseek_v4_runtime_spine::DeepSeekV4ControlProjection::WqA,
            ),
            DeepSeekV4LayerPreparationStage::AttentionControl(
                hawking_core::gravity_deepseek_v4_runtime_spine::DeepSeekV4ControlProjection::WqB,
            ),
        ] {
            let step = scheduler
                .execute_next_with_sink(&mut context, &mut sink)?
                .ok_or_else(|| failure("P3A scheduler ended before bounded Q chain"))?;
            if step.stage != expected {
                return Err(failure("P3A sink received an unexpected scheduler stage"));
            }
            scheduled.push(json!({
                "sequence": step.sequence,
                "stage": step.stage.as_str(),
                "logical_graph_node_ordinal": step.logical_graph_node_ordinal,
                "actual_command_buffers": step.actual_command_buffers,
                "actual_compute_encoders": step.actual_compute_encoders,
                "actual_gpu_dispatches": step.actual_gpu_dispatches,
                "actual_cpu_visible_waits": step.actual_cpu_visible_waits,
            }));
        }
        let report = sink.finish()?;
        validate_report(&report)?;
        let full_forward_denied = context.require_full_causal_execution().is_err();
        if !full_forward_denied {
            return Err(failure(
                "P3A scheduler sink unexpectedly admitted a full causal runtime",
            ));
        }

        let reader = context.spine().reader();
        let unsigned = json!({
            "schema": RECEIPT_SCHEMA,
            "status": RECEIPT_STATUS,
            "artifact": {
                "path": reader.artifact_root(),
                "manifest_schema": FULL_STREAM_SCHEMA,
                "manifest_status": FULL_STREAM_STATUS,
                "manifest_file_sha256": reader.manifest_file_sha256(),
                "manifest_seal_sha256": reader.manifest_seal_sha256(),
                "restart_seal_sha256": reader.restart_seal_sha256(),
                "source": {
                    "repository": reader.source_identity().repository,
                    "revision": reader.source_identity().revision,
                    "source_parent_retained": false,
                },
            },
            "scheduler_to_metal_handoff": {
                "prepared_token_id": prepared.token_id,
                "prepared_position": prepared.position,
                "layer": 0,
                "executed_scheduler_stages": scheduled,
                "scheduler_complete": scheduler.is_complete(),
                "next_unexecuted_scheduler_stage": scheduler.next_stage().map(|stage| stage.as_str()),
                "route_set_not_consumed": routes.experts,
                "control_leases_consumed_synchronously": true,
                "source_payloads_passed_directly_to_metal_sink": true,
                "host_intermediate_handoff_bytes": report.counters.host_intermediate_handoff_bytes,
            },
            "source_bindings": {
                "artifact_manifest_seal_sha256": report.source_bindings.artifact_manifest_seal_sha256,
                "repository": report.source_bindings.repository,
                "revision": report.source_bindings.revision,
                "embedding_sha256": report.source_bindings.embedding_sha256,
                "attn_norm_sha256": report.source_bindings.attn_norm_sha256,
                "q_norm_sha256": report.source_bindings.q_norm_sha256,
                "inference_model_py_sha256": report.source_bindings.inference_model_py_sha256,
                "inference_kernel_py_sha256": report.source_bindings.inference_kernel_py_sha256,
                "inference_config_json_sha256": report.source_bindings.inference_config_json_sha256,
                "model_config_json_sha256": report.source_bindings.model_config_json_sha256,
                "inference_convert_py_sha256": report.source_bindings.inference_convert_py_sha256,
            },
            "metal_execution": {
                "actual_command_buffers": report.counters.actual_command_buffers,
                "actual_compute_encoders": report.counters.actual_compute_encoders,
                "actual_gpu_dispatches": report.counters.actual_gpu_dispatches,
                "actual_cpu_visible_waits": report.counters.actual_cpu_visible_waits,
                "gpu_timestamped_dispatches": report.counters.gpu_timestamped_dispatches,
                "aggregate_gpu_duration_us": report.counters.aggregate_gpu_duration_us,
                "buffers_created": report.buffers_created,
                "device_bytes_allocated": report.device_bytes_allocated,
                "trace_samples": report.trace_samples,
                "source_upload_bytes": report.counters.source_upload_bytes,
                "static_artifact_control_reads": report.counters.static_artifact_control_reads,
                "source_control_leases_consumed": report.counters.source_control_leases_consumed,
                "host_intermediate_handoff_bytes": report.counters.host_intermediate_handoff_bytes,
                "q_head_output_device_bytes": report.q_head_output_device_bytes,
                "dispatches": report.dispatches.iter().map(|dispatch| json!({
                    "stage": dispatch.stage,
                    "kernel": dispatch.kernel,
                    "gpu_duration_us": dispatch.timing.gpu_duration_us,
                    "gpu_start_ns": dispatch.timing.gpu_start_ns,
                    "gpu_end_ns": dispatch.timing.gpu_end_ns,
                    "cpu_duration_us": dispatch.timing.host_wall_us,
                    "cpu_encode_us": dispatch.timing.encode_us,
                    "cpu_submit_us": dispatch.timing.submit_us,
                    "cpu_wait_us": dispatch.timing.wait_us,
                    "command_buffers": dispatch.timing.command_buffers,
                    "compute_encoders": dispatch.timing.compute_encoders,
                    "compute_dispatches": dispatch.timing.compute_dispatches,
                    "bytes_read": dispatch.bytes_read,
                    "bytes_written": dispatch.bytes_written,
                    "source_payloads": dispatch.source_payloads.iter().map(|payload| json!({
                        "label": payload.label,
                        "bytes": payload.bytes,
                        "sha256": payload.sha256,
                    })).collect::<Vec<_>>(),
                })).collect::<Vec<_>>(),
            },
            "execution_boundary": {
                "p3a_device_chain_complete": true,
                "registered_43_layer_engine": false,
                "causal_forward": false,
                "continuation": false,
                "kv_write": false,
                "sparse_attention": false,
                "router_logits": false,
                "routed_expert_execution": false,
                "shared_expert_execution": false,
                "lm_head": false,
                "sampling": false,
                "hcli_endpoint": false,
                "numeric_parity_v21_promoted": false,
                "base_true_tps_eligible": false,
                "full_causal_execution_denied": full_forward_denied,
                "claim": report.runtime_boundary,
            },
        });
        let receipt = seal(unsigned)?;
        write_new_receipt(&args.out, &receipt)?;
        let seal = receipt
            .get("seal_sha256")
            .and_then(Value::as_str)
            .ok_or_else(|| failure("sealed receipt has no seal_sha256"))?;
        println!(
            "status={RECEIPT_STATUS} receipt={} seal_sha256={seal}",
            args.out.display()
        );
        Ok(())
    }

    fn validate_report(report: &DeepSeekV4P3aStageSinkReport) -> ExampleResult<()> {
        if report.counters.actual_command_buffers != DSV4F_P3A_Q_CHAIN_DISPATCHES
            || report.counters.actual_compute_encoders != DSV4F_P3A_Q_CHAIN_DISPATCHES
            || report.counters.actual_gpu_dispatches != DSV4F_P3A_Q_CHAIN_DISPATCHES
            || report.counters.actual_cpu_visible_waits != DSV4F_P3A_Q_CHAIN_DISPATCHES
            || report.counters.gpu_timestamped_dispatches != DSV4F_P3A_Q_CHAIN_DISPATCHES
            || report.counters.host_intermediate_handoff_bytes != 0
            || report.dispatches.len() != DSV4F_P3A_Q_CHAIN_DISPATCHES
        {
            return Err(failure(
                "bounded P3A scheduler sink accounting is incomplete or has host handoff",
            ));
        }
        Ok(())
    }

    fn parse_args() -> ExampleResult<Args> {
        let mut artifact = None;
        let mut out = None;
        let mut args = std::env::args().skip(1);
        while let Some(argument) = args.next() {
            match argument.as_str() {
                "--artifact" => artifact = args.next().map(PathBuf::from),
                "--out" => out = args.next().map(PathBuf::from),
                "--help" | "-h" => {
                    println!(
                        "usage: gravity_deepseek_v4_p3a_scheduler_sink --artifact <absolute full Gravity dir> --out <absolute receipt.json>"
                    );
                    std::process::exit(0);
                }
                other => return Err(failure(format!("unknown argument {other:?}"))),
            }
        }
        let artifact = artifact.ok_or_else(|| failure("--artifact is required"))?;
        let out = out.ok_or_else(|| failure("--out is required"))?;
        if !artifact.is_absolute() || !out.is_absolute() {
            return Err(failure("--artifact and --out must be absolute paths"));
        }
        Ok(Args { artifact, out })
    }

    fn seal(mut value: Value) -> ExampleResult<Value> {
        if value.get("seal_sha256").is_some() {
            return Err(failure("receipt unexpectedly already has a seal"));
        }
        let seal = sha256(&serde_json::to_vec(&value)?);
        value
            .as_object_mut()
            .ok_or_else(|| failure("receipt root must be an object"))?
            .insert("seal_sha256".to_owned(), Value::String(seal));
        Ok(value)
    }

    fn write_new_receipt(path: &Path, receipt: &Value) -> ExampleResult<()> {
        if path.exists() {
            return Err(failure(format!(
                "refusing to overwrite existing P3A scheduler sink receipt {}",
                path.display()
            )));
        }
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .ok_or_else(|| failure("--out needs a parent directory"))?;
        fs::create_dir_all(parent)?;
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| failure("--out filename must be UTF-8"))?;
        let temporary = parent.join(format!(
            ".{name}.{}.p3a-scheduler-sink.tmp",
            std::process::id()
        ));
        let bytes = serde_json::to_vec_pretty(receipt)?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| failure(format!("cannot create receipt temporary file: {error}")))?;
        if let Err(error) = file
            .write_all(&bytes)
            .and_then(|_| file.write_all(b"\n"))
            .and_then(|_| file.sync_all())
        {
            let _ = fs::remove_file(&temporary);
            return Err(Box::new(error));
        }
        if let Err(error) = fs::rename(&temporary, path) {
            let _ = fs::remove_file(&temporary);
            return Err(Box::new(error));
        }
        File::open(parent)?.sync_all()?;
        Ok(())
    }

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn failure(message: impl Into<String>) -> Box<dyn Error> {
        std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}
