//! Exercise one artifact-backed DeepSeek-V4 layer preparation schedule.
//!
//! The recording sink is deliberately a source-byte consumer, not a Metal
//! encoder. A future serial authority mHC encoder and parity-gated parallel
//! replacement both attach at the same `DeepSeekV4NativeStageSink` boundary.
//! No causal layer computation, device dispatch, or TPS claim is made here.

use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use hawking_core::gravity_deepseek_v4::{FULL_STREAM_SCHEMA, FULL_STREAM_STATUS};
use hawking_core::gravity_deepseek_v4_execution_context::{
    DeepSeekV4ControlPayload, DeepSeekV4ExecutionContext, DeepSeekV4ExecutionContextConfig,
    DeepSeekV4SelectedRouteSet,
};
use hawking_core::gravity_deepseek_v4_layer_scheduler::{
    DeepSeekV4LayerPreparationResult, DeepSeekV4LayerPreparationScheduler, DeepSeekV4NativeStage,
    DeepSeekV4NativeStageConsumption, DeepSeekV4NativeStageSink,
};
use hawking_core::gravity_deepseek_v4_runtime_spine::DeepSeekV4ExpertProjection;
use hawking_core::Result as CoreResult;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const RECEIPT_SCHEMA: &str = "hawking.gravity.deepseek_v4.layer_preparation_scheduler.v1";
const RECEIPT_STATUS: &str = "PASS_BOUNDED_LAYER_SOURCE_STAGING_NOT_FORWARD";
const ONE_EXPERT_BUNDLE_BYTES: u64 = 13_369_344;

type ExampleResult<T> = Result<T, Box<dyn Error>>;

struct Args {
    artifact: PathBuf,
    out: PathBuf,
}

#[derive(Default)]
struct RecordingSourceSink {
    events: Vec<Value>,
}

impl DeepSeekV4NativeStageSink for RecordingSourceSink {
    fn consume_native_stage(
        &mut self,
        stage: DeepSeekV4NativeStage<'_>,
    ) -> CoreResult<DeepSeekV4NativeStageConsumption> {
        match stage {
            DeepSeekV4NativeStage::Control { step, payload } => {
                self.events.push(json!({
                    "stage": step.stage.as_str(),
                    "layer": step.layer,
                    "logical_graph_node_ordinal": step.logical_graph_node_ordinal,
                    "payload": control_payload_summary(payload),
                    "authority_or_optimized_kernel_dispatched": false,
                }));
            }
            DeepSeekV4NativeStage::RoutedExpertWave {
                step,
                route_set,
                context,
                accesses,
            } => {
                let expert_w1 = accesses
                    .iter()
                    .map(|access| {
                        let (weight, scale) = context.cached_routed_operator(
                            step.layer,
                            access.key.expert,
                            DeepSeekV4ExpertProjection::W1,
                        )?;
                        Ok(json!({
                            "expert": access.key.expert,
                            "cache_result": access.result.as_str(),
                            "w1_weight_bytes": weight.len(),
                            "w1_weight_sha256": sha256(weight),
                            "w1_scale_bytes": scale.len(),
                            "w1_scale_sha256": sha256(scale),
                        }))
                    })
                    .collect::<CoreResult<Vec<_>>>()?;
                self.events.push(json!({
                    "stage": step.stage.as_str(),
                    "layer": step.layer,
                    "logical_graph_node_ordinal": step.logical_graph_node_ordinal,
                    "provided_route_set": route_set.experts,
                    "router_logits_computed": false,
                    "expert_w1_native_borrows": expert_w1,
                    "expert_kernel_dispatched": false,
                }));
            }
        }
        Ok(DeepSeekV4NativeStageConsumption::default())
    }
}

fn main() -> ExampleResult<()> {
    let args = parse_args()?;
    let config = DeepSeekV4ExecutionContextConfig {
        // A native top-6 wave needs all six selected bundles borrowable at
        // once. The scheduler rejects a smaller hot tier before staging.
        routed_expert_hot_capacity_bytes: ONE_EXPERT_BUNDLE_BYTES * 6,
        routed_expert_cold_capacity_bytes: ONE_EXPERT_BUNDLE_BYTES,
        ..DeepSeekV4ExecutionContextConfig::default()
    };
    let mut context = DeepSeekV4ExecutionContext::open(&args.artifact, config)?;
    let prepared = context.prepare_decode_input(0)?;
    let route_set = DeepSeekV4SelectedRouteSet::new([0, 1, 2, 3, 4, 5])?;
    let mut scheduler = DeepSeekV4LayerPreparationScheduler::new(&context, 0, route_set)?;
    let mut sink = RecordingSourceSink::default();
    while scheduler
        .execute_next_with_sink(&mut context, &mut sink)?
        .is_some()
    {}
    if !scheduler.is_complete() || scheduler.completed().len() != 11 {
        return Err(failure(
            "layer source-staging scheduler did not complete its 11-step program",
        ));
    }
    let full_forward_denied = context.require_full_causal_execution().is_err();
    if !full_forward_denied {
        return Err(failure(
            "layer scheduler unexpectedly admitted a full causal runtime",
        ));
    }

    let reader = context.spine().reader();
    let steps = scheduler
        .completed()
        .iter()
        .map(|step| {
            let source_access_count = match &step.result {
                DeepSeekV4LayerPreparationResult::ControlLease(_) => 0,
                DeepSeekV4LayerPreparationResult::RoutedExpertAccesses(accesses) => accesses.len(),
            };
            json!({
                "sequence": step.sequence,
                "token_position": step.token_position,
                "layer": step.layer,
                "stage": step.stage.as_str(),
                "logical_graph_node_ordinal": step.logical_graph_node_ordinal,
                "is_control_stage": step.stage.is_control_stage(),
                "routed_expert_access_count": source_access_count,
                "actual_command_buffers": step.actual_command_buffers,
                "actual_gpu_dispatches": step.actual_gpu_dispatches,
                "actual_cpu_visible_waits": step.actual_cpu_visible_waits,
            })
        })
        .collect::<Vec<_>>();
    let state = context.expert_cache_state();
    let unsigned = json!({
        "schema": RECEIPT_SCHEMA,
        "status": RECEIPT_STATUS,
        "artifact": {
            "path": reader.artifact_root(),
            "manifest_schema": FULL_STREAM_SCHEMA,
            "manifest_status": FULL_STREAM_STATUS,
            "manifest_seal_sha256": reader.manifest_seal_sha256(),
            "manifest_file_sha256": reader.manifest_file_sha256(),
            "restart_seal_sha256": reader.restart_seal_sha256(),
            "source": {
                "repository": reader.source_identity().repository,
                "revision": reader.source_identity().revision,
                "source_parent_retained": false,
            },
        },
        "schedule": {
            "token_id": prepared.token_id,
            "token_position": prepared.position,
            "layer": scheduler.layer(),
            "provided_route_set": route_set.experts,
            "route_wave_hot_bytes_required": scheduler.route_wave_hot_bytes_required(),
            "router_logits_computed": false,
            "stage_count": scheduler.completed().len(),
            "steps": steps,
            "sink_events": sink.events,
            "immediate_control_lease_consumption": true,
            "swappable_mhc_operator_boundary": {
                "authority_serial_mhc_kernel_dispatched": false,
                "parallel_mhc_kernel_dispatched": false,
                "contract": "both future kernels consume the same source-native MhcControl payload through DeepSeekV4NativeStageSink and require separate parity evidence",
            },
        },
        "cache_after_schedule": {
            "hot_capacity_bytes": state.hot_capacity_bytes,
            "cold_capacity_bytes": state.cold_capacity_bytes,
            "hot_resident_bytes": state.hot_resident_bytes,
            "cold_resident_bytes": state.cold_resident_bytes,
            "source_bundle_loads": state.counters.source_bundle_loads,
            "source_chunk_reads": state.counters.source_chunk_reads,
            "source_payload_bytes_returned": state.counters.source_payload_bytes_returned,
            "source_verified_chunk_bytes": state.counters.source_verified_chunk_bytes,
        },
        "control_arena_after_schedule": {
            "capacity_bytes": context.control_arena().capacity_bytes(),
            "resident_bytes": context.control_arena().resident_bytes(),
            "eviction_count": context.control_arena().eviction_count(),
            "invariants_passed": context.control_arena().assert_invariants().is_ok(),
        },
        "execution_boundary": {
            "registered_43_layer_engine": false,
            "causal_forward": false,
            "metal_resource_allocations": 0,
            "actual_command_buffers": 0,
            "actual_gpu_dispatches": 0,
            "actual_cpu_visible_waits": 0,
            "router_logits_computed": false,
            "kv_storage_allocated": false,
            "expert_matvec_executed": false,
            "hcli_endpoint_started": false,
            "numeric_parity_v21": false,
            "base_true_tps_eligible": false,
            "full_causal_execution_denied": full_forward_denied,
            "claim": "one real source-staging schedule with a recording sink only; not a layer computation, model forward, Engine, parity result, endpoint, or TPS measurement",
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

fn control_payload_summary(payload: &DeepSeekV4ControlPayload) -> Value {
    match payload {
        DeepSeekV4ControlPayload::EmbeddingRow {
            token_id,
            bf16_bits,
        } => json!({
            "kind": "embedding_row",
            "token_id": token_id,
            "native_dtype": "BF16",
            "bytes": bf16_bits.len() * std::mem::size_of::<u16>(),
            "bytes_sha256": sha256(&bf16_bytes(bf16_bits)),
        }),
        DeepSeekV4ControlPayload::Tensor(tensor) => json!({
            "kind": "tensor",
            "name": tensor.name,
            "dtype": tensor.dtype,
            "bytes": tensor.bytes.len(),
            "bytes_sha256": sha256(&tensor.bytes),
        }),
        DeepSeekV4ControlPayload::NativePair(pair) => json!({
            "kind": "native_pair",
            "representation": pair.kind.as_str(),
            "weight": {
                "name": pair.weight.name,
                "bytes": pair.weight.bytes.len(),
                "bytes_sha256": sha256(&pair.weight.bytes),
            },
            "scale": {
                "name": pair.scale.name,
                "bytes": pair.scale.bytes.len(),
                "bytes_sha256": sha256(&pair.scale.bytes),
            },
        }),
        DeepSeekV4ControlPayload::MhcControl {
            layer,
            branch,
            tensors,
        } => json!({
            "kind": "mhc_control",
            "layer": layer,
            "branch": branch.as_str(),
            "tensors": tensors.iter().map(|tensor| json!({
                "name": tensor.name,
                "dtype": tensor.dtype,
                "bytes": tensor.bytes.len(),
                "bytes_sha256": sha256(&tensor.bytes),
            })).collect::<Vec<_>>(),
        }),
    }
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
                    "usage: gravity_deepseek_v4_layer_scheduler --artifact <absolute full Gravity dir> --out <absolute receipt.json>"
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
            "refusing to overwrite existing layer-scheduler receipt {}",
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
        ".{name}.{}.layer-scheduler.tmp",
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

fn bf16_bytes(values: &[u16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<u16>());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn failure(message: impl Into<String>) -> Box<dyn Error> {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
}
