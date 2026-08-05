//! Exercise the source-backed DeepSeek-V4 execution-context preparation seam.
//!
//! This invokes verified artifact reads, bounded host staging, and the routed
//! FP4 cache. It does not create an Engine, Metal resource, command buffer,
//! causal forward, endpoint, or TPS measurement.
//!
//! ```sh
//! cargo run --release -p hawking-core --example gravity_deepseek_v4_execution_context -- \
//!   --artifact /absolute/path/to/full-43-layer-stream.gravity \
//!   --out /absolute/path/to/DSV4F_P3_EXECUTION_CONTEXT_SCAFFOLD-v1.json
//! ```

use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use hawking_core::gravity_deepseek_v4::{FULL_STREAM_SCHEMA, FULL_STREAM_STATUS};
use hawking_core::gravity_deepseek_v4_execution_context::{
    DeepSeekV4ControlLease, DeepSeekV4ControlPayload, DeepSeekV4ExecutionContext,
    DeepSeekV4ExecutionContextConfig, DeepSeekV4MhcBranch, DeepSeekV4SelectedRouteSet,
};
use hawking_core::gravity_deepseek_v4_runtime_spine::{
    DeepSeekV4ControlProjection, DeepSeekV4ExpertProjection,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const RECEIPT_SCHEMA: &str = "hawking.gravity.deepseek_v4.execution_context_scaffold.v1";
const RECEIPT_STATUS: &str = "PASS_SOURCE_BACKED_EXECUTION_SCAFFOLD_NOT_FORWARD";
const ONE_EXPERT_BUNDLE_BYTES: u64 = 13_369_344;

type ExampleResult<T> = Result<T, Box<dyn Error>>;

struct Args {
    artifact: PathBuf,
    out: PathBuf,
}

fn main() -> ExampleResult<()> {
    let args = parse_args()?;
    let config = DeepSeekV4ExecutionContextConfig {
        // Deliberately small, real cache tiers. Six selected experts exercise
        // bounded hot/cold promotions and eviction without model materialization.
        routed_expert_hot_capacity_bytes: ONE_EXPERT_BUNDLE_BYTES * 2,
        routed_expert_cold_capacity_bytes: ONE_EXPERT_BUNDLE_BYTES,
        ..DeepSeekV4ExecutionContextConfig::default()
    };
    let mut context = DeepSeekV4ExecutionContext::open(&args.artifact, config)?;

    let prepared = context.prepare_decode_input(0)?;
    let mhc = context.stage_mhc_control(0, DeepSeekV4MhcBranch::Attention)?;
    let attention = context.stage_attention_control(0, DeepSeekV4ControlProjection::WqA)?;
    let shared = context.stage_shared_expert_control(0, DeepSeekV4ExpertProjection::W1)?;
    let routes = DeepSeekV4SelectedRouteSet::new([0, 1, 2, 3, 4, 5])?;
    let accesses = context.acquire_selected_route_set(0, routes)?;
    let (routed_weight, routed_scale) =
        context.cached_routed_operator(0, 5, DeepSeekV4ExpertProjection::W1)?;
    let routed_weight_bytes = routed_weight.len();
    let routed_weight_sha256 = sha256(routed_weight);
    let routed_scale_bytes = routed_scale.len();
    let routed_scale_sha256 = sha256(routed_scale);
    let full_forward_denied = context.require_full_causal_execution().is_err();
    if !full_forward_denied {
        return Err(failure(
            "execution context unexpectedly admitted a full causal runtime",
        ));
    }

    let reader = context.spine().reader();
    let state = context.expert_cache_state();
    let decode = context.decode_state();
    let ledger = context.command_ledger();
    let route_receipts = accesses
        .iter()
        .map(|access| {
            json!({
                "layer": access.key.layer,
                "expert": access.key.expert,
                "result": access.result.as_str(),
                "source_read": access.source_read.as_ref().map(source_read_receipt),
                "state_after": expert_cache_state_receipt(&access.state_after),
            })
        })
        .collect::<Vec<_>>();
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
        "source_hash_anchors": {
            "inference_model_py_sha256": context.spine().source_anchors().inference_model_py_sha256,
            "inference_kernel_py_sha256": context.spine().source_anchors().inference_kernel_py_sha256,
            "inference_config_json_sha256": context.spine().source_anchors().inference_config_json_sha256,
            "model_config_json_sha256": context.spine().source_anchors().model_config_json_sha256,
        },
        "context_config": {
            "max_context_tokens": context.config().max_context_tokens,
            "control_staging_capacity_bytes": context.config().control_staging_capacity_bytes,
            "control_resident_ceiling_bytes_contract": context.config().control_resident_ceiling_bytes,
            "routed_expert_hot_capacity_bytes": context.config().routed_expert_hot_capacity_bytes,
            "routed_expert_cold_capacity_bytes": context.config().routed_expert_cold_capacity_bytes,
            "command_ledger_capacity": context.config().command_ledger_capacity,
            "all_model_materialization_allowed": false,
        },
        "source_backed_preparation": {
            "prepared_decode_input": {
                "token_id": prepared.token_id,
                "position": prepared.position,
                "embedding": control_payload_receipt(&context, prepared.embedding_lease)?,
                "command_graph": graph_receipt(&prepared.command_graph),
            },
            "staged_control_payloads": [
                control_payload_receipt(&context, mhc)?,
                control_payload_receipt(&context, attention)?,
                control_payload_receipt(&context, shared)?,
            ],
            "mhc_seed_slot": {
                "copies": decode.m_hc.copies,
                "hidden_size": decode.m_hc.hidden_size,
                "bf16_elements": decode.m_hc.bf16_bits.len(),
                "bf16_bytes_sha256": sha256(&bf16_bytes(&decode.m_hc.bf16_bits)),
                "initialized": decode.m_hc.initialized,
            },
            "kv_state_slots": decode.kv_slots.iter().map(|slot| json!({
                "layer": slot.layer,
                "compression": slot.compression.as_str(),
                "sliding_window_tokens": slot.sliding_window_tokens,
                "compressed_tokens_capacity": slot.compressed_tokens_capacity,
                "logical_value_width": slot.logical_value_width,
                "logical_bf16_bytes_ceiling": slot.logical_bf16_bytes_ceiling,
                "storage_allocated": slot.storage_allocated,
                "writes_completed": slot.writes_completed,
            })).collect::<Vec<_>>(),
            "routed_expert_acquisition": {
                "provided_route_set": routes.experts,
                "router_logits_computed": false,
                "expert_kernel_executed": false,
                "accesses": route_receipts,
                "final_cache_state": expert_cache_state_receipt(&state),
                "resident_operator_handoff": {
                    "layer": 0,
                    "expert": 5,
                    "projection": "w1",
                    "weight_bytes": routed_weight_bytes,
                    "weight_bytes_sha256": routed_weight_sha256,
                    "scale_bytes": routed_scale_bytes,
                    "scale_bytes_sha256": routed_scale_sha256,
                    "executed": false,
                },
            },
            "control_arena": {
                "capacity_bytes": context.control_arena().capacity_bytes(),
                "resident_bytes": context.control_arena().resident_bytes(),
                "eviction_count": context.control_arena().eviction_count(),
                "invariants_passed": context.control_arena().assert_invariants().is_ok(),
            },
            "command_ledger": {
                "event_count": ledger.events().len(),
                "evicted_events": ledger.evicted_events(),
                "events": ledger.events().iter().map(|event| json!({
                    "sequence": event.sequence,
                    "token_position": event.token_position,
                    "kind": event.kind.as_str(),
                    "label": event.label,
                    "graph_sha256": event.graph_sha256,
                    "planned_nodes": event.planned_nodes,
                    "actual_command_buffers": event.actual_command_buffers,
                    "actual_gpu_dispatches": event.actual_gpu_dispatches,
                    "actual_cpu_visible_waits": event.actual_cpu_visible_waits,
                })).collect::<Vec<_>>(),
            },
        },
        "execution_boundary": {
            "registered_43_layer_engine": false,
            "causal_forward": false,
            "continuation": false,
            "metal_resource_allocations": 0,
            "actual_command_buffers": 0,
            "actual_gpu_dispatches": 0,
            "actual_cpu_visible_waits": 0,
            "kv_storage_allocated": false,
            "router_logits_computed": false,
            "expert_matvec_executed": false,
            "lm_head_executed": false,
            "sampling_executed": false,
            "hcli_endpoint_started": false,
            "numeric_parity_v21": false,
            "base_true_tps_eligible": false,
            "full_causal_execution_denied": full_forward_denied,
            "claim": "real source-backed preparation only; not a model forward, Engine, parity result, endpoint, or TPS measurement",
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

fn control_payload_receipt(
    context: &DeepSeekV4ExecutionContext,
    lease: DeepSeekV4ControlLease,
) -> ExampleResult<Value> {
    let value = context.control_arena().get(lease)?;
    let payload = match value {
        DeepSeekV4ControlPayload::EmbeddingRow {
            token_id,
            bf16_bits,
        } => json!({
            "kind": "embedding_row",
            "token_id": token_id,
            "native_dtype": "BF16",
            "elements": bf16_bits.len(),
            "bytes": bf16_bits.len() * std::mem::size_of::<u16>(),
            "bytes_sha256": sha256(&bf16_bytes(bf16_bits)),
        }),
        DeepSeekV4ControlPayload::Tensor(tensor) => tensor_receipt(tensor),
        DeepSeekV4ControlPayload::NativePair(pair) => json!({
            "kind": "native_pair",
            "representation": pair.kind.as_str(),
            "logical_k": pair.logical_k,
            "out_rows": pair.out_rows,
            "weight": tensor_receipt(&pair.weight),
            "scale": tensor_receipt(&pair.scale),
        }),
        DeepSeekV4ControlPayload::MhcControl {
            layer,
            branch,
            tensors,
        } => json!({
            "kind": "mhc_control",
            "layer": layer,
            "branch": branch.as_str(),
            "tensors": tensors.iter().map(tensor_receipt).collect::<Vec<_>>(),
        }),
    };
    Ok(json!({
        "lease": { "id": lease.id, "generation": lease.generation },
        "payload": payload,
    }))
}

fn tensor_receipt(
    tensor: &hawking_core::gravity_deepseek_v4_runtime_spine::DeepSeekV4StagedTensor,
) -> Value {
    json!({
        "name": tensor.name,
        "dtype": tensor.dtype,
        "shape": tensor.shape,
        "source_shard": tensor.source_shard,
        "range": { "start": tensor.range.start, "end": tensor.range.end },
        "bytes": tensor.bytes.len(),
        "bytes_sha256": sha256(&tensor.bytes),
    })
}

fn graph_receipt(
    graph: &hawking_core::gravity_deepseek_v4_execution_context::DeepSeekV4TokenCommandGraph,
) -> Value {
    json!({
        "position": graph.position,
        "node_count": graph.nodes.len(),
        "sha256": graph.graph_sha256,
        "logical_node_counts": {
            "embed_and_mhc_seed": graph.nodes.iter().filter(|node| node.kind.as_str() == "embed_and_mhc_seed").count(),
            "per_base_layer": 9,
            "base_layers": 43,
            "final_nodes": 3,
        },
        "actual_command_buffers": 0,
        "actual_gpu_dispatches": 0,
        "actual_cpu_visible_waits": 0,
    })
}

fn source_read_receipt(
    source: &hawking_core::gravity_deepseek_v4_expert_cache::ExpertBundleSourceRead,
) -> Value {
    json!({
        "payload_bytes_returned": source.payload_bytes_returned,
        "verified_chunk_bytes": source.verified_chunk_bytes,
        "source_chunk_read_count": source.source_chunk_read_count,
        "chunks": source.chunk_paths.iter().map(|chunk| json!({
            "tensor_name": chunk.tensor_name,
            "tensor_role": chunk.tensor_role,
            "chunk_relpath": chunk.chunk_relpath,
            "chunk_sha256": chunk.chunk_sha256,
            "bytes": chunk.bytes,
        })).collect::<Vec<_>>(),
    })
}

fn expert_cache_state_receipt(
    state: &hawking_core::gravity_deepseek_v4_expert_cache::ExpertCacheState,
) -> Value {
    let keys = |keys: &[hawking_core::gravity_deepseek_v4_expert_cache::ExpertBundleKey]| {
        keys.iter()
            .map(|key| json!({ "layer": key.layer, "expert": key.expert }))
            .collect::<Vec<_>>()
    };
    json!({
        "hot_capacity_bytes": state.hot_capacity_bytes,
        "cold_capacity_bytes": state.cold_capacity_bytes,
        "hot_resident_bytes": state.hot_resident_bytes,
        "cold_resident_bytes": state.cold_resident_bytes,
        "hot_keys_lru_to_mru": keys(&state.hot_keys_lru_to_mru),
        "cold_keys_lru_to_mru": keys(&state.cold_keys_lru_to_mru),
        "counters": {
            "demand_requests": state.counters.demand_requests,
            "prefetch_requests": state.counters.prefetch_requests,
            "demand_hot_hits": state.counters.demand_hot_hits,
            "demand_cold_hits": state.counters.demand_cold_hits,
            "demand_misses": state.counters.demand_misses,
            "prefetch_hot_hits": state.counters.prefetch_hot_hits,
            "prefetch_cold_hits": state.counters.prefetch_cold_hits,
            "prefetch_misses": state.counters.prefetch_misses,
            "promotions": state.counters.promotions,
            "hot_demotions": state.counters.hot_demotions,
            "hot_evictions": state.counters.hot_evictions,
            "cold_evictions": state.counters.cold_evictions,
            "demand_source_loads": state.counters.demand_source_loads,
            "prefetch_source_loads": state.counters.prefetch_source_loads,
            "source_bundle_loads": state.counters.source_bundle_loads,
            "source_tensor_reads": state.counters.source_tensor_reads,
            "source_chunk_reads": state.counters.source_chunk_reads,
            "source_payload_bytes_returned": state.counters.source_payload_bytes_returned,
            "source_verified_chunk_bytes": state.counters.source_verified_chunk_bytes,
        },
    })
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
                    "usage: gravity_deepseek_v4_execution_context --artifact <absolute full Gravity dir> --out <absolute receipt.json>"
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
            "refusing to overwrite existing execution-context receipt {}",
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
        ".{name}.{}.execution-context.tmp",
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
