//! Seal the bounded DeepSeek-V4 base-body runtime data-plane admission.
//!
//! This example validates every base-layer operator binding and stages a small
//! set of source-native payloads for future causal-executor integration.  It
//! intentionally does not construct an Engine, allocate Metal buffers,
//! execute a forward, start HCLI, or measure TPS.
//!
//! ```sh
//! cargo run --release -p hawking-core --example gravity_deepseek_v4_runtime_spine -- \
//!   --artifact /absolute/path/to/full-43-layer-stream.gravity \
//!   --out /absolute/path/to/DSV4F_RUNTIME_SPINE_ADMISSION-v1.json
//! ```

use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use hawking_core::gravity_deepseek_v4::{FULL_STREAM_SCHEMA, FULL_STREAM_STATUS};
use hawking_core::gravity_deepseek_v4_runtime_spine::{
    DeepSeekV4ControlProjection, DeepSeekV4ExpertProjection, DeepSeekV4RuntimeSpine,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const RECEIPT_SCHEMA: &str = "hawking.gravity.deepseek_v4.runtime_spine_admission.v1";
const RECEIPT_STATUS: &str = "PASS_FULL_BASE_BODY_TOPOLOGY_AND_BOUNDED_STAGING_NOT_EXECUTABLE";

type ExampleResult<T> = Result<T, Box<dyn Error>>;

struct Args {
    artifact: PathBuf,
    out: PathBuf,
}

fn main() -> ExampleResult<()> {
    let args = parse_args()?;
    let spine = DeepSeekV4RuntimeSpine::admit(&args.artifact)?;

    // These are source-native BF16 storage primitives, not hidden-state or
    // logit computations.  The reader hashes each touched source chunk.
    let embedding_bos = spine.load_embedding_row_bf16(0)?;
    let embedding_last = spine.load_embedding_row_bf16(129_279)?;
    let final_norm = spine.load_final_norm_bf16()?;
    let lm_head_bos = spine.load_lm_head_row_bf16(0)?;
    let lm_head_last = spine.load_lm_head_row_bf16(129_279)?;

    // Exercise the generic source-native staging seam for the three principal
    // representation families the future executor must consume.
    let control = spine.stage_control_pair(0, DeepSeekV4ControlProjection::WqA)?;
    let shared = spine.stage_shared_expert_pair(0, DeepSeekV4ExpertProjection::W1)?;
    let routed = spine.stage_routed_expert_pair(0, 0, DeepSeekV4ExpertProjection::W1)?;

    // Explicitly prove this surface remains below the serving boundary.
    let forward_denial = spine.capabilities().require_full_causal_runtime().is_err();
    let all_model_denial = spine.reject_all_model_materialization().is_err();
    if !forward_denial || !all_model_denial {
        return Err(failure(
            "runtime spine unexpectedly admitted execution or all-model staging",
        ));
    }

    let topology = spine.topology();
    let geometry = spine.geometry();
    let residency = spine.residency_plan();
    let capabilities = spine.capabilities();
    let reader = spine.reader();
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
            "inference_model_py_sha256": spine.source_anchors().inference_model_py_sha256,
            "inference_kernel_py_sha256": spine.source_anchors().inference_kernel_py_sha256,
            "inference_config_json_sha256": spine.source_anchors().inference_config_json_sha256,
            "model_config_json_sha256": spine.source_anchors().model_config_json_sha256,
        },
        "validated_base_body": {
            "base_decoder_layers": geometry.base_layer_count,
            "mtp_auxiliary_layers_excluded_from_base": geometry.mtp_layer_count,
            "hash_router_layers": geometry.hash_layer_count,
            "hidden_size": geometry.hidden_size,
            "vocab_size": geometry.vocab_size,
            "routed_experts_per_layer": geometry.routed_expert_count,
            "shared_experts_per_layer": geometry.shared_expert_count,
            "top_k_routed_experts": geometry.activated_experts,
            "hc_mult": geometry.hc_mult,
            "hc_sinkhorn_iters": geometry.hc_sinkhorn_iters,
            "q_lora_rank": geometry.q_lora_rank,
            "o_lora_rank": geometry.o_lora_rank,
            "base_tensor_count": topology.base_tensor_count,
            "mtp_auxiliary_tensor_count": topology.mtp_auxiliary_tensor_count,
            "admitted_stream_tensor_count": topology.global_tensor_count,
            "base_global_tensor_count": topology.base_global_tensor_count,
            "layers": topology.layers.iter().map(|layer| json!({
                "layer": layer.layer,
                "compression": layer.compression.as_str(),
                "router": layer.router.as_str(),
                "validated_tensor_count": layer.tensor_count,
                "attention_control_native_pairs": 5,
                "shared_expert_native_pairs": 3,
                "routed_expert_native_pairs": 768,
            })).collect::<Vec<_>>(),
        },
        "bounded_residency_plan": {
            "all_model_materialization_allowed": residency.all_model_materialization_allowed,
            "full_base_body_tensor_bytes": residency.full_base_body_tensor_bytes,
            "static_control_tensor_bytes": residency.static_control_tensor_bytes,
            "non_vocab_control_tensor_bytes": residency.non_vocab_control_tensor_bytes,
            "vocab_matrix_tensor_bytes": residency.vocab_matrix_tensor_bytes,
            "routed_expert_tensor_bytes": residency.routed_expert_tensor_bytes,
            "control_resident_ceiling_bytes": residency.control_resident_ceiling_bytes,
            "routed_expert_hot_ceiling_bytes": residency.routed_expert_hot_ceiling_bytes,
            "routed_expert_cold_ceiling_bytes": residency.routed_expert_cold_ceiling_bytes,
            "maximum_single_stage_bytes": residency.maximum_single_stage_bytes,
            "cache_contract": residency.cache_contract,
            "claim": "planning contract only; no control residency, cache hit rate, prefetch, GPU upload, or performance measurement occurred",
        },
        "verified_bounded_staging": {
            "bf16_rows": [
                bits_receipt("embed.weight", 0, &embedding_bos),
                bits_receipt("embed.weight", 129_279, &embedding_last),
                bits_receipt("norm.weight", 0, &final_norm),
                bits_receipt("head.weight", 0, &lm_head_bos),
                bits_receipt("head.weight", 129_279, &lm_head_last),
            ],
            "native_pairs": [
                pair_receipt("layer0_attention_control", &control),
                pair_receipt("layer0_shared_expert", &shared),
                pair_receipt("layer0_routed_expert0", &routed),
            ],
            "source_chunks_sha256_verified_before_each_return": true,
            "staged_payload_persisted": false,
        },
        "capability_gate": {
            "full_stream_reader_admitted": capabilities.full_stream_reader_admitted,
            "source_hashes_and_configs_validated": capabilities.source_hashes_and_configs_validated,
            "all_base_operator_bindings_validated": capabilities.all_base_operator_bindings_validated,
            "bounded_tensor_staging_available": capabilities.bounded_tensor_staging_available,
            "embedding_final_norm_head_primitives_available": capabilities.embedding_final_norm_head_primitives_available,
            "registered_43_layer_engine": capabilities.registered_43_layer_engine,
            "causal_forward_available": capabilities.causal_forward_available,
            "continuation_available": capabilities.continuation_available,
            "metal_dispatches_available": capabilities.metal_dispatches_available,
            "hcli_endpoint_available": capabilities.hcli_endpoint_available,
            "numeric_parity_v21_passed": capabilities.numeric_parity_v21_passed,
            "base_true_tps_eligible": capabilities.base_true_tps_eligible,
            "next_parity_rung": capabilities.next_parity_rung,
            "full_causal_runtime_denied": forward_denial,
            "all_model_materialization_denied": all_model_denial,
        },
        "execution_boundary": {
            "rust_engine_implemented": false,
            "public_cli_serve_registration_changed": false,
            "metal_allocations": 0,
            "gpu_dispatches": 0,
            "forward_tokens": 0,
            "continuation_tokens": 0,
            "hcli_endpoint_started": false,
            "numeric_parity_v21": false,
            "base_true_tps_measured": false,
            "claim": "full 43-layer topology/data-plane admission only; not a causal runtime, Engine, parity result, endpoint, or TPS result",
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
                    "usage: gravity_deepseek_v4_runtime_spine --artifact <absolute full Gravity dir> --out <absolute receipt.json>"
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

fn bits_receipt(tensor: &str, row: usize, bits: &[u16]) -> Value {
    let mut bytes = Vec::with_capacity(bits.len() * 2);
    for value in bits {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    json!({
        "tensor": tensor,
        "row": row,
        "native_dtype": "BF16",
        "elements": bits.len(),
        "bytes": bytes.len(),
        "bytes_sha256": sha256(&bytes),
    })
}

fn pair_receipt(
    label: &str,
    pair: &hawking_core::gravity_deepseek_v4_runtime_spine::DeepSeekV4StagedNativePair,
) -> Value {
    json!({
        "label": label,
        "representation": pair.kind.as_str(),
        "logical_k": pair.logical_k,
        "out_rows": pair.out_rows,
        "weight": {
            "name": pair.weight.name,
            "dtype": pair.weight.dtype,
            "shape": pair.weight.shape,
            "source_shard": pair.weight.source_shard,
            "bytes": pair.weight.bytes.len(),
            "bytes_sha256": sha256(&pair.weight.bytes),
        },
        "scale": {
            "name": pair.scale.name,
            "dtype": pair.scale.dtype,
            "shape": pair.scale.shape,
            "source_shard": pair.scale.source_shard,
            "bytes": pair.scale.bytes.len(),
            "bytes_sha256": sha256(&pair.scale.bytes),
        },
    })
}

fn seal(mut value: Value) -> ExampleResult<Value> {
    let object = value
        .as_object_mut()
        .ok_or_else(|| failure("receipt root must be a JSON object"))?;
    if object.contains_key("seal_sha256") {
        return Err(failure("receipt unexpectedly already has a seal"));
    }
    let seal = sha256(&serde_json::to_vec(&value)?);
    value
        .as_object_mut()
        .expect("receipt object was checked above")
        .insert("seal_sha256".to_owned(), Value::String(seal));
    Ok(value)
}

fn write_new_receipt(path: &Path, receipt: &Value) -> ExampleResult<()> {
    if path.exists() {
        return Err(failure(format!(
            "refusing to overwrite existing runtime-spine receipt {}",
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
    let temporary = parent.join(format!(".{name}.{}.runtime-spine.tmp", std::process::id()));
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
