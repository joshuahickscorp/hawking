//! Sealed CPU source-algorithm successor for DeepSeek-V4 layer 0 / position 1.
//!
//! This binds the sealed position-one complete-attention CPU receipt to a
//! recomputed, hash-matched four-lane BF16 HC state, then executes the source
//! FFN successor for tokenizer ID `19923` (`Hello`).  It deliberately makes no
//! Metal, registered-runtime, upstream-runtime-parity, endpoint, generated
//! token, or TPS claim.
//!
//! ```sh
//! cargo run --release -p hawking-core --example gravity_deepseek_v4_layer0_position1_full_ffn_cpu_oracle -- \
//!   --artifact /absolute/path/to/full-43-layer-stream.gravity \
//!   --predecessor /absolute/path/to/DSV4F_LAYER0_POSITION1_COMPLETE_ATTENTION_CPU_ORACLE-v1.json \
//!   --out /absolute/path/to/DSV4F_LAYER0_POSITION1_FULL_FFN_CPU_ORACLE-v1.json
//! ```

use half::bf16;
use hawking_core::gravity_deepseek_v4::{
    DeepSeekV4FullStreamReader, DeepSeekV4Segment, DeepSeekV4TensorMetadata, NativeScalePair,
    FULL_STREAM_SCHEMA, FULL_STREAM_STATUS,
};
use hawking_core::gravity_deepseek_v4_layer0_continuation::{
    POSITION1, POSITION1_TOKEN_ID, POSITION1_TOKEN_STRING,
};
use hawking_core::gravity_deepseek_v4_layer0_moe::{
    verify_layer0_moe_source_anchors, DeepSeekV4Layer0MoeSourceAnchors,
    Layer0MoeSuccessorCpuOracleResult, QuantizedLinearCpuStage, ACTIVATED_EXPERTS, HASH_LAYERS,
    LAYER0_FFN_GATE_TID2EID, LAYER0_FFN_GATE_WEIGHT, LAYER0_FFN_NORM_WEIGHT, LAYER0_HC_FFN_BASE,
    LAYER0_HC_FFN_FN, LAYER0_HC_FFN_SCALE, MOE_INTER_DIM, ROUTED_EXPERTS, ROUTE_SCALE,
    SHARED_EXPERTS, SWIGLU_LIMIT,
};
use hawking_core::gravity_deepseek_v4_layer0_position1_ffn::{
    layer0_position1_full_ffn_cpu_oracle, verify_layer0_position1_full_ffn_source_anchors,
    Layer0Position1FullFfnCpuOracleResult,
};
use hawking_core::gravity_deepseek_v4_layer0_prefix::{
    HC_FLAT_WIDTH, HC_MIX_WIDTH, HC_MULT, HIDDEN_SIZE, PREFIX_TOKEN_ID,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const RECEIPT_SCHEMA: &str = "hawking.gravity.deepseek_v4.layer0_position1_full_ffn_cpu_oracle.v1";
const RECEIPT_STATUS: &str =
    "PASS_SOURCE_DERIVED_CPU_LAYER0_POSITION1_FULL_FFN_NOT_INDEPENDENT_UPSTREAM_RUNTIME_PARITY";
const PREDECESSOR_SCHEMA: &str =
    "hawking.gravity.deepseek_v4.layer0_position1_complete_attention_cpu_oracle.v1";
const PREDECESSOR_STATUS: &str =
    "PASS_SOURCE_DERIVED_CPU_LAYER0_POSITION1_COMPLETE_ATTENTION_NOT_RUNTIME";

type ExampleResult<T> = Result<T, Box<dyn Error>>;

struct Args {
    artifact: PathBuf,
    predecessor: PathBuf,
    out: PathBuf,
}

fn main() -> ExampleResult<()> {
    let args = parse_args()?;
    let reader = DeepSeekV4FullStreamReader::admit(&args.artifact)?;
    verify_layer0_position1_full_ffn_source_anchors(&reader)?;
    let anchors = verify_layer0_moe_source_anchors(&reader)?;
    let result = layer0_position1_full_ffn_cpu_oracle(&reader)?;
    validate_result_geometry(&result)?;
    let predecessor = bind_predecessor(&reader, &args.predecessor, &result)?;

    let hcf_fn = reader.tensor_metadata(LAYER0_HC_FFN_FN)?;
    let hcf_base = reader.tensor_metadata(LAYER0_HC_FFN_BASE)?;
    let hcf_scale = reader.tensor_metadata(LAYER0_HC_FFN_SCALE)?;
    let ffn_norm = reader.tensor_metadata(LAYER0_FFN_NORM_WEIGHT)?;
    let gate_weight = reader.tensor_metadata(LAYER0_FFN_GATE_WEIGHT)?;
    let tid2eid = reader.tensor_metadata(LAYER0_FFN_GATE_TID2EID)?;
    let tid2eid_row_bytes = (ACTIVATED_EXPERTS * std::mem::size_of::<i64>()) as u64;
    let tid2eid_start = POSITION1_TOKEN_ID
        .checked_mul(tid2eid_row_bytes)
        .ok_or_else(|| failure("position-one tid2eid byte offset overflow"))?;
    let tid2eid_end = tid2eid_start
        .checked_add(tid2eid_row_bytes)
        .ok_or_else(|| failure("position-one tid2eid byte end overflow"))?;
    if tid2eid_end > tid2eid.bytes {
        return Err(failure(
            "position-one tid2eid row exceeds the source tensor",
        ));
    }

    let routed_pairs = result
        .ffn
        .routed_experts
        .iter()
        .map(|expert| routed_pair_contract_json(&reader, expert.expert_id))
        .collect::<ExampleResult<Vec<_>>>()?;
    let shared_pairs = shared_pair_contract_json(&reader)?;

    let unsigned = json!({
        "schema": RECEIPT_SCHEMA,
        "status": RECEIPT_STATUS,
        "artifact": {
            "path": reader.artifact_root().display().to_string(),
            "manifest_schema": FULL_STREAM_SCHEMA,
            "manifest_status": FULL_STREAM_STATUS,
            "manifest_seal_sha256": reader.manifest_seal_sha256(),
            "manifest_file_sha256": reader.manifest_file_sha256(),
            "restart_receipt_seal_sha256": reader.restart_seal_sha256(),
            "source": {
                "repository": reader.source_identity().repository,
                "revision": reader.source_identity().revision,
                "source_parent_retained": false,
            },
        },
        "predecessor_binding": predecessor,
        "source_algorithm_bindings": source_bindings_json(&anchors),
        "bounded_source_reads": {
            "reader_admission_verified_manifest_source_index_offsets_and_chunk_tree": true,
            "all_chunk_sha256_bytes_verified_globally": false,
            "every_listed_tensor_read_touched_chunk_sha256_verified_before_cpu_use": true,
            "parent_safetensors_materialized": false,
            "position1_complete_attention_predecessor": {
                "executor": "public layer0_position1_complete_attention_cpu_oracle",
                "token_ids": [PREFIX_TOKEN_ID, POSITION1_TOKEN_ID],
                "position": POSITION1,
                "result_input_hc_state": bf16_checkpoint_json(
                    &result.complete_attention.hc_attention_post_bf16_bits,
                    &[HC_MULT, HIDDEN_SIZE],
                )?,
                "recomputed_from_admitted_artifact_and_hash_matched_to_sealed_predecessor": true,
                "raw_hidden_state_retained": false,
            },
            "layer0_hc_ffn_fn": tensor_full_binding_json(hcf_fn),
            "layer0_hc_ffn_base": tensor_full_binding_json(hcf_base),
            "layer0_hc_ffn_scale": tensor_full_binding_json(hcf_scale),
            "layer0_ffn_norm_weight": tensor_full_binding_json(ffn_norm),
            "layer0_gate_weight": tensor_full_binding_json(gate_weight),
            "layer0_gate_tid2eid_hello_row": tensor_range_binding_json(
                tid2eid,
                tid2eid_start,
                tid2eid_end,
            ),
            "selected_routed_expert_native_fp4_pairs": routed_pairs,
            "shared_expert_native_fp8_pairs": shared_pairs,
        },
        "intermediate_receipts": intermediate_receipts_json(&result)?,
        "work_accounting": {
            "batch": 1,
            "sequence_tokens": 2,
            "position": POSITION1,
            "source_world_size": 1,
            "complete_attention_predecessor_executed_by_public_source_algorithm": true,
            "mhc_ffn_linear_dot_products": (HC_MIX_WIDTH * HC_FLAT_WIDTH) as u64,
            "gate_bf16_linear_dot_products": (ROUTED_EXPERTS * HIDDEN_SIZE) as u64,
            "hash_tid2eid_elements_read": ACTIVATED_EXPERTS,
            "routed_fp4_linear_dot_products": (ACTIVATED_EXPERTS * 3 * MOE_INTER_DIM * HIDDEN_SIZE) as u64,
            "shared_fp8_linear_dot_products": (3 * MOE_INTER_DIM * HIDDEN_SIZE) as u64,
            "route_weighted_combine_values": HIDDEN_SIZE as u64,
            "mhc_ffn_post_output_values": (HC_MULT * HIDDEN_SIZE) as u64,
            "gpu_dispatches": 0,
            "command_buffers": 0,
            "cpu_visible_waits": 0,
        },
        "execution_boundary": {
            "source_derived_cpu_algorithm_position1_ffn_successor": true,
            "independently_upstream_runtime_parity": false,
            "numeric_parity_v2_1_complete": false,
            "source_runtime_executed": false,
            "pytorch_or_tilelang_executed": false,
            "full_decoder_layer_forward": false,
            "full_model_loaded": false,
            "full_model_forward": false,
            "generated_tokens": 0,
            "metal_allocations": 0,
            "gpu_dispatches": 0,
            "command_buffers": 0,
            "hcli_endpoint_started": false,
            "base_true_tps_measured": false,
            "fallback": "NOT_APPLICABLE_CPU_ALGORITHM_ORACLE_NOT_RUNTIME",
            "claim": "one tokenizer-bound [BOS, Hello] layer-0 position-1 complete-attention-to-FFN CPU source-algorithm successor only; NOT independently executed upstream-runtime parity, a registered V4 runtime, GPU result, full decoder-layer forward, token generation, HCLI endpoint, or BASE_TRUE_TPS result",
        },
    });
    let receipt = seal(unsigned)?;
    write_new_sealed_receipt(&args.out, &receipt)?;
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

fn validate_result_geometry(result: &Layer0Position1FullFfnCpuOracleResult) -> ExampleResult<()> {
    if result.complete_attention.causal.token0_id != PREFIX_TOKEN_ID
        || result.complete_attention.causal.token1_id != POSITION1_TOKEN_ID
        || result.complete_attention.hc_attention_post_bf16_bits.len() != HC_FLAT_WIDTH
    {
        return Err(failure(
            "position-one complete-attention predecessor has unexpected identity or HC geometry",
        ));
    }
    validate_ffn_geometry(&result.ffn)
}

fn validate_ffn_geometry(result: &Layer0MoeSuccessorCpuOracleResult) -> ExampleResult<()> {
    if result.ffn_hc_pre.mixes_f32.len() != HC_MIX_WIDTH
        || result.ffn_hc_pre.pre_f32.len() != HC_MULT
        || result.ffn_hc_pre.post_f32.len() != HC_MULT
        || result.ffn_hc_pre.comb_f32.len() != HC_MULT * HC_MULT
        || result.ffn_hc_pre.reduced_bf16_bits.len() != HIDDEN_SIZE
        || result.ffn_norm_bf16_bits.len() != HIDDEN_SIZE
        || result.route.token_id != POSITION1_TOKEN_ID
        || result.route.logits_f32.len() != ROUTED_EXPERTS
        || result.route.original_scores_f32.len() != ROUTED_EXPERTS
        || result.route.selected_expert_ids.len() != ACTIVATED_EXPERTS
        || result.route.selected_weights_f32.len() != ACTIVATED_EXPERTS
        || result.routed_experts.len() != ACTIVATED_EXPERTS
        || result.moe_output_bf16_bits.len() != HIDDEN_SIZE
        || result.hc_ffn_post_bf16_bits.len() != HC_FLAT_WIDTH
    {
        return Err(failure(
            "position-one FFN successor returned unexpected geometry",
        ));
    }
    if !result.ffn_hc_pre.flat_rsqrt.is_finite()
        || result
            .ffn_hc_pre
            .mixes_f32
            .iter()
            .chain(&result.ffn_hc_pre.pre_f32)
            .chain(&result.ffn_hc_pre.post_f32)
            .chain(&result.ffn_hc_pre.comb_f32)
            .chain(&result.route.logits_f32)
            .chain(&result.route.original_scores_f32)
            .chain(&result.route.selected_weights_f32)
            .any(|value| !value.is_finite())
    {
        return Err(failure(
            "position-one FFN successor contains non-finite f32 output",
        ));
    }
    let route_sum = result.route.selected_weights_f32.iter().sum::<f32>();
    if (route_sum - ROUTE_SCALE).abs() > 1.0e-5 {
        return Err(failure(
            "position-one route weights do not sum to source route_scale",
        ));
    }
    let mut seen_slots = [false; ACTIVATED_EXPERTS];
    let mut seen_experts = [false; ROUTED_EXPERTS];
    let mut previous = None;
    for expert in &result.routed_experts {
        if expert.source_top_slot >= ACTIVATED_EXPERTS
            || seen_slots[expert.source_top_slot]
            || expert.expert_id >= ROUTED_EXPERTS as u64
            || seen_experts[expert.expert_id as usize]
            || !expert.route_weight.is_finite()
            || expert.route_weight != result.route.selected_weights_f32[expert.source_top_slot]
            || expert.expert_id != result.route.selected_expert_ids[expert.source_top_slot]
        {
            return Err(failure(
                "routed expert result has invalid source route metadata",
            ));
        }
        if let Some((previous_expert, previous_slot)) = previous {
            if (expert.expert_id, expert.source_top_slot) < (previous_expert, previous_slot) {
                return Err(failure(
                    "routed experts are not in source numeric-ID loop order",
                ));
            }
        }
        previous = Some((expert.expert_id, expert.source_top_slot));
        seen_slots[expert.source_top_slot] = true;
        seen_experts[expert.expert_id as usize] = true;
        validate_stage(&expert.gate, MOE_INTER_DIM, HIDDEN_SIZE)?;
        validate_stage(&expert.up, MOE_INTER_DIM, HIDDEN_SIZE)?;
        validate_stage(&expert.down, HIDDEN_SIZE, MOE_INTER_DIM)?;
        if expert.weighted_swiglu_bf16_bits.len() != MOE_INTER_DIM {
            return Err(failure("routed SwiGLU checkpoint has unexpected width"));
        }
    }
    if seen_slots.iter().any(|seen| !seen) {
        return Err(failure(
            "routed expert result does not cover all six source slots",
        ));
    }
    validate_stage(&result.shared_expert.gate, MOE_INTER_DIM, HIDDEN_SIZE)?;
    validate_stage(&result.shared_expert.up, MOE_INTER_DIM, HIDDEN_SIZE)?;
    validate_stage(&result.shared_expert.down, HIDDEN_SIZE, MOE_INTER_DIM)?;
    if result.shared_expert.swiglu_bf16_bits.len() != MOE_INTER_DIM {
        return Err(failure(
            "shared-expert SwiGLU checkpoint has unexpected width",
        ));
    }
    Ok(())
}

fn validate_stage(stage: &QuantizedLinearCpuStage, rows: usize, cols: usize) -> ExampleResult<()> {
    if stage.quantized_input.activation_e4m3fn.len() != cols
        || stage.quantized_input.scales_e8m0fnu.len() != cols / 128
        || stage.quantized_input.decoded_scales_f32.len() != cols / 128
        || stage.output.fp32.len() != rows
        || stage.output.bf16_bits.len() != rows
        || stage.output.fp32.iter().any(|value| !value.is_finite())
    {
        return Err(failure(
            "quantized source-linear stage has unexpected geometry or a non-finite output",
        ));
    }
    Ok(())
}

fn bind_predecessor(
    reader: &DeepSeekV4FullStreamReader,
    path: &Path,
    result: &Layer0Position1FullFfnCpuOracleResult,
) -> ExampleResult<Value> {
    let bytes = fs::read(path)?;
    let value: Value = serde_json::from_slice(&bytes)?;
    verify_seal(&value, "position-one complete-attention predecessor")?;
    if value.get("schema").and_then(Value::as_str) != Some(PREDECESSOR_SCHEMA)
        || value.get("status").and_then(Value::as_str) != Some(PREDECESSOR_STATUS)
        || value
            .pointer("/artifact/manifest_seal_sha256")
            .and_then(Value::as_str)
            != Some(reader.manifest_seal_sha256())
    {
        return Err(failure(
            "predecessor is not the matching sealed position-one complete-attention CPU receipt",
        ));
    }
    let predecessor_hash = value
        .pointer("/tail_checkpoints/attention_hc_post_bf16_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| failure("predecessor omits its complete-attention HC-state hash"))?;
    let actual_hash = sha256_hex(&u16_le_bytes(
        &result.complete_attention.hc_attention_post_bf16_bits,
    ));
    if predecessor_hash != actual_hash {
        return Err(failure(
            "recomputed position-one complete-attention HC state does not match its sealed predecessor hash",
        ));
    }
    Ok(json!({
        "path": path.display().to_string(),
        "receipt_file_sha256": sha256_hex(&bytes),
        "schema": PREDECESSOR_SCHEMA,
        "status": PREDECESSOR_STATUS,
        "seal_sha256": value.get("seal_sha256").and_then(Value::as_str),
        "manifest_seal_sha256": reader.manifest_seal_sha256(),
        "attention_hc_post_bf16_sha256": actual_hash,
        "recomputed_state_hash_matches_sealed_predecessor": true,
        "predecessor_raw_hidden_state_retained": false,
    }))
}

fn source_bindings_json(anchors: &DeepSeekV4Layer0MoeSourceAnchors) -> Value {
    json!({
        "official_assets_verified_by_admitted_full_stream_and_exact_anchor": {
            "inference/model.py": anchors.attention.prefix.act_quant.inference_model_py_sha256,
            "inference/kernel.py": anchors.attention.prefix.act_quant.inference_kernel_py_sha256,
            "inference/convert.py": anchors.attention.inference_convert_py_sha256,
            "inference/config.json": anchors.attention.prefix.act_quant.inference_config_json_sha256,
            "config.json": anchors.attention.prefix.act_quant.model_config_json_sha256,
            "tokenizer.json": anchors.attention.prefix.tokenizer_json_sha256,
            "tokenizer_config.json": anchors.attention.prefix.tokenizer_config_json_sha256,
        },
        "layer0_position1_moe_contract": {
            "token_ids": [PREFIX_TOKEN_ID, POSITION1_TOKEN_ID],
            "position1_bpe": POSITION1_TOKEN_STRING,
            "position": POSITION1,
            "layer": 0,
            "hash_routing": anchors.layer0_hash_routing,
            "artifact_tid2eid_dtype": anchors.source_tid2eid_storage_dtype,
            "tid2eid_lookup": "layers.0.ffn.gate.tid2eid[19923]",
            "n_hash_layers": HASH_LAYERS,
            "n_routed_experts": ROUTED_EXPERTS,
            "n_activated_experts": ACTIVATED_EXPERTS,
            "n_shared_experts": SHARED_EXPERTS,
            "moe_intermediate_size": MOE_INTER_DIM,
            "score_func": "sqrtsoftplus",
            "route_scale": f32_text(ROUTE_SCALE),
            "swiglu_limit": f32_text(SWIGLU_LIMIT),
            "routed_representation": "native FP4 E2M1FN x2 low-nibble then high-nibble along K; E8M0 scale per 32 logical K",
            "shared_representation": "native FP8 E4M3FN; E8M0 scale per 128x128 block",
        },
    })
}

fn intermediate_receipts_json(
    result: &Layer0Position1FullFfnCpuOracleResult,
) -> ExampleResult<Value> {
    let ffn = &result.ffn;
    let routed = ffn
        .routed_experts
        .iter()
        .map(|expert| {
            Ok(json!({
                "source_execution_order": {
                    "expert_id": expert.expert_id,
                    "source_top_slot": expert.source_top_slot,
                    "route_weight": f32_text(expert.route_weight),
                },
                "w1_fp4_linear": stage_json(&expert.gate, &[MOE_INTER_DIM])?,
                "w3_fp4_linear": stage_json(&expert.up, &[MOE_INTER_DIM])?,
                "weighted_swiglu": bf16_checkpoint_json(&expert.weighted_swiglu_bf16_bits, &[MOE_INTER_DIM])?,
                "w2_fp4_linear": stage_json(&expert.down, &[HIDDEN_SIZE])?,
            }))
        })
        .collect::<ExampleResult<Vec<_>>>()?;
    Ok(json!({
        "attention_successor_input_hc_state": bf16_checkpoint_json(
            &result.complete_attention.hc_attention_post_bf16_bits,
            &[HC_MULT, HIDDEN_SIZE],
        )?,
        "mhc_ffn_pre": {
            "flat_rsqrt": f32_text(ffn.ffn_hc_pre.flat_rsqrt),
            "mixes": f32_checkpoint_json(&ffn.ffn_hc_pre.mixes_f32, &[HC_MIX_WIDTH])?,
            "pre": f32_checkpoint_json(&ffn.ffn_hc_pre.pre_f32, &[HC_MULT])?,
            "post": f32_checkpoint_json(&ffn.ffn_hc_pre.post_f32, &[HC_MULT])?,
            "comb": f32_checkpoint_json(&ffn.ffn_hc_pre.comb_f32, &[HC_MULT, HC_MULT])?,
            "reduced": bf16_checkpoint_json(&ffn.ffn_hc_pre.reduced_bf16_bits, &[HIDDEN_SIZE])?,
        },
        "ffn_norm": bf16_checkpoint_json(&ffn.ffn_norm_bf16_bits, &[HIDDEN_SIZE])?,
        "gate": {
            "source_behavior": "scores are computed before layer-0 tid2eid[Hello] IDs are selected; gathered original sqrt-softplus scores determine route weights",
            "logits": f32_checkpoint_json(&ffn.route.logits_f32, &[ROUTED_EXPERTS])?,
            "original_scores": f32_checkpoint_json(&ffn.route.original_scores_f32, &[ROUTED_EXPERTS])?,
            "selected_expert_ids_top_slot_order": ffn.route.selected_expert_ids,
            "selected_weights_top_slot_order": f32_slice_text(&ffn.route.selected_weights_f32),
            "selected_weight_sum": f32_text(ffn.route.selected_weights_f32.iter().sum::<f32>()),
        },
        "routed_experts_source_loop_order": routed,
        "shared_expert": {
            "w1_fp8_linear": stage_json(&ffn.shared_expert.gate, &[MOE_INTER_DIM])?,
            "w3_fp8_linear": stage_json(&ffn.shared_expert.up, &[MOE_INTER_DIM])?,
            "swiglu": bf16_checkpoint_json(&ffn.shared_expert.swiglu_bf16_bits, &[MOE_INTER_DIM])?,
            "w2_fp8_linear": stage_json(&ffn.shared_expert.down, &[HIDDEN_SIZE])?,
        },
        "route_weighted_combine": {
            "source_accumulator_dtype": "F32",
            "source_order": "ascending (expert_id, source_top_slot), then shared expert",
            "output_after_type_as_input_bf16": bf16_checkpoint_json(&ffn.moe_output_bf16_bits, &[HIDDEN_SIZE])?,
        },
        "mhc_ffn_post": {
            "source_formula": "post[k] * moe_output + sum_j(comb[j,k] * position1_attention_hc_residual[j])",
            "output": bf16_checkpoint_json(&ffn.hc_ffn_post_bf16_bits, &[HC_MULT, HIDDEN_SIZE])?,
        },
    }))
}

fn stage_json(stage: &QuantizedLinearCpuStage, output_shape: &[usize]) -> ExampleResult<Value> {
    Ok(json!({
        "source_operator": "model.py::linear -> kernel.py::act_quant(block=128, ue8m0, F8_E8M0FNU scale) -> native quantized GEMM",
        "activation_e4m3fn_sha256": sha256_hex(&stage.quantized_input.activation_e4m3fn),
        "activation_bytes": stage.quantized_input.activation_e4m3fn.len(),
        "scale_e8m0fnu_sha256": sha256_hex(&stage.quantized_input.scales_e8m0fnu),
        "scale_bytes": stage.quantized_input.scales_e8m0fnu.len(),
        "output_bf16": bf16_checkpoint_json(&stage.output.bf16_bits, output_shape)?,
        "output_fp32": f32_checkpoint_json(&stage.output.fp32, output_shape)?,
    }))
}

fn routed_pair_contract_json(
    reader: &DeepSeekV4FullStreamReader,
    expert_id: u64,
) -> ExampleResult<Value> {
    let stem = format!("layers.0.ffn.experts.{expert_id}");
    let w1 = reader.native_scale_pair(&format!("{stem}.w1.weight"))?;
    let w3 = reader.native_scale_pair(&format!("{stem}.w3.weight"))?;
    let w2 = reader.native_scale_pair(&format!("{stem}.w2.weight"))?;
    Ok(json!({
        "expert_id": expert_id,
        "w1": native_pair_contract_json(&w1, true),
        "w3": native_pair_contract_json(&w3, true),
        "w2": native_pair_contract_json(&w2, true),
    }))
}

fn shared_pair_contract_json(reader: &DeepSeekV4FullStreamReader) -> ExampleResult<Value> {
    let stem = "layers.0.ffn.shared_experts";
    let w1 = reader.native_scale_pair(&format!("{stem}.w1.weight"))?;
    let w3 = reader.native_scale_pair(&format!("{stem}.w3.weight"))?;
    let w2 = reader.native_scale_pair(&format!("{stem}.w2.weight"))?;
    Ok(json!({
        "w1": native_pair_contract_json(&w1, true),
        "w3": native_pair_contract_json(&w3, true),
        "w2": native_pair_contract_json(&w2, true),
    }))
}

fn tensor_range_binding_json(tensor: &DeepSeekV4TensorMetadata, start: u64, end: u64) -> Value {
    json!({
        "name": tensor.name,
        "dtype": tensor.dtype,
        "shape": tensor.shape,
        "tensor_bytes": tensor.bytes,
        "verified_range": {"start": start, "end": end, "bytes": end - start},
        "source_shard": tensor.source_shard,
        "segments": tensor.segments.iter().map(segment_json).collect::<Vec<_>>(),
    })
}

fn tensor_full_binding_json(tensor: &DeepSeekV4TensorMetadata) -> Value {
    tensor_range_binding_json(tensor, 0, tensor.bytes)
}

fn native_pair_contract_json(pair: &NativeScalePair<'_>, read_and_executed: bool) -> Value {
    json!({
        "representation": pair.kind.as_str(),
        "weight": tensor_full_binding_json(pair.weight),
        "scale": tensor_full_binding_json(pair.scale),
        "geometry": {
            "out_rows": pair.out_rows,
            "packed_k": pair.packed_k,
            "logical_k": pair.logical_k,
            "scale_rows": pair.scale_rows,
            "scale_cols": pair.scale_cols,
        },
        "read_and_executed_by_this_bounded_cpu_oracle": read_and_executed,
    })
}

fn segment_json(segment: &DeepSeekV4Segment) -> Value {
    json!({
        "bytes": segment.bytes,
        "chunk_relpath": segment.chunk_relpath,
        "chunk_sha256": segment.sha256,
        "source_file_start": segment.source_file_start,
        "source_file_end": segment.source_file_end,
        "tensor_start": segment.tensor_start,
        "tensor_end": segment.tensor_end,
        "row_start": segment.row_start,
        "row_count": segment.row_count,
    })
}

fn bf16_checkpoint_json(bits: &[u16], shape: &[usize]) -> ExampleResult<Value> {
    if shape.iter().product::<usize>() != bits.len() {
        return Err(failure(
            "BF16 checkpoint shape does not match its element count",
        ));
    }
    let values: Vec<f32> = bits
        .iter()
        .map(|bits| bf16::from_bits(*bits).to_f32())
        .collect();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(failure("BF16 checkpoint contains a non-finite value"));
    }
    Ok(json!({
        "dtype": "BF16",
        "shape": shape,
        "raw_bytes_sha256": sha256_hex(&u16_le_bytes(bits)),
        "finite_stats": finite_stats(&values)?,
        "raw_values_retained": false,
    }))
}

fn f32_checkpoint_json(values: &[f32], shape: &[usize]) -> ExampleResult<Value> {
    if shape.iter().product::<usize>() != values.len() {
        return Err(failure(
            "F32 checkpoint shape does not match its element count",
        ));
    }
    Ok(json!({
        "dtype": "F32",
        "shape": shape,
        "raw_bytes_sha256": sha256_hex(&f32_le_bytes(values)),
        "finite_stats": finite_stats(values)?,
        "raw_values_retained": false,
    }))
}

fn finite_stats(values: &[f32]) -> ExampleResult<Value> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(failure("checkpoint does not contain finite values"));
    }
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = values.iter().map(|value| *value as f64).sum::<f64>() / values.len() as f64;
    let l2 = values
        .iter()
        .map(|value| {
            let value = *value as f64;
            value * value
        })
        .sum::<f64>()
        .sqrt();
    Ok(json!({
        "count": values.len(),
        "min": f32_text(min),
        "max": f32_text(max),
        "mean": f64_text(mean),
        "l2": f64_text(l2),
    }))
}

fn f32_text(value: f32) -> String {
    value.to_string()
}

fn f64_text(value: f64) -> String {
    value.to_string()
}

fn f32_slice_text(values: &[f32]) -> Vec<String> {
    values.iter().copied().map(f32_text).collect()
}

fn u16_le_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn parse_args() -> ExampleResult<Args> {
    let mut artifact = None;
    let mut predecessor = None;
    let mut out = None;
    let mut arguments = std::env::args_os().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_string_lossy().as_ref() {
            "--artifact" => artifact = arguments.next().map(PathBuf::from),
            "--predecessor" => predecessor = arguments.next().map(PathBuf::from),
            "--out" => out = arguments.next().map(PathBuf::from),
            "--help" | "-h" => {
                return Err(failure(
                    "usage: gravity_deepseek_v4_layer0_position1_full_ffn_cpu_oracle --artifact <absolute full Gravity dir> --predecessor <absolute P1 CPU receipt> --out <absolute receipt.json>",
                ));
            }
            other => return Err(failure(format!("unknown argument {other:?}"))),
        }
    }
    let artifact = artifact.ok_or_else(|| failure("--artifact is required"))?;
    let predecessor = predecessor.ok_or_else(|| failure("--predecessor is required"))?;
    let out = out.ok_or_else(|| failure("--out is required"))?;
    if !artifact.is_absolute() || !predecessor.is_absolute() || !out.is_absolute() {
        return Err(failure(
            "--artifact, --predecessor, and --out must be absolute paths",
        ));
    }
    Ok(Args {
        artifact,
        predecessor,
        out,
    })
}

fn verify_seal(value: &Value, label: &str) -> ExampleResult<()> {
    let recorded = value
        .get("seal_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| failure(format!("{label} has no seal_sha256")))?;
    let mut unsigned = value.clone();
    unsigned
        .as_object_mut()
        .ok_or_else(|| failure(format!("{label} is not a JSON object")))?
        .remove("seal_sha256");
    if sha256_hex(&canonical_json(&unsigned)) != recorded {
        return Err(failure(format!("{label} seal mismatch")));
    }
    Ok(())
}

fn seal(mut receipt: Value) -> ExampleResult<Value> {
    if receipt.get("seal_sha256").is_some() {
        return Err(failure("receipt already contains a seal"));
    }
    let seal = sha256_hex(&canonical_json(&receipt));
    receipt
        .as_object_mut()
        .ok_or_else(|| failure("receipt root is not an object"))?
        .insert("seal_sha256".to_owned(), Value::String(seal));
    Ok(receipt)
}

fn write_new_sealed_receipt(path: &Path, receipt: &Value) -> ExampleResult<()> {
    if path.exists() {
        return Err(failure(format!(
            "refusing to overwrite existing sealed receipt {}",
            path.display()
        )));
    }
    let parent = path
        .parent()
        .ok_or_else(|| failure("receipt output path has no parent"))?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| failure("receipt output file name is not UTF-8"))?,
        std::process::id()
    ));
    if temporary.exists() {
        return Err(failure("receipt temporary path already exists"));
    }
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    let mut file = options.open(&temporary)?;
    file.write_all(&canonical_json(receipt))?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    drop(file);
    fs::rename(&temporary, path)?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

fn canonical_json(value: &Value) -> Vec<u8> {
    let mut output = Vec::new();
    write_canonical_json(&mut output, value);
    output
}

fn write_canonical_json(output: &mut Vec<u8>, value: &Value) {
    match value {
        Value::Null => output.extend_from_slice(b"null"),
        Value::Bool(value) => output.extend_from_slice(if *value { b"true" } else { b"false" }),
        Value::Number(value) => output.extend_from_slice(value.to_string().as_bytes()),
        Value::String(value) => {
            output.extend_from_slice(serde_json::to_string(value).unwrap().as_bytes())
        }
        Value::Array(values) => {
            output.push(b'[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                write_canonical_json(output, value);
            }
            output.push(b']');
        }
        Value::Object(values) => {
            output.push(b'{');
            let mut keys: Vec<&String> = values.keys().collect();
            keys.sort();
            for (index, key) in keys.into_iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                output.extend_from_slice(serde_json::to_string(key).unwrap().as_bytes());
                output.push(b':');
                write_canonical_json(output, &values[key]);
            }
            output.push(b'}');
        }
    }
}

fn failure(message: impl Into<String>) -> Box<dyn Error> {
    std::io::Error::other(message.into()).into()
}
