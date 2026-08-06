//! Bounded complete DeepSeek-V4 layer-0 attention CPU source-algorithm receipt.
//!
//! This uses the admitted 43-layer full-stream chunks to execute exactly one
//! tokenizer-bound BOS / position-zero / ratio-zero attention path.  It is a
//! source-derived CPU checkpoint only: not upstream PyTorch/TileLang parity,
//! not a registered runtime, not a Metal result, not an endpoint, and not TPS.
//!
//! ```sh
//! cargo run --release -p hawking-core --example gravity_deepseek_v4_layer0_attention_oracle -- \
//!   --artifact /absolute/path/to/full-43-layer-stream.gravity \
//!   --out /absolute/path/to/DSV4F_LAYER0_ATTENTION_CPU_ORACLE-v1.json
//! ```

use half::bf16;
use hawking_core::gravity_deepseek_v4::{
    DeepSeekV4FullStreamReader, DeepSeekV4Segment, DeepSeekV4TensorMetadata, NativeScalePair,
    FULL_STREAM_SCHEMA, FULL_STREAM_STATUS,
};
use hawking_core::gravity_deepseek_v4_layer0_attention::{
    layer0_attention_cpu_oracle, verify_layer0_attention_source_anchors,
    DeepSeekV4Layer0AttentionSourceAnchors, Layer0AttentionCpuOracleResult, HEAD_DIM, KV_QAT_BLOCK,
    LAYER0_ATTN_SINK, LAYER0_COMPRESS_RATIO, LAYER0_KV_NORM_WEIGHT, LAYER0_Q_NORM_WEIGHT,
    LAYER0_WKV_SCALE, LAYER0_WKV_WEIGHT, LAYER0_WO_A_SCALE, LAYER0_WO_A_WEIGHT, LAYER0_WO_B_SCALE,
    LAYER0_WO_B_WEIGHT, LAYER0_WQ_B_SCALE, LAYER0_WQ_B_WEIGHT, NON_ROPE_HEAD_DIM, NUM_HEADS,
    O_GROUPS, O_LORA_RANK, Q_LORA_RANK, ROPE_HEAD_DIM, WINDOW_SIZE, WKV_ROWS, WO_A_COLS, WO_A_ROWS,
    WQ_B_ROWS,
};
use hawking_core::gravity_deepseek_v4_layer0_prefix::{
    EMBED_WEIGHT, HC_FLAT_WIDTH, HC_MIX_WIDTH, HC_MULT, HIDDEN_SIZE, LAYER0_ATTN_NORM_WEIGHT,
    LAYER0_HC_ATTN_BASE, LAYER0_HC_ATTN_FN, LAYER0_HC_ATTN_SCALE, PREFIX_TOKEN_ID,
    PREFIX_TOKEN_STRING,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const RECEIPT_SCHEMA: &str = "hawking.gravity.deepseek_v4.layer0_attention_cpu_algorithm_oracle.v1";
const RECEIPT_STATUS: &str =
    "PASS_SOURCE_DERIVED_CPU_LAYER0_ATTENTION_NOT_INDEPENDENT_UPSTREAM_RUNTIME_PARITY";

type ExampleResult<T> = Result<T, Box<dyn Error>>;

struct Args {
    artifact: PathBuf,
    out: PathBuf,
}

fn main() -> ExampleResult<()> {
    let args = parse_args()?;
    let reader = DeepSeekV4FullStreamReader::admit(&args.artifact)?;
    let anchors = verify_layer0_attention_source_anchors(&reader)?;
    let result = layer0_attention_cpu_oracle(&reader)?;
    validate_result_geometry(&result)?;

    let embed = reader.tensor_metadata(EMBED_WEIGHT)?;
    let hc_fn = reader.tensor_metadata(LAYER0_HC_ATTN_FN)?;
    let hc_base = reader.tensor_metadata(LAYER0_HC_ATTN_BASE)?;
    let hc_scale = reader.tensor_metadata(LAYER0_HC_ATTN_SCALE)?;
    let attn_norm = reader.tensor_metadata(LAYER0_ATTN_NORM_WEIGHT)?;
    let wq_a = reader.native_scale_pair("layers.0.attn.wq_a.weight")?;
    let q_norm = reader.tensor_metadata(LAYER0_Q_NORM_WEIGHT)?;
    let wq_b = reader.native_scale_pair(LAYER0_WQ_B_WEIGHT)?;
    let wkv = reader.native_scale_pair(LAYER0_WKV_WEIGHT)?;
    let kv_norm = reader.tensor_metadata(LAYER0_KV_NORM_WEIGHT)?;
    let attn_sink = reader.tensor_metadata(LAYER0_ATTN_SINK)?;
    let wo_a = reader.native_scale_pair(LAYER0_WO_A_WEIGHT)?;
    let wo_b = reader.native_scale_pair(LAYER0_WO_B_WEIGHT)?;
    let embed_row_bytes = (HIDDEN_SIZE * std::mem::size_of::<u16>()) as u64;
    let embed_row_start = PREFIX_TOKEN_ID * embed_row_bytes;

    check_pair_name(&wq_a, "layers.0.attn.wq_a.scale")?;
    check_pair_name(&wq_b, LAYER0_WQ_B_SCALE)?;
    check_pair_name(&wkv, LAYER0_WKV_SCALE)?;
    check_pair_name(&wo_a, LAYER0_WO_A_SCALE)?;
    check_pair_name(&wo_b, LAYER0_WO_B_SCALE)?;

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
        "source_algorithm_bindings": source_bindings_json(&anchors),
        "bounded_source_reads": {
            "reader_admission_verified_manifest_source_index_offsets_and_chunk_tree": true,
            "all_chunk_sha256_bytes_verified_globally": false,
            "every_listed_tensor_read_touched_chunk_sha256_verified_before_cpu_use": true,
            "parent_safetensors_materialized": false,
            "embed_token_row": tensor_range_binding_json(embed, embed_row_start, embed_row_start + embed_row_bytes),
            "layer0_hc_attn_fn": tensor_full_binding_json(hc_fn),
            "layer0_hc_attn_base": tensor_full_binding_json(hc_base),
            "layer0_hc_attn_scale": tensor_full_binding_json(hc_scale),
            "layer0_attn_norm_weight": tensor_full_binding_json(attn_norm),
            "layer0_wq_a": native_pair_contract_json(&wq_a, true),
            "layer0_q_norm_weight": tensor_full_binding_json(q_norm),
            "layer0_wq_b": native_pair_contract_json(&wq_b, true),
            "layer0_wkv": native_pair_contract_json(&wkv, true),
            "layer0_kv_norm_weight": tensor_full_binding_json(kv_norm),
            "layer0_attn_sink": tensor_full_binding_json(attn_sink),
            "layer0_wo_a_raw_then_convert_to_bf16": native_pair_contract_json(&wo_a, true),
            "layer0_wo_b": native_pair_contract_json(&wo_b, true),
        },
        "intermediate_receipts": intermediate_receipts_json(&result)?,
        "work_accounting": {
            "batch": 1,
            "sequence_tokens": 1,
            "position": 0,
            "source_world_size": 1,
            "fp8_linear_dot_products": {
                "wq_a": 1024_u64 * 4096_u64,
                "wq_b": 32768_u64 * 1024_u64,
                "wkv": 512_u64 * 4096_u64,
                "wo_b": 4096_u64 * 8192_u64,
            },
            "wo_a_bf16_einsum_dot_products": 8192_u64 * 4096_u64,
            "sparse_attention_qk_dot_products": 64_u64 * 512_u64,
            "selected_window_kv_positions": 1,
            "compressed_kv_positions": 0,
            "index_head_execution": false,
            "mHC_attn_post_output_values": (HC_MULT * HIDDEN_SIZE) as u64,
            "gpu_dispatches": 0,
            "command_buffers": 0,
            "cpu_visible_waits": 0,
        },
        "execution_boundary": {
            "source_derived_cpu_algorithm_attention": true,
            "independently_upstream_runtime_parity": false,
            "not_independently_upstream_runtime_parity": true,
            "source_runtime_executed": false,
            "pytorch_or_tilelang_executed": false,
            "full_model_loaded": false,
            "full_model_forward": false,
            "layer0_ffn_executed": false,
            "routing_executed": false,
            "generated_tokens": 0,
            "metal_allocations": 0,
            "gpu_dispatches": 0,
            "command_buffers": 0,
            "hcli_endpoint_started": false,
            "base_true_tps_measured": false,
            "claim": "one tokenizer-bound BOS, layer-0, position-0, ratio-0 translated CPU attention algorithm checkpoint only; NOT independently upstream-runtime parity, a registered V4 runtime, GPU result, full forward, token generation, HCLI, or BASE_TRUE_TPS result",
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

fn validate_result_geometry(result: &Layer0AttentionCpuOracleResult) -> ExampleResult<()> {
    if result.prefix.token_id != PREFIX_TOKEN_ID
        || result.prefix.embed_bf16_bits.len() != HIDDEN_SIZE
        || result.prefix.hc_replicated_bf16_bits.len() != HC_FLAT_WIDTH
        || result.prefix.hc_mixes_f32.len() != HC_MIX_WIDTH
        || result.prefix.hc_pre_f32.len() != HC_MULT
        || result.prefix.hc_post_f32.len() != HC_MULT
        || result.prefix.hc_comb_f32.len() != HC_MULT * HC_MULT
        || result.prefix.attn_norm_bf16_bits.len() != HIDDEN_SIZE
        || result.wq_a.output.bf16_bits.len() != Q_LORA_RANK
        || result.q_norm_bf16_bits.len() != Q_LORA_RANK
        || result.wq_b.output.bf16_bits.len() != WQ_B_ROWS
        || result.q_head_norm_bf16_bits.len() != WQ_B_ROWS
        || result.q_position0_rope_bf16_bits != result.q_head_norm_bf16_bits
        || result.wkv.output.bf16_bits.len() != WKV_ROWS
        || result.kv_norm_bf16_bits.len() != HEAD_DIM
        || result.kv_inplace_qat.output_bf16_bits.len() != HEAD_DIM
        || result.kv_inplace_qat.non_rope_activation_e4m3fn.len() != NON_ROPE_HEAD_DIM
        || result.kv_inplace_qat.non_rope_scales_e8m0fnu.len() != NON_ROPE_HEAD_DIM / KV_QAT_BLOCK
        || result.kv_position0_rope_bf16_bits != result.kv_inplace_qat.output_bf16_bits
        || result.sparse_attention_scores_f32.len() != NUM_HEADS
        || result.sparse_attention_sink_denominators_f32.len() != NUM_HEADS
        || result.sparse_attention_bf16_bits.len() != NUM_HEADS * HEAD_DIM
        || result.sparse_attention_derotated_bf16_bits != result.sparse_attention_bf16_bits
        || result.wo_a_bf16_bits.len() != O_GROUPS * O_LORA_RANK
        || result.wo_b.output.bf16_bits.len() != HIDDEN_SIZE
        || result.hc_attn_post_bf16_bits.len() != HC_MULT * HIDDEN_SIZE
    {
        return Err(failure(
            "layer-0 attention checkpoint returned unexpected geometry",
        ));
    }
    if result
        .sparse_attention_scores_f32
        .iter()
        .chain(&result.sparse_attention_sink_denominators_f32)
        .any(|value| !value.is_finite())
    {
        return Err(failure(
            "layer-0 attention sparse score/sink output is non-finite",
        ));
    }
    Ok(())
}

fn source_bindings_json(anchors: &DeepSeekV4Layer0AttentionSourceAnchors) -> Value {
    json!({
        "official_assets_verified_by_admitted_full_stream_and_exact_anchor": {
            "inference/model.py": anchors.prefix.act_quant.inference_model_py_sha256,
            "inference/kernel.py": anchors.prefix.act_quant.inference_kernel_py_sha256,
            "inference/convert.py": anchors.inference_convert_py_sha256,
            "inference/config.json": anchors.prefix.act_quant.inference_config_json_sha256,
            "config.json": anchors.prefix.act_quant.model_config_json_sha256,
            "tokenizer.json": anchors.prefix.tokenizer_json_sha256,
            "tokenizer_config.json": anchors.prefix.tokenizer_config_json_sha256,
        },
        "tokenizer_binding": {
            "fixed_token_id": PREFIX_TOKEN_ID,
            "fixed_bpe_token": PREFIX_TOKEN_STRING,
            "no_prompt_text_or_prompt_derived_hidden_state_recorded": true,
        },
        "pinned_position_zero_attention_contract": {
            "batch": 1,
            "sequence_tokens": 1,
            "position": 0,
            "hidden_size": HIDDEN_SIZE,
            "n_heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
            "rope_head_dim": ROPE_HEAD_DIM,
            "q_lora_rank": Q_LORA_RANK,
            "o_groups": O_GROUPS,
            "o_lora_rank": O_LORA_RANK,
            "window_size": WINDOW_SIZE,
            "layer0_compress_ratio": LAYER0_COMPRESS_RATIO,
            "indexer_executed": false,
            "source_topk_window_indices": [0],
            "position_zero_rope_identity_verified_by_geometry": true,
            "source_fp8_weight_block": [128, 128],
            "source_kv_inplace_qat_block": KV_QAT_BLOCK,
            "source_scale_format": "ue8m0",
            "source_scale_dtype": "F8_E8M0FNU",
        },
        "source_algorithm_path": [
            "model.py::Transformer.forward / Block.hc_pre / RMSNorm through the sealed prefix rung",
            "model.py::Attention.forward: q_norm(wq_a(x)) -> wq_b -> per-head rsqrt RMS normalization",
            "model.py::apply_rotary_emb at position 0: phase is zero, therefore BF16 identity copy",
            "model.py::Attention.forward: kv_norm(wkv(x)) -> kernel.py::act_quant(..., block_size=64, inplace=True) over non-RoPE KV dimensions only",
            "model.py::get_window_topk_idxs at start_pos=0,seqlen=1: one causal index 0; compress_ratio=0 so no compressor/indexer branch",
            "kernel.py::sparse_attn_kernel: single selected KV online softmax plus learned attn_sink denominator",
            "inference/convert.py: raw WO-A FP8/E8M0 block dequantization and BF16 materialization",
            "model.py::Attention.forward: grouped BF16 WO-A einsum then FP8 model.linear WO-B",
            "model.py::Block.hc_post: post-weighted attention plus comb-column-weighted residual lanes, BF16 output cast",
        ],
    })
}

fn intermediate_receipts_json(result: &Layer0AttentionCpuOracleResult) -> ExampleResult<Value> {
    let prefix = &result.prefix;
    Ok(json!({
        "prefix_continuity": {
            "token_id": prefix.token_id,
            "embedding": bf16_checkpoint_json(&prefix.embed_bf16_bits, &[HIDDEN_SIZE])?,
            "hc_replicate": {
                "shape": [HC_MULT, HIDDEN_SIZE],
                "dtype": "BF16",
                "sha256_bf16_le": sha256_hex(&u16_le_bytes(&prefix.hc_replicated_bf16_bits)),
                "four_lanes_exactly_equal_to_embedding_row": prefix.hc_replicated_bf16_bits.chunks_exact(HIDDEN_SIZE).all(|lane| lane == prefix.embed_bf16_bits.as_slice()),
            },
            "hc_attn_pre": {
                "mixes": f32_checkpoint_json(&prefix.hc_mixes_f32, &[HC_MIX_WIDTH])?,
                "pre": f32_checkpoint_json(&prefix.hc_pre_f32, &[HC_MULT])?,
                "post": f32_checkpoint_json(&prefix.hc_post_f32, &[HC_MULT])?,
                "comb": f32_checkpoint_json(&prefix.hc_comb_f32, &[HC_MULT, HC_MULT])?,
                "reduced_bf16": bf16_checkpoint_json(&prefix.hc_attn_pre_bf16_bits, &[HIDDEN_SIZE])?,
            },
            "attn_norm_output_and_wq_a_input": bf16_checkpoint_json(&prefix.attn_norm_bf16_bits, &[HIDDEN_SIZE])?,
        },
        "q_path": {
            "wq_a_fp8_linear": fp8_stage_json(&result.wq_a.quantized_input, &result.wq_a.output, &[Q_LORA_RANK])?,
            "q_norm": bf16_checkpoint_json(&result.q_norm_bf16_bits, &[Q_LORA_RANK])?,
            "wq_b_fp8_linear": fp8_stage_json(&result.wq_b.quantized_input, &result.wq_b.output, &[NUM_HEADS, HEAD_DIM])?,
            "per_head_rmsnorm": bf16_checkpoint_json(&result.q_head_norm_bf16_bits, &[NUM_HEADS, HEAD_DIM])?,
            "position_zero_rope": {
                "identity": true,
                "output": bf16_checkpoint_json(&result.q_position0_rope_bf16_bits, &[NUM_HEADS, HEAD_DIM])?,
            },
        },
        "kv_path": {
            "wkv_fp8_linear": fp8_stage_json(&result.wkv.quantized_input, &result.wkv.output, &[HEAD_DIM])?,
            "kv_norm": bf16_checkpoint_json(&result.kv_norm_bf16_bits, &[HEAD_DIM])?,
            "non_rope_inplace_fp8_qat": {
                "quantized_dimensions": NON_ROPE_HEAD_DIM,
                "protected_rope_dimensions": ROPE_HEAD_DIM,
                "block_size": KV_QAT_BLOCK,
                "activation_e4m3fn_sha256": sha256_hex(&result.kv_inplace_qat.non_rope_activation_e4m3fn),
                "scale_e8m0fnu_sha256": sha256_hex(&result.kv_inplace_qat.non_rope_scales_e8m0fnu),
                "scale_count": result.kv_inplace_qat.non_rope_scales_e8m0fnu.len(),
                "output": bf16_checkpoint_json(&result.kv_inplace_qat.output_bf16_bits, &[HEAD_DIM])?,
            },
            "position_zero_rope": {
                "identity": true,
                "output": bf16_checkpoint_json(&result.kv_position0_rope_bf16_bits, &[HEAD_DIM])?,
            },
        },
        "sparse_attention": {
            "ratio_zero_no_compression_or_indexer": true,
            "window_topk_indices": [0],
            "selected_kv_count": 1,
            "attn_sink_included": true,
            "softmax_scale": (HEAD_DIM as f32).powf(-0.5).to_string(),
            "per_head_scores": f32_checkpoint_json(&result.sparse_attention_scores_f32, &[NUM_HEADS])?,
            "per_head_sink_denominators": f32_checkpoint_json(&result.sparse_attention_sink_denominators_f32, &[NUM_HEADS])?,
            "output_before_derotation": bf16_checkpoint_json(&result.sparse_attention_bf16_bits, &[NUM_HEADS, HEAD_DIM])?,
            "position_zero_derotation": {
                "identity": true,
                "output": bf16_checkpoint_json(&result.sparse_attention_derotated_bf16_bits, &[NUM_HEADS, HEAD_DIM])?,
            },
        },
        "o_path": {
            "wo_a": {
                "source_operator": "inference/convert.py raw FP8/E8M0 -> BF16 materialization, then model.py grouped einsum",
                "raw_native_shape": [WO_A_ROWS, WO_A_COLS],
                "converted_weight_dtype": "BF16",
                "grouped_einsum_input_shape": [O_GROUPS, WO_A_COLS],
                "grouped_einsum_output": bf16_checkpoint_json(&result.wo_a_bf16_bits, &[O_GROUPS, O_LORA_RANK])?,
            },
            "wo_b_fp8_linear": fp8_stage_json(&result.wo_b.quantized_input, &result.wo_b.output, &[HIDDEN_SIZE])?,
        },
        "mhc_attn_post": {
            "source_formula": "post[k] * attention + sum_j(comb[j,k] * residual_hc[j])",
            "attention_input": bf16_checkpoint_json(&result.wo_b.output.bf16_bits, &[HIDDEN_SIZE])?,
            "residual_hc_shape": [HC_MULT, HIDDEN_SIZE],
            "output": bf16_checkpoint_json(&result.hc_attn_post_bf16_bits, &[HC_MULT, HIDDEN_SIZE])?,
        },
    }))
}

fn fp8_stage_json(
    quantized: &hawking_core::gravity_deepseek_v4_act_quant::ActQuantizedBf16Row,
    output: &hawking_core::gravity_deepseek_v4_act_quant::Fp8MatvecCpuResult,
    output_shape: &[usize],
) -> ExampleResult<Value> {
    Ok(json!({
        "source_operator": "model.py::linear -> kernel.py::act_quant(block=128, ue8m0, F8_E8M0FNU scale) -> fp8_gemm",
        "activation_e4m3fn_sha256": sha256_hex(&quantized.activation_e4m3fn),
        "activation_bytes": quantized.activation_e4m3fn.len(),
        "scale_e8m0fnu_sha256": sha256_hex(&quantized.scales_e8m0fnu),
        "scale_bytes": quantized.scales_e8m0fnu.len(),
        "output_bf16": bf16_checkpoint_json(&output.bf16_bits, output_shape)?,
        "output_fp32": f32_checkpoint_json(&output.fp32, output_shape)?,
    }))
}

fn check_pair_name(pair: &NativeScalePair<'_>, expected_scale: &str) -> ExampleResult<()> {
    if pair.scale.name != expected_scale {
        return Err(failure(
            "native FP8 pair scale name changed after CPU execution",
        ));
    }
    Ok(())
}

fn tensor_range_binding_json(tensor: &DeepSeekV4TensorMetadata, start: u64, end: u64) -> Value {
    json!({
        "name": tensor.name,
        "dtype": tensor.dtype,
        "shape": tensor.shape,
        "tensor_bytes": tensor.bytes,
        "range": {"start": start, "end_exclusive": end},
        "returned_bytes": end - start,
        "touched_content_addressed_chunks": tensor.segments.iter()
            .filter(|segment| segment.tensor_end > start && segment.tensor_start < end)
            .map(segment_json)
            .collect::<Vec<_>>(),
    })
}

fn tensor_full_binding_json(tensor: &DeepSeekV4TensorMetadata) -> Value {
    tensor_range_binding_json(tensor, 0, tensor.bytes)
}

fn native_pair_contract_json(pair: &NativeScalePair<'_>, read_and_executed: bool) -> Value {
    json!({
        "kind": pair.kind.as_str(),
        "weight": tensor_full_binding_json(pair.weight),
        "scale": tensor_full_binding_json(pair.scale),
        "geometry": {
            "out_rows": pair.out_rows,
            "packed_k": pair.packed_k,
            "logical_k": pair.logical_k,
            "scale_rows": pair.scale_rows,
            "scale_cols": pair.scale_cols,
        },
        "read_and_executed_by_this_checkpoint": read_and_executed,
    })
}

fn segment_json(segment: &DeepSeekV4Segment) -> Value {
    json!({
        "bytes": segment.bytes,
        "chunk_relpath": segment.chunk_relpath,
        "sha256": segment.sha256,
        "tensor_start": segment.tensor_start,
        "tensor_end": segment.tensor_end,
        "source_file_start": segment.source_file_start,
        "source_file_end": segment.source_file_end,
        "row_start": segment.row_start,
        "row_count": segment.row_count,
    })
}

fn bf16_checkpoint_json(bits: &[u16], shape: &[usize]) -> ExampleResult<Value> {
    let values: Vec<f32> = bits
        .iter()
        .map(|bits| bf16::from_bits(*bits).to_f32())
        .collect();
    Ok(json!({
        "shape": shape,
        "dtype": "BF16",
        "element_count": bits.len(),
        "sha256_bf16_le": sha256_hex(&u16_le_bytes(bits)),
        "decoded_f32_stats": finite_stats(&values)?,
    }))
}

fn f32_checkpoint_json(values: &[f32], shape: &[usize]) -> ExampleResult<Value> {
    Ok(json!({
        "shape": shape,
        "dtype": "F32",
        "element_count": values.len(),
        "sha256_f32_le": sha256_hex(&f32_le_bytes(values)),
        "stats": finite_stats(values)?,
    }))
}

fn finite_stats(values: &[f32]) -> ExampleResult<Value> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(failure("checkpoint is empty or non-finite"));
    }
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    let mut maximum_abs = 0.0_f32;
    let mut sum = 0.0_f64;
    for &value in values {
        minimum = minimum.min(value);
        maximum = maximum.max(value);
        maximum_abs = maximum_abs.max(value.abs());
        sum += f64::from(value);
    }
    Ok(json!({
        "min_f32": minimum.to_string(),
        "max_f32": maximum.to_string(),
        "max_abs_f32": maximum_abs.to_string(),
        "mean_f64": (sum / values.len() as f64).to_string(),
    }))
}

fn u16_le_bytes(values: &[u16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<u16>());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    bytes
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn parse_args() -> ExampleResult<Args> {
    let mut artifact = None::<PathBuf>;
    let mut out = None::<PathBuf>;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--artifact" => artifact = args.next().map(PathBuf::from),
            "--out" => out = args.next().map(PathBuf::from),
            "--help" | "-h" => {
                println!(
                    "usage: gravity_deepseek_v4_layer0_attention_oracle --artifact <absolute full Gravity dir> --out <absolute receipt.json>"
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

fn seal(mut receipt: Value) -> ExampleResult<Value> {
    if !receipt.is_object() || receipt.get("seal_sha256").is_some() {
        return Err(failure("receipt must be an unsealed JSON object"));
    }
    let seal = sha256_hex(&canonical_json(&receipt));
    receipt
        .as_object_mut()
        .expect("receipt object was checked")
        .insert("seal_sha256".to_owned(), Value::String(seal));
    Ok(receipt)
}

/// Create without overwriting an older scientific receipt.  A hard-link
/// publish gives the final name create-new semantics after a durable temporary
/// write succeeds.
fn write_new_sealed_receipt(path: &Path, receipt: &Value) -> ExampleResult<()> {
    if path.exists() {
        return Err(failure(format!(
            "refusing to overwrite existing layer-0 attention receipt {}",
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
        ".{name}.{}.layer0-attention.tmp",
        std::process::id()
    ));
    let bytes = serde_json::to_vec_pretty(receipt)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| failure(format!("cannot create temporary receipt: {error}")))?;
    if let Err(error) = file
        .write_all(&bytes)
        .and_then(|_| file.write_all(b"\n"))
        .and_then(|_| file.sync_all())
    {
        let _ = fs::remove_file(&temporary);
        return Err(Box::new(error));
    }
    drop(file);
    if let Err(error) = fs::hard_link(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(failure(format!(
            "refusing to overwrite or link layer-0 attention receipt {}: {error}",
            path.display()
        )));
    }
    fs::remove_file(&temporary)?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

fn canonical_json(value: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    write_canonical_json(&mut out, value);
    out
}

fn write_canonical_json(out: &mut Vec<u8>, value: &Value) {
    match value {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(true) => out.extend_from_slice(b"true"),
        Value::Bool(false) => out.extend_from_slice(b"false"),
        Value::Number(number) => out.extend_from_slice(number.to_string().as_bytes()),
        Value::String(string) => out.extend_from_slice(
            serde_json::to_string(string)
                .expect("JSON string serialization is infallible")
                .as_bytes(),
        ),
        Value::Array(values) => {
            out.push(b'[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    out.push(b',');
                }
                write_canonical_json(out, value);
            }
            out.push(b']');
        }
        Value::Object(object) => {
            out.push(b'{');
            let mut keys: Vec<&String> = object.keys().collect();
            keys.sort();
            for (index, key) in keys.into_iter().enumerate() {
                if index != 0 {
                    out.push(b',');
                }
                out.extend_from_slice(
                    serde_json::to_string(key)
                        .expect("JSON string serialization is infallible")
                        .as_bytes(),
                );
                out.push(b':');
                write_canonical_json(out, &object[key]);
            }
            out.push(b'}');
        }
    }
}

fn failure(message: impl Into<String>) -> Box<dyn Error> {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
}
