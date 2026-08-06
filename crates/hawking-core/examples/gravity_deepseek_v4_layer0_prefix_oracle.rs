//! Bounded source-algorithm DeepSeek-V4 layer-0 prefix checkpoint.
//!
//! This consumes the sealed full stream only through its admitted reader.  It
//! resolves the pinned tokenizer BOS id, reads one real embedding row, runs
//! scalar CPU Hyper-Connection/Sinkhorn/RMSNorm formulae, then records the
//! BF16 input accepted by layer-0 `wq_a`.  It intentionally does not execute
//! upstream Python/CUDA/TileLang, an attention projection, a full forward,
//! Metal work, an endpoint, or a TPS benchmark.
//!
//! ```sh
//! cargo run --release -p hawking-core --example gravity_deepseek_v4_layer0_prefix_oracle -- \
//!   --artifact /absolute/path/to/full-43-layer-stream.gravity \
//!   --out /absolute/path/to/DSV4F_LAYER0_PREFIX_CPU_ORACLE.json
//! ```

use half::bf16;
use hawking_core::gravity_deepseek_v4::{
    DeepSeekV4FullStreamReader, DeepSeekV4Segment, DeepSeekV4TensorMetadata, NativeScalePair,
    FULL_STREAM_SCHEMA, FULL_STREAM_STATUS,
};
use hawking_core::gravity_deepseek_v4_act_quant::{
    ACT_QUANT_BLOCK, LAYER0_WQ_A_COLS, LAYER0_WQ_A_ROWS, LAYER0_WQ_A_SCALE, LAYER0_WQ_A_WEIGHT,
};
use hawking_core::gravity_deepseek_v4_layer0_prefix::{
    layer0_prefix_cpu_oracle, verify_layer0_prefix_source_anchors, HC_FLAT_WIDTH, HC_MIX_WIDTH,
    HC_MULT, HC_SINKHORN_ITERS, HIDDEN_SIZE, LAYER0_ATTN_NORM_WEIGHT, LAYER0_HC_ATTN_BASE,
    LAYER0_HC_ATTN_FN, LAYER0_HC_ATTN_SCALE, PREFIX_TOKEN_ID, PREFIX_TOKEN_STRING, VOCAB_SIZE,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const RECEIPT_SCHEMA: &str = "hawking.gravity.deepseek_v4.layer0_prefix_cpu_algorithm_oracle.v1";
const RECEIPT_STATUS: &str =
    "PASS_SOURCE_DERIVED_CPU_PREFIX_NOT_INDEPENDENT_UPSTREAM_RUNTIME_PARITY";

type ExampleResult<T> = Result<T, Box<dyn Error>>;

struct Args {
    artifact: PathBuf,
    out: PathBuf,
}

fn main() -> ExampleResult<()> {
    let args = parse_args()?;
    let reader = DeepSeekV4FullStreamReader::admit(&args.artifact)?;
    let anchors = verify_layer0_prefix_source_anchors(&reader)?;
    let result = layer0_prefix_cpu_oracle(&reader)?;

    if result.token_id != PREFIX_TOKEN_ID
        || result.embed_bf16_bits.len() != HIDDEN_SIZE
        || result.hc_replicated_bf16_bits.len() != HC_FLAT_WIDTH
        || result.hc_mixes_f32.len() != HC_MIX_WIDTH
        || result.hc_pre_f32.len() != HC_MULT
        || result.hc_post_f32.len() != HC_MULT
        || result.hc_comb_f32.len() != HC_MULT * HC_MULT
        || result.hc_attn_pre_bf16_bits.len() != HIDDEN_SIZE
        || result.attn_norm_bf16_bits.len() != LAYER0_WQ_A_COLS
        || result.wq_a_input_act_quant.activation_e4m3fn.len() != LAYER0_WQ_A_COLS
        || result.wq_a_input_act_quant.scales_e8m0fnu.len() != LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK
    {
        return Err(failure(
            "layer-0 prefix checkpoint returned unexpected geometry",
        ));
    }

    let embed = reader.tensor_metadata("embed.weight")?;
    let hc_fn = reader.tensor_metadata(LAYER0_HC_ATTN_FN)?;
    let hc_base = reader.tensor_metadata(LAYER0_HC_ATTN_BASE)?;
    let hc_scale = reader.tensor_metadata(LAYER0_HC_ATTN_SCALE)?;
    let attn_norm = reader.tensor_metadata(LAYER0_ATTN_NORM_WEIGHT)?;
    let wq_a = reader.native_scale_pair(LAYER0_WQ_A_WEIGHT)?;
    if wq_a.weight.shape != [LAYER0_WQ_A_ROWS as u64, LAYER0_WQ_A_COLS as u64]
        || wq_a.scale.name != LAYER0_WQ_A_SCALE
    {
        return Err(failure(
            "layer-0 WQ-A source contract changed after prefix execution",
        ));
    }

    let embed_row_bytes = (HIDDEN_SIZE * std::mem::size_of::<u16>()) as u64;
    let embed_row_start = PREFIX_TOKEN_ID * embed_row_bytes;
    let sinkhorn_row_sums = matrix_sums(&result.hc_comb_f32, true)?;
    let sinkhorn_column_sums = matrix_sums(&result.hc_comb_f32, false)?;
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
        "source_algorithm_bindings": {
            "official_assets_verified_by_admitted_full_stream_and_exact_anchor": {
                "inference/model.py": anchors.act_quant.inference_model_py_sha256,
                "inference/kernel.py": anchors.act_quant.inference_kernel_py_sha256,
                "inference/config.json": anchors.act_quant.inference_config_json_sha256,
                "config.json": anchors.act_quant.model_config_json_sha256,
                "tokenizer.json": anchors.tokenizer_json_sha256,
                "tokenizer_config.json": anchors.tokenizer_config_json_sha256,
            },
            "tokenizer_binding": {
                "fixed_token_id": PREFIX_TOKEN_ID,
                "fixed_bpe_token": PREFIX_TOKEN_STRING,
                "tokenizer_json_model_vocab_exact_mapping_verified": true,
                "tokenizer_config_bos_content_exact_mapping_verified": true,
                "config_bos_token_id_exact_mapping_verified": true,
                "no_prompt_text_or_prompt-derived_hidden_state_recorded": true,
            },
            "source_algorithm_path": [
                "model.py::Transformer.forward: self.embed(input_ids)",
                "model.py::Transformer.forward: h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)",
                "model.py::Block.hc_pre: flatten BF16 HC lanes to FP32, F.linear(hc_attn_fn), rsqrt(mean(square)+norm_eps), hc_split_sinkhorn, lane reduction, BF16 cast",
                "kernel.py::hc_split_sinkhorn_kernel: sigmoid pre/post, row softmax plus eps, first column normalization, then 19 row/column Sinkhorn passes",
                "model.py::RMSNorm.forward: FP32 variance/multiply against BF16-loaded norm weight, then BF16 cast",
                "model.py::linear: the resulting BF16 [4096] is the input accepted by FP8 layer0 attn.wq_a act_quant(block=128, ue8m0, E8M0FNU)",
            ],
            "pinned_constants_verified_from_config_assets": {
                "vocab_size": VOCAB_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "hc_mult": HC_MULT,
                "hc_sinkhorn_iters": HC_SINKHORN_ITERS,
                "hc_eps": "1e-6",
                "rms_norm_eps": "1e-6",
            },
        },
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
            "next_wq_a_input_contract_not_read_or_executed": native_pair_contract_json(&wq_a),
        },
        "intermediate_receipts": {
            "embedding": bf16_checkpoint_json(&result.embed_bf16_bits, &[HIDDEN_SIZE]),
            "hc_replicate": {
                "shape": [HC_MULT, HIDDEN_SIZE],
                "dtype": "BF16",
                "sha256_bf16_le": sha256_hex(&u16_le_bytes(&result.hc_replicated_bf16_bits)),
                "four_lanes_exactly_equal_to_embedding_row": result.hc_replicated_bf16_bits.chunks_exact(HIDDEN_SIZE).all(|lane| lane == result.embed_bf16_bits.as_slice()),
            },
            "hc_attn_pre": {
                "flat_shape": [HC_MULT, HIDDEN_SIZE],
                "flat_fp32_rsqrt": result.hc_flat_rsqrt.to_string(),
                "mixes_shape": [HC_MIX_WIDTH],
                "mixes_f32": f32_checkpoint_json(&result.hc_mixes_f32, &[HC_MIX_WIDTH])?,
                "sinkhorn": {
                    "hc_mult": HC_MULT,
                    "iterations": HC_SINKHORN_ITERS,
                    "eps": "1e-6",
                    "first_pass": "row softmax element = exp(logit-row_max)/row_sum + eps; then column divide by (column_sum + eps)",
                    "remaining_passes": 19,
                    "remaining_order": "row divide by (row_sum + eps), then column divide by (column_sum + eps)",
                    "pre_f32": f32_checkpoint_json(&result.hc_pre_f32, &[HC_MULT])?,
                    "post_f32": f32_checkpoint_json(&result.hc_post_f32, &[HC_MULT])?,
                    "comb_f32": f32_checkpoint_json(&result.hc_comb_f32, &[HC_MULT, HC_MULT])?,
                    "row_sums_after_20": sinkhorn_row_sums,
                    "column_sums_after_20": sinkhorn_column_sums,
                },
                "reduced_bf16": bf16_checkpoint_json(&result.hc_attn_pre_bf16_bits, &[HIDDEN_SIZE]),
            },
            "attention_rmsnorm": {
                "input": "hc_attn_pre reduced BF16 [4096]",
                "weight": "real layer0 attn_norm.weight BF16 [4096] loaded as FP32 values by source RMSNorm",
                "eps": "1e-6",
                "output": bf16_checkpoint_json(&result.attn_norm_bf16_bits, &[HIDDEN_SIZE]),
            },
            "wq_a_input": {
                "shape": [LAYER0_WQ_A_COLS],
                "dtype": "BF16",
                "equals_attention_rmsnorm_output": true,
                "sha256_bf16_le": sha256_hex(&u16_le_bytes(&result.attn_norm_bf16_bits)),
                "source_linear_handoff_act_quant": {
                    "block_size": ACT_QUANT_BLOCK,
                    "activation_dtype": "F8_E4M3FN",
                    "scale_dtype": "F8_E8M0FNU",
                    "scale_format": "ue8m0",
                    "activation_sha256": sha256_hex(&result.wq_a_input_act_quant.activation_e4m3fn),
                    "scale_sha256": sha256_hex(&result.wq_a_input_act_quant.scales_e8m0fnu),
                    "scale_e8m0fnu_bytes": result.wq_a_input_act_quant.scales_e8m0fnu,
                    "fp8_wq_a_weight_read": false,
                    "fp8_wq_a_gemm_executed": false,
                },
            },
        },
        "execution_boundary": {
            "source_derived_cpu_algorithm_prefix": true,
            "independently_upstream_runtime_parity": false,
            "not_independently_upstream_runtime_parity": true,
            "source_runtime_executed": false,
            "pytorch_or_tilelang_executed": false,
            "full_model_loaded": false,
            "full_model_forward": false,
            "attention_projection_executed": false,
            "generated_tokens": 0,
            "metal_allocations": 0,
            "gpu_dispatches": 0,
            "command_buffers": 0,
            "cpu_visible_waits": 0,
            "hcli_endpoint_started": false,
            "base_true_tps_measured": false,
            "claim": "bounded translated CPU source-algorithm prefix only; NOT independently upstream-runtime parity, a GPU result, full forward, token-generation, HCLI, or BASE_TRUE_TPS result",
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
                    "usage: gravity_deepseek_v4_layer0_prefix_oracle --artifact <absolute full Gravity dir> --out <absolute receipt.json>"
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

fn native_pair_contract_json(pair: &NativeScalePair<'_>) -> Value {
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
        "read_or_executed_by_this_prefix": false,
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

fn bf16_checkpoint_json(bits: &[u16], shape: &[usize]) -> Value {
    let values: Vec<f32> = bits
        .iter()
        .map(|bits| bf16::from_bits(*bits).to_f32())
        .collect();
    json!({
        "shape": shape,
        "dtype": "BF16",
        "element_count": bits.len(),
        "sha256_bf16_le": sha256_hex(&u16_le_bytes(bits)),
        "decoded_f32_stats": finite_stats(&values).expect("prefix BF16 checkpoint was validated finite"),
    })
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

fn matrix_sums(values: &[f32], row_sums: bool) -> ExampleResult<Vec<String>> {
    if values.len() != HC_MULT * HC_MULT || values.iter().any(|value| !value.is_finite()) {
        return Err(failure("Sinkhorn combination matrix is not finite 4x4"));
    }
    Ok((0..HC_MULT)
        .map(|axis| {
            let sum = (0..HC_MULT)
                .map(|offset| {
                    if row_sums {
                        values[axis * HC_MULT + offset]
                    } else {
                        values[offset * HC_MULT + axis]
                    }
                })
                .sum::<f32>();
            sum.to_string()
        })
        .collect())
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

/// Create without overwriting a prior scientific receipt.  The hard-link
/// publish gives the final name create-new semantics after a durable temporary
/// write has succeeded.
fn write_new_sealed_receipt(path: &Path, receipt: &Value) -> ExampleResult<()> {
    if path.exists() {
        return Err(failure(format!(
            "refusing to overwrite existing layer-0 prefix receipt {}",
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
    let temporary = parent.join(format!(".{name}.{}.layer0-prefix.tmp", std::process::id()));
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
            "refusing to overwrite or link layer-0 prefix receipt {}: {error}",
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
