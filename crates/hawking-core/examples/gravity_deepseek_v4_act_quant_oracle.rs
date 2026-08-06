//! Bounded CPU source-algorithm oracle for the next DeepSeek-V4 `Linear` checkpoint.
//!
//! It consumes one deterministic, exact-BF16 input row and the real sealed
//! `layers.0.attn.wq_a` FP8/E8M0 source pair.  The run is intentionally not a
//! model load, forward, generation, endpoint, Metal dispatch, or TPS result.
//! It writes a new sealed receipt only after verifying the source artifact,
//! source-code/config anchors, bounded tensor bytes, source-derived
//! `act_quant`, and scalar CPU FP8 GEMV result.
//!
//! ```sh
//! cargo run --release -p hawking-core --example gravity_deepseek_v4_act_quant_oracle -- \
//!   --artifact /absolute/path/to/full-43-layer-stream.gravity \
//!   --out /absolute/path/to/DSV4F_ACT_QUANT_WQ_A_CPU_ORACLE.json
//! ```

use hawking_core::gravity_deepseek_v4::{
    DeepSeekV4FullStreamReader, FULL_STREAM_SCHEMA, FULL_STREAM_STATUS,
};
use hawking_core::gravity_deepseek_v4_act_quant::{
    deterministic_wq_a_input_bf16, layer0_wq_a_cpu_oracle, verify_source_algorithm_anchors,
    ACT_QUANT_BLOCK, LAYER0_WQ_A_COLS, LAYER0_WQ_A_ROWS, LAYER0_WQ_A_SCALE, LAYER0_WQ_A_WEIGHT,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const RECEIPT_SCHEMA: &str =
    "hawking.gravity.deepseek_v4.act_quant_fp8_wq_a_cpu_algorithm_oracle.v1";
const RECEIPT_STATUS: &str =
    "PASS_SOURCE_DERIVED_CPU_ALGORITHM_ORACLE_NOT_INDEPENDENT_SOURCE_RUNTIME_PARITY";

type ExampleResult<T> = Result<T, Box<dyn Error>>;

struct Args {
    artifact: PathBuf,
    out: PathBuf,
}

fn main() -> ExampleResult<()> {
    let args = parse_args()?;
    let reader = DeepSeekV4FullStreamReader::admit(&args.artifact)?;
    let anchors = verify_source_algorithm_anchors(&reader)?;

    let (weight, scale) = {
        let pair = reader.native_scale_pair(LAYER0_WQ_A_WEIGHT)?;
        if pair.weight.name != LAYER0_WQ_A_WEIGHT || pair.scale.name != LAYER0_WQ_A_SCALE {
            return Err(failure(
                "layer-0 WQ-A pair name changed after reader admission",
            ));
        }
        (pair.weight.clone(), pair.scale.clone())
    };
    let input = deterministic_wq_a_input_bf16();
    if input.len() != LAYER0_WQ_A_COLS {
        return Err(failure(
            "deterministic BF16 oracle input does not match WQ-A K",
        ));
    }
    let result = layer0_wq_a_cpu_oracle(&reader, &input)?;
    if result.output.fp32.len() != LAYER0_WQ_A_ROWS
        || result.output.bf16_bits.len() != LAYER0_WQ_A_ROWS
        || result.quantized_input.activation_e4m3fn.len() != LAYER0_WQ_A_COLS
        || result.quantized_input.scales_e8m0fnu.len() != LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK
    {
        return Err(failure(
            "bounded CPU oracle returned unexpected WQ-A geometry",
        ));
    }

    let output_stats = finite_stats(&result.output.fp32)?;
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
                "inference/model.py": anchors.inference_model_py_sha256,
                "inference/kernel.py": anchors.inference_kernel_py_sha256,
                "inference/config.json": anchors.inference_config_json_sha256,
                "config.json": anchors.model_config_json_sha256,
            },
            "model_linear_path": "model.py::linear selects act_quant(x, block_size=128, scale_fmt=ue8m0, scale_dtype=float8_e8m0fnu) then fp8_gemm for F8_E4M3 weights",
            "act_quant_path": "kernel.py::act_quant_kernel performs BF16 block absmax, max(amax, 1e-4), fast_round_scale, clamp[-448,448], E4M3FN cast, and E8M0FNU scale storage",
            "fp8_gemm_path": "kernel.py::fp8_gemm_kernel accumulates 128-wide E4M3FN dot blocks in FP32 and multiplies activation and [out/128,K/128] E8M0FNU scales",
        },
        "bounded_source_reads": {
            "reader_admission_verified_manifest_source_index_offsets_and_chunk_tree": true,
            "all_chunk_sha256_bytes_verified_globally": false,
            "touched_tensor_chunks_sha256_verified_before_cpu_use": true,
            "weight": tensor_binding_json(&weight),
            "scale": tensor_binding_json(&scale),
            "maximum_temporary_source_weight_bytes": LAYER0_WQ_A_ROWS * LAYER0_WQ_A_COLS,
            "maximum_temporary_source_scale_bytes": (LAYER0_WQ_A_ROWS / ACT_QUANT_BLOCK) * (LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK),
            "parent_safetensors_materialized": false,
        },
        "input": {
            "kind": "deterministic_exact_bf16_bitpattern_vector_v1",
            "captured_from_model_forward": false,
            "length": input.len(),
            "dtype": "BF16",
            "byte_order": "little-endian",
            "sha256_bf16_le": sha256_hex(&u16_le_bytes(&input)),
        },
        "act_quant": {
            "source_derived": true,
            "input_dtype": "BF16",
            "block_size": ACT_QUANT_BLOCK,
            "activation_dtype": "F8_E4M3FN",
            // Keep human-readable values as integers/strings in the sealed
            // document.  That lets the Rust producer and the campaign's
            // Python canonical-JSON verifier agree exactly on the receipt
            // bytes rather than relying on two language-specific float
            // renderers.
            "activation_finite_range": [-448, 448],
            "scale_dtype": "F8_E8M0FNU",
            "scale_format": "ue8m0",
            "scale_rounding": "2^ceil(log2(max(abs(block))/448)) after max(amax,1e-4)",
            "amax_floor": "1e-4",
            "activation_bytes": result.quantized_input.activation_e4m3fn.len(),
            "activation_sha256": sha256_hex(&result.quantized_input.activation_e4m3fn),
            "scale_bytes": result.quantized_input.scales_e8m0fnu.len(),
            "scale_sha256": sha256_hex(&result.quantized_input.scales_e8m0fnu),
            "scale_e8m0fnu_bytes": result.quantized_input.scales_e8m0fnu,
        },
        "cpu_fp8_gemv": {
            "operator": "layer0_wq_a",
            "shape": [LAYER0_WQ_A_ROWS, LAYER0_WQ_A_COLS],
            "accumulation": "scalar f32 product_then_add within K=128 blocks; scaled block accumulators added row-major",
            "output_fp32_count": result.output.fp32.len(),
            "output_fp32_le_sha256": sha256_hex(&f32_le_bytes(&result.output.fp32)),
            "output_bf16_count": result.output.bf16_bits.len(),
            "output_bf16_le_sha256": sha256_hex(&u16_le_bytes(&result.output.bf16_bits)),
            "output_fp32_stats": output_stats,
        },
        "execution_boundary": {
            "source_derived_algorithm_oracle": true,
            "independently_source_runtime_parity": false,
            "not_independently_source_runtime_parity": true,
            "source_runtime_executed": false,
            "full_model_loaded": false,
            "full_model_forward": false,
            "generated_tokens": 0,
            "metal_allocations": 0,
            "gpu_dispatches": 0,
            "command_buffers": 0,
            "cpu_visible_waits": 0,
            "hcli_endpoint_started": false,
            "base_true_tps_measured": false,
            "claim": "bounded source-derived CPU algorithm oracle only; NOT independently source runtime parity; not a GPU, full-forward, token-generation, HCLI, or BASE_TRUE_TPS result",
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
                    "usage: gravity_deepseek_v4_act_quant_oracle --artifact <absolute full Gravity dir> --out <absolute receipt.json>"
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

fn tensor_binding_json(
    tensor: &hawking_core::gravity_deepseek_v4::DeepSeekV4TensorMetadata,
) -> Value {
    json!({
        "name": tensor.name,
        "dtype": tensor.dtype,
        "shape": tensor.shape,
        "bytes": tensor.bytes,
        "source_shard": tensor.source_shard,
        "source_file_start": tensor.source_file_start,
        "source_file_end": tensor.source_file_end,
        "segments": tensor.segments.iter().map(|segment| json!({
            "bytes": segment.bytes,
            "chunk_relpath": segment.chunk_relpath,
            "sha256": segment.sha256,
            "tensor_start": segment.tensor_start,
            "tensor_end": segment.tensor_end,
            "row_start": segment.row_start,
            "row_count": segment.row_count,
        })).collect::<Vec<_>>(),
    })
}

fn finite_stats(values: &[f32]) -> ExampleResult<Value> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(failure(
            "CPU FP8 GEMV produced an empty or non-finite output",
        ));
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
        // Float values are hashes' explanatory companions, not numeric inputs
        // to a later computation.  Serialize them as strings so the receipt's
        // canonical seal is identical under Rust and Python JSON rules.
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

/// Create the final receipt without replacing a prior measurement.  A hard
/// link in the same directory gives the final name `create_new` semantics
/// after the temporary file has been durably written.
fn write_new_sealed_receipt(path: &Path, receipt: &Value) -> ExampleResult<()> {
    if path.exists() {
        return Err(failure(format!(
            "refusing to overwrite existing CPU oracle receipt {}",
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
        ".{name}.{}.act-quant-oracle.tmp",
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
            "refusing to overwrite or link CPU oracle receipt {}: {error}",
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
