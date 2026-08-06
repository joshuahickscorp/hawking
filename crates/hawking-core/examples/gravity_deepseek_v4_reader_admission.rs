//! Bounded, read-only admission receipt for a sealed DeepSeek-V4 full stream.
//!
//! This example does not create an engine, allocate Metal memory, execute a
//! forward, start an endpoint, or measure TPS.  It admits the complete
//! content-addressed source stream and performs a small, explicit set of
//! verified reads to prove the reusable reader boundary.
//!
//! ```sh
//! cargo run --release -p hawking-core --example gravity_deepseek_v4_reader_admission -- \
//!   --artifact /absolute/path/to/full-43-layer-stream.gravity \
//!   --out /absolute/path/to/DSV4F_FULL_STREAM_READER_ADMISSION.json
//! ```

use hawking_core::gravity_deepseek_v4::{
    DeepSeekV4FullStreamReader, NativeScalePair, NativeScalePairKind, FULL_STREAM_SCHEMA,
    FULL_STREAM_STATUS,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const RECEIPT_SCHEMA: &str = "hawking.gravity.deepseek_v4.full_stream_reader_admission.v1";
const RECEIPT_STATUS: &str = "PASS_FULL_STREAM_READER_ADMISSION_NOT_FORWARD_OR_RUNTIME";
const FP8_WEIGHT: &str = "layers.0.attn.wq_a.weight";
const FP4_WEIGHT: &str = "layers.0.ffn.experts.0.w1.weight";
const HEAD: &str = "head.weight";
const SMALL_RANGE_BYTES: u64 = 65_536;
const FULL_COMPONENT_READ_BOUND: usize = 8 * 1024 * 1024;
const RANGE_READ_BOUND: usize = SMALL_RANGE_BYTES as usize;

type ExampleResult<T> = Result<T, Box<dyn Error>>;

struct Args {
    artifact: PathBuf,
    out: PathBuf,
}

fn main() -> ExampleResult<()> {
    let args = parse_args()?;
    let reader = DeepSeekV4FullStreamReader::admit(&args.artifact)?;

    let fp8 = reader.native_scale_pair(FP8_WEIGHT)?;
    let fp4 = reader.native_scale_pair(FP4_WEIGHT)?;
    if fp8.kind != NativeScalePairKind::Fp8E4M3fn
        || fp8.weight.shape != [1024, 4096]
        || fp8.scale.shape != [8, 32]
        || fp4.kind != NativeScalePairKind::Fp4E2M1fnX2
        || fp4.weight.shape != [2048, 2048]
        || fp4.scale.shape != [2048, 128]
    {
        return Err(failure(
            "reader admission did not bind the expected source-native FP8/FP4 probes",
        ));
    }

    // Full reads remain component-bounded and use the reader's explicit
    // allocation ceiling.  The reader verifies every touched physical chunk
    // before returning any bytes.
    let fp8_weight = reader.read_verified_full(FP8_WEIGHT, FULL_COMPONENT_READ_BOUND)?;
    let fp8_scale = reader.read_verified_full(&fp8.scale.name, FULL_COMPONENT_READ_BOUND)?;
    let fp4_weight = reader.read_verified_full(FP4_WEIGHT, FULL_COMPONENT_READ_BOUND)?;
    let fp4_scale = reader.read_verified_full(&fp4.scale.name, FULL_COMPONENT_READ_BOUND)?;
    let head_range = reader.read_verified_range(HEAD, 0..SMALL_RANGE_BYTES, RANGE_READ_BOUND)?;

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
        "admission_validity": {
            "manifest_seal_verified": true,
            "schema_status_pinned_source_verified": true,
            "restart_receipt_and_journal_bindings_verified": true,
            "all_named_tensor_source_index_bindings_verified": true,
            "all_tensor_segment_contiguity_and_source_offset_mappings_verified": true,
            "all_referenced_chunk_paths_regular_non_symlink_and_exact_tree_verified": true,
            "all_chunk_sha256_bytes_verified": false,
            "all_chunk_sha256_bytes_verification_method": "call DeepSeekV4FullStreamReader::verify_all_chunks explicitly; this bounded receipt verifies every chunk touched by its listed reads",
            "tensor_count": reader.tensor_count(),
            "source_native_tensor_bytes": reader.tensor_bytes(),
            "content_addressed_chunk_count": reader.chunk_count(),
            "native_scale_pair_count": reader.native_scale_pair_count(),
            "native_fp8_pair_count": reader.native_scale_pair_count_for(NativeScalePairKind::Fp8E4M3fn),
            "native_fp4_pair_count": reader.native_scale_pair_count_for(NativeScalePairKind::Fp4E2M1fnX2),
            "metadata_assets_sha256_verified": reader.source_metadata_asset_count(),
            "pinned_codec_assets": {
                "inference_model_py_sha256": reader.source_metadata_asset_sha256("inference/model.py")?,
                "inference_kernel_py_sha256": reader.source_metadata_asset_sha256("inference/kernel.py")?,
                "inference_convert_py_sha256": reader.source_metadata_asset_sha256("inference/convert.py")?,
            },
        },
        "validated_native_scale_contracts": [
            pair_receipt(&fp8),
            pair_receipt(&fp4),
        ],
        "verified_reads": [
            read_receipt(FP8_WEIGHT, "full_tensor", 0, fp8_weight.len() as u64, &fp8_weight),
            read_receipt(&fp8.scale.name, "full_tensor", 0, fp8_scale.len() as u64, &fp8_scale),
            read_receipt(FP4_WEIGHT, "full_tensor", 0, fp4_weight.len() as u64, &fp4_weight),
            read_receipt(&fp4.scale.name, "full_tensor", 0, fp4_scale.len() as u64, &fp4_scale),
            read_receipt(HEAD, "bounded_range", 0, SMALL_RANGE_BYTES, &head_range),
        ],
        "execution_boundary": {
            "reader_only": true,
            "engine_created": false,
            "metal_allocations": 0,
            "gpu_dispatches": 0,
            "forward_tokens": 0,
            "hcli_endpoint_started": false,
            "base_true_tps_measured": false,
            "public_cli_serve_admission_changed": false,
            "claim": "source-stream reader admission only; not a runtime, forward, endpoint, parity, or TPS result",
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
                    "usage: gravity_deepseek_v4_reader_admission --artifact <absolute full Gravity dir> --out <absolute receipt.json>"
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

fn pair_receipt(pair: &NativeScalePair<'_>) -> Value {
    json!({
        "kind": pair.kind.as_str(),
        "weight": {
            "name": pair.weight.name,
            "dtype": pair.weight.dtype,
            "shape": pair.weight.shape,
            "bytes": pair.weight.bytes,
            "source_shard": pair.weight.source_shard,
        },
        "scale": {
            "name": pair.scale.name,
            "dtype": pair.scale.dtype,
            "shape": pair.scale.shape,
            "bytes": pair.scale.bytes,
            "source_shard": pair.scale.source_shard,
        },
        "geometry": {
            "out_rows": pair.out_rows,
            "packed_k": pair.packed_k,
            "logical_k": pair.logical_k,
            "scale_rows": pair.scale_rows,
            "scale_cols": pair.scale_cols,
        },
    })
}

fn read_receipt(name: &str, kind: &str, start: u64, end: u64, bytes: &[u8]) -> Value {
    json!({
        "tensor": name,
        "kind": kind,
        "range": {"start": start, "end_exclusive": end},
        "returned_bytes": bytes.len(),
        "returned_bytes_sha256": sha256(bytes),
        "touched_chunks_sha256_verified_before_return": true,
    })
}

fn seal(mut value: Value) -> ExampleResult<Value> {
    let object = value
        .as_object_mut()
        .ok_or_else(|| failure("receipt root must be a JSON object"))?;
    if object.contains_key("seal_sha256") {
        return Err(failure("receipt unexpectedly already has a seal"));
    }
    // serde_json's default map is sorted and the receipt carries no floating
    // point values, so this is the compact sort-key canonical form used by the
    // campaign receipts.
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
            "refusing to overwrite existing reader admission receipt {}",
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
        ".{name}.{}.reader-admission.tmp",
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
