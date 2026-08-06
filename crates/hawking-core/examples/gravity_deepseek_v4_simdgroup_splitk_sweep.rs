//! Artifact-bound optional SIMDgroup/split-K sweep for two DeepSeek-V4 raw
//! weight component kernels.
//!
//! This benchmark deliberately stops at the raw-weight component boundary.
//! It does **not** execute `model.py` activation quantization, a transformer
//! forward, routing, MoE scheduling, HCLI, or a TPS benchmark.  Its job is to
//! compare serial source-native authority kernels with optional parallel
//! candidates against the *same* sealed bytes, deterministic FP32 input, and
//! byte-decoded CPU reference.
//!
//! ```sh
//! cargo run --release -p hawking-core --example gravity_deepseek_v4_simdgroup_splitk_sweep -- \
//!   --artifact /absolute/path/to/full-43-layer-stream.gravity \
//!   --out /absolute/path/to/DSV4F_RAW_WEIGHT_SIMDGROUP_SPLITK_SWEEP.json
//! ```

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(
        std::io::Error::other("gravity_deepseek_v4_simdgroup_splitk_sweep requires macOS Metal")
            .into(),
    )
}

#[cfg(target_os = "macos")]
#[path = "gravity_deepseek_v4_fp8_metal_probe.rs"]
mod fp8_authority;

#[cfg(target_os = "macos")]
#[path = "gravity_deepseek_v4_fp4_metal_probe.rs"]
mod fp4_authority;

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::metal::{MetalContext, PhysicalTraceGuard, PhysicalTraceIdentity};
    use serde_json::{json, Map, Value};
    use sha2::{Digest, Sha256};
    use std::cmp::Ordering;
    use std::error::Error;
    use std::fs::{self, File, OpenOptions};
    use std::io::Write;
    use std::path::{Component, Path, PathBuf};

    const ARTIFACT_SCHEMA: &str = "hawking.gravity.deepseek_v4.full_stream.v1";
    const ARTIFACT_STATUS: &str = "FULL_MODEL_STREAMED_SEALED_NOT_RUNTIME_READY";
    const REPOSITORY: &str = "deepseek-ai/DeepSeek-V4-Flash";
    const REVISION: &str = "60d8d70770c6776ff598c94bb586a859a38244f1";
    const FP8_KERNEL: &str = "deepseek_v4_fp8_e4m3fn_e8m0_matvec_simdgroup_v4_splitk_candidate";
    const FP4_KERNEL: &str = "deepseek_v4_fp4_e2m1fn_x2_e8m0_matvec_simdgroup_v4_splitk_candidate";
    const AUTHORITY_THREADGROUPS: &[u32] = &[32, 64, 128, 256, 512, 1024, 2048];
    const DEFAULT_WARMUPS: usize = 3;
    const DEFAULT_TRIALS: usize = 9;
    const MAX_ROWS_PER_THREADGROUP: u32 = 4;
    const PARTIAL_BYTES_PER_ROW: u64 = 32 * std::mem::size_of::<f32>() as u64;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Geometry {
        threads_x: u32,
        rows_per_threadgroup: u32,
    }

    // A deliberately finite ladder, ordered by total group size then row
    // reuse. Unsupported rungs remain in the receipt rather than being hidden.
    const GEOMETRIES: &[Geometry] = &[
        Geometry {
            threads_x: 32,
            rows_per_threadgroup: 1,
        },
        Geometry {
            threads_x: 32,
            rows_per_threadgroup: 2,
        },
        Geometry {
            threads_x: 32,
            rows_per_threadgroup: 4,
        },
        Geometry {
            threads_x: 64,
            rows_per_threadgroup: 1,
        },
        Geometry {
            threads_x: 64,
            rows_per_threadgroup: 2,
        },
        Geometry {
            threads_x: 64,
            rows_per_threadgroup: 4,
        },
        Geometry {
            threads_x: 128,
            rows_per_threadgroup: 1,
        },
        Geometry {
            threads_x: 128,
            rows_per_threadgroup: 2,
        },
        Geometry {
            threads_x: 128,
            rows_per_threadgroup: 4,
        },
        Geometry {
            threads_x: 256,
            rows_per_threadgroup: 1,
        },
        Geometry {
            threads_x: 256,
            rows_per_threadgroup: 2,
        },
        Geometry {
            threads_x: 256,
            rows_per_threadgroup: 4,
        },
        Geometry {
            threads_x: 512,
            rows_per_threadgroup: 1,
        },
        Geometry {
            threads_x: 512,
            rows_per_threadgroup: 2,
        },
        Geometry {
            threads_x: 1024,
            rows_per_threadgroup: 1,
        },
        // Explicitly invalid on normal 1024-thread Metal pipelines; retained
        // so the evidence exposes the requested depth of the ladder.
        Geometry {
            threads_x: 1024,
            rows_per_threadgroup: 2,
        },
    ];

    type SweepResult<T> = Result<T, Box<dyn Error>>;

    struct Args {
        artifact: PathBuf,
        out: PathBuf,
        warmups: usize,
        trials: usize,
    }

    struct BoundTensor {
        name: String,
        dtype: String,
        shape: Vec<u64>,
        raw: Vec<u8>,
        source_shard: String,
        source_window: Value,
        descriptor: Value,
        segments: Vec<Value>,
    }

    fn failure(message: impl Into<String>) -> Box<dyn Error> {
        std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
    }

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn sha256_join(parts: &[&str]) -> String {
        let mut digest = Sha256::new();
        for part in parts {
            digest.update(part.as_bytes());
            digest.update([0]);
        }
        format!("{:x}", digest.finalize())
    }

    fn is_sha256(value: &str) -> bool {
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }

    fn parse_positive(value: Option<String>, flag: &str) -> SweepResult<usize> {
        let value = value.ok_or_else(|| failure(format!("{flag} needs a value")))?;
        let parsed = value
            .parse::<usize>()
            .map_err(|_| failure(format!("{flag} must be a positive integer")))?;
        if parsed == 0 {
            return Err(failure(format!("{flag} must be positive")));
        }
        Ok(parsed)
    }

    fn parse_args() -> SweepResult<Args> {
        let mut artifact = None::<PathBuf>;
        let mut out = None::<PathBuf>;
        let mut warmups = DEFAULT_WARMUPS;
        let mut trials = DEFAULT_TRIALS;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--artifact" => artifact = args.next().map(PathBuf::from),
                "--out" => out = args.next().map(PathBuf::from),
                "--warmups" => warmups = parse_positive(args.next(), "--warmups")?,
                "--trials" => trials = parse_positive(args.next(), "--trials")?,
                "--help" | "-h" => {
                    println!(
                        "usage: gravity_deepseek_v4_simdgroup_splitk_sweep --artifact <absolute full Gravity dir> --out <absolute receipt.json> [--warmups N] [--trials N]"
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
        Ok(Args {
            artifact,
            out,
            warmups,
            trials,
        })
    }

    fn checked_regular_path(root: &Path, relative: &str, label: &str) -> SweepResult<PathBuf> {
        let relative_path = Path::new(relative);
        if relative_path.is_absolute()
            || relative_path.components().any(|component| {
                matches!(
                    component,
                    Component::ParentDir | Component::RootDir | Component::Prefix(_)
                )
            })
        {
            return Err(failure(format!(
                "{label} path escapes artifact: {relative}"
            )));
        }
        let path = root.join(relative_path);
        let metadata = fs::symlink_metadata(&path).map_err(|error| {
            failure(format!(
                "cannot inspect {label} {}: {error}",
                path.display()
            ))
        })?;
        if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
            return Err(failure(format!(
                "{label} must be a regular non-symlink file: {}",
                path.display()
            )));
        }
        Ok(path)
    }

    fn object<'a>(value: &'a Value, label: &str) -> SweepResult<&'a Map<String, Value>> {
        value
            .as_object()
            .ok_or_else(|| failure(format!("{label} must be a JSON object")))
    }

    fn array<'a>(value: &'a Value, label: &str) -> SweepResult<&'a Vec<Value>> {
        value
            .as_array()
            .ok_or_else(|| failure(format!("{label} must be a JSON array")))
    }

    fn field<'a>(object: &'a Value, key: &str, label: &str) -> SweepResult<&'a Value> {
        object
            .get(key)
            .ok_or_else(|| failure(format!("{label} lacks {key}")))
    }

    fn string_field<'a>(object: &'a Value, key: &str, label: &str) -> SweepResult<&'a str> {
        field(object, key, label)?
            .as_str()
            .ok_or_else(|| failure(format!("{label}.{key} must be a string")))
    }

    fn u64_field(object: &Value, key: &str, label: &str) -> SweepResult<u64> {
        field(object, key, label)?
            .as_u64()
            .ok_or_else(|| failure(format!("{label}.{key} must be an unsigned integer")))
    }

    fn bool_field(object: &Value, key: &str, label: &str) -> SweepResult<bool> {
        field(object, key, label)?
            .as_bool()
            .ok_or_else(|| failure(format!("{label}.{key} must be a boolean")))
    }

    fn verify_full_manifest(artifact_arg: &Path) -> SweepResult<(PathBuf, Value, String)> {
        let artifact_metadata = fs::symlink_metadata(artifact_arg)?;
        if artifact_metadata.file_type().is_symlink() || !artifact_metadata.file_type().is_dir() {
            return Err(failure("--artifact must be a non-symlink directory"));
        }
        let artifact = fs::canonicalize(artifact_arg)?;
        let manifest_path =
            checked_regular_path(&artifact, "manifest.json", "full Gravity manifest")?;
        let manifest_raw = fs::read(&manifest_path)?;
        let manifest_file_sha256 = sha256(&manifest_raw);
        let mut manifest: Value = serde_json::from_slice(&manifest_raw)
            .map_err(|error| failure(format!("full Gravity manifest is not JSON: {error}")))?;
        let recorded_seal = object(&manifest, "full Gravity manifest")?
            .get("seal_sha256")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| failure("full Gravity manifest lacks seal_sha256"))?;
        if !is_sha256(&recorded_seal) {
            return Err(failure(
                "full Gravity manifest seal is not lowercase SHA-256",
            ));
        }
        object(&manifest, "full Gravity manifest")?;
        manifest
            .as_object_mut()
            .expect("manifest object was checked")
            .remove("seal_sha256");
        let observed_seal = sha256(&serde_json::to_vec(&manifest)?);
        if observed_seal != recorded_seal {
            return Err(failure(format!(
                "full Gravity manifest seal mismatch: recorded={recorded_seal} observed={observed_seal}"
            )));
        }
        let manifest_object = manifest
            .as_object_mut()
            .expect("manifest object was checked");
        manifest_object.insert("seal_sha256".to_owned(), Value::String(recorded_seal));

        if string_field(&manifest, "schema", "full Gravity manifest")? != ARTIFACT_SCHEMA
            || string_field(&manifest, "status", "full Gravity manifest")? != ARTIFACT_STATUS
        {
            return Err(failure(
                "artifact is not the sealed DeepSeek-V4 full stream runtime-pending state",
            ));
        }
        let source = field(&manifest, "source", "full Gravity manifest")?;
        if string_field(source, "repository", "manifest.source")? != REPOSITORY
            || string_field(source, "revision", "manifest.source")? != REVISION
            || bool_field(source, "source_parent_persisted", "manifest.source")?
        {
            return Err(failure(
                "full Gravity artifact source identity/storage contract is not eligible",
            ));
        }
        Ok((artifact, manifest, manifest_file_sha256))
    }

    fn bind_tensor(
        artifact: &Path,
        manifest: &Value,
        name: &str,
        expected_dtype: &str,
        expected_shape: &[u64],
    ) -> SweepResult<BoundTensor> {
        let source = field(manifest, "source", "full Gravity manifest")?;
        let assets = field(source, "metadata_assets", "manifest.source")?;
        let index_asset = object(assets, "manifest.source.metadata_assets")?
            .get("model.safetensors.index.json")
            .ok_or_else(|| failure("manifest source lacks model.safetensors.index.json"))?;
        let index_path = string_field(index_asset, "path", "source index asset")?;
        let index_bytes = u64_field(index_asset, "bytes", "source index asset")?;
        let index_sha256 = string_field(index_asset, "sha256", "source index asset")?;
        if index_path != "model.safetensors.index.json" || !is_sha256(index_sha256) {
            return Err(failure("source index asset binding is malformed"));
        }
        let raw_index = fs::read(checked_regular_path(
            &artifact.join("metadata"),
            index_path,
            "model source tensor index",
        )?)?;
        if raw_index.len() as u64 != index_bytes || sha256(&raw_index) != index_sha256 {
            return Err(failure(
                "source tensor index bytes/hash differ from manifest",
            ));
        }
        let index: Value = serde_json::from_slice(&raw_index)
            .map_err(|error| failure(format!("source tensor index is invalid: {error}")))?;
        let source_shard = index
            .pointer(&format!("/weight_map/{name}"))
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| failure(format!("source tensor index lacks {name}")))?;

        let descriptor = object(
            field(manifest, "tensors", "full Gravity manifest")?,
            "manifest.tensors",
        )?
        .get(name)
        .cloned()
        .ok_or_else(|| failure(format!("manifest lacks {name}")))?;
        if string_field(&descriptor, "name", "tensor descriptor")? != name
            || string_field(&descriptor, "dtype", "tensor descriptor")? != expected_dtype
        {
            return Err(failure(format!(
                "{name}: descriptor identity/dtype mismatch"
            )));
        }
        let shape = array(
            field(&descriptor, "shape", "tensor descriptor")?,
            "tensor descriptor.shape",
        )?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .ok_or_else(|| failure(format!("{name}: shape is malformed")))
        })
        .collect::<Result<Vec<_>, _>>()?;
        if shape != expected_shape {
            return Err(failure(format!(
                "{name}: shape {shape:?} differs from required {expected_shape:?}"
            )));
        }
        let expected_bytes = expected_shape
            .iter()
            .try_fold(1_u64, |acc, value| acc.checked_mul(*value))
            .ok_or_else(|| failure(format!("{name}: expected byte count overflow")))?;
        let tensor_bytes = u64_field(&descriptor, "bytes", "tensor descriptor")?;
        if tensor_bytes != expected_bytes {
            return Err(failure(format!(
                "{name}: tensor byte count is not source-native"
            )));
        }
        let offsets = array(
            field(&descriptor, "data_offsets", "tensor descriptor")?,
            "tensor descriptor.data_offsets",
        )?;
        if offsets.len() != 2 {
            return Err(failure(format!(
                "{name}: data_offsets must have length two"
            )));
        }
        let data_start = offsets[0]
            .as_u64()
            .ok_or_else(|| failure(format!("{name}: data_offsets[0] malformed")))?;
        let data_end = offsets[1]
            .as_u64()
            .ok_or_else(|| failure(format!("{name}: data_offsets[1] malformed")))?;
        if data_end <= data_start || data_end - data_start != tensor_bytes {
            return Err(failure(format!("{name}: data_offsets do not bind bytes")));
        }

        let windows = array(
            field(source, "source_windows", "manifest.source")?,
            "source_windows",
        )?;
        let matching_windows = windows
            .iter()
            .filter(|window| {
                window.pointer("/source/shard").and_then(Value::as_str)
                    == Some(source_shard.as_str())
            })
            .collect::<Vec<_>>();
        if matching_windows.len() != 1 {
            return Err(failure(format!(
                "{name}: source shard must bind exactly one source window"
            )));
        }
        let source_window = matching_windows[0].clone();
        let window_source = field(&source_window, "source", "source window")?;
        if string_field(window_source, "repository", "source window.source")? != REPOSITORY
            || string_field(window_source, "revision", "source window.source")? != REVISION
            || string_field(window_source, "commit_hash", "source window.source")? != REVISION
        {
            return Err(failure(format!("{name}: source window identity mismatch")));
        }
        let header_bytes = u64_field(&source_window, "header_bytes", "source window")?;

        let mut raw = Vec::with_capacity(usize::try_from(expected_bytes)?);
        let mut segments_json = Vec::new();
        let mut cursor = 0_u64;
        for segment in array(
            field(&descriptor, "segments", "tensor descriptor")?,
            "tensor segments",
        )? {
            let segment_sha = string_field(segment, "sha256", "tensor segment")?;
            let segment_path = string_field(segment, "chunk_relpath", "tensor segment")?;
            let segment_bytes = u64_field(segment, "bytes", "tensor segment")?;
            let tensor_start = u64_field(segment, "tensor_start", "tensor segment")?;
            let tensor_end = u64_field(segment, "tensor_end", "tensor segment")?;
            let source_start = u64_field(segment, "source_file_start", "tensor segment")?;
            let source_end = u64_field(segment, "source_file_end", "tensor segment")?;
            if !is_sha256(segment_sha)
                || segment_path != format!("chunks/{}/{}", &segment_sha[..2], segment_sha)
                || tensor_start != cursor
                || tensor_end <= tensor_start
                || tensor_end - tensor_start != segment_bytes
                || source_end <= source_start
                || source_end - source_start != segment_bytes
            {
                return Err(failure(format!(
                    "{name}: malformed content-addressed segment"
                )));
            }
            let expected_source_start = header_bytes
                .checked_add(data_start)
                .and_then(|offset| offset.checked_add(tensor_start))
                .ok_or_else(|| failure(format!("{name}: source offset overflow")))?;
            if source_start != expected_source_start
                || source_end != header_bytes + data_start + tensor_end
            {
                return Err(failure(format!("{name}: source segment range mismatch")));
            }
            let chunk = fs::read(checked_regular_path(
                artifact,
                segment_path,
                "content-addressed chunk",
            )?)?;
            if chunk.len() as u64 != segment_bytes || sha256(&chunk) != segment_sha {
                return Err(failure(format!(
                    "{name}: content-addressed chunk hash/size mismatch"
                )));
            }
            raw.extend_from_slice(&chunk);
            segments_json.push(json!({
                "chunk_relpath": segment_path,
                "sha256": segment_sha,
                "bytes": segment_bytes,
                "tensor_start": tensor_start,
                "tensor_end": tensor_end,
                "source_file_start": source_start,
                "source_file_end": source_end,
                "row_start": field(segment, "row_start", "tensor segment")?,
                "row_count": field(segment, "row_count", "tensor segment")?,
            }));
            cursor = tensor_end;
        }
        if cursor != tensor_bytes || raw.len() as u64 != expected_bytes {
            return Err(failure(format!(
                "{name}: chunks do not reconstruct source tensor"
            )));
        }
        Ok(BoundTensor {
            name: name.to_owned(),
            dtype: expected_dtype.to_owned(),
            shape,
            raw,
            source_shard,
            source_window,
            descriptor,
            segments: segments_json,
        })
    }

    fn bound_tensor_json(tensor: &BoundTensor) -> Value {
        json!({
            "name": tensor.name,
            "dtype": tensor.dtype,
            "shape": tensor.shape,
            "bytes": tensor.raw.len(),
            "logical_tensor_sha256": sha256(&tensor.raw),
            "source_shard": tensor.source_shard,
            "source_window": tensor.source_window,
            "descriptor_data_offsets": tensor.descriptor.get("data_offsets"),
            "segments": tensor.segments,
        })
    }

    fn deterministic_input(cols: usize) -> Vec<f32> {
        (0..cols)
            .map(|index| {
                let signed = ((index.wrapping_mul(73).wrapping_add(19)) % 257) as i32 - 128;
                signed as f32 * (1.0 / 128.0)
            })
            .collect()
    }

    fn decode_e4m3fn(bits: u8) -> SweepResult<f32> {
        let exponent = (bits >> 3) & 0x0f;
        let mantissa = bits & 0x07;
        if exponent == 0x0f && mantissa == 0x07 {
            return Err(failure("E4M3FN contains its reserved NaN encoding"));
        }
        let magnitude = if exponent == 0 {
            mantissa as f32 * 0.001_953_125_f32
        } else {
            f32::from_bits(((u32::from(exponent) + 120) << 23) | (u32::from(mantissa) << 20))
        };
        Ok(if bits & 0x80 == 0 {
            magnitude
        } else {
            -magnitude
        })
    }

    fn decode_e8m0fnu(bits: u8) -> SweepResult<f32> {
        if bits == 0xff {
            return Err(failure("E8M0FNU contains its reserved NaN encoding"));
        }
        Ok(if bits == 0 {
            f32::from_bits(0x0040_0000)
        } else {
            f32::from_bits(u32::from(bits) << 23)
        })
    }

    fn decode_e2m1fn(packed: u8, high_nibble: bool) -> f32 {
        let nibble = if high_nibble {
            (packed >> 4) & 0x0f
        } else {
            packed & 0x0f
        };
        let magnitude = match nibble & 0x07 {
            0 => 0.0,
            1 => 0.5,
            2 => 1.0,
            3 => 1.5,
            4 => 2.0,
            5 => 3.0,
            6 => 4.0,
            _ => 6.0,
        };
        if nibble & 0x08 == 0 {
            magnitude
        } else {
            -magnitude
        }
    }

    fn f32_sha256(values: &[f32]) -> String {
        sha256(bytemuck::cast_slice(values))
    }

    fn fp8_reference(
        weights: &[u8],
        scales: &[u8],
        rows: usize,
        cols: usize,
        input: &[f32],
    ) -> SweepResult<Vec<f32>> {
        if weights.len() != rows * cols || scales.len() != (rows / 128) * (cols / 128) {
            return Err(failure("FP8 source-native tensor geometry is malformed"));
        }
        let mut output = vec![0.0_f32; rows];
        for row in 0..rows {
            let mut acc = 0.0_f32;
            for col in 0..cols {
                let unit = decode_e4m3fn(weights[row * cols + col])?;
                let scale = decode_e8m0fnu(scales[(row / 128) * (cols / 128) + col / 128])?;
                acc += unit * scale * input[col];
            }
            if !acc.is_finite() {
                return Err(failure("FP8 CPU reference produced a non-finite output"));
            }
            output[row] = acc;
        }
        Ok(output)
    }

    fn fp4_reference(
        packed_weights: &[u8],
        scales: &[u8],
        rows: usize,
        packed_cols: usize,
        input: &[f32],
    ) -> SweepResult<Vec<f32>> {
        let logical_cols = packed_cols
            .checked_mul(2)
            .ok_or_else(|| failure("FP4 logical K overflow"))?;
        if packed_weights.len() != rows * packed_cols
            || scales.len() != rows * (logical_cols / 32)
            || input.len() != logical_cols
        {
            return Err(failure("FP4 source-native tensor geometry is malformed"));
        }
        let mut output = vec![0.0_f32; rows];
        for row in 0..rows {
            let mut acc = 0.0_f32;
            for col in 0..logical_cols {
                let unit = decode_e2m1fn(packed_weights[row * packed_cols + col / 2], col & 1 != 0);
                let scale = decode_e8m0fnu(scales[row * (logical_cols / 32) + col / 32])?;
                acc += unit * scale * input[col];
            }
            if !acc.is_finite() {
                return Err(failure("FP4 CPU reference produced a non-finite output"));
            }
            output[row] = acc;
        }
        Ok(output)
    }

    fn parity(cpu: &[f32], gpu: &[f32]) -> SweepResult<Value> {
        if cpu.len() != gpu.len() || cpu.is_empty() {
            return Err(failure("CPU/GPU parity vectors have incompatible length"));
        }
        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        let mut mean_abs = 0.0_f64;
        let mut worst = 0_usize;
        let mut passing = true;
        for (index, (&reference, &observed)) in cpu.iter().zip(gpu).enumerate() {
            if !reference.is_finite() || !observed.is_finite() {
                return Err(failure("CPU/GPU parity saw a non-finite output"));
            }
            let absolute = (reference - observed).abs();
            let relative = absolute / reference.abs().max(1.0e-6);
            if absolute > max_abs {
                max_abs = absolute;
                worst = index;
            }
            max_rel = max_rel.max(relative);
            mean_abs += f64::from(absolute);
            if absolute > 1.0e-4 + 1.0e-4 * reference.abs() {
                passing = false;
            }
        }
        Ok(json!({
            "status": if passing { "PASS" } else { "FAIL" },
            "comparison": "per-output abs_error <= 1e-4 + 1e-4 * abs(raw_weight_cpu_reference)",
            "max_abs_error": max_abs,
            "mean_abs_error": mean_abs / cpu.len() as f64,
            "max_relative_error": max_rel,
            "worst_output_row": worst,
            "cpu_value_at_worst_row": cpu[worst],
            "gpu_value_at_worst_row": gpu[worst],
        }))
    }

    fn percentile_us(samples: &[u64], percentile: u64) -> SweepResult<u64> {
        if samples.is_empty() || percentile == 0 || percentile > 100 {
            return Err(failure("invalid timing percentile request"));
        }
        let mut ordered = samples.to_vec();
        ordered.sort_unstable();
        let rank = (ordered.len() as u64)
            .checked_mul(percentile)
            .and_then(|value| value.checked_add(99))
            .ok_or_else(|| failure("timing percentile rank overflow"))?
            / 100;
        Ok(ordered[rank.saturating_sub(1) as usize])
    }

    fn timing_summary(samples: &[u64]) -> SweepResult<Value> {
        if samples.is_empty() {
            return Err(failure("candidate has no measured timing samples"));
        }
        Ok(json!({
            "samples_us": samples,
            "p50_us": percentile_us(samples, 50)?,
            "p95_us": percentile_us(samples, 95)?,
            "p99_us": percentile_us(samples, 99)?,
            "minimum_us": samples.iter().copied().min(),
            "maximum_us": samples.iter().copied().max(),
        }))
    }

    fn geometry_json(
        geometry: Geometry,
        logical_cols: u32,
        vector_width_logical: u32,
        packed_vector_width_bytes: Option<u32>,
    ) -> Value {
        let total_threads =
            u64::from(geometry.threads_x) * u64::from(geometry.rows_per_threadgroup);
        let logical_per_lane = (logical_cols + geometry.threads_x * vector_width_logical - 1)
            / (geometry.threads_x * vector_width_logical)
            * vector_width_logical;
        json!({
            "threads_x": geometry.threads_x,
            "rows_per_threadgroup": geometry.rows_per_threadgroup,
            "threadgroup": [geometry.threads_x, geometry.rows_per_threadgroup, 1],
            "total_threads_per_threadgroup": total_threads,
            "SIMDgroup_width": 32,
            "SIMDgroups_per_output_row": geometry.threads_x / 32,
            "K_split_lanes_per_output_row": geometry.threads_x,
            "logical_vector_width": vector_width_logical,
            "packed_vector_width_bytes": packed_vector_width_bytes,
            "logical_K_nominal_per_lane": logical_per_lane,
            // The shader declares room for four rows even on a one-row rung;
            // distinguish physical allocation from the number of partials
            // logically populated by this geometry.
            "threadgroup_memory_allocated_bytes": PARTIAL_BYTES_PER_ROW * u64::from(MAX_ROWS_PER_THREADGROUP),
            "threadgroup_partial_storage_bytes_logically_used": PARTIAL_BYTES_PER_ROW * u64::from(geometry.rows_per_threadgroup),
            "register_pressure": "not exposed by this Metal API; not inferred",
            "occupancy": "not exposed by this Metal API; not inferred",
            "pipeline_overlap": "none; one timed component dispatch then explicit completed-command-buffer wait",
        })
    }

    fn geometry_cmp(
        left: &(Geometry, u64, u64, u64),
        right: &(Geometry, u64, u64, u64),
    ) -> Ordering {
        (
            left.1,
            left.2,
            left.3,
            left.0.threads_x,
            left.0.rows_per_threadgroup,
        )
            .cmp(&(
                right.1,
                right.2,
                right.3,
                right.0.threads_x,
                right.0.rows_per_threadgroup,
            ))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_parallel_candidate(
        artifact: &Path,
        manifest: &Value,
        manifest_file_sha256: &str,
        component: &str,
        kernel: &str,
        role: &str,
        weights: &BoundTensor,
        scales: &BoundTensor,
        input: &[f32],
        cpu_output: &[f32],
        rows: u32,
        kernel_cols: u32,
        scale_cols: u32,
        logical_cols: u32,
        logical_vector_width: u32,
        packed_vector_width_bytes: Option<u32>,
        warmups: usize,
        trials: usize,
    ) -> SweepResult<Value> {
        if warmups == 0 || trials == 0 || rows == 0 || kernel_cols == 0 || scale_cols == 0 {
            return Err(failure(
                "parallel component sweep has invalid zero geometry",
            ));
        }
        if logical_vector_width == 0 || logical_cols % logical_vector_width != 0 {
            return Err(failure(
                "parallel component vector width does not divide logical K",
            ));
        }
        if GEOMETRIES.iter().any(|geometry| {
            geometry.threads_x == 0
                || geometry.threads_x % 32 != 0
                || geometry.rows_per_threadgroup == 0
                || geometry.rows_per_threadgroup > MAX_ROWS_PER_THREADGROUP
        }) {
            return Err(failure("parallel component geometry ladder is malformed"));
        }
        let manifest_seal = string_field(manifest, "seal_sha256", "full Gravity manifest")?;
        let run_nonce = sha256_join(&[
            manifest_seal,
            &sha256(&weights.raw),
            &sha256(&scales.raw),
            &f32_sha256(input),
            kernel,
            "raw_weight_simdgroup_splitk_v1",
        ]);
        let context = MetalContext::new_with_trace(true)?;
        let device = context.device_name();
        let pipeline = context.pipeline(kernel)?;
        let pipeline_thread_execution_width = pipeline.thread_execution_width() as u64;
        let pipeline_max_total_threads_per_threadgroup =
            pipeline.max_total_threads_per_threadgroup() as u64;
        drop(pipeline);
        let weight_buffer = context.new_buffer_with_bytes_checked(&weights.raw)?;
        let scale_buffer = context.new_buffer_with_bytes_checked(&scales.raw)?;
        let input_buffer = context.new_buffer_with_bytes_checked(bytemuck::cast_slice(input))?;
        let output_buffer =
            context.new_buffer_checked(rows as usize * std::mem::size_of::<f32>())?;
        let logical_bytes_read_per_dispatch = weights.raw.len() as u64
            + scales.raw.len() as u64
            + (input.len() * std::mem::size_of::<f32>()) as u64;
        let logical_bytes_written_per_dispatch = rows as u64 * std::mem::size_of::<f32>() as u64;
        let interval_id = sha256_join(&[&run_nonce, kernel, "parallel_geometry_sweep"]);
        let trace_identity = PhysicalTraceIdentity::new(
            interval_id.clone(),
            run_nonce.clone(),
            "raw_weight_parallel_geometry_sweep".to_owned(),
            role.to_owned(),
            Some(1),
            0,
        )?;
        let physical_trace = PhysicalTraceGuard::begin(trace_identity)?;
        let mut candidates = Vec::with_capacity(GEOMETRIES.len());
        let mut stable = Vec::<(Geometry, u64, u64, u64)>::new();
        let mut observed_dispatches = 0_u64;
        let mut observed_waits = 0_u64;
        let dispatches_per_supported_geometry = warmups + trials;

        for &geometry in GEOMETRIES {
            let total_threads =
                u64::from(geometry.threads_x) * u64::from(geometry.rows_per_threadgroup);
            let geometry_value = geometry_json(
                geometry,
                logical_cols,
                logical_vector_width,
                packed_vector_width_bytes,
            );
            if total_threads > pipeline_max_total_threads_per_threadgroup {
                candidates.push(json!({
                    "geometry": geometry_value,
                    "status": "UNSUPPORTED_MAX_TOTAL_THREADS",
                    "not_dispatched_reason": "requested threadgroup exceeds this compiled pipeline's maximum",
                    "pipeline_max_total_threads_per_threadgroup": pipeline_max_total_threads_per_threadgroup,
                    "gpu_dispatches": 0,
                    "command_buffers": 0,
                    "compute_encoders": 0,
                    "cpu_visible_waits": 0,
                    "empty_command_buffers": 0,
                    "fallback": false,
                    "fallback_reason": null,
                }));
                continue;
            }

            let mut gpu_us = Vec::with_capacity(trials);
            let mut encode_us = Vec::with_capacity(trials);
            let mut submit_us = Vec::with_capacity(trials);
            let mut wait_us = Vec::with_capacity(trials);
            let mut host_wall_us = Vec::with_capacity(trials);
            for dispatch_index in 0..dispatches_per_supported_geometry {
                let timing = context.dispatch_threads_timed(
                    kernel,
                    (geometry.threads_x, rows, 1),
                    (geometry.threads_x, geometry.rows_per_threadgroup, 1),
                    |encoder| {
                        encoder.set_buffer(0, Some(&weight_buffer), 0);
                        encoder.set_buffer(1, Some(&scale_buffer), 0);
                        encoder.set_buffer(2, Some(&input_buffer), 0);
                        encoder.set_buffer(3, Some(&output_buffer), 0);
                        encoder.set_bytes(
                            4,
                            std::mem::size_of::<u32>() as u64,
                            &rows as *const u32 as *const _,
                        );
                        encoder.set_bytes(
                            5,
                            std::mem::size_of::<u32>() as u64,
                            &kernel_cols as *const u32 as *const _,
                        );
                        encoder.set_bytes(
                            6,
                            std::mem::size_of::<u32>() as u64,
                            &scale_cols as *const u32 as *const _,
                        );
                        encoder.set_bytes(
                            7,
                            std::mem::size_of::<u32>() as u64,
                            &geometry.threads_x as *const u32 as *const _,
                        );
                    },
                )?;
                if timing.compute_dispatches != 1
                    || timing.command_buffers != 1
                    || timing.compute_encoders != 1
                {
                    return Err(failure(
                        "parallel component candidate did not use one command buffer/encoder/dispatch",
                    ));
                }
                observed_dispatches += 1;
                observed_waits += 1;
                if dispatch_index >= warmups {
                    gpu_us.push(
                        timing
                            .gpu_duration_us
                            .filter(|duration| *duration > 0)
                            .ok_or_else(|| {
                                failure(
                                    "parallel component measured dispatch has no positive GPU timestamp",
                                )
                            })?,
                    );
                    encode_us.push(timing.encode_us);
                    submit_us.push(timing.submit_us);
                    wait_us.push(timing.wait_us);
                    host_wall_us.push(timing.host_wall_us);
                }
            }
            let gpu_output = unsafe {
                std::slice::from_raw_parts(output_buffer.contents() as *const f32, rows as usize)
                    .to_vec()
            };
            let comparison = parity(cpu_output, &gpu_output)?;
            let gpu_summary = timing_summary(&gpu_us)?;
            let p50 = u64_field(&gpu_summary, "p50_us", "candidate GPU timing")?;
            let p95 = u64_field(&gpu_summary, "p95_us", "candidate GPU timing")?;
            let p99 = u64_field(&gpu_summary, "p99_us", "candidate GPU timing")?;
            let parity_pass =
                string_field(&comparison, "status", "candidate CPU parity")? == "PASS";
            if parity_pass {
                stable.push((geometry, p50, p95, p99));
            }
            let logical_read_gib_s = logical_bytes_read_per_dispatch as f64 * 1_000_000.0
                / p50 as f64
                / (1024.0 * 1024.0 * 1024.0);
            candidates.push(json!({
                "geometry": geometry_value,
                "status": if parity_pass { "PASS_GPU_TIMESTAMPED_RAW_WEIGHT_CPU_PARITY" } else { "FAIL_RAW_WEIGHT_CPU_PARITY" },
                "raw_weight_component_only": true,
                "warmup_dispatches": warmups,
                "measured_gpu_timestamped_dispatches": trials,
                "gpu_duration": gpu_summary,
                "host_encode_duration": timing_summary(&encode_us)?,
                "host_submit_duration": timing_summary(&submit_us)?,
                "host_wait_duration": timing_summary(&wait_us)?,
                "host_wall_duration": timing_summary(&host_wall_us)?,
                "raw_weight_cpu_parity": comparison,
                "gpu_output_sha256_f32_le": f32_sha256(&gpu_output),
                "logical_bytes_read_per_dispatch": logical_bytes_read_per_dispatch,
                "logical_bytes_written_per_dispatch": logical_bytes_written_per_dispatch,
                "fp_operations_estimate_per_dispatch": u64::from(rows) * u64::from(logical_cols) * 3,
                "integer_or_bit_operations": "not independently counter-exposed; source byte decode is in the candidate kernel",
                "logical_read_bandwidth_gib_s_at_gpu_p50": logical_read_gib_s,
                "gpu_dispatches": dispatches_per_supported_geometry,
                "command_buffers": dispatches_per_supported_geometry,
                "compute_encoders": dispatches_per_supported_geometry,
                "cpu_visible_waits": dispatches_per_supported_geometry,
                "empty_command_buffers": 0,
                "fallback": false,
                "fallback_reason": null,
            }));
        }
        let physical_counts = physical_trace.counts();
        drop(physical_trace);
        let (buffers_created, bytes_allocated, commits) = context.drain_stats();
        let trace_samples = context.drain_trace();
        if stable.is_empty() {
            return Err(failure(
                "no parallel component geometry passed raw-weight CPU parity",
            ));
        }
        if physical_counts.command_count != observed_dispatches
            || physical_counts.encoder_count != observed_dispatches
            || commits as u64 != observed_dispatches
            || trace_samples.len() as u64 != observed_dispatches
        {
            return Err(failure(
                "parallel component physical command/encoder/dispatch accounting mismatch",
            ));
        }
        let deepest = stable
            .iter()
            .max_by_key(|(geometry, _, _, _)| {
                (
                    u64::from(geometry.threads_x) * u64::from(geometry.rows_per_threadgroup),
                    geometry.threads_x,
                    geometry.rows_per_threadgroup,
                )
            })
            .copied()
            .ok_or_else(|| failure("parallel component sweep lacks deepest stable rung"))?;
        let winner = stable
            .iter()
            .min_by(|left, right| geometry_cmp(left, right))
            .copied()
            .ok_or_else(|| failure("parallel component sweep lacks winner"))?;

        Ok(json!({
            "component": component,
            "kernel": kernel,
            "status": "PASS_RAW_WEIGHT_SIMDGROUP_SPLITK_COMPONENT_SWEEP_NOT_SOURCE_FORWARD_OR_RUNTIME",
            "scope": {
                "source_hash_bound_full_gravity_artifact": true,
                "raw_weight_component_cpu_parity": true,
                "activation_quantization_not_executed": true,
                "not_source_forward_parity": true,
                "not_a_full_model_load": true,
                "not_a_full_token_or_generation": true,
                "not_a_route_or_MoE_execution_claim": true,
                "not_a_HCLI_measurement": true,
                "not_a_BASE_TRUE_TPS_measurement": true,
                "not_a_runtime_kernel_promotion": true,
            },
            "artifact": {
                "path": artifact,
                "manifest_file_sha256": manifest_file_sha256,
                "manifest_seal_sha256": manifest_seal,
                "manifest_seal_verified": true,
                "full_stream_schema": string_field(manifest, "schema", "full Gravity manifest")?,
                "full_stream_status": string_field(manifest, "status", "full Gravity manifest")?,
            },
            "source": {
                "repository": REPOSITORY,
                "revision": REVISION,
                "weight": bound_tensor_json(weights),
                "scale": bound_tensor_json(scales),
            },
            "input": {
                "kind": "deterministic_exact_binary_rational_vector_v1",
                "length": input.len(),
                "sha256_f32_le": f32_sha256(input),
            },
            "raw_weight_cpu_reference": {
                "accumulation": "row-major f32 product_then_add",
                "output_sha256_f32_le": f32_sha256(cpu_output),
            },
            "metal": {
                "device": device,
                "pipeline_thread_execution_width": pipeline_thread_execution_width,
                "pipeline_max_total_threads_per_threadgroup": pipeline_max_total_threads_per_threadgroup,
                "buffers_created": buffers_created,
                "bytes_allocated": bytes_allocated,
                "physical_trace_command_buffers": physical_counts.command_count,
                "physical_trace_compute_encoders": physical_counts.encoder_count,
                "observed_gpu_dispatches": observed_dispatches,
                "observed_cpu_visible_waits": observed_waits,
                "commits": commits,
                "fallback": false,
                "fallback_count": 0,
            },
            "sweep": {
                "requested_geometries": GEOMETRIES.iter().map(|geometry| geometry_json(*geometry, logical_cols, logical_vector_width, packed_vector_width_bytes)).collect::<Vec<_>>(),
                "candidates": candidates,
                "deepest_stable_threadgroup_rung": {
                    "geometry": geometry_json(deepest.0, logical_cols, logical_vector_width, packed_vector_width_bytes),
                    "gpu_p50_us": deepest.1,
                    "gpu_p95_us": deepest.2,
                    "gpu_p99_us": deepest.3,
                },
                "winner": {
                    "selection": "minimum GPU p50, then p95, then p99, then lower threads_x, then lower rows_per_threadgroup",
                    "geometry": geometry_json(winner.0, logical_cols, logical_vector_width, packed_vector_width_bytes),
                    "gpu_p50_us": winner.1,
                    "gpu_p95_us": winner.2,
                    "gpu_p99_us": winner.3,
                },
            },
            "physical_trace": {
                "interval_id": interval_id,
                "run_nonce": run_nonce,
                "phase": "raw_weight_parallel_geometry_sweep",
                "role": role,
            },
            "next_boundary": "This raw-weight candidate leaves model.py act_quant unexecuted. It cannot establish source-forward parity, register a V4 runtime, execute a token, or satisfy BASE_TRUE_TPS.",
        }))
    }

    fn pointer_string<'a>(value: &'a Value, pointer: &str, label: &str) -> SweepResult<&'a str> {
        value
            .pointer(pointer)
            .and_then(Value::as_str)
            .ok_or_else(|| failure(format!("{label} missing string at {pointer}")))
    }

    fn pointer_u64(value: &Value, pointer: &str, label: &str) -> SweepResult<u64> {
        value
            .pointer(pointer)
            .and_then(Value::as_u64)
            .ok_or_else(|| failure(format!("{label} missing integer at {pointer}")))
    }

    fn component_comparison(
        authority: &Value,
        candidate: &Value,
        label: &str,
    ) -> SweepResult<Value> {
        let authority_input = pointer_string(authority, "/input/sha256_f32_le", label)?;
        let candidate_input = pointer_string(candidate, "/input/sha256_f32_le", label)?;
        let authority_cpu =
            pointer_string(authority, "/cpu_reference/output_sha256_f32_le", label)?;
        let candidate_cpu = pointer_string(
            candidate,
            "/raw_weight_cpu_reference/output_sha256_f32_le",
            label,
        )?;
        if authority_input != candidate_input || authority_cpu != candidate_cpu {
            return Err(failure(format!(
                "{label}: authority and SIMDgroup candidate did not use the same input/CPU reference"
            )));
        }
        let authority_p50 = pointer_u64(authority, "/ladder/winner/gpu_p50_us", label)?;
        let candidate_p50 = pointer_u64(candidate, "/sweep/winner/gpu_p50_us", label)?;
        let authority_p95 = pointer_u64(authority, "/ladder/winner/gpu_p95_us", label)?;
        let candidate_p95 = pointer_u64(candidate, "/sweep/winner/gpu_p95_us", label)?;
        let authority_p99 = pointer_u64(authority, "/ladder/winner/gpu_p99_us", label)?;
        let candidate_p99 = pointer_u64(candidate, "/sweep/winner/gpu_p99_us", label)?;
        let p50_relation = if candidate_p50 < authority_p50 {
            "CANDIDATE_GPU_P50_WIN_NOT_PROMOTED"
        } else if candidate_p50 > authority_p50 {
            "CANDIDATE_GPU_P50_LOSS"
        } else {
            "GPU_P50_TIE_NOT_PROMOTED"
        };
        Ok(json!({
            "same_raw_weight_input_and_cpu_reference": true,
            "authority_serial_winner_gpu_p50_us": authority_p50,
            "candidate_parallel_winner_gpu_p50_us": candidate_p50,
            "authority_serial_winner_gpu_p95_us": authority_p95,
            "candidate_parallel_winner_gpu_p95_us": candidate_p95,
            "authority_serial_winner_gpu_p99_us": authority_p99,
            "candidate_parallel_winner_gpu_p99_us": candidate_p99,
            "p50_speedup_authority_divided_by_candidate": authority_p50 as f64 / candidate_p50 as f64,
            "p50_outcome": p50_relation,
            "promotion": "NOT_PROMOTED; optional raw-weight component candidate only",
        }))
    }

    fn sealed_receipt(mut receipt: Value) -> SweepResult<(Value, String)> {
        if !receipt.is_object() || receipt.get("seal_sha256").is_some() {
            return Err(failure("sweep receipt cannot be sealed"));
        }
        let seal = sha256(&serde_json::to_vec(&receipt)?);
        receipt
            .as_object_mut()
            .expect("receipt object was checked")
            .insert("seal_sha256".to_owned(), Value::String(seal.clone()));
        Ok((receipt, seal))
    }

    fn write_new_receipt(path: &Path, receipt: &Value) -> SweepResult<()> {
        if path.exists() {
            return Err(failure(format!(
                "refusing to overwrite existing SIMDgroup sweep receipt {}",
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
            ".{name}.{}.simdgroup-sweep.tmp",
            std::process::id()
        ));
        let bytes = serde_json::to_vec_pretty(receipt)?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| failure(format!("cannot create sweep temporary: {error}")))?;
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

    pub fn run() -> SweepResult<()> {
        let args = parse_args()?;
        if args.out.exists() {
            return Err(failure(format!(
                "refusing to overwrite existing SIMDgroup sweep receipt {}",
                args.out.display()
            )));
        }

        // Before: serial source-native raw-weight authorities. Their own
        // artifact/chunk verification is rerun; their output is embedded in
        // the sealed comparison rather than being treated as a stale receipt.
        let fp8_authority = super::fp8_authority::macos::sweep_component(
            &args.artifact,
            args.warmups,
            args.trials,
            AUTHORITY_THREADGROUPS,
        )?;
        let fp4_authority = super::fp4_authority::macos::sweep_component(
            &args.artifact,
            args.warmups,
            args.trials,
            AUTHORITY_THREADGROUPS,
        )?;

        let (artifact, manifest, manifest_file_sha256) = verify_full_manifest(&args.artifact)?;
        let fp8_weights = bind_tensor(
            &artifact,
            &manifest,
            "layers.0.attn.wq_a.weight",
            "F8_E4M3",
            &[1024, 4096],
        )?;
        let fp8_scales = bind_tensor(
            &artifact,
            &manifest,
            "layers.0.attn.wq_a.scale",
            "F8_E8M0",
            &[8, 32],
        )?;
        let fp8_input = deterministic_input(4096);
        let fp8_cpu = fp8_reference(&fp8_weights.raw, &fp8_scales.raw, 1024, 4096, &fp8_input)?;
        let fp8_candidate = run_parallel_candidate(
            &artifact,
            &manifest,
            &manifest_file_sha256,
            "layers.0.attn.wq_a source-native FP8 control linear",
            FP8_KERNEL,
            "fp8_control_raw_weight_matvec",
            &fp8_weights,
            &fp8_scales,
            &fp8_input,
            &fp8_cpu,
            1024,
            4096,
            32,
            4096,
            4,
            None,
            args.warmups,
            args.trials,
        )?;

        let fp4_weights = bind_tensor(
            &artifact,
            &manifest,
            "layers.0.ffn.experts.0.w1.weight",
            "I8",
            &[2048, 2048],
        )?;
        let fp4_scales = bind_tensor(
            &artifact,
            &manifest,
            "layers.0.ffn.experts.0.w1.scale",
            "F8_E8M0",
            &[2048, 128],
        )?;
        let fp4_input = deterministic_input(4096);
        let fp4_cpu = fp4_reference(&fp4_weights.raw, &fp4_scales.raw, 2048, 2048, &fp4_input)?;
        let fp4_candidate = run_parallel_candidate(
            &artifact,
            &manifest,
            &manifest_file_sha256,
            "layers.0.ffn.experts.0.w1 source-native FP4 routed-expert gate linear",
            FP4_KERNEL,
            "fp4_routed_expert_raw_weight_matvec",
            &fp4_weights,
            &fp4_scales,
            &fp4_input,
            &fp4_cpu,
            2048,
            2048,
            128,
            4096,
            8,
            Some(4),
            args.warmups,
            args.trials,
        )?;

        let fp8_authority_device =
            pointer_string(&fp8_authority, "/metal/device", "FP8 authority")?;
        let fp4_authority_device =
            pointer_string(&fp4_authority, "/metal/device", "FP4 authority")?;
        let fp8_candidate_device =
            pointer_string(&fp8_candidate, "/metal/device", "FP8 candidate")?;
        let fp4_candidate_device =
            pointer_string(&fp4_candidate, "/metal/device", "FP4 candidate")?;
        if fp8_authority_device != fp4_authority_device
            || fp8_authority_device != fp8_candidate_device
            || fp8_authority_device != fp4_candidate_device
        {
            return Err(failure(
                "serial and candidate sweeps used different Metal devices",
            ));
        }
        if !fp8_authority_device.contains("M3") {
            return Err(failure(format!(
                "this bounded campaign requires an Apple M3 Metal run, found {fp8_authority_device:?}"
            )));
        }
        let manifest_seal = string_field(&manifest, "seal_sha256", "full Gravity manifest")?;
        for (label, result) in [
            ("FP8 authority", &fp8_authority),
            ("FP4 authority", &fp4_authority),
            ("FP8 candidate", &fp8_candidate),
            ("FP4 candidate", &fp4_candidate),
        ] {
            if pointer_string(result, "/artifact/manifest_seal_sha256", label)? != manifest_seal {
                return Err(failure(format!(
                    "{label} bound a different full Gravity manifest"
                )));
            }
        }
        let fp8_comparison = component_comparison(&fp8_authority, &fp8_candidate, "FP8 control")?;
        let fp4_comparison =
            component_comparison(&fp4_authority, &fp4_candidate, "FP4 routed expert")?;
        let aggregate_dispatches = pointer_u64(
            &fp8_authority,
            "/metal/observed_gpu_dispatches",
            "FP8 authority",
        )? + pointer_u64(
            &fp4_authority,
            "/metal/observed_gpu_dispatches",
            "FP4 authority",
        )? + pointer_u64(
            &fp8_candidate,
            "/metal/observed_gpu_dispatches",
            "FP8 candidate",
        )? + pointer_u64(
            &fp4_candidate,
            "/metal/observed_gpu_dispatches",
            "FP4 candidate",
        )?;
        let aggregate_waits = pointer_u64(
            &fp8_authority,
            "/metal/observed_cpu_visible_waits",
            "FP8 authority",
        )? + pointer_u64(
            &fp4_authority,
            "/metal/observed_cpu_visible_waits",
            "FP4 authority",
        )? + pointer_u64(
            &fp8_candidate,
            "/metal/observed_cpu_visible_waits",
            "FP8 candidate",
        )? + pointer_u64(
            &fp4_candidate,
            "/metal/observed_cpu_visible_waits",
            "FP4 candidate",
        )?;
        let aggregate_command_buffers = pointer_u64(
            &fp8_authority,
            "/metal/physical_trace_command_buffers",
            "FP8 authority",
        )? + pointer_u64(
            &fp4_authority,
            "/metal/physical_trace_command_buffers",
            "FP4 authority",
        )? + pointer_u64(
            &fp8_candidate,
            "/metal/physical_trace_command_buffers",
            "FP8 candidate",
        )? + pointer_u64(
            &fp4_candidate,
            "/metal/physical_trace_command_buffers",
            "FP4 candidate",
        )?;
        if aggregate_dispatches == 0
            || aggregate_dispatches != aggregate_waits
            || aggregate_dispatches != aggregate_command_buffers
        {
            return Err(failure(
                "aggregate command-buffer/dispatch/wait accounting did not reconcile",
            ));
        }

        let unsigned = json!({
            "schema": "hawking.gravity.deepseek_v4.raw_weight_simdgroup_splitk_sweep.v1",
            "status": "PASS_REAL_M3_METAL_RAW_WEIGHT_SIMDGROUP_SPLITK_SWEEP_NOT_SOURCE_FORWARD_OR_RUNTIME",
            "scope": {
                "same_sealed_full_gravity_artifact_before_and_after": true,
                "same_deterministic_input_and_raw_weight_cpu_reference_before_and_after": true,
                "raw_weight_component_only": true,
                "model_py_activation_quantization_not_executed": true,
                "not_source_forward_parity": true,
                "not_a_full_model_load": true,
                "not_a_full_43_layer_runtime_adapter": true,
                "not_a_token_or_generation": true,
                "not_a_route_or_MoE_execution_claim": true,
                "not_a_HCLI_measurement": true,
                "not_a_BASE_TRUE_TPS_measurement": true,
                "not_a_runtime_kernel_promotion": true,
            },
            "reproduction": {
                "example": "cargo run --release -p hawking-core --example gravity_deepseek_v4_simdgroup_splitk_sweep",
                "warmup_dispatches_per_supported_geometry": args.warmups,
                "measured_gpu_timestamped_dispatches_per_supported_geometry": args.trials,
                "authority_threadgroup_ladder": AUTHORITY_THREADGROUPS,
                "parallel_geometry_ladder": GEOMETRIES.iter().map(|geometry| json!({"threads_x": geometry.threads_x, "rows_per_threadgroup": geometry.rows_per_threadgroup})).collect::<Vec<_>>(),
                "timing_authority": "completed MTLCommandBuffer GPUStartTime/GPUEndTime only; host encode/submit/wait/wall are reported separately",
                "parity_authority": "same source-native raw-byte CPU decoders and deterministic FP32 input; no activation-quantized source-forward claim",
            },
            "artifact_binding": {
                "path": artifact,
                "manifest_file_sha256": manifest_file_sha256,
                "manifest_seal_sha256": manifest_seal,
                "manifest_seal_verified": true,
                "source_parent_persisted": false,
            },
            "metal": {
                "device": fp8_authority_device,
                "aggregate_real_gpu_dispatches": aggregate_dispatches,
                "aggregate_command_buffers": aggregate_command_buffers,
                "aggregate_cpu_visible_waits": aggregate_waits,
                "empty_command_buffers": 0,
                "fallback": false,
                "fallback_count": 0,
                "accounting_reconciled": true,
            },
            "before_after": {
                "fp8_control": fp8_comparison,
                "fp4_routed_expert": fp4_comparison,
            },
            "serial_authority_before": {
                "fp8_control": fp8_authority,
                "fp4_routed_expert": fp4_authority,
            },
            "optional_parallel_candidates_after": {
                "fp8_control": fp8_candidate,
                "fp4_routed_expert": fp4_candidate,
            },
            "next_boundary": "The next faithful unit is act_quant -> source-native FP8 projection. This receipt only proves or rejects raw-weight component candidates; it leaves activation quantization, source-forward parity, full V4 runtime integration, routing, state, HCLI, and BASE_TRUE_TPS unmeasured.",
        });
        let (receipt, seal) = sealed_receipt(unsigned)?;
        write_new_receipt(&args.out, &receipt)?;
        println!(
            "{}",
            serde_json::to_string(&json!({
                "status": "PASS_REAL_M3_METAL_RAW_WEIGHT_SIMDGROUP_SPLITK_SWEEP_NOT_SOURCE_FORWARD_OR_RUNTIME",
                "receipt": args.out,
                "seal_sha256": seal,
                "aggregate_real_gpu_dispatches": aggregate_dispatches,
                "fp8_p50_outcome": fp8_comparison.get("p50_outcome"),
                "fp4_p50_outcome": fp4_comparison.get("p50_outcome"),
            }))?
        );
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}
