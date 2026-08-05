//! Source-hash-bound Metal parity probe for one DeepSeek-V4-Flash FP4 expert.
//!
//! This binary is deliberately a narrow component authority.  It does *not*
//! load a model, execute a token, register a DeepSeek-V4 runtime, or make a
//! TPS/HCLI claim.  It binds the sealed full-stream artifact to one real
//! routed expert gate linear and proves its packed source-native FP4 matvec on
//! Metal against an exact CPU byte decoder:
//!
//! ```text
//! layers.0.ffn.experts.0.w1.weight [2048, 2048] I8 = E2M1FN x2 packed
//! layers.0.ffn.experts.0.w1.scale  [2048,  128] F8_E8M0 = per logical K=32
//! ```
//!
//! ```sh
//! cargo run --release -p hawking-core --example gravity_deepseek_v4_fp4_metal_probe -- \
//!   --artifact /absolute/path/to/full-43-layer-stream.gravity \
//!   --out /absolute/path/to/DSV4F_FP4_METAL_PROBE.json
//! ```

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("gravity_deepseek_v4_fp4_metal_probe requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
pub mod macos {
    use hawking_core::metal::{MetalContext, PhysicalTraceGuard, PhysicalTraceIdentity};
    use serde::Deserialize;
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::collections::BTreeMap;
    use std::error::Error;
    use std::fs::{self, File, OpenOptions};
    use std::io::Write;
    use std::path::{Component, Path, PathBuf};

    const ARTIFACT_SCHEMA: &str = "hawking.gravity.deepseek_v4.full_stream.v1";
    const ARTIFACT_STATUS: &str = "FULL_MODEL_STREAMED_SEALED_NOT_RUNTIME_READY";
    const REPOSITORY: &str = "deepseek-ai/DeepSeek-V4-Flash";
    const REVISION: &str = "60d8d70770c6776ff598c94bb586a859a38244f1";
    const WEIGHT_NAME: &str = "layers.0.ffn.experts.0.w1.weight";
    const SCALE_NAME: &str = "layers.0.ffn.experts.0.w1.scale";
    const KERNEL_NAME: &str = "deepseek_v4_fp4_e2m1fn_x2_e8m0_matvec_authority";
    const ROWS: usize = 2048;
    const PACKED_COLS: usize = 2048;
    const LOGICAL_COLS: usize = PACKED_COLS * 2;
    const FP4_BLOCK: usize = 32;
    const SCALE_COLS: usize = LOGICAL_COLS / FP4_BLOCK;
    const ABS_TOLERANCE: f32 = 1.0e-4;
    const REL_TOLERANCE: f32 = 1.0e-4;

    type ProbeResult<T> = Result<T, Box<dyn Error>>;

    fn failure(message: impl Into<String>) -> Box<dyn Error> {
        std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
    }

    #[derive(Debug, Deserialize)]
    struct FullManifest {
        schema: String,
        status: String,
        seal_sha256: String,
        source: Source,
        representation_and_kernel_grammar: KernelGrammar,
        tensors: BTreeMap<String, TensorDescriptor>,
    }

    #[derive(Debug, Deserialize)]
    struct Source {
        repository: String,
        revision: String,
        metadata_assets: BTreeMap<String, MetadataAsset>,
        source_windows: Vec<SourceWindow>,
        source_parent_persisted: bool,
    }

    #[derive(Debug, Deserialize)]
    struct MetadataAsset {
        path: String,
        bytes: u64,
        sha256: String,
    }

    #[derive(Debug, Deserialize)]
    struct SourceWindow {
        header_bytes: u64,
        streamed_full_file_sha256: String,
        source: SourceFile,
    }

    #[derive(Debug, Deserialize)]
    struct SourceFile {
        repository: String,
        revision: String,
        commit_hash: String,
        shard: String,
        etag_sha256: String,
        xet_file_hash: String,
    }

    #[derive(Debug, Deserialize)]
    struct KernelGrammar {
        fp4: String,
        official_convert_py_sha256: String,
        official_kernel_py_sha256: String,
    }

    #[derive(Debug, Deserialize)]
    struct TensorDescriptor {
        name: String,
        dtype: String,
        shape: Vec<u64>,
        data_offsets: Vec<u64>,
        bytes: u64,
        segments: Vec<Segment>,
    }

    #[derive(Debug, Deserialize)]
    struct Segment {
        bytes: u64,
        chunk_relpath: String,
        sha256: String,
        source_file_start: u64,
        source_file_end: u64,
        tensor_start: u64,
        tensor_end: u64,
        row_start: u64,
        row_count: u64,
    }

    #[derive(Debug, Deserialize)]
    struct SourceIndex {
        weight_map: BTreeMap<String, String>,
    }

    struct Args {
        artifact: PathBuf,
        out: PathBuf,
    }

    struct BoundTensor<'a> {
        descriptor: &'a TensorDescriptor,
        source_shard: String,
        source_window: &'a SourceWindow,
        raw: Vec<u8>,
        segments_json: Vec<Value>,
    }

    fn parse_args() -> ProbeResult<Args> {
        let mut artifact = None::<PathBuf>;
        let mut out = None::<PathBuf>;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--artifact" => artifact = args.next().map(PathBuf::from),
                "--out" => out = args.next().map(PathBuf::from),
                "--help" | "-h" => {
                    println!(
                        "usage: gravity_deepseek_v4_fp4_metal_probe --artifact <absolute full Gravity dir> --out <absolute receipt.json>"
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

    fn checked_regular_path(root: &Path, relative: &str, label: &str) -> ProbeResult<PathBuf> {
        let rel = Path::new(relative);
        if rel.is_absolute()
            || rel.components().any(|component| {
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
        let path = root.join(rel);
        let meta = fs::symlink_metadata(&path).map_err(|error| {
            failure(format!(
                "cannot inspect {label} {}: {error}",
                path.display()
            ))
        })?;
        if meta.file_type().is_symlink() || !meta.file_type().is_file() {
            return Err(failure(format!(
                "{label} must be a regular non-symlink file: {}",
                path.display()
            )));
        }
        Ok(path)
    }

    fn verify_manifest_seal(raw: &[u8]) -> ProbeResult<Value> {
        let mut value: Value = serde_json::from_slice(raw)
            .map_err(|error| failure(format!("manifest is not JSON: {error}")))?;
        let recorded = {
            let object = value
                .as_object_mut()
                .ok_or_else(|| failure("manifest root must be an object"))?;
            object
                .remove("seal_sha256")
                .and_then(|value| value.as_str().map(str::to_owned))
                .ok_or_else(|| failure("manifest lacks a string seal_sha256"))?
        };
        if !is_sha256(&recorded) {
            return Err(failure("manifest seal_sha256 is not lowercase SHA-256"));
        }
        let observed = sha256(&serde_json::to_vec(&value)?);
        if observed != recorded {
            return Err(failure(format!(
                "manifest seal mismatch: recorded={recorded} observed={observed}"
            )));
        }
        value
            .as_object_mut()
            .expect("manifest object was already checked")
            .insert("seal_sha256".to_owned(), Value::String(recorded));
        Ok(value)
    }

    fn load_index(artifact: &Path, source: &Source) -> ProbeResult<SourceIndex> {
        let asset = source
            .metadata_assets
            .get("model.safetensors.index.json")
            .ok_or_else(|| failure("manifest lacks model.safetensors.index.json source asset"))?;
        if asset.path != "model.safetensors.index.json" || !is_sha256(&asset.sha256) {
            return Err(failure("manifest source index asset binding is malformed"));
        }
        let path = checked_regular_path(
            &artifact.join("metadata"),
            &asset.path,
            "model source tensor index",
        )?;
        let raw = fs::read(&path)?;
        if raw.len() as u64 != asset.bytes || sha256(&raw) != asset.sha256 {
            return Err(failure(
                "model source tensor index bytes/hash differ from manifest",
            ));
        }
        serde_json::from_slice(&raw)
            .map_err(|error| failure(format!("model source tensor index is invalid: {error}")))
    }

    fn source_window<'a>(manifest: &'a FullManifest, shard: &str) -> ProbeResult<&'a SourceWindow> {
        let matches: Vec<&SourceWindow> = manifest
            .source
            .source_windows
            .iter()
            .filter(|window| window.source.shard == shard)
            .collect();
        if matches.len() != 1 {
            return Err(failure(format!(
                "expected exactly one sealed source window for {shard}, found {}",
                matches.len()
            )));
        }
        let window = matches[0];
        if window.source.repository != manifest.source.repository
            || window.source.revision != manifest.source.revision
            || window.source.commit_hash != manifest.source.revision
            || !is_sha256(&window.streamed_full_file_sha256)
            || !is_sha256(&window.source.etag_sha256)
            || !is_sha256(&window.source.xet_file_hash)
            || window.streamed_full_file_sha256 != window.source.etag_sha256
        {
            return Err(failure(format!(
                "source window {shard} does not bind the pinned source identity"
            )));
        }
        Ok(window)
    }

    fn bind_tensor<'a>(
        artifact: &Path,
        manifest: &'a FullManifest,
        index: &SourceIndex,
        name: &str,
        expected_dtype: &str,
        expected_shape: &[usize],
    ) -> ProbeResult<BoundTensor<'a>> {
        let descriptor = manifest
            .tensors
            .get(name)
            .ok_or_else(|| failure(format!("manifest lacks required tensor {name}")))?;
        if descriptor.name != name || descriptor.dtype != expected_dtype {
            return Err(failure(format!(
                "{name}: descriptor identity/dtype differs from required {expected_dtype}"
            )));
        }
        let shape: Vec<usize> = descriptor
            .shape
            .iter()
            .map(|value| {
                usize::try_from(*value).map_err(|_| failure(format!("{name}: shape exceeds usize")))
            })
            .collect::<Result<_, _>>()?;
        if shape != expected_shape {
            return Err(failure(format!(
                "{name}: shape {:?} differs from required {expected_shape:?}",
                descriptor.shape
            )));
        }
        let expected_bytes = expected_shape
            .iter()
            .try_fold(1usize, |acc, value| acc.checked_mul(*value))
            .ok_or_else(|| failure(format!("{name}: expected byte count overflow")))?;
        if descriptor.data_offsets.len() != 2
            || descriptor.data_offsets[0] >= descriptor.data_offsets[1]
            || descriptor.data_offsets[1] - descriptor.data_offsets[0] != descriptor.bytes
            || descriptor.bytes != expected_bytes as u64
        {
            return Err(failure(format!(
                "{name}: invalid source data offsets or byte length"
            )));
        }
        let source_shard = index
            .weight_map
            .get(name)
            .cloned()
            .ok_or_else(|| failure(format!("source tensor index lacks {name}")))?;
        let window = source_window(manifest, &source_shard)?;

        let mut raw = Vec::with_capacity(expected_bytes);
        let mut cursor = 0u64;
        let mut segments_json = Vec::with_capacity(descriptor.segments.len());
        for segment in &descriptor.segments {
            if !is_sha256(&segment.sha256)
                || segment.chunk_relpath
                    != format!("chunks/{}/{}", &segment.sha256[..2], segment.sha256)
                || segment.tensor_start != cursor
                || segment.tensor_end <= segment.tensor_start
                || segment.tensor_end - segment.tensor_start != segment.bytes
                || segment.source_file_end <= segment.source_file_start
                || segment.source_file_end - segment.source_file_start != segment.bytes
            {
                return Err(failure(format!(
                    "{name}: malformed content-addressed segment"
                )));
            }
            let expected_start = window
                .header_bytes
                .checked_add(descriptor.data_offsets[0])
                .and_then(|value| value.checked_add(segment.tensor_start))
                .ok_or_else(|| failure(format!("{name}: source offset overflow")))?;
            let expected_end = window
                .header_bytes
                .checked_add(descriptor.data_offsets[0])
                .and_then(|value| value.checked_add(segment.tensor_end))
                .ok_or_else(|| failure(format!("{name}: source offset overflow")))?;
            if segment.source_file_start != expected_start
                || segment.source_file_end != expected_end
            {
                return Err(failure(format!(
                    "{name}: source range does not bind descriptor offsets"
                )));
            }
            let chunk =
                checked_regular_path(artifact, &segment.chunk_relpath, "content-addressed chunk")?;
            let bytes = fs::read(&chunk)?;
            if bytes.len() as u64 != segment.bytes || sha256(&bytes) != segment.sha256 {
                return Err(failure(format!(
                    "{name}: content-addressed chunk hash/size mismatch"
                )));
            }
            raw.extend_from_slice(&bytes);
            segments_json.push(json!({
                "chunk_relpath": segment.chunk_relpath,
                "sha256": segment.sha256,
                "bytes": segment.bytes,
                "tensor_start": segment.tensor_start,
                "tensor_end": segment.tensor_end,
                "source_file_start": segment.source_file_start,
                "source_file_end": segment.source_file_end,
                "row_start": segment.row_start,
                "row_count": segment.row_count,
            }));
            cursor = segment.tensor_end;
        }
        if cursor != descriptor.bytes || raw.len() != expected_bytes {
            return Err(failure(format!(
                "{name}: content-addressed segments have a gap or wrong total"
            )));
        }
        Ok(BoundTensor {
            descriptor,
            source_shard,
            source_window: window,
            raw,
            segments_json,
        })
    }

    fn decode_e2m1fn(nibble: u8) -> ProbeResult<f32> {
        if nibble > 0x0f {
            return Err(failure("FP4 E2M1FN value is not a nibble"));
        }
        const TABLE: [f32; 16] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        Ok(TABLE[nibble as usize])
    }

    fn decode_e8m0fnu(bits: u8) -> ProbeResult<f32> {
        if bits == 0xff {
            return Err(failure("E8M0FNU scale contains the 0xff NaN encoding"));
        }
        Ok(if bits == 0 {
            f32::from_bits(0x0040_0000)
        } else {
            f32::from_bits(u32::from(bits) << 23)
        })
    }

    fn decoder_self_check() -> ProbeResult<()> {
        let expected = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        for (nibble, value) in expected.into_iter().enumerate() {
            if decode_e2m1fn(nibble as u8)? != value {
                return Err(failure(format!(
                    "E2M1FN decoder self-check failed for 0x{nibble:x}"
                )));
            }
        }
        if decode_e8m0fnu(0)? != 2.0_f32.powi(-127)
            || decode_e8m0fnu(0x7f)? != 1.0
            || decode_e8m0fnu(0x80)? != 2.0
            || decode_e8m0fnu(0xff).is_ok()
        {
            return Err(failure("E8M0FNU decoder self-check failed"));
        }
        Ok(())
    }

    fn deterministic_input(cols: usize) -> Vec<f32> {
        (0..cols)
            .map(|index| {
                let signed = ((index.wrapping_mul(73).wrapping_add(19)) % 257) as i32 - 128;
                signed as f32 * (1.0 / 128.0)
            })
            .collect()
    }

    fn native_encoding_stats(weights: &[u8], scales: &[u8]) -> ProbeResult<Value> {
        let mut nibble_histogram = [0u64; 16];
        for &packed in weights {
            nibble_histogram[(packed & 0x0f) as usize] += 1;
            nibble_histogram[(packed >> 4) as usize] += 1;
        }
        let mut min_scale = u8::MAX;
        let mut max_scale = u8::MIN;
        for &scale in scales {
            min_scale = min_scale.min(scale);
            max_scale = max_scale.max(scale);
            if !decode_e8m0fnu(scale)?.is_finite() {
                return Err(failure("finite E8M0FNU decoder returned non-finite scale"));
            }
        }
        Ok(json!({
            "packed_weight_bytes_checked": weights.len(),
            "logical_fp4_values_checked": weights.len() * 2,
            "fp4_nibble_histogram_low_to_high": nibble_histogram,
            "scale_bytes_checked": scales.len(),
            "minimum_scale_raw_byte": min_scale,
            "maximum_scale_raw_byte": max_scale,
            "reserved_nan_scale_encodings": 0,
            "all_native_values_finite": true,
        }))
    }

    fn cpu_reference(weights: &[u8], scales: &[u8], input: &[f32]) -> ProbeResult<Vec<f32>> {
        if weights.len() != ROWS * PACKED_COLS
            || scales.len() != ROWS * SCALE_COLS
            || input.len() != LOGICAL_COLS
        {
            return Err(failure(
                "source-native FP4 pair has an unexpected compact geometry",
            ));
        }
        let mut output = Vec::with_capacity(ROWS);
        for row in 0..ROWS {
            let mut accumulation = 0.0_f32;
            let packed_row = row * PACKED_COLS;
            let scale_row = row * SCALE_COLS;
            for col in 0..LOGICAL_COLS {
                let packed = weights[packed_row + col / 2];
                let nibble = if col & 1 == 0 {
                    packed & 0x0f
                } else {
                    packed >> 4
                };
                let unit = decode_e2m1fn(nibble)?;
                let scale = decode_e8m0fnu(scales[scale_row + col / FP4_BLOCK])?;
                let product = unit * scale * input[col];
                accumulation += product;
            }
            if !accumulation.is_finite() {
                return Err(failure(
                    "CPU source-native FP4 reference produced a non-finite output",
                ));
            }
            output.push(accumulation);
        }
        Ok(output)
    }

    fn f32_sha256(values: &[f32]) -> String {
        sha256(bytemuck::cast_slice(values))
    }

    fn first8(values: &[f32]) -> Vec<f32> {
        values.iter().take(8).copied().collect()
    }

    fn parity(cpu: &[f32], gpu: &[f32]) -> ProbeResult<Value> {
        if cpu.len() != gpu.len() || cpu.is_empty() {
            return Err(failure("CPU/GPU parity vectors have incompatible length"));
        }
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        let mut mean_abs = 0.0f64;
        let mut worst_index = 0usize;
        let mut passing = true;
        for (index, (&reference, &observed)) in cpu.iter().zip(gpu).enumerate() {
            if !reference.is_finite() || !observed.is_finite() {
                return Err(failure("CPU/GPU parity encountered a non-finite output"));
            }
            let abs = (reference - observed).abs();
            let rel = abs / reference.abs().max(1.0e-6);
            if abs > max_abs {
                max_abs = abs;
                worst_index = index;
            }
            max_rel = max_rel.max(rel);
            mean_abs += f64::from(abs);
            if abs > ABS_TOLERANCE + REL_TOLERANCE * reference.abs() {
                passing = false;
            }
        }
        mean_abs /= cpu.len() as f64;
        Ok(json!({
            "status": if passing { "PASS" } else { "FAIL" },
            "comparison": "per-output abs_error <= 1e-4 + 1e-4 * abs(cpu_reference)",
            "max_abs_error": max_abs,
            "mean_abs_error": mean_abs,
            "max_relative_error": max_rel,
            "worst_output_row": worst_index,
            "cpu_value_at_worst_row": cpu[worst_index],
            "gpu_value_at_worst_row": gpu[worst_index],
        }))
    }

    fn tensor_receipt_json(bound: &BoundTensor<'_>) -> Value {
        json!({
            "name": bound.descriptor.name,
            "dtype": bound.descriptor.dtype,
            "shape": bound.descriptor.shape,
            "data_offsets": bound.descriptor.data_offsets,
            "bytes": bound.descriptor.bytes,
            "source_shard": bound.source_shard,
            "source_window": {
                "header_bytes": bound.source_window.header_bytes,
                "streamed_full_file_sha256": bound.source_window.streamed_full_file_sha256,
                "etag_sha256": bound.source_window.source.etag_sha256,
                "xet_file_hash": bound.source_window.source.xet_file_hash,
            },
            "segments": bound.segments_json,
            "logical_tensor_sha256": sha256(&bound.raw),
        })
    }

    fn sealed_receipt(mut receipt: Value) -> ProbeResult<(Value, String)> {
        if receipt.get("seal_sha256").is_some() {
            return Err(failure(
                "probe receipt unexpectedly already contains a seal",
            ));
        }
        let seal = sha256(&serde_json::to_vec(&receipt)?);
        receipt
            .as_object_mut()
            .ok_or_else(|| failure("probe receipt root must be an object"))?
            .insert("seal_sha256".to_owned(), Value::String(seal.clone()));
        Ok((receipt, seal))
    }

    fn write_new_receipt(path: &Path, receipt: &Value) -> ProbeResult<()> {
        if path.exists() {
            return Err(failure(format!(
                "refusing to overwrite existing probe receipt {}",
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
        let temporary = parent.join(format!(".{name}.{}.fp4-metal.tmp", std::process::id()));
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

    fn percentile_us(samples: &[u64], percentile: u64) -> ProbeResult<u64> {
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

    fn timing_summary(samples: &[u64]) -> ProbeResult<Value> {
        if samples.is_empty() {
            return Err(failure("component sweep candidate has no measured samples"));
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

    /// Run a bounded threadgroup sweep over the already-proven source-native
    /// FP4 routed-expert component.  It deliberately does not turn the expert
    /// linear into a route, a layer, a model, or a token benchmark.
    pub fn sweep_component(
        artifact_arg: &Path,
        warmup_dispatches: usize,
        measured_dispatches: usize,
        threadgroups: &[u32],
    ) -> Result<Value, Box<dyn Error>> {
        if warmup_dispatches == 0 || measured_dispatches == 0 || threadgroups.is_empty() {
            return Err(failure(
                "component sweep needs positive warmups, measurements, and threadgroups",
            ));
        }
        if !artifact_arg.is_absolute() {
            return Err(failure("component sweep artifact path must be absolute"));
        }
        if threadgroups.iter().any(|threads| *threads == 0) {
            return Err(failure("component sweep threadgroup ladder contains zero"));
        }
        if threadgroups.windows(2).any(|window| window[0] >= window[1]) {
            return Err(failure(
                "component sweep threadgroup ladder must be strictly increasing",
            ));
        }

        let artifact_meta = fs::symlink_metadata(artifact_arg)?;
        if artifact_meta.file_type().is_symlink() || !artifact_meta.file_type().is_dir() {
            return Err(failure(
                "component sweep artifact must be a non-symlink directory",
            ));
        }
        let artifact = fs::canonicalize(artifact_arg)?;
        let manifest_path =
            checked_regular_path(&artifact, "manifest.json", "full Gravity manifest")?;
        let manifest_raw = fs::read(&manifest_path)?;
        let manifest_file_sha256 = sha256(&manifest_raw);
        verify_manifest_seal(&manifest_raw)?;
        let manifest: FullManifest = serde_json::from_slice(&manifest_raw).map_err(|error| {
            failure(format!(
                "sealed full Gravity manifest has invalid schema: {error}"
            ))
        })?;
        if manifest.schema != ARTIFACT_SCHEMA || manifest.status != ARTIFACT_STATUS {
            return Err(failure(
                "artifact is not the sealed DeepSeek-V4 full stream runtime-pending state",
            ));
        }
        if manifest.source.repository != REPOSITORY || manifest.source.revision != REVISION {
            return Err(failure(
                "artifact source identity is not the pinned DeepSeek-V4-Flash revision",
            ));
        }
        if manifest.source.source_parent_persisted {
            return Err(failure(
                "full artifact violates the source-parent-evicted storage contract",
            ));
        }
        if !manifest.representation_and_kernel_grammar.fp4.contains(
            "E2M1FN x2 packed low-nibble then high-nibble along K; E8M0FNU scale per 32 logical K",
        ) {
            return Err(failure(
                "artifact does not carry the required FP4 kernel grammar authority",
            ));
        }
        decoder_self_check()?;
        let index = load_index(&artifact, &manifest.source)?;
        let weights = bind_tensor(
            &artifact,
            &manifest,
            &index,
            WEIGHT_NAME,
            "I8",
            &[ROWS, PACKED_COLS],
        )?;
        let scales = bind_tensor(
            &artifact,
            &manifest,
            &index,
            SCALE_NAME,
            "F8_E8M0",
            &[ROWS, SCALE_COLS],
        )?;
        if weights.source_shard != scales.source_shard {
            return Err(failure(
                "selected source-native FP4 weight and scale live in different shards",
            ));
        }
        let native_stats = native_encoding_stats(&weights.raw, &scales.raw)?;
        let input = deterministic_input(LOGICAL_COLS);
        let input_sha256 = f32_sha256(&input);
        let cpu_output = cpu_reference(&weights.raw, &scales.raw, &input)?;

        let run_nonce = sha256_join(&[
            &manifest.seal_sha256,
            &weights.descriptor.segments[0].sha256,
            &scales.descriptor.segments[0].sha256,
            &input_sha256,
            "threadgroup_sweep_v1",
        ]);
        let context = MetalContext::new_with_trace(true)?;
        let device_name = context.device_name();
        let pipeline = context.pipeline(KERNEL_NAME)?;
        let pipeline_thread_execution_width = pipeline.thread_execution_width() as u64;
        let pipeline_max_total_threads_per_threadgroup =
            pipeline.max_total_threads_per_threadgroup() as u64;
        drop(pipeline);
        let weight_buffer = context.new_buffer_with_bytes_checked(&weights.raw)?;
        let scale_buffer = context.new_buffer_with_bytes_checked(&scales.raw)?;
        let input_buffer = context.new_buffer_with_bytes_checked(bytemuck::cast_slice(&input))?;
        let output_buffer = context.new_buffer_checked(ROWS * std::mem::size_of::<f32>())?;
        let rows = ROWS as u32;
        let packed_cols = PACKED_COLS as u32;
        let scale_cols = SCALE_COLS as u32;
        let logical_bytes_read_per_dispatch =
            weights.raw.len() + scales.raw.len() + input.len() * std::mem::size_of::<f32>();
        let logical_bytes_written_per_dispatch = ROWS * std::mem::size_of::<f32>();
        let interval_id = sha256_join(&[&run_nonce, KERNEL_NAME, "threadgroup_sweep"]);
        let trace_identity = PhysicalTraceIdentity::new(
            interval_id.clone(),
            run_nonce.clone(),
            "component_threadgroup_sweep".to_owned(),
            "fp4_routed_expert_gate_matvec".to_owned(),
            Some(1),
            0,
        )?;
        let physical_trace = PhysicalTraceGuard::begin(trace_identity)?;
        let mut candidates = Vec::with_capacity(threadgroups.len());
        let mut stable = Vec::<(u32, u64, u64, u64)>::new();
        let mut observed_dispatches = 0u64;
        let mut observed_waits = 0u64;

        for &threadgroup in threadgroups {
            if u64::from(threadgroup) > pipeline_max_total_threads_per_threadgroup {
                candidates.push(json!({
                    "threadgroup": [threadgroup, 1, 1],
                    "status": "UNSUPPORTED_MAX_TOTAL_THREADS",
                    "not_dispatched_reason": "requested threadgroup exceeds this compiled pipeline's maximum",
                    "pipeline_max_total_threads_per_threadgroup": pipeline_max_total_threads_per_threadgroup,
                    "gpu_dispatches": 0,
                    "command_buffers": 0,
                    "compute_encoders": 0,
                    "cpu_visible_waits": 0,
                    "fallback": false,
                    "fallback_reason": null,
                }));
                continue;
            }

            let mut gpu_us = Vec::with_capacity(measured_dispatches);
            let mut encode_us = Vec::with_capacity(measured_dispatches);
            let mut submit_us = Vec::with_capacity(measured_dispatches);
            let mut wait_us = Vec::with_capacity(measured_dispatches);
            let mut host_wall_us = Vec::with_capacity(measured_dispatches);
            let dispatches_for_candidate = warmup_dispatches + measured_dispatches;
            for dispatch_index in 0..dispatches_for_candidate {
                let timing = context.dispatch_threads_timed(
                    KERNEL_NAME,
                    (rows, 1, 1),
                    (threadgroup, 1, 1),
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
                            &packed_cols as *const u32 as *const _,
                        );
                        encoder.set_bytes(
                            6,
                            std::mem::size_of::<u32>() as u64,
                            &scale_cols as *const u32 as *const _,
                        );
                    },
                )?;
                if timing.compute_dispatches != 1
                    || timing.command_buffers != 1
                    || timing.compute_encoders != 1
                {
                    return Err(failure(
                        "FP4 component sweep dispatch did not use exactly one command buffer/encoder/dispatch",
                    ));
                }
                observed_dispatches += 1;
                observed_waits += 1;
                if dispatch_index >= warmup_dispatches {
                    gpu_us.push(timing.gpu_duration_us.filter(|duration| *duration > 0).ok_or_else(|| {
                        failure("FP4 component sweep measured dispatch has no positive GPU timestamp")
                    })?);
                    encode_us.push(timing.encode_us);
                    submit_us.push(timing.submit_us);
                    wait_us.push(timing.wait_us);
                    host_wall_us.push(timing.host_wall_us);
                }
            }
            let gpu_output = unsafe {
                std::slice::from_raw_parts(output_buffer.contents() as *const f32, ROWS).to_vec()
            };
            let comparison = parity(&cpu_output, &gpu_output)?;
            if comparison.get("status").and_then(Value::as_str) != Some("PASS") {
                return Err(failure(
                    "source-native FP4 CPU/GPU parity failed for a measured threadgroup geometry",
                ));
            }
            let gpu_summary = timing_summary(&gpu_us)?;
            let p50 = gpu_summary
                .get("p50_us")
                .and_then(Value::as_u64)
                .ok_or_else(|| failure("FP4 timing summary lacks p50"))?;
            let p95 = gpu_summary
                .get("p95_us")
                .and_then(Value::as_u64)
                .ok_or_else(|| failure("FP4 timing summary lacks p95"))?;
            let p99 = gpu_summary
                .get("p99_us")
                .and_then(Value::as_u64)
                .ok_or_else(|| failure("FP4 timing summary lacks p99"))?;
            stable.push((threadgroup, p50, p95, p99));
            let logical_read_gib_s = logical_bytes_read_per_dispatch as f64 * 1_000_000.0
                / p50 as f64
                / (1024.0 * 1024.0 * 1024.0);
            candidates.push(json!({
                "threadgroup": [threadgroup, 1, 1],
                "status": "PASS_GPU_TIMESTAMPED_CPU_PARITY",
                "geometry_kind": "one_output_row_per_thread; threadgroup-only sweep",
                "warmup_dispatches": warmup_dispatches,
                "measured_dispatches": measured_dispatches,
                "gpu_duration": gpu_summary,
                "host_encode_duration": timing_summary(&encode_us)?,
                "host_submit_duration": timing_summary(&submit_us)?,
                "host_wait_duration": timing_summary(&wait_us)?,
                "host_wall_duration": timing_summary(&host_wall_us)?,
                "cpu_parity": comparison,
                "gpu_output_sha256_f32_le": f32_sha256(&gpu_output),
                "logical_bytes_read_per_dispatch": logical_bytes_read_per_dispatch,
                "logical_bytes_written_per_dispatch": logical_bytes_written_per_dispatch,
                "logical_read_bandwidth_gib_s_at_gpu_p50": logical_read_gib_s,
                "gpu_dispatches": dispatches_for_candidate,
                "command_buffers": dispatches_for_candidate,
                "compute_encoders": dispatches_for_candidate,
                "cpu_visible_waits": dispatches_for_candidate,
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
            return Err(failure("no supported FP4 component sweep geometry passed"));
        }
        if physical_counts.command_count != observed_dispatches
            || physical_counts.encoder_count != observed_dispatches
            || commits as u64 != observed_dispatches
            || trace_samples.len() as u64 != observed_dispatches
        {
            return Err(failure(
                "FP4 component sweep physical command/encoder/dispatch accounting mismatch",
            ));
        }
        let deepest = stable
            .iter()
            .max_by_key(|(threadgroup, _, _, _)| *threadgroup)
            .copied()
            .ok_or_else(|| failure("FP4 sweep lacks a deepest stable geometry"))?;
        let winner = stable
            .iter()
            .min_by_key(|(threadgroup, p50, p95, p99)| (*p50, *p95, *p99, *threadgroup))
            .copied()
            .ok_or_else(|| failure("FP4 sweep lacks a winning geometry"))?;

        Ok(json!({
            "component": "layers.0.ffn.experts.0.w1 source-native FP4 routed-expert gate linear",
            "kernel": KERNEL_NAME,
            "status": "PASS_COMPONENT_THREADGROUP_SWEEP_NOT_FULL_RUNTIME",
            "scope": {
                "component_microbenchmark_only": true,
                "not_a_full_model_load": true,
                "not_an_MoE_route_or_expert_selection_claim": true,
                "not_a_full_token_or_generation": true,
                "not_a_full_43_layer_runtime_adapter": true,
                "not_a_BASE_TRUE_TPS_measurement": true,
            },
            "artifact": {
                "path": artifact,
                "manifest_file_sha256": manifest_file_sha256,
                "manifest_seal_sha256": manifest.seal_sha256,
                "manifest_seal_verified": true,
                "full_stream_schema": manifest.schema,
                "full_stream_status": manifest.status,
                "source_parent_persisted": manifest.source.source_parent_persisted,
            },
            "source": {
                "repository": manifest.source.repository,
                "revision": manifest.source.revision,
                "source_index_binding_verified": true,
                "source_window_binding_verified": true,
                "chunk_mapping_verified": true,
                "weight": tensor_receipt_json(&weights),
                "scale": tensor_receipt_json(&scales),
            },
            "native_codec": {
                "grammar": "E2M1FN x2 packed low-nibble then high-nibble along K; E8M0FNU scale per 32 logical K",
                "encoding_validation": native_stats,
            },
            "input": {
                "kind": "deterministic_exact_binary_rational_vector_v1",
                "length": input.len(),
                "sha256_f32_le": input_sha256,
            },
            "cpu_reference": {
                "operation": "one full 2048x4096 logical source-native FP4 routed-expert gate linear",
                "accumulation": "row-major f32 product_then_add",
                "output_sha256_f32_le": f32_sha256(&cpu_output),
            },
            "metal": {
                "device": device_name,
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
            "ladder": {
                "requested_threadgroups": threadgroups,
                "candidates": candidates,
                "deepest_stable_threadgroup_rung": {
                    "threadgroup": [deepest.0, 1, 1],
                    "gpu_p50_us": deepest.1,
                    "gpu_p95_us": deepest.2,
                    "gpu_p99_us": deepest.3,
                },
                "winner": {
                    "selection": "minimum GPU p50, then p95, then p99, then lower threadgroup",
                    "threadgroup": [winner.0, 1, 1],
                    "gpu_p50_us": winner.1,
                    "gpu_p95_us": winner.2,
                    "gpu_p99_us": winner.3,
                },
            },
            "physical_trace": {
                "interval_id": interval_id,
                "run_nonce": run_nonce,
                "phase": "component_threadgroup_sweep",
                "role": "fp4_routed_expert_gate_matvec",
            },
            "next_boundary": "This is a source-hash-bound FP4 routed-expert component microbenchmark only. It does not execute routing, load the full model, execute a token, validate a full runtime graph, or satisfy BASE_TRUE_TPS.",
        }))
    }

    pub fn run() -> ProbeResult<()> {
        let args = parse_args()?;
        let artifact_meta = fs::symlink_metadata(&args.artifact)?;
        if artifact_meta.file_type().is_symlink() || !artifact_meta.file_type().is_dir() {
            return Err(failure("--artifact must be a non-symlink directory"));
        }
        let artifact = fs::canonicalize(&args.artifact)?;
        let manifest_path =
            checked_regular_path(&artifact, "manifest.json", "full Gravity manifest")?;
        let manifest_raw = fs::read(&manifest_path)?;
        let manifest_file_sha256 = sha256(&manifest_raw);
        verify_manifest_seal(&manifest_raw)?;
        let manifest: FullManifest = serde_json::from_slice(&manifest_raw).map_err(|error| {
            failure(format!(
                "sealed full Gravity manifest has invalid schema: {error}"
            ))
        })?;
        if manifest.schema != ARTIFACT_SCHEMA || manifest.status != ARTIFACT_STATUS {
            return Err(failure(
                "artifact is not the sealed DeepSeek-V4 full stream runtime-pending state",
            ));
        }
        if manifest.source.repository != REPOSITORY || manifest.source.revision != REVISION {
            return Err(failure(
                "artifact source identity is not the pinned DeepSeek-V4-Flash revision",
            ));
        }
        if manifest.source.source_parent_persisted {
            return Err(failure(
                "full artifact violates the source-parent-evicted storage contract",
            ));
        }
        if !manifest.representation_and_kernel_grammar.fp4.contains(
            "E2M1FN x2 packed low-nibble then high-nibble along K; E8M0FNU scale per 32 logical K",
        ) || !is_sha256(
            &manifest
                .representation_and_kernel_grammar
                .official_convert_py_sha256,
        ) || !is_sha256(
            &manifest
                .representation_and_kernel_grammar
                .official_kernel_py_sha256,
        ) {
            return Err(failure(
                "artifact does not carry the required FP4 kernel grammar authority",
            ));
        }
        decoder_self_check()?;
        let index = load_index(&artifact, &manifest.source)?;
        let weights = bind_tensor(
            &artifact,
            &manifest,
            &index,
            WEIGHT_NAME,
            "I8",
            &[ROWS, PACKED_COLS],
        )?;
        let scales = bind_tensor(
            &artifact,
            &manifest,
            &index,
            SCALE_NAME,
            "F8_E8M0",
            &[ROWS, SCALE_COLS],
        )?;
        if weights.source_shard != scales.source_shard {
            return Err(failure(
                "selected source-native FP4 weight and scale live in different shards",
            ));
        }
        let native_stats = native_encoding_stats(&weights.raw, &scales.raw)?;
        let input = deterministic_input(LOGICAL_COLS);
        let input_sha256 = f32_sha256(&input);
        let cpu_output = cpu_reference(&weights.raw, &scales.raw, &input)?;

        let run_nonce = sha256_join(&[
            &manifest.seal_sha256,
            &weights.descriptor.segments[0].sha256,
            &scales.descriptor.segments[0].sha256,
            &input_sha256,
        ]);
        let interval_id = sha256_join(&[&run_nonce, KERNEL_NAME, "component_parity"]);
        let trace_identity = PhysicalTraceIdentity::new(
            interval_id.clone(),
            run_nonce.clone(),
            "component_parity".to_owned(),
            "fp4_routed_expert_gate_matvec".to_owned(),
            Some(1),
            0,
        )?;
        let physical_trace = PhysicalTraceGuard::begin(trace_identity)?;

        let context = MetalContext::new_with_trace(true)?;
        let device_name = context.device_name();
        let weight_buffer = context.new_buffer_with_bytes_checked(&weights.raw)?;
        let scale_buffer = context.new_buffer_with_bytes_checked(&scales.raw)?;
        let input_buffer = context.new_buffer_with_bytes_checked(bytemuck::cast_slice(&input))?;
        let output_buffer = context.new_buffer_checked(ROWS * std::mem::size_of::<f32>())?;
        let rows = ROWS as u32;
        let packed_cols = PACKED_COLS as u32;
        let scale_cols = SCALE_COLS as u32;
        let timing =
            context.dispatch_threads_timed(KERNEL_NAME, (rows, 1, 1), (256, 1, 1), |encoder| {
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
                    &packed_cols as *const u32 as *const _,
                );
                encoder.set_bytes(
                    6,
                    std::mem::size_of::<u32>() as u64,
                    &scale_cols as *const u32 as *const _,
                );
            })?;
        let gpu_output = unsafe {
            std::slice::from_raw_parts(output_buffer.contents() as *const f32, ROWS).to_vec()
        };
        let physical_counts = physical_trace.counts();
        drop(physical_trace);
        let (buffers_created, bytes_allocated, commits) = context.drain_stats();
        let trace_samples = context.drain_trace();
        if timing.compute_dispatches != 1
            || timing.command_buffers != 1
            || timing.compute_encoders != 1
            || physical_counts.command_count != 1
            || physical_counts.encoder_count != 1
            || commits != 1
            || trace_samples.len() != 1
        {
            return Err(failure(
                "Metal probe topology did not contain exactly one real compute dispatch",
            ));
        }
        let gpu_duration_us = timing
            .gpu_duration_us
            .filter(|duration| *duration > 0)
            .ok_or_else(|| {
                failure(
                    "Metal completed without a usable positive GPU timestamp; no receipt emitted",
                )
            })?;
        let comparison = parity(&cpu_output, &gpu_output)?;
        if comparison.get("status").and_then(Value::as_str) != Some("PASS") {
            return Err(failure(
                "source-native FP4 CPU/GPU parity failed; no receipt emitted",
            ));
        }

        let unsigned = json!({
            "schema": "hawking.gravity.deepseek_v4.fp4_e2m1fn_x2_e8m0_metal_component_probe.v1",
            "status": "PASS_REAL_METAL_COMPONENT_PARITY_NOT_FULL_RUNTIME",
            "scope": {
                "component": "layers.0.ffn.experts.0.w1 routed-expert gate linear",
                "logical_operation": "2048 output rows x 4096 logical K source-native FP4 matvec",
                "not_a_full_model_load": true,
                "not_a_generation_or_TPS_claim": true,
                "not_a_registered_43_layer_runtime_adapter": true,
                "not_an_MoE_route_or_expert_selection_claim": true,
            },
            "artifact": {
                "path": artifact,
                "manifest_file_sha256": manifest_file_sha256,
                "manifest_seal_sha256": manifest.seal_sha256,
                "manifest_seal_verified": true,
                "full_stream_schema": manifest.schema,
                "full_stream_status": manifest.status,
                "source_parent_persisted": manifest.source.source_parent_persisted,
            },
            "source": {
                "repository": manifest.source.repository,
                "revision": manifest.source.revision,
                "source_index_binding_verified": true,
                "source_window_binding_verified": true,
                "chunk_mapping_verified": true,
                "weight": tensor_receipt_json(&weights),
                "scale": tensor_receipt_json(&scales),
            },
            "native_codec": {
                "grammar": "E2M1FN x2 packed low-nibble then high-nibble along K; E8M0FNU scale per 32 logical K",
                "cpu_decoder": "exact 16-entry pinned-source E2M1FN table and finite E8M0FNU bit-layout decode",
                "metal_decoder": "exact E2M1FN nibble switch and E8M0FNU bit-layout decode in deepseek_v4_fp4_e2m1fn_x2_e8m0_matvec_authority",
                "official_convert_py_sha256": manifest.representation_and_kernel_grammar.official_convert_py_sha256,
                "official_kernel_py_sha256": manifest.representation_and_kernel_grammar.official_kernel_py_sha256,
                "encoding_validation": native_stats,
            },
            "input": {
                "kind": "deterministic_exact_binary_rational_vector_v1",
                "length": input.len(),
                "sha256_f32_le": input_sha256,
                "first8": first8(&input),
            },
            "cpu_reference": {
                "operation": "one full 2048x4096 logical source-native FP4 routed-expert gate linear",
                "accumulation": "row-major f32 product_then_add",
                "output_sha256_f32_le": f32_sha256(&cpu_output),
                "first8": first8(&cpu_output),
            },
            "metal": {
                "device": device_name,
                "kernel": KERNEL_NAME,
                "dispatch_geometry": { "grid": [rows, 1, 1], "threadgroup": [256, 1, 1] },
                "gpu_dispatches": timing.compute_dispatches,
                "command_buffers": timing.command_buffers,
                "compute_encoders": timing.compute_encoders,
                "physical_trace_command_buffers": physical_counts.command_count,
                "physical_trace_compute_encoders": physical_counts.encoder_count,
                "buffers_created": buffers_created,
                "bytes_allocated": bytes_allocated,
                "timing": {
                    "authority": "completed MTLCommandBuffer GPUStartTime/GPUEndTime",
                    "gpu_duration_us": gpu_duration_us,
                    "gpu_start_ns": timing.gpu_start_ns,
                    "gpu_end_ns": timing.gpu_end_ns,
                    "pipeline_lookup_us": timing.pipeline_lookup_us,
                    "encode_us": timing.encode_us,
                    "submit_us": timing.submit_us,
                    "wait_us": timing.wait_us,
                    "host_wall_us": timing.host_wall_us,
                },
                "gpu_output_sha256_f32_le": f32_sha256(&gpu_output),
                "first8": first8(&gpu_output),
                "fallback": false,
                "fallback_reason": null,
            },
            "parity": comparison,
            "physical_trace": {
                "interval_id": interval_id,
                "run_nonce": run_nonce,
                "phase": "component_parity",
                "role": "fp4_routed_expert_gate_matvec",
            },
            "next_boundary": "This proves one source-native FP4 routed-expert linear on Metal only. It does not clear full-model adapter, full-runtime parity, routing, generation, HCLI, or BASE_TRUE_TPS gates.",
        });
        let (receipt, seal) = sealed_receipt(unsigned)?;
        write_new_receipt(&args.out, &receipt)?;
        println!(
            "{}",
            serde_json::to_string(&json!({
                "status": "PASS_REAL_METAL_COMPONENT_PARITY_NOT_FULL_RUNTIME",
                "receipt": args.out,
                "seal_sha256": seal,
                "gpu_duration_us": gpu_duration_us,
                "gpu_dispatches": timing.compute_dispatches,
            }))?
        );
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}
