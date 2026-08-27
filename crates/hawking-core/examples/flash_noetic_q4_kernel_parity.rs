//! Bounded native-kernel parity for the pinned Flash-Next noetic Q4 descriptor.
//!
//! This probe reads a real routed-expert block from the exact ModelLake
//! specimen, encodes it with the descriptor already exercised by the Python
//! loader round-trip, and runs the existing uniform-Q4/group-64 Metal matvec.
//! It is intentionally one source tensor block, not a model loader, decoder,
//! token loop, capability test, complete-system EBPW result, or Flash TPS
//! claim.

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("Flash noetic kernel parity requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
mod macos {
    use half::{bf16, f16};
    use hawking_core::metal::{MetalContext, MetalDispatchTiming};
    use metal::Buffer;
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::env;
    use std::error::Error;
    use std::fs::{self, File};
    use std::io::{Read, Seek, SeekFrom};
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    const SCHEMA: &str = "hawking.flash_noetic_q4_kernel_parity.v1";
    const NOMENCLATURE_VERSION: &str = "HAWKING_NOMENCLATURE_V1";
    const REPO_ID: &str = "Qwen/Qwen3.8-Flash-Next";
    const PINNED_REVISION: &str = "34567a4712bc9766c4449e2e98e4468bfa24d915";
    const TENSOR_NAME: &str = "model.language_model.layers.0.mlp.experts.gate_up_proj";
    const GROUP_SIZE: usize = 64;
    const CODE_BYTES_PER_GROUP: usize = GROUP_SIZE / 2;
    const KERNEL_NAME: &str = "qwen_uniform_q4_group64_matvec";
    const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;
    const MAX_BLOCK_BYTES: usize = 8 * 1024 * 1024;
    const DEFAULT_ROW_COUNT: usize = 128;
    const DEFAULT_WARMUP: usize = 2;
    const DEFAULT_REPS: usize = 7;
    const OUTPUT_ERROR_TOLERANCE: f32 = 2.0e-3;

    struct Args {
        root: PathBuf,
        descriptor: PathBuf,
        candidate_body: Option<PathBuf>,
        body_receipt: Option<PathBuf>,
        tensor_name: String,
        expert_index: usize,
        row_start: usize,
        row_count: usize,
        warmup: usize,
        reps: usize,
        out: PathBuf,
    }

    struct NoeticDescriptor {
        path: PathBuf,
        sha256: String,
        body: Value,
    }

    struct TensorLocation {
        tensor_name: String,
        shard: PathBuf,
        shard_name: String,
        shard_size: u64,
        header_bytes: Vec<u8>,
        dtype: String,
        shape: Vec<usize>,
        data_start: u64,
        data_begin: u64,
        data_end: u64,
        index_sha256: String,
    }

    struct PackedQ4 {
        codes: Vec<u8>,
        scales_bits: Vec<u16>,
        candidate_bytes: Vec<u8>,
        source_rmse: f64,
        source_max_abs_error: f32,
    }

    fn repository_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."))
    }

    fn default_lake_root() -> PathBuf {
        if let Some(value) = env::var_os("HCLI_FLASH_NEXT_ROOT") {
            return PathBuf::from(value);
        }
        PathBuf::from(
            "/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc",
        )
    }

    fn parse_usize(value: Option<String>, flag: &str) -> Result<usize, Box<dyn Error>> {
        value
            .ok_or_else(|| format!("missing value after {flag}"))?
            .parse::<usize>()
            .map_err(|error| format!("invalid {flag}: {error}").into())
    }

    fn parse_args() -> Result<Args, Box<dyn Error>> {
        let root = repository_root();
        let mut args = Args {
            root: default_lake_root(),
            descriptor: root.join("receipts/headless/FLASH_ROUTED_EXPERT_LOADER_ROUNDTRIP.json"),
            candidate_body: None,
            body_receipt: None,
            tensor_name: TENSOR_NAME.to_owned(),
            expert_index: 0,
            row_start: 0,
            row_count: DEFAULT_ROW_COUNT,
            warmup: DEFAULT_WARMUP,
            reps: DEFAULT_REPS,
            out: root.join("receipts/headless/FLASH_NOETIC_Q4_KERNEL_PARITY.json"),
        };
        let mut values = env::args().skip(1);
        while let Some(flag) = values.next() {
            match flag.as_str() {
                "--root" => args.root = PathBuf::from(values.next().ok_or("missing --root")?),
                "--descriptor" => {
                    args.descriptor = PathBuf::from(values.next().ok_or("missing --descriptor")?)
                }
                "--candidate-body" => {
                    args.candidate_body = Some(PathBuf::from(
                        values.next().ok_or("missing --candidate-body")?,
                    ))
                }
                "--body-receipt" => {
                    args.body_receipt = Some(PathBuf::from(
                        values.next().ok_or("missing --body-receipt")?,
                    ))
                }
                "--tensor-name" => {
                    args.tensor_name = values.next().ok_or("missing --tensor-name")?
                }
                "--expert-index" => args.expert_index = parse_usize(values.next(), &flag)?,
                "--row-start" => args.row_start = parse_usize(values.next(), &flag)?,
                "--row-count" => args.row_count = parse_usize(values.next(), &flag)?,
                "--warmup" => args.warmup = parse_usize(values.next(), &flag)?,
                "--reps" => args.reps = parse_usize(values.next(), &flag)?,
                "--out" => args.out = PathBuf::from(values.next().ok_or("missing --out")?),
                "--help" | "-h" => {
                    println!(
                        "usage: flash_noetic_q4_kernel_parity [--root DIR] [--tensor-name NAME] \
                         [--descriptor FILE] [--candidate-body FILE] [--body-receipt FILE] \
                         [--expert-index N] [--row-start N] [--row-count N] [--warmup N] \
                         [--reps N] [--out FILE]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument {other}").into()),
            }
        }
        if args.reps == 0 || args.reps > 128 {
            return Err("--reps must be in 1..=128".into());
        }
        if args.warmup > 64 {
            return Err("--warmup must be <= 64".into());
        }
        if args.row_count == 0 {
            return Err("--row-count must be positive".into());
        }
        Ok(args)
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn read_json(path: &Path) -> Result<(Value, Vec<u8>), Box<dyn Error>> {
        let bytes = fs::read(path)?;
        Ok((serde_json::from_slice(&bytes)?, bytes))
    }

    fn descriptor_string<'a>(value: &'a Value, field: &str) -> Result<&'a str, Box<dyn Error>> {
        value
            .get(field)
            .and_then(Value::as_str)
            .ok_or_else(|| format!("Noetic descriptor is missing string field {field}").into())
    }

    fn descriptor_usize(value: &Value, field: &str) -> Result<usize, Box<dyn Error>> {
        value
            .get(field)
            .and_then(Value::as_u64)
            .and_then(|number| usize::try_from(number).ok())
            .ok_or_else(|| format!("Noetic descriptor is missing integer field {field}").into())
    }

    fn load_noetic_descriptor(
        path: &Path,
        tensor_name: &str,
    ) -> Result<NoeticDescriptor, Box<dyn Error>> {
        let canonical_path = path.canonicalize()?;
        let (receipt, bytes) = read_json(&canonical_path)?;
        if receipt.get("status").and_then(Value::as_str) != Some("PASSED") {
            return Err("Noetic loader receipt is not PASSED".into());
        }
        if receipt.get("nomenclature_version").and_then(Value::as_str) != Some(NOMENCLATURE_VERSION)
        {
            return Err("Noetic loader receipt has an unsupported nomenclature version".into());
        }
        let descriptor = receipt
            .get("representation_descriptor")
            .ok_or("Noetic loader receipt has no representation_descriptor")?;
        if descriptor_string(descriptor, "schema")? != "hcli.noetic.representation_descriptor.v1" {
            return Err("unsupported Noetic representation descriptor schema".into());
        }
        if descriptor_string(descriptor, "candidate_id")? != "independent_q4_g64" {
            return Err("bounded Metal probe currently accepts independent_q4_g64 only".into());
        }
        let source = descriptor
            .get("source_tensor")
            .ok_or("Noetic descriptor has no source_tensor")?;
        if descriptor_string(source, "tensor_name")? != tensor_name
            || descriptor_string(source, "dtype")?.to_uppercase() != "BF16"
        {
            return Err(
                "Noetic descriptor source tensor does not match the selected BF16 tensor".into(),
            );
        }
        let shape = source
            .get("shape")
            .and_then(Value::as_array)
            .ok_or("Noetic descriptor source_tensor has no shape")?;
        let shape_values: Vec<usize> = shape
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|number| usize::try_from(number).ok())
                    .ok_or_else(|| "Noetic descriptor source shape is not an integer vector".into())
            })
            .collect::<Result<_, Box<dyn Error>>>()?;
        if shape_values.len() != 2 && shape_values.len() != 3 {
            return Err("Noetic descriptor source shape must be rank two or three".into());
        }
        let columns = *shape_values
            .last()
            .ok_or("Noetic descriptor source shape is empty")?;
        if columns == 0 || columns % GROUP_SIZE != 0 {
            return Err("Noetic descriptor source columns are not divisible by 64".into());
        }
        let expected_layout = if shape_values.len() == 3 {
            "row-major [expert, row, column]"
        } else {
            "row-major [row, column]"
        };
        if descriptor_string(source, "layout")? != expected_layout {
            return Err("Noetic descriptor source layout does not match its rank".into());
        }
        if descriptor_usize(source, "group_size")? != GROUP_SIZE {
            return Err("Noetic descriptor group size is not 64".into());
        }
        let storage = descriptor
            .get("storage")
            .ok_or("Noetic descriptor has no storage policy")?;
        if descriptor_string(storage, "code_dtype")? != "uint4_packed"
            || descriptor_usize(storage, "code_offset")? != 8
            || descriptor_string(storage, "nibble_order")?
                != "low_nibble_then_high_nibble_row_major"
            || descriptor_string(storage, "scale_dtype")? != "little_endian_float16"
        {
            return Err(
                "Noetic descriptor storage policy is not the bounded native Q4 layout".into(),
            );
        }
        let policy = descriptor
            .get("loader_policy")
            .ok_or("Noetic descriptor has no loader_policy")?;
        if policy.get("source_mutation").and_then(Value::as_bool) != Some(false)
            || policy.get("model_load").and_then(Value::as_bool) != Some(false)
            || descriptor_string(policy, "dense_rematerialization")? != "forbidden"
        {
            return Err("Noetic descriptor loader policy permits an unsafe fallback".into());
        }
        let transform_reference = descriptor
            .get("full_transform_reference")
            .or_else(|| descriptor.get("transform_reference"))
            .ok_or("Noetic descriptor has no transform_reference")?;
        let transform_status = descriptor_string(transform_reference, "status")?;
        if transform_status != "FULL_TENSOR_TRANSFORM_ONLY"
            && transform_status != "BOUNDED_TENSOR_TRANSFORM_ONLY"
        {
            return Err(
                "Noetic descriptor transform reference is not a verified Q4 candidate".into(),
            );
        }
        Ok(NoeticDescriptor {
            path: canonical_path,
            sha256: sha256_hex(&bytes),
            body: descriptor.clone(),
        })
    }

    fn validate_manifest(root: &Path) -> Result<Value, Box<dyn Error>> {
        let manifest_path = Path::new("/Volumes/corpdrive/hawking-modellake/manifests/Qwen--Qwen3.8-Flash-Next@34567a4712bc.json");
        let (manifest, bytes) = read_json(manifest_path)?;
        let repo = manifest.get("repo").and_then(Value::as_str);
        let revision = manifest.get("revision").and_then(Value::as_str);
        let resolved = manifest.get("resolved_sha").and_then(Value::as_str);
        if repo != Some(REPO_ID)
            || revision != Some(PINNED_REVISION)
            || resolved != Some(PINNED_REVISION)
        {
            return Err("ModelLake manifest is not the exact pinned Flash target".into());
        }
        if let Some(path) = manifest.get("path").and_then(Value::as_str) {
            if Path::new(path).canonicalize()? != root.canonicalize()? {
                return Err("selected specimen root does not match the pinned manifest".into());
            }
        }
        Ok(json!({
            "path": manifest_path.display().to_string(),
            "sha256": sha256_hex(&bytes),
            "repo": repo,
            "revision": revision,
            "resolved_sha": resolved,
            "n_files": manifest.get("n_files"),
            "bytes": manifest.get("bytes"),
            "label": "[V]",
        }))
    }

    fn load_tensor_location(
        root: &Path,
        tensor_name: &str,
    ) -> Result<TensorLocation, Box<dyn Error>> {
        let index_path = root.join("model.safetensors.index.json");
        let (index, index_bytes) = read_json(&index_path)?;
        let weight_map = index
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or("safetensors index has no weight_map")?;
        let shard_name = weight_map
            .get(tensor_name)
            .and_then(Value::as_str)
            .ok_or_else(|| format!("safetensors index has no {tensor_name}"))?;
        let shard = root.join(shard_name);
        if shard.canonicalize()?.parent() != Some(root.canonicalize()?.as_path()) {
            return Err("safetensors shard escapes the selected specimen".into());
        }
        let mut file = File::open(&shard)?;
        let mut length_bytes = [0u8; 8];
        file.read_exact(&mut length_bytes)?;
        let header_len = u64::from_le_bytes(length_bytes);
        if header_len == 0 || header_len > MAX_HEADER_BYTES {
            return Err(format!("unsafe safetensors header length {header_len}").into());
        }
        let mut header_bytes = vec![0u8; header_len as usize];
        file.read_exact(&mut header_bytes)?;
        let header: Value = serde_json::from_slice(&header_bytes)?;
        let tensor = header
            .get(tensor_name)
            .ok_or_else(|| format!("shard header has no {tensor_name}"))?;
        let dtype = tensor
            .get("dtype")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let shape_values = tensor
            .get("shape")
            .and_then(Value::as_array)
            .ok_or("tensor header has no shape")?;
        if shape_values.len() != 2 && shape_values.len() != 3 {
            return Err("Flash matrix tensor must be rank two or three".into());
        }
        let shape: Vec<usize> = shape_values
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .ok_or("invalid tensor dimension")
                    .map(|v| v as usize)
            })
            .collect::<Result<_, _>>()?;
        if dtype.to_uppercase() != "BF16" || shape.last().copied().unwrap_or(0) % GROUP_SIZE != 0 {
            return Err(format!(
                "unexpected Flash tensor contract: dtype={dtype}, shape={shape:?}"
            )
            .into());
        }
        let offsets = tensor
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or("tensor header has no data_offsets")?;
        if offsets.len() != 2 {
            return Err("tensor data_offsets must have two values".into());
        }
        let data_begin = offsets[0].as_u64().ok_or("invalid tensor data start")?;
        let data_end = offsets[1].as_u64().ok_or("invalid tensor data end")?;
        let expected_bytes = shape.iter().product::<usize>() * 2;
        if data_end.checked_sub(data_begin) != Some(expected_bytes as u64) {
            return Err("tensor data range disagrees with BF16 shape".into());
        }
        let shard_size = file.metadata()?.len();
        let data_start = 8u64 + header_len;
        if data_start.checked_add(data_end).is_none() || data_start + data_end > shard_size {
            return Err("tensor data range exceeds shard".into());
        }
        Ok(TensorLocation {
            tensor_name: tensor_name.to_owned(),
            shard,
            shard_name: shard_name.to_owned(),
            shard_size,
            header_bytes,
            dtype,
            shape,
            data_start,
            data_begin,
            data_end,
            index_sha256: sha256_hex(&index_bytes),
        })
    }

    fn read_source_block(
        location: &TensorLocation,
        expert: usize,
        row_start: usize,
        row_count: usize,
    ) -> Result<Vec<u8>, Box<dyn Error>> {
        let cols = *location.shape.last().ok_or("tensor shape has no columns")?;
        let row_total = if location.shape.len() == 3 {
            location.shape[1]
        } else {
            location.shape[0]
        };
        let expert_total = if location.shape.len() == 3 {
            location.shape[0]
        } else {
            1
        };
        if expert >= expert_total || row_start >= row_total || row_count > row_total - row_start {
            return Err("selected matrix row window is outside the pinned tensor".into());
        }
        let row_width = cols * 2;
        let first_row = if location.shape.len() == 3 {
            expert * location.shape[1] + row_start
        } else {
            row_start
        };
        let first_element = first_row
            .checked_mul(cols)
            .ok_or("source element offset overflow")?;
        let offset = location
            .data_start
            .checked_add(location.data_begin)
            .and_then(|value| value.checked_add((first_element * 2) as u64))
            .ok_or("source byte offset overflow")?;
        let byte_count = row_count * row_width;
        let mut file = File::open(&location.shard)?;
        file.seek(SeekFrom::Start(offset))?;
        let mut raw = vec![0u8; byte_count];
        file.read_exact(&mut raw)?;
        Ok(raw)
    }

    fn bf16_at(raw: &[u8], element: usize) -> f32 {
        let offset = element * 2;
        bf16::from_bits(u16::from_le_bytes([raw[offset], raw[offset + 1]])).to_f32()
    }

    fn pack_q4(raw: &[u8], rows: usize, cols: usize) -> Result<PackedQ4, Box<dyn Error>> {
        if raw.len() != rows * cols * 2 || cols % GROUP_SIZE != 0 {
            return Err("source block does not match packed Q4 dimensions".into());
        }
        let groups_per_row = cols / GROUP_SIZE;
        let group_count = rows * groups_per_row;
        let mut codes = vec![0u8; group_count * CODE_BYTES_PER_GROUP];
        let mut scales_bits = Vec::with_capacity(group_count);
        let mut squared_error = 0.0f64;
        let mut max_abs_error = 0.0f32;
        for row in 0..rows {
            for group in 0..groups_per_row {
                let begin = row * cols + group * GROUP_SIZE;
                let mut peak = 0.0f32;
                for col in 0..GROUP_SIZE {
                    let value = bf16_at(raw, begin + col);
                    if !value.is_finite() {
                        return Err("source BF16 block contains a non-finite value".into());
                    }
                    peak = peak.max(value.abs());
                }
                let stored_scale = f16::from_f32(if peak == 0.0 { 0.0 } else { peak / 7.0 });
                let scale = stored_scale.to_f32();
                if !scale.is_finite() {
                    return Err("Q4 group scale is not finite".into());
                }
                scales_bits.push(stored_scale.to_bits());
                let code_base = (row * groups_per_row + group) * CODE_BYTES_PER_GROUP;
                for col in 0..GROUP_SIZE {
                    let value = bf16_at(raw, begin + col);
                    let q = if scale == 0.0 {
                        0i32
                    } else {
                        (value / scale).round().clamp(-8.0, 7.0) as i32
                    };
                    let nibble = (q + 8) as u8;
                    let slot = &mut codes[code_base + col / 2];
                    if col % 2 == 0 {
                        *slot |= nibble;
                    } else {
                        *slot |= nibble << 4;
                    }
                    let error = value - q as f32 * scale;
                    squared_error += f64::from(error) * f64::from(error);
                    max_abs_error = max_abs_error.max(error.abs());
                }
            }
        }
        let mut candidate_bytes = codes.clone();
        for bits in &scales_bits {
            candidate_bytes.extend_from_slice(&bits.to_le_bytes());
        }
        Ok(PackedQ4 {
            codes,
            scales_bits,
            candidate_bytes,
            source_rmse: (squared_error / (rows * cols) as f64).sqrt(),
            source_max_abs_error: max_abs_error,
        })
    }

    fn load_candidate_body(
        body_path: &Path,
        receipt_path: &Path,
        rows: usize,
        cols: usize,
    ) -> Result<PackedQ4, Box<dyn Error>> {
        let body = fs::read(body_path.canonicalize()?)?;
        let (receipt, _receipt_bytes) = read_json(&receipt_path.canonicalize()?)?;
        if receipt.get("schema").and_then(Value::as_str)
            != Some("hcli.agentos.flash_noetic_component_body.v1")
            || receipt.get("status").and_then(Value::as_str) != Some("PASSED")
        {
            return Err("candidate body receipt is not a PASSED Noetic component body".into());
        }
        if receipt.get("candidate_id").and_then(Value::as_str) != Some("independent_q4_g64")
            || receipt.get("source_independent").and_then(Value::as_bool) != Some(true)
            || receipt
                .get("candidate_body_persisted")
                .and_then(Value::as_bool)
                != Some(true)
        {
            return Err(
                "candidate body receipt does not declare a source-independent Q4 body".into(),
            );
        }
        let receipt_body = receipt
            .get("body")
            .ok_or("candidate body receipt has no body record")?;
        let recorded_path = receipt_body
            .get("path")
            .and_then(Value::as_str)
            .ok_or("candidate body receipt has no body path")?;
        if Path::new(recorded_path).canonicalize()? != body_path.canonicalize()? {
            return Err("candidate body path does not match its receipt".into());
        }
        let recorded_sha = receipt_body
            .get("sha256")
            .and_then(Value::as_str)
            .ok_or("candidate body receipt has no body sha256")?;
        if recorded_sha != sha256_hex(&body) {
            return Err("candidate body sha256 does not match its receipt".into());
        }
        let code_bytes = rows * (cols / 2);
        let scale_count = rows * (cols / GROUP_SIZE);
        let scale_bytes = scale_count * 2;
        if body.len() != code_bytes + scale_bytes {
            return Err(format!(
                "candidate body size {} does not match rows={} cols={} expected={}",
                body.len(),
                rows,
                cols,
                code_bytes + scale_bytes
            )
            .into());
        }
        let codes = body[..code_bytes].to_vec();
        let scales_bits = body[code_bytes..]
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect::<Vec<_>>();
        if scales_bits.len() != scale_count {
            return Err("candidate body scale count is invalid".into());
        }
        Ok(PackedQ4 {
            codes,
            scales_bits,
            candidate_bytes: body,
            source_rmse: 0.0,
            source_max_abs_error: 0.0,
        })
    }

    fn source_error(packed: &PackedQ4, raw: &[u8], rows: usize, cols: usize) -> (f64, f32) {
        let mut squared_error = 0.0f64;
        let mut max_abs_error = 0.0f32;
        let groups_per_row = cols / GROUP_SIZE;
        for row in 0..rows {
            for group in 0..groups_per_row {
                let group_base = row * groups_per_row + group;
                let scale = f16::from_bits(packed.scales_bits[group_base]).to_f32();
                let code_base = group_base * CODE_BYTES_PER_GROUP;
                for local in 0..GROUP_SIZE {
                    let byte = packed.codes[code_base + local / 2];
                    let nibble = if local % 2 == 0 {
                        byte & 0x0f
                    } else {
                        byte >> 4
                    };
                    let observed = (nibble as i32 - 8) as f32 * scale;
                    let error = bf16_at(raw, row * cols + group * GROUP_SIZE + local) - observed;
                    squared_error += f64::from(error) * f64::from(error);
                    max_abs_error = max_abs_error.max(error.abs());
                }
            }
        }
        ((squared_error / (rows * cols) as f64).sqrt(), max_abs_error)
    }

    fn deterministic_input(cols: usize) -> Vec<f32> {
        (0..cols)
            .map(|index| ((index * 71 % 509) as f32 - 254.0) / 509.0)
            .collect()
    }

    fn cpu_matvec(packed: &PackedQ4, rows: usize, cols: usize, input: &[f32]) -> Vec<f32> {
        let groups_per_row = cols / GROUP_SIZE;
        let mut output = vec![0.0f32; rows];
        for row in 0..rows {
            for group in 0..groups_per_row {
                let group_base = row * groups_per_row + group;
                let scale = f16::from_bits(packed.scales_bits[group_base]).to_f32();
                let code_base = group_base * CODE_BYTES_PER_GROUP;
                for local in 0..GROUP_SIZE {
                    let byte = packed.codes[code_base + local / 2];
                    let nibble = if local % 2 == 0 {
                        byte & 0x0f
                    } else {
                        byte >> 4
                    };
                    output[row] +=
                        (nibble as i32 - 8) as f32 * scale * input[group * GROUP_SIZE + local];
                }
            }
        }
        output
    }

    fn read_f32(buffer: &Buffer, count: usize) -> Vec<f32> {
        let pointer = buffer.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(pointer, count) }.to_vec()
    }

    fn dispatch(
        context: &MetalContext,
        codes: &Buffer,
        scales: &Buffer,
        input: &Buffer,
        output: &Buffer,
        rows: usize,
        cols: usize,
    ) -> Result<MetalDispatchTiming, Box<dyn Error>> {
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let groups_u32 = (cols / GROUP_SIZE) as u32;
        Ok(
            context.dispatch_threads_timed(
                KERNEL_NAME,
                (rows_u32, 1, 1),
                (1, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(codes), 0);
                    encoder.set_buffer(1, Some(scales), 0);
                    encoder.set_buffer(2, Some(input), 0);
                    encoder.set_buffer(3, Some(output), 0);
                    encoder.set_bytes(4, 4, &rows_u32 as *const u32 as *const _);
                    encoder.set_bytes(5, 4, &cols_u32 as *const u32 as *const _);
                    encoder.set_bytes(6, 4, &groups_u32 as *const u32 as *const _);
                },
            )?,
        )
    }

    fn gpu_ns(timing: MetalDispatchTiming) -> Result<u64, Box<dyn Error>> {
        match (timing.gpu_start_ns, timing.gpu_end_ns) {
            (Some(start), Some(end)) if end > start => Ok(end - start),
            _ => Err("Metal completed without a usable GPU timestamp".into()),
        }
    }

    fn output_metrics(reference: &[f32], observed: &[f32]) -> Value {
        let mut max_abs = 0.0f32;
        let mut squared = 0.0f64;
        let mut left_norm = 0.0f64;
        let mut right_norm = 0.0f64;
        let mut dot = 0.0f64;
        for (left, right) in reference.iter().zip(observed) {
            let error = *left - *right;
            max_abs = max_abs.max(error.abs());
            squared += f64::from(error) * f64::from(error);
            left_norm += f64::from(*left) * f64::from(*left);
            right_norm += f64::from(*right) * f64::from(*right);
            dot += f64::from(*left) * f64::from(*right);
        }
        let count = reference.len();
        json!({
            "count": count,
            "max_abs_error": max_abs,
            "rmse": (squared / count as f64).sqrt(),
            "cosine": if left_norm > 0.0 && right_norm > 0.0 { Some(dot / (left_norm.sqrt() * right_norm.sqrt())) } else { None },
            "finite": observed.iter().all(|value| value.is_finite()),
            "tolerance": OUTPUT_ERROR_TOLERANCE,
            "within_tolerance": max_abs <= OUTPUT_ERROR_TOLERANCE,
        })
    }

    fn write_atomic(path: &Path, value: &Value) -> Result<(), Box<dyn Error>> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let temporary = path.with_extension(format!("json.tmp-{}", std::process::id()));
        fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
        fs::rename(temporary, path)?;
        Ok(())
    }

    fn run(args: &Args) -> Result<Value, Box<dyn Error>> {
        let started = Instant::now();
        let root = args.root.canonicalize()?;
        let descriptor = load_noetic_descriptor(&args.descriptor, &args.tensor_name)?;
        let descriptor_path = descriptor.path.display().to_string();
        let descriptor_sha256 = descriptor.sha256.clone();
        let descriptor_schema = descriptor_string(&descriptor.body, "schema")?.to_owned();
        let descriptor_candidate_id =
            descriptor_string(&descriptor.body, "candidate_id")?.to_owned();
        let manifest = validate_manifest(&root)?;
        let location = load_tensor_location(&root, &args.tensor_name)?;
        let (expert_total, row_total, columns) = if location.shape.len() == 3 {
            (location.shape[0], location.shape[1], location.shape[2])
        } else {
            (1, location.shape[0], location.shape[1])
        };
        if args.expert_index >= expert_total
            || args.row_start >= row_total
            || args.row_count > row_total - args.row_start
        {
            return Err("selected matrix row window is outside the pinned tensor".into());
        }
        if args.row_count * columns * 2 > MAX_BLOCK_BYTES {
            return Err("selected source block exceeds the safety limit".into());
        }
        let before = fs::metadata(&location.shard)?;
        let raw = read_source_block(&location, args.expert_index, args.row_start, args.row_count)?;
        let source_hash = sha256_hex(&raw);
        let mut packed = if let Some(body_path) = &args.candidate_body {
            let receipt_path = args
                .body_receipt
                .as_ref()
                .ok_or("--body-receipt is required with --candidate-body")?;
            load_candidate_body(body_path, receipt_path, args.row_count, columns)?
        } else {
            pack_q4(&raw, args.row_count, columns)?
        };
        let (source_rmse, source_max_abs_error) =
            source_error(&packed, &raw, args.row_count, columns);
        packed.source_rmse = source_rmse;
        packed.source_max_abs_error = source_max_abs_error;
        let input = deterministic_input(columns);
        let input_hash = sha256_hex(bytemuck::cast_slice(&input));
        let expected = cpu_matvec(&packed, args.row_count, columns, &input);
        let context = MetalContext::new_with_trace(true)?;
        let codes = context.new_buffer_with_bytes_checked(&packed.codes)?;
        let scales =
            context.new_buffer_with_bytes_checked(bytemuck::cast_slice(&packed.scales_bits))?;
        let input_buffer = context.new_buffer_with_bytes_checked(bytemuck::cast_slice(&input))?;
        let output_buffer =
            context.new_buffer_checked(args.row_count * std::mem::size_of::<f32>())?;

        for _ in 0..args.warmup {
            let timing = dispatch(
                &context,
                &codes,
                &scales,
                &input_buffer,
                &output_buffer,
                args.row_count,
                columns,
            )?;
            let _ = gpu_ns(timing)?;
        }
        let mut gpu_samples = Vec::with_capacity(args.reps);
        let mut host_samples = Vec::with_capacity(args.reps);
        for _ in 0..args.reps {
            let host_started = Instant::now();
            let timing = dispatch(
                &context,
                &codes,
                &scales,
                &input_buffer,
                &output_buffer,
                args.row_count,
                columns,
            )?;
            host_samples.push(host_started.elapsed().as_nanos() as u64);
            gpu_samples.push(gpu_ns(timing)?);
        }
        let observed = read_f32(&output_buffer, args.row_count);
        let parity = output_metrics(&expected, &observed);
        if parity.get("finite").and_then(Value::as_bool) != Some(true)
            || parity.get("within_tolerance").and_then(Value::as_bool) != Some(true)
        {
            return Err(format!("Flash Q4 Metal parity failed: {parity}").into());
        }
        let after = fs::metadata(&location.shard)?;
        if before.len() != after.len() || before.modified().ok() != after.modified().ok() {
            return Err("source shard changed during the bounded probe".into());
        }
        let mut gpu_sorted = gpu_samples.clone();
        gpu_sorted.sort_unstable();
        let mut host_sorted = host_samples.clone();
        host_sorted.sort_unstable();
        let memory = context.device_memory_limits();
        let candidate_sha256 = sha256_hex(&packed.candidate_bytes);
        let values = args.row_count * columns;
        let elapsed_s = started.elapsed().as_secs_f64();
        let body_info = if let Some(body_path) = &args.candidate_body {
            let receipt_path = args.body_receipt.as_ref().unwrap();
            let receipt_bytes = fs::read(receipt_path.canonicalize()?)?;
            Some(json!({
                "path": body_path.canonicalize()?.display().to_string(),
                "sha256": candidate_sha256,
                "bytes": packed.candidate_bytes.len(),
                "receipt_path": receipt_path.canonicalize()?.display().to_string(),
                "receipt_sha256": sha256_hex(&receipt_bytes),
                "source_independent": true,
            }))
        } else {
            None
        };
        let has_body = body_info.is_some();
        let native_loader_status = if has_body {
            "BOUNDED_NOETIC_DESCRIPTOR_AND_BODY_LOAD"
        } else {
            "BOUNDED_NOETIC_DESCRIPTOR_LOAD"
        };
        let layout = if location.shape.len() == 3 {
            "row-major [expert, row, column]"
        } else {
            "row-major [row, column]"
        };
        let component_scope = if location.shape.len() == 3 {
            "one Flash routed-expert source block matvec"
        } else {
            "one Flash matrix row-block matvec"
        };
        Ok(json!({
            "schema": SCHEMA,
            "nomenclature_version": NOMENCLATURE_VERSION,
            "status": "PASSED",
            "repo": REPO_ID,
            "pinned_revision": PINNED_REVISION,
            "root": root,
            "model_lake_manifest": manifest,
            "source_label": "[V]",
            "derived_label": "[D]",
            "source_tensor": {
                "tensor_name": location.tensor_name,
                "dtype": location.dtype,
                "shape": location.shape,
                "shard": location.shard,
                "shard_name": location.shard_name,
                "shard_size": location.shard_size,
                "header_sha256": sha256_hex(&location.header_bytes),
                "index_sha256": location.index_sha256,
                "tensor_data_offsets": [location.data_begin, location.data_end],
                "selected_expert": if location.shape.len() == 3 { args.expert_index } else { 0 },
                "selected_row_start": args.row_start,
                "selected_row_count": args.row_count,
                "selected_block_bytes": raw.len(),
                "selected_block_sha256": source_hash,
                "selected_shard_full_hash_recomputed": false,
            },
            "noetic_descriptor": {
                "path": descriptor_path,
                "sha256": descriptor_sha256,
                "schema": descriptor_schema,
                "candidate_id": descriptor_candidate_id,
                "load_policy": "validated serialized descriptor; candidate body is not persisted by this bounded probe",
            },
            "noetic_representation": {
                "schema": "hcli.noetic.representation_descriptor.v1",
                "candidate_id": descriptor_candidate_id,
                "layout": layout,
                "group_size": GROUP_SIZE,
                "code_dtype": "uint4_packed",
                "nibble_order": "low_nibble_then_high_nibble_row_major",
                "code_offset": 8,
                "scale_dtype": "little_endian_float16",
                "scale_scope": "one scale per 64 source values",
                "candidate_bytes": packed.candidate_bytes.len(),
                "code_bytes": packed.codes.len(),
                "scale_bytes": packed.scales_bits.len() * 2,
                "effective_bits_per_value": packed.candidate_bytes.len() as f64 * 8.0 / values as f64,
                "candidate_sha256": candidate_sha256,
                "source_to_candidate_rmse": packed.source_rmse,
                "source_to_candidate_max_abs_error": packed.source_max_abs_error,
                "label": "[D]",
            },
            "candidate_body": body_info,
            "native_kernel": {
                "kernel": KERNEL_NAME,
                "shader_family": "qwen_uniform_q4.metal",
                "kernel_registered": true,
                "dispatches_per_sample": 1,
                "grid": [args.row_count, 1, 1],
                "threadgroup": [1, 1, 1],
                "scope": component_scope,
                "whole_model_capability": "NOT_TESTED",
                "whole_model_runtime": "NOT_TESTED",
                "label": "[V]/[D]",
            },
            "native_loader": {
                "status": native_loader_status,
                "descriptor_path": descriptor_path,
                "descriptor_sha256": descriptor_sha256,
                "candidate_id": descriptor_candidate_id,
                "source_block_streamed_for_reference": true,
                "candidate_body_persisted": has_body,
                "source_independent_execution": has_body,
                "dense_rematerialization": "forbidden",
                "whole_model_capability": "NOT_TESTED",
                "whole_model_runtime": "NOT_TESTED",
            },
            "gpu_timing": {
                "device": context.device_name(),
                "warmup_runs": args.warmup,
                "measured_runs": args.reps,
                "gpu_ns": gpu_samples,
                "gpu_ns_min": gpu_sorted[0],
                "gpu_ns_median": gpu_sorted[gpu_sorted.len() / 2],
                "gpu_ns_max": *gpu_sorted.last().unwrap(),
                "host_wall_ns": host_samples,
                "host_wall_ns_min": host_sorted[0],
                "host_wall_ns_median": host_sorted[host_sorted.len() / 2],
                "host_wall_ns_max": *host_sorted.last().unwrap(),
                "timing_authority": "Metal completed-command-buffer GPUStartTime/GPUEndTime; host wall reported separately",
                "memory_limits": {
                    "max_buffer_length": memory.max_buffer_length,
                    "recommended_max_working_set_size": memory.recommended_max_working_set_size,
                    "current_allocated_size": memory.current_allocated_size,
                    "has_unified_memory": memory.has_unified_memory,
                },
            },
            "parity": parity,
            "input": {
                "values": input.len(),
                "deterministic_sha256": input_hash,
                "definition": "((index * 71) mod 509 - 254) / 509",
            },
            "body_mutated": false,
            "model_loaded": false,
            "complete_system_ebpw": null,
            "flash_tps": null,
            "promotion_allowed": false,
            "claim_boundary": if has_body { "PASSED bounded source-independent Noetic component-body load, source-reference Q4 Metal kernel parity, and GPU timing only; no whole-model loader, capability, complete-token runtime, EBPW, or Flash TPS claim" } else { "PASSED bounded serialized Noetic descriptor load, source-tensor Q4 Metal kernel parity, and GPU timing only; no whole-model loader, capability, complete-token runtime, EBPW, or Flash TPS claim" },
            "next_action": if has_body { "compose the source-independent component body into the bounded Noetic graph while keeping protected complete-token timing gated" } else { "persist a source-independent component body, then compose it into the bounded Noetic graph while keeping protected complete-token timing gated" },
            "elapsed_s": elapsed_s,
        }))
    }

    pub fn main() -> Result<(), Box<dyn Error>> {
        let args = parse_args()?;
        let destination = args.out.clone();
        let report = match run(&args) {
            Ok(report) => report,
            Err(error) => json!({
                "schema": SCHEMA,
                "nomenclature_version": NOMENCLATURE_VERSION,
                "status": "FAILED",
                "repo": REPO_ID,
                "pinned_revision": PINNED_REVISION,
                "error": {"type": "ProbeError", "message": error.to_string()},
                "body_mutated": false,
                "model_loaded": false,
                "whole_model_capability": "NOT_TESTED",
                "whole_model_runtime": "NOT_TESTED",
                "promotion_allowed": false,
            }),
        };
        write_atomic(&destination, &report)?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        if report.get("status").and_then(Value::as_str) == Some("PASSED") {
            Ok(())
        } else {
            Err("Flash noetic Q4 kernel parity failed; see the receipt".into())
        }
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::main()
}
