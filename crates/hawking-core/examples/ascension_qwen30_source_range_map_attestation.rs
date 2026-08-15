#![allow(dead_code)] // The authority builder intentionally exposes a test-only synthetic surface.

//! Metadata-only Qwen30 source range-map authority for the streamed-oracle contract.
//!
//! This executable is deliberately narrower than a source oracle.  It reads
//! only the Hugging Face safetensors index and each shard's 8-byte safetensors
//! prefix plus JSON header.  It never reads a tensor payload byte, never maps
//! a shard, never instantiates a model, and has no GPU, server, HCLI, or lease
//! integration.  The resulting create-new authority is therefore useful for
//! constraining a future separately leased reader, but is not source-oracle,
//! numerical, coherence, throughput, or promotion evidence.

use serde::Serialize;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process;

const SCHEMA: &str =
    "hawking.ascension.qwen30_streamed_oracle_metadata_only_range_map_authority.v1";
const STATUS: &str = "PREPARED_QWEN30_STREAMED_ORACLE_SOURCE_RANGE_MAP_AUTHORITY_NOT_EXECUTED";
const SOURCE_MODEL_ID: &str = "Qwen3-Coder-30B-A3B-Instruct";
const SOURCE_INDEX_FILE: &str = "model.safetensors.index.json";
const SOURCE_INDEX_FORMAT: &str = "huggingface.safetensors.index.json";

const PREFIX_TOKEN_COUNT: u64 = 369;
const FORCED_TOKEN_ID: u64 = 949;
const TRACE_FORWARD_COUNT: u64 = PREFIX_TOKEN_COUNT + 1;
const LAYER_COUNT: u64 = 48;
const HIDDEN_SIZE: u64 = 2_048;
const VOCAB_ROWS: u64 = 151_936;
const ATTENTION_HEADS: u64 = 32;
const KV_HEADS: u64 = 4;
const HEAD_DIM: u64 = 128;
const EXPERT_COUNT: u64 = 128;
const TOP_K: u64 = 8;
const MOE_INTERMEDIATE: u64 = 768;
const BF16_BYTES: u64 = 2;
const ROW_TILE_ROWS: u64 = 128;
const EXPECTED_TENSOR_COUNT: usize = 18_867;
const MAX_SOURCE_INDEX_BYTES: u64 = 32 * 1024 * 1024;
const MAX_SAFETENSORS_HEADER_BYTES: u64 = 32 * 1024 * 1024;

#[derive(Debug)]
struct Args {
    source_root: PathBuf,
    source_revision: String,
    out: PathBuf,
}

fn usage() -> &'static str {
    "usage: ascension_qwen30_source_range_map_attestation --source-root ABSOLUTE_SOURCE_DIR --source-revision PINNED_40_HEX_REVISION --out NEW_ABSOLUTE_JSON"
}

fn parse_args_from<I>(arguments: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut source_root = None;
    let mut source_revision = None;
    let mut out = None;
    let mut values = arguments.into_iter();
    while let Some(flag) = values.next() {
        let destination = match flag.as_str() {
            "--source-root" => &mut source_root,
            "--source-revision" => &mut source_revision,
            "--out" => &mut out,
            "--help" | "-h" => return Err(usage().to_owned()),
            _ => return Err(format!("unsupported option {flag:?}; {}", usage())),
        };
        let value = values
            .next()
            .ok_or_else(|| format!("missing value for {flag}; {}", usage()))?;
        if destination.replace(value).is_some() {
            return Err(format!("{flag} was supplied more than once; {}", usage()));
        }
    }
    let source_root = PathBuf::from(
        source_root.ok_or_else(|| format!("--source-root is required; {}", usage()))?,
    );
    let source_revision =
        source_revision.ok_or_else(|| format!("--source-revision is required; {}", usage()))?;
    let out = PathBuf::from(out.ok_or_else(|| format!("--out is required; {}", usage()))?);
    if !source_root.is_absolute() {
        return Err("--source-root must be absolute".to_owned());
    }
    if !out.is_absolute() {
        return Err("--out must be absolute".to_owned());
    }
    if !is_pinned_revision(&source_revision) {
        return Err(
            "--source-revision must be exactly 40 lowercase hexadecimal characters".to_owned(),
        );
    }
    Ok(Args {
        source_root,
        source_revision,
        out,
    })
}

fn parse_args() -> Result<Args, String> {
    parse_args_from(env::args().skip(1))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_pinned_revision(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn checked_u64_add(left: u64, right: u64, label: &str) -> Result<u64, String> {
    left.checked_add(right)
        .ok_or_else(|| format!("{label} overflows u64"))
}

fn checked_u64_mul(left: u64, right: u64, label: &str) -> Result<u64, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("{label} overflows u64"))
}

fn checked_usize(value: u64, label: &str) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("{label} exceeds this platform"))
}

fn checked_relative_path(value: &str, label: &str) -> Result<PathBuf, String> {
    let path = Path::new(value);
    if value.is_empty() || path.is_absolute() {
        return Err(format!("{label} must be a non-empty relative path"));
    }
    if path
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!(
            "{label} must not contain parent/current/root components"
        ));
    }
    Ok(path.to_path_buf())
}

fn canonical_real_directory(path: &Path, label: &str) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err(format!("{label} must be absolute"));
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_dir() {
        return Err(format!("{label} must be a non-symlink directory"));
    }
    let canonical = fs::canonicalize(path)
        .map_err(|error| format!("cannot canonicalize {label} {}: {error}", path.display()))?;
    // A system-provided ancestor such as macOS's `/var` -> `/private/var`
    // alias is not a source-root escape.  The final root itself still must
    // not be a symlink, and all descendants are checked against this returned
    // canonical root below.
    Ok(canonical)
}

fn checked_regular_file_under_root(
    root: &Path,
    relative: &Path,
    label: &str,
) -> Result<PathBuf, String> {
    let joined = root.join(relative);
    let metadata = fs::symlink_metadata(&joined)
        .map_err(|error| format!("cannot stat {label} {}: {error}", joined.display()))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    let canonical = fs::canonicalize(&joined)
        .map_err(|error| format!("cannot canonicalize {label} {}: {error}", joined.display()))?;
    if !canonical.starts_with(root) {
        return Err(format!(
            "{label} contains a symlink or escapes the source root"
        ));
    }
    Ok(joined)
}

fn bounded_metadata_bytes(path: &Path, label: &str, maximum: u64) -> Result<Vec<u8>, String> {
    let metadata = fs::metadata(path)
        .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?;
    let length = metadata.len();
    if length == 0 || length > maximum {
        return Err(format!(
            "{label} must be 1..={maximum} bytes of metadata, observed {length}"
        ));
    }
    let mut file = File::open(path)
        .map_err(|error| format!("cannot open {label} {}: {error}", path.display()))?;
    let mut bytes = vec![0u8; checked_usize(length, label)?];
    file.read_exact(&mut bytes)
        .map_err(|error| format!("cannot read {label} {}: {error}", path.display()))?;
    if fs::metadata(path)
        .map_err(|error| format!("cannot restat {label} {}: {error}", path.display()))?
        .len()
        != length
    {
        return Err(format!("{label} changed while it was read"));
    }
    Ok(bytes)
}

/// This function stops at the end of the JSON header.  In particular, it has
/// no offset/length parameter that could be pointed into the tensor payload.
fn read_safetensors_header_metadata(path: &Path) -> Result<(u64, [u8; 8], Vec<u8>, u64), String> {
    let file_length = fs::metadata(path)
        .map_err(|error| format!("cannot stat safetensors shard {}: {error}", path.display()))?
        .len();
    if file_length < 9 {
        return Err(format!(
            "safetensors shard {} is too short for a header",
            path.display()
        ));
    }
    let mut file = File::open(path)
        .map_err(|error| format!("cannot open safetensors shard {}: {error}", path.display()))?;
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix).map_err(|error| {
        format!(
            "cannot read safetensors header prefix {}: {error}",
            path.display()
        )
    })?;
    let header_length = u64::from_le_bytes(prefix);
    if header_length == 0 || header_length > MAX_SAFETENSORS_HEADER_BYTES {
        return Err(format!(
            "safetensors shard {} header length must be 1..={MAX_SAFETENSORS_HEADER_BYTES}, observed {header_length}",
            path.display()
        ));
    }
    let data_start = checked_u64_add(8, header_length, "safetensors data start")?;
    if data_start > file_length {
        return Err(format!(
            "safetensors shard {} header exceeds file length",
            path.display()
        ));
    }
    let mut header = vec![0u8; checked_usize(header_length, "safetensors header length")?];
    file.read_exact(&mut header).map_err(|error| {
        format!(
            "cannot read safetensors JSON header {}: {error}",
            path.display()
        )
    })?;
    Ok((header_length, prefix, header, file_length))
}

fn json_object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be a JSON object"))
}

fn json_string<'a>(value: &'a Value, label: &str) -> Result<&'a str, String> {
    value
        .as_str()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{label} must be a non-empty string"))
}

fn json_u64(value: &Value, label: &str) -> Result<u64, String> {
    value
        .as_u64()
        .ok_or_else(|| format!("{label} must be a non-negative integer"))
}

fn shape_elements(shape: &[u64], label: &str) -> Result<u64, String> {
    if shape.is_empty() || shape.iter().any(|dimension| *dimension == 0) {
        return Err(format!("{label} must have a non-empty positive shape"));
    }
    shape.iter().try_fold(1u64, |count, dimension| {
        checked_u64_mul(count, *dimension, &format!("{label} element count"))
    })
}

#[derive(Clone, Debug)]
struct TensorSpec {
    tensor_name: String,
    role: &'static str,
    shape: Vec<u64>,
    window_access: &'static str,
    row_window_rows: u64,
    selected_expert_only: bool,
}

impl TensorSpec {
    fn new(
        tensor_name: impl Into<String>,
        role: &'static str,
        shape: Vec<u64>,
        window_access: &'static str,
        row_window_rows: u64,
        selected_expert_only: bool,
    ) -> Self {
        Self {
            tensor_name: tensor_name.into(),
            role,
            shape,
            window_access,
            row_window_rows,
            selected_expert_only,
        }
    }

    fn row_window_shape(&self) -> Vec<u64> {
        let mut shape = self.shape.clone();
        shape[0] = self.row_window_rows;
        shape
    }
}

fn canonical_specs() -> Vec<TensorSpec> {
    let mut specs = Vec::with_capacity(EXPECTED_TENSOR_COUNT);
    specs.push(TensorSpec::new(
        "model.embed_tokens.weight",
        "embedding_row",
        vec![VOCAB_ROWS, HIDDEN_SIZE],
        "exact_token_id_row_only",
        1,
        false,
    ));
    for layer in 0..LAYER_COUNT {
        let prefix = format!("model.layers.{layer}");
        specs.push(TensorSpec::new(
            format!("{prefix}.input_layernorm.weight"),
            "input_rmsnorm",
            vec![HIDDEN_SIZE],
            "full_vector",
            HIDDEN_SIZE,
            false,
        ));
        specs.push(TensorSpec::new(
            format!("{prefix}.self_attn.q_norm.weight"),
            "q_rmsnorm",
            vec![HEAD_DIM],
            "full_vector",
            HEAD_DIM,
            false,
        ));
        specs.push(TensorSpec::new(
            format!("{prefix}.self_attn.k_norm.weight"),
            "k_rmsnorm",
            vec![HEAD_DIM],
            "full_vector",
            HEAD_DIM,
            false,
        ));
        specs.push(TensorSpec::new(
            format!("{prefix}.self_attn.q_proj.weight"),
            "q_projection",
            vec![ATTENTION_HEADS * HEAD_DIM, HIDDEN_SIZE],
            "contiguous_row_tile",
            ROW_TILE_ROWS,
            false,
        ));
        specs.push(TensorSpec::new(
            format!("{prefix}.self_attn.k_proj.weight"),
            "k_projection",
            vec![KV_HEADS * HEAD_DIM, HIDDEN_SIZE],
            "contiguous_row_tile",
            ROW_TILE_ROWS,
            false,
        ));
        specs.push(TensorSpec::new(
            format!("{prefix}.self_attn.v_proj.weight"),
            "v_projection",
            vec![KV_HEADS * HEAD_DIM, HIDDEN_SIZE],
            "contiguous_row_tile",
            ROW_TILE_ROWS,
            false,
        ));
        specs.push(TensorSpec::new(
            format!("{prefix}.self_attn.o_proj.weight"),
            "o_projection",
            vec![HIDDEN_SIZE, ATTENTION_HEADS * HEAD_DIM],
            "contiguous_row_tile",
            ROW_TILE_ROWS,
            false,
        ));
        specs.push(TensorSpec::new(
            format!("{prefix}.post_attention_layernorm.weight"),
            "post_attention_rmsnorm",
            vec![HIDDEN_SIZE],
            "full_vector",
            HIDDEN_SIZE,
            false,
        ));
        specs.push(TensorSpec::new(
            format!("{prefix}.mlp.gate.weight"),
            "router_all_128_logits",
            vec![EXPERT_COUNT, HIDDEN_SIZE],
            "full_router_rows",
            EXPERT_COUNT,
            false,
        ));
        for expert in 0..EXPERT_COUNT {
            let expert_prefix = format!("{prefix}.mlp.experts.{expert}");
            specs.push(TensorSpec::new(
                format!("{expert_prefix}.gate_proj.weight"),
                "selected_expert_gate_projection",
                vec![MOE_INTERMEDIATE, HIDDEN_SIZE],
                "selected_route_expert_contiguous_row_tile",
                ROW_TILE_ROWS,
                true,
            ));
            specs.push(TensorSpec::new(
                format!("{expert_prefix}.up_proj.weight"),
                "selected_expert_up_projection",
                vec![MOE_INTERMEDIATE, HIDDEN_SIZE],
                "selected_route_expert_contiguous_row_tile",
                ROW_TILE_ROWS,
                true,
            ));
            specs.push(TensorSpec::new(
                format!("{expert_prefix}.down_proj.weight"),
                "selected_expert_down_projection",
                vec![HIDDEN_SIZE, MOE_INTERMEDIATE],
                "selected_route_expert_contiguous_row_tile",
                ROW_TILE_ROWS,
                true,
            ));
        }
    }
    specs.push(TensorSpec::new(
        "model.norm.weight",
        "final_rmsnorm",
        vec![HIDDEN_SIZE],
        "full_vector",
        HIDDEN_SIZE,
        false,
    ));
    specs.push(TensorSpec::new(
        "lm_head.weight",
        "lm_head_all_rows",
        vec![VOCAB_ROWS, HIDDEN_SIZE],
        "contiguous_row_tile",
        ROW_TILE_ROWS,
        false,
    ));
    debug_assert_eq!(specs.len(), EXPECTED_TENSOR_COUNT);
    specs
}

#[derive(Clone, Debug, Serialize)]
struct BoundTensor {
    tensor_name: String,
    role: &'static str,
    source_dtype: &'static str,
    full_shape: Vec<u64>,
    window_access: &'static str,
    row_window_shape: Vec<u64>,
    selected_expert_only: bool,
    dynamic_route_guard: bool,
    shard_relative_path: String,
    safetensors_relative_data_offsets: [u64; 2],
    absolute_data_offset: u64,
    data_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
struct ShardAuthority {
    relative_path: String,
    file_bytes: u64,
    safetensors_header_bytes: u64,
    safetensors_header_sha256: String,
    safetensors_prefix_sha256: String,
    shard_metadata_sha256: String,
    tensor_count: usize,
    payload_bytes_declared_by_header: u64,
    header_only_bytes_read: u64,
    tensor_payload_bytes_read: u64,
}

fn parse_weight_map(source_index: &Value) -> Result<BTreeMap<String, String>, String> {
    let root = json_object(source_index, "source safetensors index")?;
    let weight_map = root
        .get("weight_map")
        .ok_or_else(|| "source safetensors index lacks weight_map".to_owned())?;
    let weight_map = json_object(weight_map, "source safetensors index weight_map")?;
    let mut result = BTreeMap::new();
    for (tensor_name, path) in weight_map {
        if tensor_name.is_empty() {
            return Err("source safetensors index has an empty tensor name".to_owned());
        }
        let path = json_string(path, "source safetensors index shard path")?;
        checked_relative_path(path, "source safetensors index shard path")?;
        if result
            .insert(tensor_name.clone(), path.to_owned())
            .is_some()
        {
            return Err(format!(
                "source safetensors index duplicates tensor {tensor_name}"
            ));
        }
    }
    Ok(result)
}

fn expected_specs_by_shard(
    specs: &[TensorSpec],
    weight_map: &BTreeMap<String, String>,
) -> Result<BTreeMap<String, Vec<TensorSpec>>, String> {
    if specs.is_empty() {
        return Err("source-range specification must not be empty".to_owned());
    }
    if weight_map.len() != specs.len() {
        return Err(format!(
            "source index has {} tensors; this exact range-map scope requires {}",
            weight_map.len(),
            specs.len()
        ));
    }
    let mut expected_names = BTreeSet::new();
    let mut grouped = BTreeMap::<String, Vec<TensorSpec>>::new();
    for spec in specs {
        if !expected_names.insert(spec.tensor_name.as_str()) {
            return Err(format!(
                "internal Q30 specification duplicates {}",
                spec.tensor_name
            ));
        }
        let shard = weight_map.get(&spec.tensor_name).ok_or_else(|| {
            format!(
                "source index lacks Q30 streamed-oracle tensor {}",
                spec.tensor_name
            )
        })?;
        grouped.entry(shard.clone()).or_default().push(spec.clone());
    }
    for tensor_name in weight_map.keys() {
        if !expected_names.contains(tensor_name.as_str()) {
            return Err(format!(
                "source index contains tensor {tensor_name} outside the exact 369+forced Q30 contract"
            ));
        }
    }
    Ok(grouped)
}

fn header_tensor_entry<'a>(
    header: &'a Map<String, Value>,
    tensor_name: &str,
) -> Result<&'a Map<String, Value>, String> {
    let entry = header
        .get(tensor_name)
        .ok_or_else(|| format!("safetensors header lacks required tensor {tensor_name}"))?;
    json_object(entry, &format!("safetensors header tensor {tensor_name}"))
}

fn parse_header_shape(entry: &Map<String, Value>, tensor_name: &str) -> Result<Vec<u64>, String> {
    let values = entry
        .get("shape")
        .ok_or_else(|| format!("safetensors tensor {tensor_name} lacks shape"))?
        .as_array()
        .ok_or_else(|| format!("safetensors tensor {tensor_name} shape must be an array"))?;
    values
        .iter()
        .map(|value| {
            json_u64(
                value,
                &format!("safetensors tensor {tensor_name} shape dimension"),
            )
        })
        .collect()
}

fn parse_header_offsets(entry: &Map<String, Value>, tensor_name: &str) -> Result<[u64; 2], String> {
    let offsets = entry
        .get("data_offsets")
        .ok_or_else(|| format!("safetensors tensor {tensor_name} lacks data_offsets"))?
        .as_array()
        .ok_or_else(|| format!("safetensors tensor {tensor_name} offsets must be an array"))?;
    if offsets.len() != 2 {
        return Err(format!(
            "safetensors tensor {tensor_name} needs exactly two offsets"
        ));
    }
    let start = json_u64(
        &offsets[0],
        &format!("safetensors tensor {tensor_name} data start"),
    )?;
    let end = json_u64(
        &offsets[1],
        &format!("safetensors tensor {tensor_name} data end"),
    )?;
    if end < start {
        return Err(format!(
            "safetensors tensor {tensor_name} has inverted data offsets"
        ));
    }
    Ok([start, end])
}

fn bind_one_shard(
    root: &Path,
    relative_path: &str,
    specs: &[TensorSpec],
) -> Result<(ShardAuthority, Vec<BoundTensor>), String> {
    let relative = checked_relative_path(relative_path, "source index shard path")?;
    let shard_path = checked_regular_file_under_root(root, &relative, "source safetensors shard")?;
    let (header_length, prefix, header_bytes, file_length) =
        read_safetensors_header_metadata(&shard_path)?;
    let data_start = checked_u64_add(8, header_length, "safetensors data start")?;
    let payload_length = file_length
        .checked_sub(data_start)
        .ok_or_else(|| "safetensors header exceeds its shard".to_owned())?;
    let header_value: Value = serde_json::from_slice(&header_bytes).map_err(|error| {
        format!(
            "safetensors shard {} has invalid JSON header: {error}",
            shard_path.display()
        )
    })?;
    let header = json_object(&header_value, "safetensors header")?;
    let expected_names = specs
        .iter()
        .map(|spec| spec.tensor_name.as_str())
        .collect::<BTreeSet<_>>();
    for name in header.keys() {
        if name == "__metadata__" {
            continue;
        }
        if !expected_names.contains(name.as_str()) {
            return Err(format!(
                "safetensors header {} names tensor {name} outside the exact source range map",
                relative_path
            ));
        }
    }

    let mut intervals = Vec::<(u64, u64, String)>::with_capacity(specs.len());
    let mut bound = Vec::with_capacity(specs.len());
    for spec in specs {
        let entry = header_tensor_entry(header, &spec.tensor_name)?;
        let dtype = json_string(
            entry
                .get("dtype")
                .ok_or_else(|| format!("safetensors tensor {} lacks dtype", spec.tensor_name))?,
            &format!("safetensors tensor {} dtype", spec.tensor_name),
        )?;
        if dtype != "BF16" {
            return Err(format!(
                "safetensors tensor {} dtype is {dtype:?}, expected BF16",
                spec.tensor_name
            ));
        }
        let observed_shape = parse_header_shape(entry, &spec.tensor_name)?;
        if observed_shape != spec.shape {
            return Err(format!(
                "safetensors tensor {} shape {:?} differs from required {:?}",
                spec.tensor_name, observed_shape, spec.shape
            ));
        }
        let offsets = parse_header_offsets(entry, &spec.tensor_name)?;
        let expected_bytes = checked_u64_mul(
            shape_elements(&spec.shape, &format!("{} shape", spec.tensor_name))?,
            BF16_BYTES,
            &format!("{} BF16 byte count", spec.tensor_name),
        )?;
        let observed_bytes = offsets[1] - offsets[0];
        if observed_bytes != expected_bytes {
            return Err(format!(
                "safetensors tensor {} offset span {observed_bytes} differs from BF16 shape byte count {expected_bytes}",
                spec.tensor_name
            ));
        }
        if offsets[1] > payload_length {
            return Err(format!(
                "safetensors tensor {} offsets escape shard payload metadata",
                spec.tensor_name
            ));
        }
        let absolute_data_offset =
            checked_u64_add(data_start, offsets[0], "absolute tensor data offset")?;
        intervals.push((offsets[0], offsets[1], spec.tensor_name.clone()));
        bound.push(BoundTensor {
            tensor_name: spec.tensor_name.clone(),
            role: spec.role,
            source_dtype: "BF16",
            full_shape: spec.shape.clone(),
            window_access: spec.window_access,
            row_window_shape: spec.row_window_shape(),
            selected_expert_only: spec.selected_expert_only,
            dynamic_route_guard: spec.selected_expert_only,
            shard_relative_path: relative_path.to_owned(),
            safetensors_relative_data_offsets: offsets,
            absolute_data_offset,
            data_bytes: observed_bytes,
        });
    }
    intervals.sort_by_key(|(start, _, _)| *start);
    let mut previous_end = 0u64;
    for (start, end, name) in &intervals {
        if *start != previous_end {
            return Err(format!(
                "safetensors tensor {name} has a gap or overlap before offset {start}; exact contiguous range mapping is required"
            ));
        }
        previous_end = *end;
    }
    if previous_end != payload_length {
        return Err(format!(
            "safetensors shard {relative_path} payload metadata ends at {previous_end}, expected {payload_length}"
        ));
    }
    bound.sort_by(|left, right| left.tensor_name.cmp(&right.tensor_name));
    let metadata_material = json!({
        "relative_path": relative_path,
        "file_bytes": file_length,
        "safetensors_header_bytes": header_length,
        "safetensors_header_sha256": sha256_hex(&header_bytes),
        "safetensors_prefix_sha256": sha256_hex(&prefix),
        "tensor_offsets": intervals.iter().map(|(start, end, name)| {
            json!({"tensor_name": name, "start": start, "end": end})
        }).collect::<Vec<_>>(),
    });
    let metadata_bytes = serde_json::to_vec(&metadata_material)
        .map_err(|error| format!("cannot serialize shard metadata binding: {error}"))?;
    let authority = ShardAuthority {
        relative_path: relative_path.to_owned(),
        file_bytes: file_length,
        safetensors_header_bytes: header_length,
        safetensors_header_sha256: sha256_hex(&header_bytes),
        safetensors_prefix_sha256: sha256_hex(&prefix),
        shard_metadata_sha256: sha256_hex(&metadata_bytes),
        tensor_count: bound.len(),
        payload_bytes_declared_by_header: payload_length,
        header_only_bytes_read: data_start,
        tensor_payload_bytes_read: 0,
    };
    Ok((authority, bound))
}

fn build_authority_from_specs(
    source_root: &Path,
    source_revision: &str,
    specs: &[TensorSpec],
) -> Result<Value, String> {
    let source_root = canonical_real_directory(source_root, "source root")?;
    if !is_pinned_revision(source_revision) {
        return Err(
            "source revision must be exactly 40 lowercase hexadecimal characters".to_owned(),
        );
    }
    let index_relative = checked_relative_path(SOURCE_INDEX_FILE, "source index path")?;
    let index_path =
        checked_regular_file_under_root(&source_root, &index_relative, "source safetensors index")?;
    let index_bytes = bounded_metadata_bytes(
        &index_path,
        "source safetensors index",
        MAX_SOURCE_INDEX_BYTES,
    )?;
    let index_value: Value = serde_json::from_slice(&index_bytes)
        .map_err(|error| format!("source safetensors index is invalid JSON: {error}"))?;
    let weight_map = parse_weight_map(&index_value)?;
    let grouped_specs = expected_specs_by_shard(specs, &weight_map)?;

    let mut shards = Vec::with_capacity(grouped_specs.len());
    let mut tensors = Vec::with_capacity(specs.len());
    for (relative_path, shard_specs) in &grouped_specs {
        let (shard, mut shard_tensors) = bind_one_shard(&source_root, relative_path, shard_specs)?;
        shards.push(shard);
        tensors.append(&mut shard_tensors);
    }
    tensors.sort_by(|left, right| left.tensor_name.cmp(&right.tensor_name));
    if tensors.len() != specs.len() {
        return Err("metadata range-map tensor count drifted while binding shards".to_owned());
    }
    let source_weight_bytes_from_shard_metadata = shards.iter().try_fold(0u64, |sum, shard| {
        checked_u64_add(sum, shard.file_bytes, "source shard metadata byte sum")
    })?;
    let all_header_only_bytes_read = shards.iter().try_fold(0u64, |sum, shard| {
        checked_u64_add(
            sum,
            shard.header_only_bytes_read,
            "source header metadata byte sum",
        )
    })?;
    let material = json!({
        "schema": SCHEMA,
        "status": STATUS,
        "source": {
            "model_id": SOURCE_MODEL_ID,
            "source_revision": source_revision,
            "source_root_absolute": source_root,
            "source_index": {
                "relative_path": SOURCE_INDEX_FILE,
                "format": SOURCE_INDEX_FORMAT,
                "bytes": index_bytes.len(),
                "sha256": sha256_hex(&index_bytes),
                "weight_map_tensor_count": weight_map.len(),
            },
            "source_shard_count": shards.len(),
            "source_shard_file_bytes_from_metadata": source_weight_bytes_from_shard_metadata,
            "source_tensor_count": tensors.len(),
        },
        "exact_streamed_oracle_scope": {
            "source_template_token_count": PREFIX_TOKEN_COUNT,
            "forced_identical_continuation_token_id": FORCED_TOKEN_ID,
            "total_forwards_per_replay_arm": TRACE_FORWARD_COUNT,
            "layers": LAYER_COUNT,
            "top_k_routes_per_token": TOP_K,
            "all_128_expert_tensor_names_are_pre-authorized_but_payload_reads_must_still_be_route_selected": true,
            "row_tile_rows": ROW_TILE_ROWS,
            "sampling_or_autoregressive_feedback_forbidden": true,
        },
        "metadata_access_boundary": {
            "source_index_bytes_read": index_bytes.len(),
            "safetensors_prefix_and_json_header_bytes_read": all_header_only_bytes_read,
            "source_tensor_payload_bytes_read": 0,
            "tensor_payload_hashes_collected": false,
            "whole_shard_payload_checksum_collected": false,
            "mmap_or_memory_map_used": false,
            "source_model_instantiated": false,
            "gpu_or_metal_invoked": false,
            "server_started": false,
            "hcli_invoked": false,
            "lease_requested": false,
        },
        "path_safety": {
            "source_root_must_be_absolute_real_non_symlink_directory": true,
            "index_and_shards_must_be_regular_non_symlink_files_under_source_root": true,
            "relative_paths_reject_parent_current_and_root_components": true,
            "canonical_paths_must_not_escape_source_root": true,
        },
        "shards": shards,
        "tensors": tensors,
        "claim_boundary": "Prepared metadata-only range-map authority. It does not execute a source oracle, read any source tensor payload, prove numerical semantics, establish coherence, or report HCLI, TG, TPS, tournament, or serving capability.",
    });
    let material_bytes = serde_json::to_vec(&material).map_err(|error| {
        format!("cannot serialize source range-map authority material: {error}")
    })?;
    Ok(json!({
        "authority_content_sha256": sha256_hex(&material_bytes),
        "authority": material,
    }))
}

fn write_new_json(path: &Path, value: &Value) -> Result<(), String> {
    if !path.is_absolute() {
        return Err("output path must be absolute".to_owned());
    }
    let parent = path
        .parent()
        .ok_or_else(|| "output path must have a parent directory".to_owned())?;
    let canonical_parent = canonical_real_directory(parent, "output parent")?;
    let file_name = path
        .file_name()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| "output path must name a file".to_owned())?;
    let canonical_output = canonical_parent.join(file_name);
    if fs::symlink_metadata(&canonical_output).is_ok() {
        return Err(format!(
            "output path already exists: {}",
            canonical_output.display()
        ));
    }
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("cannot serialize source range-map authority: {error}"))?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&canonical_output)
        .map_err(|error| {
            format!(
                "cannot create new output {}: {error}",
                canonical_output.display()
            )
        })?;
    file.write_all(&bytes)
        .and_then(|_| file.write_all(b"\n"))
        .and_then(|_| file.sync_all())
        .map_err(|error| {
            format!(
                "cannot durably write output {}: {error}",
                canonical_output.display()
            )
        })
}

fn run(arguments: Args) -> Result<Value, String> {
    let specs = canonical_specs();
    if specs.len() != EXPECTED_TENSOR_COUNT {
        return Err(
            "internal Q30 source-range specification has the wrong tensor count".to_owned(),
        );
    }
    let authority =
        build_authority_from_specs(&arguments.source_root, &arguments.source_revision, &specs)?;
    write_new_json(&arguments.out, &authority)?;
    Ok(authority)
}

fn main() {
    match parse_args().and_then(run) {
        Ok(document) => match serde_json::to_string_pretty(&document) {
            Ok(text) => println!("{text}"),
            Err(error) => {
                eprintln!("cannot serialize authority result: {error}");
                process::exit(1);
            }
        },
        Err(error) => {
            eprintln!("{error}");
            process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn fixture_specs() -> Vec<TensorSpec> {
        vec![
            TensorSpec::new(
                "tensor.a",
                "fixture_a",
                vec![2, 2],
                "contiguous_row_tile",
                1,
                false,
            ),
            TensorSpec::new("tensor.b", "fixture_b", vec![2], "full_vector", 2, false),
        ]
    }

    fn write_fixture(
        dtype_a: &str,
        shape_a: Value,
        offsets_a: Value,
        index_path_a: &str,
    ) -> (TempDir, PathBuf) {
        let directory = TempDir::new().expect("temporary fixture directory");
        let root = directory.path().join("source");
        fs::create_dir(&root).expect("create fixture source root");
        let header = serde_json::to_vec(&json!({
            "tensor.a": {"dtype": dtype_a, "shape": shape_a, "data_offsets": offsets_a},
            "tensor.b": {"dtype": "BF16", "shape": [2], "data_offsets": [8, 12]},
        }))
        .expect("serialize synthetic safetensors header");
        let shard_path = root.join("model-00001-of-00001.safetensors");
        let mut shard = File::create(&shard_path).expect("create synthetic shard");
        shard
            .write_all(&(header.len() as u64).to_le_bytes())
            .and_then(|_| shard.write_all(&header))
            .expect("write synthetic safetensors metadata");
        shard
            .set_len(8 + header.len() as u64 + 12)
            .expect("declare synthetic payload extent without writing it");
        let index = json!({
            "metadata": {"total_size": 12},
            "weight_map": {
                "tensor.a": index_path_a,
                "tensor.b": "model-00001-of-00001.safetensors",
            }
        });
        fs::write(
            root.join(SOURCE_INDEX_FILE),
            serde_json::to_vec(&index).expect("serialize synthetic index"),
        )
        .expect("write synthetic source index");
        (directory, root)
    }

    #[test]
    fn canonical_q30_domain_is_exactly_the_streamed_oracle_tensor_set() {
        let specs = canonical_specs();
        assert_eq!(specs.len(), EXPECTED_TENSOR_COUNT);
        assert_eq!(
            specs
                .iter()
                .filter(|spec| spec.selected_expert_only)
                .count(),
            (LAYER_COUNT * EXPERT_COUNT * 3) as usize
        );
        assert!(specs.iter().any(|spec| {
            spec.tensor_name == "model.embed_tokens.weight"
                && spec.window_access == "exact_token_id_row_only"
                && spec.row_window_rows == 1
        }));
        assert!(specs.iter().any(|spec| {
            spec.tensor_name == "model.layers.47.mlp.experts.127.down_proj.weight"
                && spec.shape == vec![HIDDEN_SIZE, MOE_INTERMEDIATE]
                && spec.selected_expert_only
        }));
        assert!(specs.iter().any(|spec| {
            spec.tensor_name == "lm_head.weight"
                && spec.shape == vec![VOCAB_ROWS, HIDDEN_SIZE]
                && spec.row_window_rows == ROW_TILE_ROWS
        }));
    }

    #[test]
    fn metadata_only_fixture_binds_index_headers_shapes_and_offsets() {
        let (_directory, root) = write_fixture(
            "BF16",
            json!([2, 2]),
            json!([0, 8]),
            "model-00001-of-00001.safetensors",
        );
        let authority = build_authority_from_specs(&root, &"a".repeat(40), &fixture_specs())
            .expect("metadata-only fixture authority");
        assert_eq!(authority["authority"]["status"], STATUS);
        assert_eq!(
            authority["authority"]["metadata_access_boundary"]["source_tensor_payload_bytes_read"],
            0
        );
        assert_eq!(
            authority["authority"]["metadata_access_boundary"]["mmap_or_memory_map_used"],
            false
        );
        assert_eq!(authority["authority"]["source"]["source_tensor_count"], 2);
        assert_eq!(
            authority["authority"]["shards"][0]["tensor_payload_bytes_read"],
            0
        );
        assert_eq!(authority["authority"]["tensors"][0]["source_dtype"], "BF16");
        assert_eq!(authority["authority"]["tensors"][0]["data_bytes"], 8);
        assert!(authority["authority_content_sha256"]
            .as_str()
            .is_some_and(|value| value.len() == 64));
    }

    #[test]
    fn metadata_authority_rejects_dtype_shape_and_offset_drift() {
        let (_directory, root) = write_fixture(
            "F32",
            json!([2, 2]),
            json!([0, 8]),
            "model-00001-of-00001.safetensors",
        );
        let error = build_authority_from_specs(&root, &"a".repeat(40), &fixture_specs())
            .expect_err("F32 must be rejected");
        assert!(error.contains("expected BF16"));

        let (_directory, root) = write_fixture(
            "BF16",
            json!([4]),
            json!([0, 8]),
            "model-00001-of-00001.safetensors",
        );
        let error = build_authority_from_specs(&root, &"a".repeat(40), &fixture_specs())
            .expect_err("shape mismatch must be rejected");
        assert!(error.contains("shape"));

        let (_directory, root) = write_fixture(
            "BF16",
            json!([2, 2]),
            json!([0, 6]),
            "model-00001-of-00001.safetensors",
        );
        let error = build_authority_from_specs(&root, &"a".repeat(40), &fixture_specs())
            .expect_err("offset span mismatch must be rejected");
        assert!(error.contains("offset span"));
    }

    #[test]
    fn metadata_authority_rejects_path_escape_and_create_new_replay() {
        let (_directory, root) = write_fixture(
            "BF16",
            json!([2, 2]),
            json!([0, 8]),
            "../outside.safetensors",
        );
        let error = build_authority_from_specs(&root, &"a".repeat(40), &fixture_specs())
            .expect_err("path escape must be rejected");
        assert!(error.contains("parent/current/root"));

        let (directory, root) = write_fixture(
            "BF16",
            json!([2, 2]),
            json!([0, 8]),
            "model-00001-of-00001.safetensors",
        );
        let authority = build_authority_from_specs(&root, &"a".repeat(40), &fixture_specs())
            .expect("valid synthetic authority");
        let output = directory.path().join("authority.json");
        write_new_json(&output, &authority).expect("first create-new write");
        let error = write_new_json(&output, &authority).expect_err("replay output must fail");
        assert!(error.contains("already exists"));
    }

    #[cfg(unix)]
    #[test]
    fn metadata_authority_rejects_symlinked_shards() {
        use std::os::unix::fs::symlink;

        let (directory, root) = write_fixture(
            "BF16",
            json!([2, 2]),
            json!([0, 8]),
            "model-00001-of-00001.safetensors",
        );
        let real = root.join("real.safetensors");
        fs::rename(root.join("model-00001-of-00001.safetensors"), &real)
            .expect("rename fixture shard");
        symlink(&real, root.join("model-00001-of-00001.safetensors"))
            .expect("create fixture symlink");
        let error = build_authority_from_specs(&root, &"a".repeat(40), &fixture_specs())
            .expect_err("symlinked shard must be rejected");
        assert!(error.contains("non-symlink"));
        drop(directory);
    }

    #[test]
    fn cli_requires_absolute_create_new_authority_inputs() {
        let error = parse_args_from([
            "--source-root".to_owned(),
            "relative".to_owned(),
            "--source-revision".to_owned(),
            "a".repeat(40),
            "--out".to_owned(),
            "/tmp/out.json".to_owned(),
        ])
        .expect_err("relative source root must fail");
        assert!(error.contains("source-root must be absolute"));
    }
}
