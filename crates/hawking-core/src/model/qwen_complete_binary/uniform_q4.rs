//! Diagnostic HQ30UQ4 (uniform Q4 group-64 / group-128) complete-body admission.
//!
//! Layout matches `qwen_uniform_q4.metal`:
//!   * magic `HQ30UQ4\\0`, version 1, group_size 64 or 128
//!   * FP16 scale per group of `group_size` flat elements
//!   * `group_size/2` code bytes per group (even nibble low, odd high; q = nibble - 8)
//!
//! The sealed G0 admission path still requires group_size=64. The parser
//! accepts 128 so a group-128 payload can bind `geo_tpr64`. The runtime must
//! dispatch `qwen_uniform_q4_*` kernels, never binary matvec.

use super::*;
use std::sync::Arc;

pub const UNIFORM_Q4_MAGIC: [u8; 8] = *b"HQ30UQ4\0";
pub const UNIFORM_Q4_VERSION: u32 = 1;
pub const UNIFORM_Q4_GROUP_SIZE: usize = 64;
pub const UNIFORM_Q4_GROUP_SIZE_128: usize = 128;
pub const UNIFORM_Q4_CODE_BYTES_PER_GROUP: usize = UNIFORM_Q4_GROUP_SIZE / 2;
/// Group sizes the HQ30UQ4 header parser will admit. Unsupported sizes still
/// refuse. The sealed G0 admission contract remains group 64 only.
pub const UNIFORM_Q4_SUPPORTED_GROUP_SIZES: [usize; 2] =
    [UNIFORM_Q4_GROUP_SIZE, UNIFORM_Q4_GROUP_SIZE_128];

pub fn uniform_q4_group_size_supported(group_size: usize) -> bool {
    UNIFORM_Q4_SUPPORTED_GROUP_SIZES.contains(&group_size)
}

pub fn uniform_q4_code_bytes_per_group(group_size: usize) -> Result<usize> {
    if !uniform_q4_group_size_supported(group_size) {
        return Err(Error::Model(format!(
            "uniform Q4 group_size={group_size} must be 64 or 128"
        )));
    }
    Ok(group_size / 2)
}
pub const QWEN30_UNIFORM_Q4_SCHEMA: &str =
    "hawking.ascension.qwen30_uniform_q4_group64_candidate.v1";
pub const QWEN30_UNIFORM_Q4_CANDIDATE_STATUS: &str =
    "CANDIDATE_UNIFORM_Q4_GROUP64_DIAGNOSTIC_UNQUALIFIED";
pub const QWEN30_UNIFORM_Q4_MODEL_ID: &str =
    "Qwen3-Coder-30B-A3B-Instruct-uniform-q4-group64-v1";

/// Admission bindings for the Lane-N uniform-Q4 diagnostic candidate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen30UniformQ4Admission {
    pub expected_manifest_seal_sha256: String,
    pub expected_source_audit_seal_sha256: String,
    pub expected_source_revision: String,
    pub expected_revalidation_path: PathBuf,
    pub expected_revalidation_seal_sha256: String,
    pub expected_terminal_path: PathBuf,
    pub expected_terminal_seal_sha256: String,
}

/// Parse one HQ30UQ4 payload into the shared header geometry.
///
/// `sign_offset` is the code-body offset (same field, different encoding).
pub fn parse_uniform_q4_header(payload: &[u8]) -> Result<CompleteBinaryHeader> {
    if payload.len() < COMPLETE_BINARY_HEADER_BYTES {
        return Err(Error::Model(format!(
            "uniform Q4 payload is {} bytes; header requires {COMPLETE_BINARY_HEADER_BYTES}",
            payload.len()
        )));
    }
    if payload[..8] != UNIFORM_Q4_MAGIC {
        return Err(Error::Model(
            "uniform Q4 magic does not match HQ30UQ4".into(),
        ));
    }
    let version = read_u32(payload, 8)?;
    if version != UNIFORM_Q4_VERSION {
        return Err(Error::Model(format!(
            "uniform Q4 version {version} is unsupported (expected {UNIFORM_Q4_VERSION})"
        )));
    }
    let group_size = read_u32(payload, 12)? as usize;
    if !uniform_q4_group_size_supported(group_size) {
        return Err(Error::Model(format!(
            "uniform Q4 group_size={group_size} must be 64 or 128"
        )));
    }
    let rank = read_u16(payload, 16)? as usize;
    let reserved = read_u16(payload, 18)?;
    let element_u64 = read_u64(payload, 20)?;
    let reserved_tail = read_u32(payload, 28)?;
    if reserved != 0 || reserved_tail != 0 {
        return Err(Error::Model(
            "uniform Q4 reserved header fields must be zero".into(),
        ));
    }
    if rank == 0 {
        return Err(Error::Model(
            "uniform Q4 tensor rank must be positive".into(),
        ));
    }
    let dimensions_offset = COMPLETE_BINARY_HEADER_BYTES;
    let dimensions_bytes = rank
        .checked_mul(4)
        .ok_or_else(|| Error::Model("uniform Q4 rank byte count overflow".into()))?;
    let after_dimensions = dimensions_offset
        .checked_add(dimensions_bytes)
        .ok_or_else(|| Error::Model("uniform Q4 dimension offset overflow".into()))?;
    if after_dimensions > payload.len() {
        return Err(Error::Model(
            "uniform Q4 is truncated in tensor dimensions".into(),
        ));
    }
    let mut shape = Vec::with_capacity(rank);
    let mut derived_elements = 1usize;
    for dimension in 0..rank {
        let value = read_u32(payload, dimensions_offset + dimension * 4)? as usize;
        if value == 0 {
            return Err(Error::Model(
                "uniform Q4 tensor dimensions must be positive".into(),
            ));
        }
        derived_elements = derived_elements
            .checked_mul(value)
            .ok_or_else(|| Error::Model("uniform Q4 tensor element count overflow".into()))?;
        shape.push(value);
    }
    let elements = usize::try_from(element_u64)
        .map_err(|_| Error::Model("uniform Q4 element count exceeds this platform".into()))?;
    if elements != derived_elements {
        return Err(Error::Model(format!(
            "uniform Q4 element count {elements} disagrees with shape product {derived_elements}"
        )));
    }
    let groups = elements
        .checked_add(group_size - 1)
        .ok_or_else(|| Error::Model("uniform Q4 group count overflow".into()))?
        / group_size;
    let scale_bytes = groups
        .checked_mul(2)
        .ok_or_else(|| Error::Model("uniform Q4 scale byte count overflow".into()))?;
    let code_bytes_per_group = uniform_q4_code_bytes_per_group(group_size)?;
    let code_bytes = groups
        .checked_mul(code_bytes_per_group)
        .ok_or_else(|| Error::Model("uniform Q4 code byte count overflow".into()))?;
    let scale_offset = after_dimensions;
    let sign_offset = scale_offset
        .checked_add(scale_bytes)
        .ok_or_else(|| Error::Model("uniform Q4 code offset overflow".into()))?;
    let payload_bytes = sign_offset
        .checked_add(code_bytes)
        .ok_or_else(|| Error::Model("uniform Q4 payload byte count overflow".into()))?;
    if payload_bytes != payload.len() {
        return Err(Error::Model(format!(
            "uniform Q4 payload size {} does not equal fixed-layout expectation {payload_bytes}",
            payload.len()
        )));
    }
    for group in 0..groups {
        let scale = f16::from_bits(read_u16(payload, scale_offset + group * 2)?);
        if !scale.is_finite() {
            return Err(Error::Model(format!(
                "uniform Q4 scale for group {group} is not finite"
            )));
        }
    }
    Ok(CompleteBinaryHeader {
        version,
        group_size,
        shape,
        elements,
        groups,
        scale_offset,
        sign_offset,
        payload_bytes,
    })
}

fn expected_q4_tensor_path(root: &Path, tensor_name: &str) -> Result<PathBuf> {
    let tensors = root.join("tensors");
    let metadata = fs::symlink_metadata(&tensors).map_err(|error| {
        Error::Model(format!(
            "uniform Q4 artifact root: cannot stat {}: {error}",
            tensors.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(Error::Model(format!(
            "uniform Q4 artifact root: {} must be a non-symlink tensors directory",
            tensors.display()
        )));
    }
    let filename = format!("{}.hq30uq4", sha256_hex(tensor_name.as_bytes()));
    Ok(tensors.join(filename))
}

fn validate_q4_tensor_row(
    row: &Map<String, Value>,
    root: &Path,
    source: &SourceChain,
) -> Result<(CompleteBinaryTensor, Arc<[u8]>)> {
    let label = "uniform Q4 manifest tensor";
    let tensor_name = required_string(row, "tensor_name", label)?;
    if tensor_name.contains('\0') {
        return Err(Error::Model(format!("{label}: tensor_name contains a NUL byte")));
    }
    let source_shard = required_string(row, "source_shard", label)?;
    require_safe_filename(source_shard, label)?;
    let source_shard_sha256 = required_sha256(row, "source_shard_sha256", label)?;
    let expected_source_hash = source.shard_hashes.get(source_shard).ok_or_else(|| {
        Error::Model(format!(
            "{label}: source shard {source_shard:?} is not in the sealed source receipt"
        ))
    })?;
    if &source_shard_sha256 != expected_source_hash {
        return Err(Error::Model(format!(
            "{label}: source shard hash for {tensor_name:?} differs from the source receipt"
        )));
    }
    if source.weight_map.get(tensor_name).map(String::as_str) != Some(source_shard) {
        return Err(Error::Model(format!(
            "{label}: source index does not bind tensor {tensor_name:?} to shard {source_shard:?}"
        )));
    }
    let source_dtype = required_string(row, "source_dtype", label)?;
    if !matches!(
        source_dtype,
        "BF16" | "BFLOAT16" | "F32" | "FLOAT32" | "F16" | "FLOAT16"
    ) {
        return Err(Error::Model(format!(
            "{label}: tensor {tensor_name:?} has unsupported source_dtype {source_dtype:?}"
        )));
    }
    let shape = declared_tensor_shape(row, label)?;
    let declared_elements = required_u64(row, "elements", label)?;
    let shape_elements = shape.iter().try_fold(1u64, |total, dimension| {
        total
            .checked_mul(u64::try_from(*dimension).unwrap_or(u64::MAX))
            .ok_or_else(|| Error::Model(format!("{label}: shape element product overflows u64")))
    })?;
    if declared_elements == 0 || declared_elements != shape_elements {
        return Err(Error::Model(format!(
            "{label}: tensor {tensor_name:?} elements does not equal its shape product"
        )));
    }

    let expected_path = expected_q4_tensor_path(root, tensor_name)?;
    let expected_path = canonical_regular_path(&expected_path, label)?;
    let declared_path =
        manifest_descendant_file(root, required_string(row, "artifact_path", label)?, label)?;
    if declared_path != expected_path {
        return Err(Error::Model(format!(
            "{label}: tensor {tensor_name:?} artifact_path does not equal its deterministic tensor path"
        )));
    }
    let payload = read_regular_file(&expected_path, label)?;
    if required_u64(row, "artifact_bytes", label)?
        != u64::try_from(payload.len()).unwrap_or(u64::MAX)
    {
        return Err(Error::Model(format!(
            "{label}: tensor {tensor_name:?} artifact_bytes does not equal physical payload bytes"
        )));
    }
    let artifact_sha256 = required_sha256(row, "artifact_sha256", label)?;
    if sha256_hex(&payload) != artifact_sha256 {
        return Err(Error::Model(format!(
            "{label}: tensor {tensor_name:?} payload SHA-256 does not match the manifest"
        )));
    }
    let layout = required_object(row, "layout", label)?;
    require_exact_string(layout, "magic", "HQ30UQ4\0", label)?;
    if required_u64(layout, "version", label)? != u64::from(UNIFORM_Q4_VERSION)
        || required_u64(layout, "group_size", label)? != u64::try_from(UNIFORM_Q4_GROUP_SIZE).unwrap()
    {
        return Err(Error::Model(format!(
            "{label}: tensor {tensor_name:?} layout does not identify version-1 64-value Q4 groups"
        )));
    }
    require_exact_string(layout, "scale_dtype", "float16", label)?;

    let header = parse_uniform_q4_header(&payload)?;
    if header.group_size != UNIFORM_Q4_GROUP_SIZE
        || header.version != UNIFORM_Q4_VERSION
        || header.shape != shape
        || u64::try_from(header.elements).ok() != Some(declared_elements)
    {
        return Err(Error::Model(format!(
            "{label}: tensor {tensor_name:?} manifest geometry disagrees with its Q4 payload header"
        )));
    }
    Ok((
        CompleteBinaryTensor {
            tensor_name: tensor_name.to_owned(),
            source_shard: source_shard.to_owned(),
            source_shard_sha256,
            source_dtype: source_dtype.to_owned(),
            artifact_path: expected_path,
            artifact_sha256,
            header,
        },
        Arc::from(payload),
    ))
}

/// Admit the sealed uniform-Q4 diagnostic complete candidate.
///
/// Reuses the protected baseline revalidation receipt (source shards are
/// unchanged). Every payload is SHA-256 verified and HQ30UQ4-parsed.
pub fn admit_qwen30_uniform_q4_artifact(
    manifest_path: impl AsRef<Path>,
    admission: &Qwen30UniformQ4Admission,
) -> Result<CompleteBinaryArtifact> {
    crate::startup_timing::time_ms_result("admit_uniform_q4_total", || {
        admit_qwen30_uniform_q4_artifact_inner(manifest_path, admission)
    })
}

fn admit_qwen30_uniform_q4_artifact_inner(
    manifest_path: impl AsRef<Path>,
    admission: &Qwen30UniformQ4Admission,
) -> Result<CompleteBinaryArtifact> {
    if !is_sha256(&admission.expected_manifest_seal_sha256)
        || !is_sha256(&admission.expected_source_audit_seal_sha256)
        || !is_sha256(&admission.expected_revalidation_seal_sha256)
        || !is_sha256(&admission.expected_terminal_seal_sha256)
        || admission.expected_source_revision.is_empty()
    {
        return Err(Error::Model(
            "uniform Q4 admission requires protected lowercase SHA-256 seals and a source revision"
                .into(),
        ));
    }
    let manifest_path =
        canonical_regular_path(manifest_path.as_ref(), "uniform Q4 manifest")?;
    let root = manifest_path.parent().ok_or_else(|| {
        Error::Model("uniform Q4 manifest has no parent artifact root".into())
    })?;
    let manifest_raw = read_regular_file(&manifest_path, "uniform Q4 manifest")?;
    let manifest = parse_json_no_duplicate_keys(&manifest_raw, "uniform Q4 manifest")?;
    let manifest_object = manifest
        .as_object()
        .ok_or_else(|| Error::Model("uniform Q4 manifest: root must be an object".into()))?;
    let manifest_seal = verify_sealed_document(&manifest, "uniform Q4 manifest")?;
    if manifest_seal != admission.expected_manifest_seal_sha256 {
        return Err(Error::Model(
            "uniform Q4 manifest seal does not match the protected admission binding".into(),
        ));
    }
    require_exact_string(
        manifest_object,
        "schema",
        QWEN30_UNIFORM_Q4_SCHEMA,
        "uniform Q4 manifest",
    )?;
    require_exact_string(
        manifest_object,
        "status",
        QWEN30_UNIFORM_Q4_CANDIDATE_STATUS,
        "uniform Q4 manifest",
    )?;
    let manifest_audit_seal = required_sha256(
        manifest_object,
        "source_body_audit_seal_sha256",
        "uniform Q4 manifest",
    )?;
    if manifest_audit_seal != admission.expected_source_audit_seal_sha256 {
        return Err(Error::Model(
            "uniform Q4 manifest source audit seal does not match protected admission binding"
                .into(),
        ));
    }

    // Terminal receipt (diagnostic completion binding).
    let terminal_path = canonical_regular_path(
        &admission.expected_terminal_path,
        "uniform Q4 terminal receipt",
    )?;
    let terminal_raw = read_regular_file(&terminal_path, "uniform Q4 terminal receipt")?;
    let terminal = parse_json_no_duplicate_keys(&terminal_raw, "uniform Q4 terminal receipt")?;
    let terminal_seal = verify_sealed_document(&terminal, "uniform Q4 terminal receipt")?;
    if terminal_seal != admission.expected_terminal_seal_sha256 {
        return Err(Error::Model(
            "uniform Q4 terminal seal does not match protected admission binding".into(),
        ));
    }

    // Source chain via the protected baseline revalidation receipt (absolute).
    let revalidation_path = canonical_regular_path(
        &admission.expected_revalidation_path,
        "uniform Q4 source revalidation receipt",
    )?;
    let revalidation_raw =
        read_regular_file(&revalidation_path, "uniform Q4 source revalidation receipt")?;
    let revalidation =
        parse_json_no_duplicate_keys(&revalidation_raw, "uniform Q4 source revalidation receipt")?;
    let revalidation_seal =
        verify_sealed_document(&revalidation, "uniform Q4 source revalidation receipt")?;
    if revalidation_seal != admission.expected_revalidation_seal_sha256 {
        return Err(Error::Model(
            "uniform Q4 revalidation seal does not match protected admission binding".into(),
        ));
    }
    let declared_revalidation = required_string(
        manifest_object,
        "source_revalidation_receipt_path",
        "uniform Q4 manifest",
    )?;
    if Path::new(declared_revalidation) != revalidation_path.as_path()
        && canonical_regular_path(Path::new(declared_revalidation), "declared revalidation")
            .ok()
            .as_ref()
            != Some(&revalidation_path)
    {
        // Accept exact path string or canonical form.
        let declared_canon = absolute_path(declared_revalidation, "declared revalidation")?;
        if declared_canon != revalidation_path {
            return Err(Error::Model(
                "uniform Q4 manifest revalidation path differs from protected handoff".into(),
            ));
        }
    }
    if required_sha256(
        manifest_object,
        "source_revalidation_receipt_seal_sha256",
        "uniform Q4 manifest",
    )? != admission.expected_revalidation_seal_sha256
    {
        return Err(Error::Model(
            "uniform Q4 manifest revalidation seal differs from protected handoff".into(),
        ));
    }

    // Reuse baseline source-chain validation against Qwen30 model identity.
    let baseline_admission = CompleteBinaryAdmission {
        model: QwenCompleteBinaryModel::Qwen30Coder,
        expected_manifest_seal_sha256: admission.expected_manifest_seal_sha256.clone(),
        expected_source_audit_seal_sha256: admission.expected_source_audit_seal_sha256.clone(),
        expected_source_revision: admission.expected_source_revision.clone(),
    };
    // validate_source_chain requires revalidation parent == expected root for
    // baseline; here revalidation lives under complete-gravity. Call the
    // inner validators by reconstructing SourceChain from the revalidation
    // receipt the same way baseline does, using the receipt's parent as the
    // authority root for the path check only.
    let revalidation_parent = revalidation_path.parent().ok_or_else(|| {
        Error::Model("uniform Q4 revalidation receipt has no parent".into())
    })?;
    let source = validate_source_chain(
        &revalidation,
        &revalidation_path,
        revalidation_parent,
        &baseline_admission,
        &manifest_audit_seal,
    )?;

    // Representation must identify uniform Q4 group-64.
    let representation =
        required_object(manifest_object, "representation", "uniform Q4 manifest")?;
    require_exact_string(
        representation,
        "family",
        "uniform_q4_group64_fp16_scale",
        "uniform Q4 representation",
    )?;
    if required_u64(representation, "group_size", "uniform Q4 representation")?
        != u64::try_from(UNIFORM_Q4_GROUP_SIZE).unwrap()
        || !required_bool(
            representation,
            "physical_direct_layout",
            "uniform Q4 representation",
        )?
    {
        return Err(Error::Model(
            "uniform Q4 representation requires group_size=64 and physical_direct_layout".into(),
        ));
    }

    let rows = required_array(manifest_object, "tensors", "uniform Q4 manifest")?;
    if rows.len() != source.weight_map.len() {
        return Err(Error::Model(
            "uniform Q4 manifest tensor count does not match the revalidated source index".into(),
        ));
    }

    let mut tensors = BTreeMap::new();
    let mut verified_payloads = BTreeMap::new();
    // Bounded parallel by source-shard lanes, same shape as baseline cold path.
    let parallel = match std::env::var("HAWKING_ADMISSION_PARALLEL") {
        Ok(v) if matches!(v.as_str(), "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF") => {
            false
        }
        _ => true,
    };
    if parallel {
        let (lanes, workers) = quality_payload_verification_lanes(rows);
        let mut handles = Vec::with_capacity(workers);
        for lane in lanes {
            let lane_rows: Vec<Value> = lane.iter().map(|&i| rows[i].clone()).collect();
            let root = root.to_path_buf();
            let source = source.clone();
            handles.push(thread::spawn(move || {
                let mut local_tensors = BTreeMap::new();
                let mut local_payloads = BTreeMap::new();
                for value in lane_rows {
                    let row = value.as_object().ok_or_else(|| {
                        Error::Model("uniform Q4 manifest tensor entry must be an object".into())
                    })?;
                    let (tensor, payload) = validate_q4_tensor_row(row, &root, &source)?;
                    let name = tensor.tensor_name.clone();
                    if local_tensors.insert(name.clone(), tensor).is_some() {
                        return Err(Error::Model(
                            "uniform Q4 manifest contains a duplicate tensor_name".into(),
                        ));
                    }
                    local_payloads.insert(name, payload);
                }
                Ok((local_tensors, local_payloads))
            }));
        }
        for handle in handles {
            let (local_tensors, local_payloads) = handle
                .join()
                .map_err(|_| Error::Model("uniform Q4 admission worker panicked".into()))??;
            for (name, tensor) in local_tensors {
                let payload = local_payloads.get(&name).cloned().ok_or_else(|| {
                    Error::Model("uniform Q4 admission lane missing payload".into())
                })?;
                if tensors.insert(name.clone(), tensor).is_some() {
                    return Err(Error::Model(
                        "uniform Q4 manifest contains a duplicate tensor_name across lanes".into(),
                    ));
                }
                verified_payloads.insert(name, payload);
            }
        }
    } else {
        for value in rows {
            let row = value.as_object().ok_or_else(|| {
                Error::Model("uniform Q4 manifest tensor entry must be an object".into())
            })?;
            let (tensor, payload) = validate_q4_tensor_row(row, root, &source)?;
            let name = tensor.tensor_name.clone();
            if tensors.insert(name.clone(), tensor).is_some() {
                return Err(Error::Model(
                    "uniform Q4 manifest contains a duplicate tensor_name".into(),
                ));
            }
            verified_payloads.insert(name, payload);
        }
    }

    if tensors.keys().ne(source.weight_map.keys()) {
        return Err(Error::Model(
            "uniform Q4 manifest tensor set does not exactly match the revalidated source index"
                .into(),
        ));
    }
    if verified_payloads.len() != tensors.len() {
        return Err(Error::Model(
            "uniform Q4 admission did not retain one verified immutable payload per tensor".into(),
        ));
    }

    let (elements, payload_bytes) =
        validate_ledger(manifest_object, &tensors, manifest_raw.len())?;

    Ok(CompleteBinaryArtifact {
        model: QwenCompleteBinaryModel::Qwen30Coder,
        manifest_path,
        manifest_seal_sha256: manifest_seal,
        source_audit_path: source.source_audit_path,
        source_audit_seal_sha256: source.source_audit_seal_sha256,
        source_revision: source.source_revision,
        source_index_path: source.source_index_path,
        source_weight_elements: elements,
        tensor_payload_bytes: payload_bytes,
        tensors,
        verified_payloads,
    })
}

#[cfg(test)]
mod parse_group_size_tests {
    use super::*;

    fn hq30uq4_payload(shape: &[usize], group_size: usize) -> Vec<u8> {
        let elements = shape.iter().product::<usize>();
        let groups = elements.div_ceil(group_size.max(1));
        let code_bytes_per_group = group_size / 2;
        let mut payload = Vec::new();
        payload.extend_from_slice(&UNIFORM_Q4_MAGIC);
        payload.extend_from_slice(&UNIFORM_Q4_VERSION.to_le_bytes());
        payload.extend_from_slice(&(group_size as u32).to_le_bytes());
        payload.extend_from_slice(&(shape.len() as u16).to_le_bytes());
        payload.extend_from_slice(&0u16.to_le_bytes());
        payload.extend_from_slice(&(elements as u64).to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());
        for dimension in shape {
            payload.extend_from_slice(&(*dimension as u32).to_le_bytes());
        }
        payload.resize(payload.len() + groups * 2, 0);
        payload.resize(payload.len() + groups * code_bytes_per_group, 0);
        payload
    }

    #[test]
    fn parse_accepts_group_64_and_128_and_refuses_the_rest() {
        let g64 = parse_uniform_q4_header(&hq30uq4_payload(&[4, 64], 64)).unwrap();
        assert_eq!(g64.group_size, 64);
        assert_eq!(g64.groups, 4);
        assert_eq!(g64.elements, 256);

        let g128 = parse_uniform_q4_header(&hq30uq4_payload(&[4, 128], 128)).unwrap();
        assert_eq!(g128.group_size, 128);
        assert_eq!(g128.groups, 4);
        assert_eq!(g128.elements, 512);
        assert_eq!(
            g128.payload_bytes,
            COMPLETE_BINARY_HEADER_BYTES + 8 + 4 * 2 + 4 * 64
        );

        for group_size in [0usize, 32, 96, 256, 512] {
            let err = parse_uniform_q4_header(&hq30uq4_payload(&[4, 256], group_size))
                .expect_err("unsupported group size must refuse");
            let msg = format!("{err}");
            assert!(
                msg.contains("must be 64 or 128"),
                "group_size={group_size} refuse message was {msg}"
            );
            assert!(
                msg.contains(&format!("group_size={group_size}")),
                "group_size={group_size} refuse message was {msg}"
            );
        }
    }
}
