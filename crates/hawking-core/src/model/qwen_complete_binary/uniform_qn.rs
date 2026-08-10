//! Diagnostic HQ30UQ2 / HQ30UQ3 (uniform Qn group-128) admission for Lane-N.
//!
//! Little-endian bit-stream codes + FP16 group scale. Bound = (1<<(bits-1))-1.
//! Produces CompleteBinaryArtifact; runtime dispatches qwen_uniform_qn_*.

use super::*;
use std::sync::Arc;

pub const UNIFORM_QN_VERSION: u32 = 1;
pub const UNIFORM_QN_GROUP_SIZE: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UniformQnBits {
    Two = 2,
    Three = 3,
}

impl UniformQnBits {
    pub fn as_u32(self) -> u32 {
        self as u32
    }
    pub fn bound(self) -> i32 {
        (1 << (self as u32 - 1)) - 1
    }
    pub fn magic(self) -> [u8; 8] {
        match self {
            Self::Two => *b"HQ30UQ2\0",
            Self::Three => *b"HQ30UQ3\0",
        }
    }
    pub fn magic_text(self) -> &'static str {
        match self {
            Self::Two => "HQ30UQ2\0",
            Self::Three => "HQ30UQ3\0",
        }
    }
    pub fn schema(self) -> &'static str {
        match self {
            Self::Two => "hawking.ascension.qwen30_uniform_q2_group128_candidate.v1",
            Self::Three => "hawking.ascension.qwen30_uniform_q3_group128_candidate.v1",
        }
    }
    pub fn candidate_status(self) -> &'static str {
        match self {
            Self::Two => "CANDIDATE_UNIFORM_Q2_GROUP128_DIAGNOSTIC_UNQUALIFIED",
            Self::Three => "CANDIDATE_UNIFORM_Q3_GROUP128_DIAGNOSTIC_UNQUALIFIED",
        }
    }
    pub fn family(self) -> &'static str {
        match self {
            Self::Two => "uniform_q2_group128_fp16_scale",
            Self::Three => "uniform_q3_group128_fp16_scale",
        }
    }
    pub fn ext(self) -> &'static str {
        match self {
            Self::Two => "hq30uq2",
            Self::Three => "hq30uq3",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen30UniformQnAdmission {
    pub bits: UniformQnBits,
    pub expected_manifest_seal_sha256: String,
    pub expected_source_audit_seal_sha256: String,
    pub expected_source_revision: String,
    pub expected_revalidation_path: PathBuf,
    pub expected_revalidation_seal_sha256: String,
    pub expected_terminal_path: PathBuf,
    pub expected_terminal_seal_sha256: String,
}

pub fn parse_uniform_qn_header(
    payload: &[u8],
    bits: UniformQnBits,
) -> Result<CompleteBinaryHeader> {
    if payload.len() < COMPLETE_BINARY_HEADER_BYTES {
        return Err(Error::Model("uniform Qn payload truncated at header".into()));
    }
    if payload[..8] != bits.magic() {
        return Err(Error::Model(format!(
            "uniform Qn magic does not match {:?}",
            bits.magic_text()
        )));
    }
    let version = read_u32(payload, 8)?;
    if version != UNIFORM_QN_VERSION {
        return Err(Error::Model(format!(
            "uniform Qn version {version} unsupported"
        )));
    }
    let group_size = read_u32(payload, 12)? as usize;
    if group_size != UNIFORM_QN_GROUP_SIZE {
        return Err(Error::Model(format!(
            "uniform Qn group_size={group_size} must be {UNIFORM_QN_GROUP_SIZE}"
        )));
    }
    let rank = read_u16(payload, 16)? as usize;
    let reserved = read_u16(payload, 18)?;
    let element_u64 = read_u64(payload, 20)?;
    let reserved_tail = read_u32(payload, 28)?;
    if reserved != 0 || reserved_tail != 0 {
        return Err(Error::Model("uniform Qn reserved fields must be zero".into()));
    }
    if rank == 0 {
        return Err(Error::Model("uniform Qn rank must be positive".into()));
    }
    let dimensions_offset = COMPLETE_BINARY_HEADER_BYTES;
    let dimensions_bytes = rank
        .checked_mul(4)
        .ok_or_else(|| Error::Model("uniform Qn rank overflow".into()))?;
    let after_dimensions = dimensions_offset
        .checked_add(dimensions_bytes)
        .ok_or_else(|| Error::Model("uniform Qn dims overflow".into()))?;
    if after_dimensions > payload.len() {
        return Err(Error::Model("uniform Qn truncated in dimensions".into()));
    }
    let mut shape = Vec::with_capacity(rank);
    let mut derived_elements = 1usize;
    for dimension in 0..rank {
        let value = read_u32(payload, dimensions_offset + dimension * 4)? as usize;
        if value == 0 {
            return Err(Error::Model("uniform Qn dimension must be positive".into()));
        }
        derived_elements = derived_elements
            .checked_mul(value)
            .ok_or_else(|| Error::Model("uniform Qn element overflow".into()))?;
        shape.push(value);
    }
    let elements = usize::try_from(element_u64)
        .map_err(|_| Error::Model("uniform Qn elements exceed platform".into()))?;
    if elements != derived_elements {
        return Err(Error::Model(format!(
            "uniform Qn elements {elements} != shape product {derived_elements}"
        )));
    }
    let groups = elements
        .checked_add(group_size - 1)
        .ok_or_else(|| Error::Model("uniform Qn groups overflow".into()))?
        / group_size;
    let scale_bytes = groups
        .checked_mul(2)
        .ok_or_else(|| Error::Model("uniform Qn scale bytes overflow".into()))?;
    let padded = groups
        .checked_mul(group_size)
        .ok_or_else(|| Error::Model("uniform Qn padded overflow".into()))?;
    let code_bits = padded
        .checked_mul(bits as usize)
        .ok_or_else(|| Error::Model("uniform Qn code bits overflow".into()))?;
    let code_bytes = code_bits.div_ceil(8);
    let scale_offset = after_dimensions;
    let sign_offset = scale_offset
        .checked_add(scale_bytes)
        .ok_or_else(|| Error::Model("uniform Qn code offset overflow".into()))?;
    let payload_bytes = sign_offset
        .checked_add(code_bytes)
        .ok_or_else(|| Error::Model("uniform Qn payload overflow".into()))?;
    if payload_bytes != payload.len() {
        return Err(Error::Model(format!(
            "uniform Qn payload size {} != expected {payload_bytes}",
            payload.len()
        )));
    }
    for group in 0..groups {
        let scale = f16::from_bits(read_u16(payload, scale_offset + group * 2)?);
        if !scale.is_finite() {
            return Err(Error::Model(format!(
                "uniform Qn scale group {group} non-finite"
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

fn expected_qn_tensor_path(root: &Path, tensor_name: &str, bits: UniformQnBits) -> Result<PathBuf> {
    let tensors = root.join("tensors");
    let metadata = fs::symlink_metadata(&tensors).map_err(|error| {
        Error::Model(format!("uniform Qn tensors dir: {error}"))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(Error::Model("uniform Qn tensors must be a real directory".into()));
    }
    Ok(tensors.join(format!(
        "{}.{}",
        sha256_hex(tensor_name.as_bytes()),
        bits.ext()
    )))
}

fn validate_qn_tensor_row(
    row: &Map<String, Value>,
    root: &Path,
    source: &SourceChain,
    bits: UniformQnBits,
) -> Result<(CompleteBinaryTensor, Arc<[u8]>)> {
    let label = "uniform Qn manifest tensor";
    let tensor_name = required_string(row, "tensor_name", label)?;
    let source_shard = required_string(row, "source_shard", label)?;
    require_safe_filename(source_shard, label)?;
    let source_shard_sha256 = required_sha256(row, "source_shard_sha256", label)?;
    let expected_source_hash = source.shard_hashes.get(source_shard).ok_or_else(|| {
        Error::Model(format!("{label}: shard {source_shard:?} missing from receipt"))
    })?;
    if &source_shard_sha256 != expected_source_hash {
        return Err(Error::Model(format!(
            "{label}: shard hash mismatch for {tensor_name:?}"
        )));
    }
    if source.weight_map.get(tensor_name).map(String::as_str) != Some(source_shard) {
        return Err(Error::Model(format!(
            "{label}: index does not bind {tensor_name:?} to {source_shard:?}"
        )));
    }
    let source_dtype = required_string(row, "source_dtype", label)?;
    let shape = declared_tensor_shape(row, label)?;
    let declared_elements = required_u64(row, "elements", label)?;
    let shape_elements = shape.iter().try_fold(1u64, |total, dimension| {
        total
            .checked_mul(u64::try_from(*dimension).unwrap_or(u64::MAX))
            .ok_or_else(|| Error::Model(format!("{label}: shape overflow")))
    })?;
    if declared_elements == 0 || declared_elements != shape_elements {
        return Err(Error::Model(format!(
            "{label}: elements mismatch for {tensor_name:?}"
        )));
    }
    let expected_path = expected_qn_tensor_path(root, tensor_name, bits)?;
    let expected_path = canonical_regular_path(&expected_path, label)?;
    let declared_path =
        manifest_descendant_file(root, required_string(row, "artifact_path", label)?, label)?;
    if declared_path != expected_path {
        return Err(Error::Model(format!(
            "{label}: artifact_path mismatch for {tensor_name:?}"
        )));
    }
    let payload = read_regular_file(&expected_path, label)?;
    if required_u64(row, "artifact_bytes", label)?
        != u64::try_from(payload.len()).unwrap_or(u64::MAX)
    {
        return Err(Error::Model(format!(
            "{label}: artifact_bytes mismatch for {tensor_name:?}"
        )));
    }
    let artifact_sha256 = required_sha256(row, "artifact_sha256", label)?;
    if sha256_hex(&payload) != artifact_sha256 {
        return Err(Error::Model(format!(
            "{label}: SHA-256 mismatch for {tensor_name:?}"
        )));
    }
    let layout = required_object(row, "layout", label)?;
    require_exact_string(layout, "magic", bits.magic_text(), label)?;
    if required_u64(layout, "version", label)? != u64::from(UNIFORM_QN_VERSION)
        || required_u64(layout, "group_size", label)?
            != u64::try_from(UNIFORM_QN_GROUP_SIZE).unwrap()
        || required_u64(layout, "bits", label)? != u64::from(bits.as_u32())
    {
        return Err(Error::Model(format!(
            "{label}: layout fields wrong for {tensor_name:?}"
        )));
    }
    let header = parse_uniform_qn_header(&payload, bits)?;
    if header.group_size != UNIFORM_QN_GROUP_SIZE
        || header.shape != shape
        || u64::try_from(header.elements).ok() != Some(declared_elements)
    {
        return Err(Error::Model(format!(
            "{label}: geometry mismatch for {tensor_name:?}"
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

pub fn admit_qwen30_uniform_qn_artifact(
    manifest_path: impl AsRef<Path>,
    admission: &Qwen30UniformQnAdmission,
) -> Result<CompleteBinaryArtifact> {
    crate::startup_timing::time_ms_result("admit_uniform_qn_total", || {
        admit_qwen30_uniform_qn_artifact_inner(manifest_path, admission)
    })
}

fn admit_qwen30_uniform_qn_artifact_inner(
    manifest_path: impl AsRef<Path>,
    admission: &Qwen30UniformQnAdmission,
) -> Result<CompleteBinaryArtifact> {
    let bits = admission.bits;
    if !is_sha256(&admission.expected_manifest_seal_sha256)
        || !is_sha256(&admission.expected_source_audit_seal_sha256)
        || !is_sha256(&admission.expected_revalidation_seal_sha256)
        || !is_sha256(&admission.expected_terminal_seal_sha256)
        || admission.expected_source_revision.is_empty()
    {
        return Err(Error::Model(
            "uniform Qn admission requires protected seals and revision".into(),
        ));
    }
    let manifest_path =
        canonical_regular_path(manifest_path.as_ref(), "uniform Qn manifest")?;
    let root = manifest_path
        .parent()
        .ok_or_else(|| Error::Model("uniform Qn manifest has no parent".into()))?;
    let manifest_raw = read_regular_file(&manifest_path, "uniform Qn manifest")?;
    let manifest = parse_json_no_duplicate_keys(&manifest_raw, "uniform Qn manifest")?;
    let manifest_object = manifest
        .as_object()
        .ok_or_else(|| Error::Model("uniform Qn manifest root must be object".into()))?;
    let manifest_seal = verify_sealed_document(&manifest, "uniform Qn manifest")?;
    if manifest_seal != admission.expected_manifest_seal_sha256 {
        return Err(Error::Model("uniform Qn manifest seal mismatch".into()));
    }
    require_exact_string(manifest_object, "schema", bits.schema(), "uniform Qn manifest")?;
    require_exact_string(
        manifest_object,
        "status",
        bits.candidate_status(),
        "uniform Qn manifest",
    )?;
    let manifest_audit_seal = required_sha256(
        manifest_object,
        "source_body_audit_seal_sha256",
        "uniform Qn manifest",
    )?;
    if manifest_audit_seal != admission.expected_source_audit_seal_sha256 {
        return Err(Error::Model("uniform Qn audit seal mismatch".into()));
    }

    let terminal_path = canonical_regular_path(
        &admission.expected_terminal_path,
        "uniform Qn terminal",
    )?;
    let terminal_raw = read_regular_file(&terminal_path, "uniform Qn terminal")?;
    let terminal = parse_json_no_duplicate_keys(&terminal_raw, "uniform Qn terminal")?;
    let terminal_seal = verify_sealed_document(&terminal, "uniform Qn terminal")?;
    if terminal_seal != admission.expected_terminal_seal_sha256 {
        return Err(Error::Model("uniform Qn terminal seal mismatch".into()));
    }

    let revalidation_path = canonical_regular_path(
        &admission.expected_revalidation_path,
        "uniform Qn revalidation",
    )?;
    let revalidation_raw = read_regular_file(&revalidation_path, "uniform Qn revalidation")?;
    let revalidation =
        parse_json_no_duplicate_keys(&revalidation_raw, "uniform Qn revalidation")?;
    let revalidation_seal = verify_sealed_document(&revalidation, "uniform Qn revalidation")?;
    if revalidation_seal != admission.expected_revalidation_seal_sha256 {
        return Err(Error::Model("uniform Qn revalidation seal mismatch".into()));
    }
    if required_sha256(
        manifest_object,
        "source_revalidation_receipt_seal_sha256",
        "uniform Qn manifest",
    )? != admission.expected_revalidation_seal_sha256
    {
        return Err(Error::Model(
            "uniform Qn manifest revalidation seal differs from handoff".into(),
        ));
    }

    let baseline_admission = CompleteBinaryAdmission {
        model: QwenCompleteBinaryModel::Qwen30Coder,
        expected_manifest_seal_sha256: admission.expected_manifest_seal_sha256.clone(),
        expected_source_audit_seal_sha256: admission.expected_source_audit_seal_sha256.clone(),
        expected_source_revision: admission.expected_source_revision.clone(),
    };
    let revalidation_parent = revalidation_path
        .parent()
        .ok_or_else(|| Error::Model("revalidation has no parent".into()))?;
    let source = validate_source_chain(
        &revalidation,
        &revalidation_path,
        revalidation_parent,
        &baseline_admission,
        &manifest_audit_seal,
    )?;

    let representation =
        required_object(manifest_object, "representation", "uniform Qn manifest")?;
    require_exact_string(representation, "family", bits.family(), "uniform Qn representation")?;
    if required_u64(representation, "group_size", "uniform Qn representation")?
        != u64::try_from(UNIFORM_QN_GROUP_SIZE).unwrap()
        || required_u64(representation, "bits_per_weight", "uniform Qn representation")?
            != u64::from(bits.as_u32())
    {
        return Err(Error::Model(
            "uniform Qn representation group/bits mismatch".into(),
        ));
    }

    let rows = required_array(manifest_object, "tensors", "uniform Qn manifest")?;
    if rows.len() != source.weight_map.len() {
        return Err(Error::Model(
            "uniform Qn tensor count mismatches source index".into(),
        ));
    }

    let mut tensors = BTreeMap::new();
    let mut verified_payloads = BTreeMap::new();
    let (lanes, workers) = quality_payload_verification_lanes(rows);
    let mut handles: Vec<
        thread::JoinHandle<Result<(BTreeMap<String, CompleteBinaryTensor>, BTreeMap<String, Arc<[u8]>>)>>,
    > = Vec::with_capacity(workers);
    for lane in lanes {
        let lane_rows: Vec<Value> = lane.iter().map(|&i| rows[i].clone()).collect();
        let root = root.to_path_buf();
        let source = source.clone();
        handles.push(thread::spawn(move || -> Result<(
            BTreeMap<String, CompleteBinaryTensor>,
            BTreeMap<String, Arc<[u8]>>,
        )> {
            let mut local_tensors = BTreeMap::new();
            let mut local_payloads = BTreeMap::new();
            for value in lane_rows {
                let row = value.as_object().ok_or_else(|| {
                    Error::Model("uniform Qn tensor entry must be object".into())
                })?;
                let (tensor, payload) = validate_qn_tensor_row(row, &root, &source, bits)?;
                let name = tensor.tensor_name.clone();
                local_tensors.insert(name.clone(), tensor);
                local_payloads.insert(name, payload);
            }
            Ok((local_tensors, local_payloads))
        }));
    }
    for handle in handles {
        let (local_tensors, local_payloads) = handle
            .join()
            .map_err(|_| Error::Model("uniform Qn worker panicked".into()))??;
        for (name, tensor) in local_tensors {
            let payload = local_payloads.get(&name).cloned().ok_or_else(|| {
                Error::Model("uniform Qn lane missing payload".into())
            })?;
            if tensors.insert(name.clone(), tensor).is_some() {
                return Err(Error::Model("uniform Qn duplicate tensor".into()));
            }
            verified_payloads.insert(name, payload);
        }
    }
    if tensors.keys().ne(source.weight_map.keys()) {
        return Err(Error::Model(
            "uniform Qn tensor set != source index".into(),
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
