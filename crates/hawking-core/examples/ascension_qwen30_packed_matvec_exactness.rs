//! CPU-only exactness laboratory for the rejected Qwen30 packed-matvec SIMD
//! candidate.
//!
//! This target is deliberately isolated from the Qwen30 runtime, watcher,
//! server, kernel registry, and Metal library.  It binds an experiment to the
//! currently admitted complete-binary manifest by both its protected seal and
//! raw document digest, verifies the selected packed payload digests, then
//! characterizes three *host models* of reduction order:
//!
//! * the current scalar one-row-at-a-time accumulation order;
//! * the rejected 32-lane strided SIMD / balanced-tree proxy; and
//! * an unregistered group-tiled Neumaier-compensated candidate.
//!
//! It never creates a Metal device or dispatches GPU work.  The tiled shader
//! paired with this target is intentionally not embedded in `metal::library`:
//! it can only be tried later in a separately leased component experiment.
//! This report is numerical research, not a decoder, generation, HCLI, TPS,
//! TG, capability, or tournament receipt.
//!
//! Example:
//! ```text
//! cargo run --release -p hawking-core --example ascension_qwen30_packed_matvec_exactness -- \
//!   --manifest /abs/.../QWEN30_COMPLETE_BINARY_GRAVITY_CANDIDATE.json \
//!   --expected-manifest-seal-sha256 3321... \
//!   --expected-manifest-document-sha256 68f8... \
//!   --out /abs/.../QWEN30_PACKED_MATVEC_REDUCTION_EXACTNESS_CPU.json
//! ```

use half::f16;
use hawking_core::model::qwen_complete_binary::parse_complete_binary_header;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

const SCHEMA: &str = "hawking.ascension.qwen30_packed_matvec_reduction_exactness_cpu.v1";
const COMPLETE_BINARY_SCHEMA: &str = "hawking.ascension.qwen30_complete_binary_gravity.v1";
const COMPLETE_BINARY_MAGIC: &str = "HQ30G1B1";
const GROUP_SIZE: usize = 128;
const SIMD_LANES: usize = 32;
const DEFAULT_SAMPLE_ROWS: usize = 8;

#[derive(Clone, Copy)]
struct ProjectionSpec {
    category: &'static str,
    tensor_name: &'static str,
    shape: [usize; 2],
}

// These cover the Qwen30 projection geometries that are relevant to the
// current direct-packed path, including the real routed-expert gate/up/down
// chain used for the error-amplification experiment below.
const PROJECTION_SPECS: &[ProjectionSpec] = &[
    ProjectionSpec {
        category: "attention_q",
        tensor_name: "model.layers.0.self_attn.q_proj.weight",
        shape: [4096, 2048],
    },
    ProjectionSpec {
        category: "attention_o",
        tensor_name: "model.layers.24.self_attn.o_proj.weight",
        shape: [2048, 4096],
    },
    ProjectionSpec {
        category: "router",
        tensor_name: "model.layers.47.mlp.gate.weight",
        shape: [128, 2048],
    },
    ProjectionSpec {
        category: "expert_gate",
        tensor_name: "model.layers.0.mlp.experts.0.gate_proj.weight",
        shape: [768, 2048],
    },
    ProjectionSpec {
        category: "expert_up",
        tensor_name: "model.layers.0.mlp.experts.0.up_proj.weight",
        shape: [768, 2048],
    },
    ProjectionSpec {
        category: "expert_down",
        tensor_name: "model.layers.0.mlp.experts.0.down_proj.weight",
        shape: [2048, 768],
    },
    ProjectionSpec {
        category: "lm_head",
        tensor_name: "lm_head.weight",
        shape: [151_936, 2048],
    },
];

#[derive(Debug)]
struct Args {
    manifest: PathBuf,
    expected_manifest_seal_sha256: String,
    expected_manifest_document_sha256: String,
    out: PathBuf,
    sample_rows: usize,
}

#[derive(Clone)]
struct PackedMatrix {
    category: &'static str,
    tensor_name: String,
    artifact_path: PathBuf,
    artifact_sha256: String,
    source_shard_sha256: String,
    rows: usize,
    cols: usize,
    group_size: usize,
    groups_per_row: usize,
    scale_offset: usize,
    sign_offset: usize,
    payload: Vec<u8>,
}

#[derive(Clone, Copy, Debug)]
enum InputFamily {
    Bounded,
    WideExponent,
    Cancellation,
}

impl InputFamily {
    const ALL: [Self; 3] = [Self::Bounded, Self::WideExponent, Self::Cancellation];

    fn name(self) -> &'static str {
        match self {
            Self::Bounded => "bounded_seeded_f32",
            Self::WideExponent => "wide_exponent_seeded_f32",
            Self::Cancellation => "alternating_cancellation_seeded_f32",
        }
    }

    fn seed(self) -> u64 {
        match self {
            Self::Bounded => 0x9e37_79b9_7f4a_7c15,
            Self::WideExponent => 0xd1b5_4a32_d192_ed03,
            Self::Cancellation => 0x94d0_49bb_1331_11eb,
        }
    }
}

#[derive(Default, Clone)]
struct ErrorAccumulator {
    count: usize,
    nonfinite: usize,
    max_abs_vs_f64: f64,
    max_abs_vs_scalar: f64,
    max_ulp_vs_scalar: u64,
    sum_sq_vs_f64: f64,
    sum_sq_reference: f64,
}

#[derive(Serialize)]
struct ErrorReport {
    values: usize,
    all_finite: bool,
    max_abs_vs_fp64_reference: f64,
    max_abs_vs_scalar_control: f64,
    max_ulp_vs_scalar_control: u64,
    relative_l2_vs_fp64_reference: f64,
}

#[derive(Serialize)]
struct MatrixBindingReport {
    category: &'static str,
    tensor_name: String,
    artifact_path: String,
    artifact_sha256: String,
    source_shard_sha256: String,
    shape: [usize; 2],
    group_size: usize,
    groups_per_row: usize,
    layout_magic: &'static str,
}

#[derive(Serialize)]
struct InputProjectionReport {
    input_family: &'static str,
    sampled_rows: Vec<usize>,
    scalar_control: ErrorReport,
    rejected_simdgroup_balanced_tree_proxy: ErrorReport,
    tiled_neumaier_candidate: ErrorReport,
    tiled_precision_not_worse_than_scalar_control: bool,
}

#[derive(Serialize)]
struct ProjectionReport {
    binding: MatrixBindingReport,
    input_reports: Vec<InputProjectionReport>,
}

#[derive(Serialize)]
struct ChainInputReport {
    input_family: &'static str,
    swiglu_max_abs_delta_simd_vs_scalar: f64,
    swiglu_max_abs_delta_tiled_vs_scalar: f64,
    down_scalar_control: ErrorReport,
    down_rejected_simdgroup_balanced_tree_proxy: ErrorReport,
    down_tiled_neumaier_candidate: ErrorReport,
    tiled_precision_not_worse_than_scalar_control: bool,
    downstream_projection_argmax_not_lm_head_or_token: BTreeMap<&'static str, usize>,
    same_downstream_projection_argmax_as_scalar: BTreeMap<&'static str, bool>,
}

#[derive(Serialize)]
struct ExpertChainReport {
    description: &'static str,
    source_bound_organs: Vec<MatrixBindingReport>,
    input_reports: Vec<ChainInputReport>,
}

#[derive(Serialize)]
struct CandidateContractReport {
    shader_path: String,
    shader_sha256: String,
    shader_entry_point: &'static str,
    geometry: &'static str,
    cpu_algorithm: &'static str,
    unregistered: bool,
    gpu_execution_performed: bool,
}

#[derive(Serialize)]
struct CpuPrecisionGateReport {
    status: &'static str,
    projection_input_cases: usize,
    real_expert_chain_cases: usize,
    all_cases_passed: bool,
    rule: &'static str,
}

#[derive(Serialize)]
struct SourceBindingReport {
    manifest_path: String,
    manifest_document_sha256: String,
    expected_manifest_document_sha256: String,
    manifest_seal_sha256: String,
    expected_manifest_seal_sha256: String,
    manifest_schema: String,
    source_body_audit_seal_sha256: String,
    source_revision: String,
}

#[derive(Serialize)]
struct IntegrationCriteria {
    precision_gate: Vec<&'static str>,
    gpu_component_gate: Vec<&'static str>,
    full_model_gate: Vec<&'static str>,
    disallowed_promotions: Vec<&'static str>,
}

#[derive(Serialize)]
struct Report {
    schema: &'static str,
    status: &'static str,
    source_binding: SourceBindingReport,
    candidate_contract: CandidateContractReport,
    cpu_precision_gate: CpuPrecisionGateReport,
    projection_reports: Vec<ProjectionReport>,
    real_expert_chain_error_amplification: ExpertChainReport,
    integration_criteria: IntegrationCriteria,
    claim_boundary: Vec<&'static str>,
    unsealed_preimage_sha256: String,
}

fn usage() -> &'static str {
    "usage: ascension_qwen30_packed_matvec_exactness \\
        --manifest ABSOLUTE_PATH \\
        --expected-manifest-seal-sha256 SHA256 \\
        --expected-manifest-document-sha256 SHA256 \\
        --out ABSOLUTE_PATH \\
        [--sample-rows POSITIVE_INTEGER]"
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut manifest = None;
    let mut expected_manifest_seal_sha256 = None;
    let mut expected_manifest_document_sha256 = None;
    let mut out = None;
    let mut sample_rows = DEFAULT_SAMPLE_ROWS;
    let mut values = env::args().skip(1);
    while let Some(flag) = values.next() {
        let value = values
            .next()
            .ok_or_else(|| format!("missing value after {flag}; {}", usage()))?;
        match flag.as_str() {
            "--manifest" => {
                if manifest.replace(PathBuf::from(value)).is_some() {
                    return Err(format!("--manifest repeated; {}", usage()).into());
                }
            }
            "--expected-manifest-seal-sha256" => {
                if expected_manifest_seal_sha256.replace(value).is_some() {
                    return Err(
                        format!("--expected-manifest-seal-sha256 repeated; {}", usage()).into(),
                    );
                }
            }
            "--expected-manifest-document-sha256" => {
                if expected_manifest_document_sha256.replace(value).is_some() {
                    return Err(format!(
                        "--expected-manifest-document-sha256 repeated; {}",
                        usage()
                    )
                    .into());
                }
            }
            "--out" => {
                if out.replace(PathBuf::from(value)).is_some() {
                    return Err(format!("--out repeated; {}", usage()).into());
                }
            }
            "--sample-rows" => {
                sample_rows = value
                    .parse::<usize>()
                    .map_err(|_| format!("--sample-rows must be an integer; {}", usage()))?;
            }
            _ => return Err(format!("unknown option {flag}; {}", usage()).into()),
        }
    }
    let args = Args {
        manifest: manifest.ok_or_else(|| format!("missing --manifest; {}", usage()))?,
        expected_manifest_seal_sha256: expected_manifest_seal_sha256
            .ok_or_else(|| format!("missing --expected-manifest-seal-sha256; {}", usage()))?,
        expected_manifest_document_sha256: expected_manifest_document_sha256
            .ok_or_else(|| format!("missing --expected-manifest-document-sha256; {}", usage()))?,
        out: out.ok_or_else(|| format!("missing --out; {}", usage()))?,
        sample_rows,
    };
    if !args.manifest.is_absolute() || !args.out.is_absolute() {
        return Err(format!("--manifest and --out must be absolute paths; {}", usage()).into());
    }
    if args.sample_rows == 0 {
        return Err("--sample-rows must be positive".into());
    }
    for (label, value) in [
        (
            "--expected-manifest-seal-sha256",
            &args.expected_manifest_seal_sha256,
        ),
        (
            "--expected-manifest-document-sha256",
            &args.expected_manifest_document_sha256,
        ),
    ] {
        if !is_sha256(value) {
            return Err(format!("{label} must be lowercase SHA-256 hex").into());
        }
    }
    Ok(args)
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn required_string<'a>(
    object: &'a serde_json::Map<String, Value>,
    key: &str,
) -> Result<&'a str, Box<dyn Error>> {
    object
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("manifest missing string {key:?}").into())
}

fn required_usize(
    object: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<usize, Box<dyn Error>> {
    object
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| format!("manifest missing usize {key:?}").into())
}

fn source_revision(
    manifest: &serde_json::Map<String, Value>,
    manifest_parent: &Path,
) -> Result<String, Box<dyn Error>> {
    let source = manifest
        .get("source")
        .and_then(Value::as_object)
        .ok_or("manifest missing source object")?;
    for key in ["revision", "source_revision", "commit"] {
        if let Some(value) = source.get(key).and_then(Value::as_str) {
            return Ok(value.to_owned());
        }
    }
    if let Some(revalidation_path) = manifest
        .get("source_revalidation_receipt_path")
        .and_then(Value::as_str)
    {
        let receipt_path = PathBuf::from(revalidation_path).canonicalize()?;
        if !receipt_path.starts_with(manifest_parent) {
            return Err("manifest revalidation receipt escapes artifact parent".into());
        }
        let receipt: Value = serde_json::from_slice(&fs::read(receipt_path)?)?;
        if let Some(revision) = receipt.get("source_revision").and_then(Value::as_str) {
            return Ok(revision.to_owned());
        }
    }
    Err("manifest source object and its revalidation receipt lack source revision".into())
}

fn selected_rows(rows: usize, limit: usize) -> Vec<usize> {
    let mut selected = BTreeSet::new();
    for row in [
        0,
        1,
        rows / 7,
        rows / 3,
        rows / 2,
        rows.saturating_mul(2) / 3,
        rows.saturating_sub(2),
        rows.saturating_sub(1),
    ] {
        if row < rows {
            selected.insert(row);
        }
    }
    if selected.len() > limit {
        selected.into_iter().take(limit).collect()
    } else {
        selected.into_iter().collect()
    }
}

fn matrix_binding(matrix: &PackedMatrix) -> MatrixBindingReport {
    MatrixBindingReport {
        category: matrix.category,
        tensor_name: matrix.tensor_name.clone(),
        artifact_path: matrix.artifact_path.display().to_string(),
        artifact_sha256: matrix.artifact_sha256.clone(),
        source_shard_sha256: matrix.source_shard_sha256.clone(),
        shape: [matrix.rows, matrix.cols],
        group_size: matrix.group_size,
        groups_per_row: matrix.groups_per_row,
        layout_magic: COMPLETE_BINARY_MAGIC,
    }
}

fn load_selected_matrices(
    manifest_path: &Path,
    expected_document_sha: &str,
    expected_seal: &str,
) -> Result<(SourceBindingReport, BTreeMap<&'static str, PackedMatrix>), Box<dyn Error>> {
    let manifest_path = manifest_path.canonicalize()?;
    let manifest_parent = manifest_path
        .parent()
        .ok_or("manifest has no parent")?
        .canonicalize()?;
    let manifest_bytes = fs::read(&manifest_path)?;
    let manifest_document_sha256 = sha256_hex(&manifest_bytes);
    if manifest_document_sha256 != expected_document_sha {
        return Err(format!(
            "manifest document SHA mismatch: expected {expected_document_sha}, observed {manifest_document_sha256}"
        )
        .into());
    }
    let manifest_value: Value = serde_json::from_slice(&manifest_bytes)?;
    let manifest = manifest_value
        .as_object()
        .ok_or("manifest root must be object")?;
    if required_string(manifest, "schema")? != COMPLETE_BINARY_SCHEMA {
        return Err(
            "manifest schema does not identify the Qwen30 complete-binary control artifact".into(),
        );
    }
    let recorded_seal = required_string(manifest, "seal_sha256")?;
    if recorded_seal != expected_seal {
        return Err(format!(
            "manifest seal mismatch: expected {expected_seal}, observed {recorded_seal}"
        )
        .into());
    }
    let tensors = manifest
        .get("tensors")
        .and_then(Value::as_array)
        .ok_or("manifest missing tensors array")?;
    let mut by_name = BTreeMap::new();
    for tensor in tensors {
        let object = tensor
            .as_object()
            .ok_or("manifest tensor row must be object")?;
        by_name.insert(required_string(object, "tensor_name")?, object);
    }
    let mut selected = BTreeMap::new();
    for spec in PROJECTION_SPECS {
        let row = by_name
            .get(spec.tensor_name)
            .ok_or_else(|| format!("manifest missing selected tensor {}", spec.tensor_name))?;
        let artifact_path_text = required_string(row, "artifact_path")?;
        let artifact_path = PathBuf::from(artifact_path_text).canonicalize()?;
        if !artifact_path.starts_with(&manifest_parent) {
            return Err(format!(
                "selected artifact {} escapes manifest parent {}",
                artifact_path.display(),
                manifest_parent.display()
            )
            .into());
        }
        let expected_payload_sha = required_string(row, "artifact_sha256")?;
        if !is_sha256(expected_payload_sha) {
            return Err(format!(
                "selected tensor {} has invalid artifact SHA",
                spec.tensor_name
            )
            .into());
        }
        let payload = fs::read(&artifact_path)?;
        let observed_payload_sha = sha256_hex(&payload);
        if observed_payload_sha != expected_payload_sha {
            return Err(format!(
                "selected tensor {} payload SHA mismatch: expected {}, observed {}",
                spec.tensor_name, expected_payload_sha, observed_payload_sha
            )
            .into());
        }
        let header = parse_complete_binary_header(&payload)?;
        if header.group_size != GROUP_SIZE {
            return Err(format!(
                "selected tensor {} group size {} differs from admitted Qwen30 geometry {}",
                spec.tensor_name, header.group_size, GROUP_SIZE
            )
            .into());
        }
        if header.shape.as_slice() != spec.shape {
            return Err(format!(
                "selected tensor {} shape {:?} differs from expected {:?}",
                spec.tensor_name, header.shape, spec.shape
            )
            .into());
        }
        if header.elements != spec.shape[0] * spec.shape[1]
            || header.groups != header.elements.div_ceil(GROUP_SIZE)
        {
            return Err(format!(
                "selected tensor {} header geometry is inconsistent",
                spec.tensor_name
            )
            .into());
        }
        if required_usize(row, "elements")? != header.elements {
            return Err(format!(
                "selected tensor {} manifest element count drifted",
                spec.tensor_name
            )
            .into());
        }
        selected.insert(
            spec.category,
            PackedMatrix {
                category: spec.category,
                tensor_name: spec.tensor_name.to_owned(),
                artifact_path,
                artifact_sha256: expected_payload_sha.to_owned(),
                source_shard_sha256: required_string(row, "source_shard_sha256")?.to_owned(),
                rows: spec.shape[0],
                cols: spec.shape[1],
                group_size: header.group_size,
                groups_per_row: spec.shape[1].div_ceil(header.group_size),
                scale_offset: header.scale_offset,
                sign_offset: header.sign_offset,
                payload,
            },
        );
    }
    let binding = SourceBindingReport {
        manifest_path: manifest_path.display().to_string(),
        manifest_document_sha256,
        expected_manifest_document_sha256: expected_document_sha.to_owned(),
        manifest_seal_sha256: recorded_seal.to_owned(),
        expected_manifest_seal_sha256: expected_seal.to_owned(),
        manifest_schema: required_string(manifest, "schema")?.to_owned(),
        source_body_audit_seal_sha256: required_string(manifest, "source_body_audit_seal_sha256")?
            .to_owned(),
        source_revision: source_revision(manifest, &manifest_parent)?,
    };
    Ok((binding, selected))
}

impl PackedMatrix {
    fn term(&self, row: usize, col: usize, input: &[f32]) -> f32 {
        debug_assert!(row < self.rows && col < self.cols && input.len() == self.cols);
        let flat = row * self.cols + col;
        let group = flat / self.group_size;
        let scale_offset = self.scale_offset + group * std::mem::size_of::<u16>();
        let scale = f16::from_bits(u16::from_le_bytes([
            self.payload[scale_offset],
            self.payload[scale_offset + 1],
        ]))
        .to_f32();
        let group_bit = flat % self.group_size;
        let sign_byte =
            self.payload[self.sign_offset + group * (self.group_size / 8) + group_bit / 8];
        let positive = ((sign_byte >> (group_bit % 8)) & 1) != 0;
        (if positive { scale } else { -scale }) * input[col]
    }

    fn terms(&self, row: usize, input: &[f32]) -> Vec<f32> {
        (0..self.cols)
            .map(|col| self.term(row, col, input))
            .collect()
    }
}

fn scalar_reduce(terms: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &term in terms {
        // This intentionally mirrors the scalar shader's source-level
        // `sum += term` order, not a more accurate reduction.
        sum += term;
    }
    sum
}

fn balanced_tree_reduce(mut partials: [f32; SIMD_LANES]) -> f32 {
    let mut active = SIMD_LANES;
    while active > 1 {
        for lane in 0..(active / 2) {
            partials[lane] += partials[lane + active / 2];
        }
        active /= 2;
    }
    partials[0]
}

fn rejected_simdgroup_proxy_reduce(terms: &[f32]) -> f32 {
    let mut partials = [0.0f32; SIMD_LANES];
    for (lane, partial) in partials.iter_mut().enumerate() {
        let mut col = lane;
        while col < terms.len() {
            *partial += terms[col];
            col += SIMD_LANES;
        }
    }
    // Metal's `simd_sum` reduction tree is device/compiler-specific. This is
    // an intentionally deterministic balanced-tree proxy, not a claim that
    // it reproduces every implementation detail of a specific GPU.
    balanced_tree_reduce(partials)
}

fn neumaier_add(sum: &mut f32, compensation: &mut f32, value: f32) {
    let next = *sum + value;
    if sum.abs() >= value.abs() {
        *compensation += (*sum - next) + value;
    } else {
        *compensation += (value - next) + *sum;
    }
    *sum = next;
}

fn tiled_neumaier_reduce(terms: &[f32], group_size: usize) -> f32 {
    assert!(group_size > 0 && terms.len().is_multiple_of(group_size));
    let groups = terms.len() / group_size;
    assert!(
        groups <= SIMD_LANES,
        "Qwen30 precision candidate is deliberately bounded to <=32 groups per row"
    );
    let mut group_sums = [0.0f32; SIMD_LANES];
    for group in 0..groups {
        let mut sum = 0.0f32;
        let mut compensation = 0.0f32;
        for &term in &terms[group * group_size..(group + 1) * group_size] {
            neumaier_add(&mut sum, &mut compensation, term);
        }
        group_sums[group] = sum + compensation;
    }
    let mut sum = 0.0f32;
    let mut compensation = 0.0f32;
    for &group_sum in &group_sums[..groups] {
        neumaier_add(&mut sum, &mut compensation, group_sum);
    }
    sum + compensation
}

fn fp64_reference(terms: &[f32]) -> f64 {
    terms
        .iter()
        .fold(0.0f64, |sum, &term| sum + f64::from(term))
}

fn ulp_distance(a: f32, b: f32) -> u64 {
    if !a.is_finite() || !b.is_finite() {
        return u64::MAX;
    }
    let ordered = |value: f32| {
        let bits = value.to_bits();
        if bits & 0x8000_0000 != 0 {
            (!bits) as i64
        } else {
            (bits | 0x8000_0000) as i64
        }
    };
    ordered(a).abs_diff(ordered(b)) as u64
}

impl ErrorAccumulator {
    fn observe(&mut self, value: f32, scalar: f32, reference: f64) {
        self.count += 1;
        if !value.is_finite() || !scalar.is_finite() || !reference.is_finite() {
            self.nonfinite += 1;
            return;
        }
        let delta_ref = f64::from(value) - reference;
        let delta_scalar = f64::from(value) - f64::from(scalar);
        self.max_abs_vs_f64 = self.max_abs_vs_f64.max(delta_ref.abs());
        self.max_abs_vs_scalar = self.max_abs_vs_scalar.max(delta_scalar.abs());
        self.max_ulp_vs_scalar = self.max_ulp_vs_scalar.max(ulp_distance(value, scalar));
        self.sum_sq_vs_f64 += delta_ref * delta_ref;
        self.sum_sq_reference += reference * reference;
    }

    fn finish(&self) -> ErrorReport {
        ErrorReport {
            values: self.count,
            all_finite: self.nonfinite == 0,
            max_abs_vs_fp64_reference: self.max_abs_vs_f64,
            max_abs_vs_scalar_control: self.max_abs_vs_scalar,
            max_ulp_vs_scalar_control: self.max_ulp_vs_scalar,
            relative_l2_vs_fp64_reference: if self.sum_sq_reference == 0.0 {
                self.sum_sq_vs_f64.sqrt()
            } else {
                (self.sum_sq_vs_f64 / self.sum_sq_reference).sqrt()
            },
        }
    }
}

fn tiled_not_worse_than_scalar(tiled: &ErrorReport, scalar: &ErrorReport) -> bool {
    const NUMERIC_SLACK: f64 = 1.0e-18;
    tiled.all_finite
        && scalar.all_finite
        && tiled.max_abs_vs_fp64_reference <= scalar.max_abs_vs_fp64_reference + NUMERIC_SLACK
        && tiled.relative_l2_vs_fp64_reference
            <= scalar.relative_l2_vs_fp64_reference + NUMERIC_SLACK
}

fn deterministic_input(family: InputFamily, width: usize) -> Vec<f32> {
    let mut state = family.seed();
    (0..width)
        .map(|index| {
            state ^= state << 7;
            state ^= state >> 9;
            state ^= state << 8;
            let unit = ((state >> 40) as f32) / ((1u32 << 24) as f32);
            let sign = if state & 1 == 0 { -1.0 } else { 1.0 };
            match family {
                InputFamily::Bounded => sign * (0.03125 + unit),
                InputFamily::WideExponent => {
                    let exponent = ((state >> 16) % 25) as i32 - 12;
                    sign * (0.5 + unit) * 2.0f32.powi(exponent)
                }
                InputFamily::Cancellation => {
                    let alternating = if index % 2 == 0 { 1.0 } else { -1.0 };
                    alternating * sign * (0.25 + unit) * 2.0f32.powi(((index % 9) as i32) - 4)
                }
            }
        })
        .collect()
}

fn projection_report(matrix: &PackedMatrix, sample_rows: usize) -> ProjectionReport {
    let rows = selected_rows(matrix.rows, sample_rows);
    let mut input_reports = Vec::new();
    for family in InputFamily::ALL {
        let input = deterministic_input(family, matrix.cols);
        let mut scalar_stats = ErrorAccumulator::default();
        let mut simd_stats = ErrorAccumulator::default();
        let mut tiled_stats = ErrorAccumulator::default();
        for &row in &rows {
            let terms = matrix.terms(row, &input);
            let scalar = scalar_reduce(&terms);
            let simd = rejected_simdgroup_proxy_reduce(&terms);
            let tiled = tiled_neumaier_reduce(&terms, matrix.group_size);
            let reference = fp64_reference(&terms);
            scalar_stats.observe(scalar, scalar, reference);
            simd_stats.observe(simd, scalar, reference);
            tiled_stats.observe(tiled, scalar, reference);
        }
        let scalar_control = scalar_stats.finish();
        let rejected_simdgroup_balanced_tree_proxy = simd_stats.finish();
        let tiled_neumaier_candidate = tiled_stats.finish();
        input_reports.push(InputProjectionReport {
            input_family: family.name(),
            sampled_rows: rows.clone(),
            tiled_precision_not_worse_than_scalar_control: tiled_not_worse_than_scalar(
                &tiled_neumaier_candidate,
                &scalar_control,
            ),
            scalar_control,
            rejected_simdgroup_balanced_tree_proxy,
            tiled_neumaier_candidate,
        });
    }
    ProjectionReport {
        binding: matrix_binding(matrix),
        input_reports,
    }
}

fn silu_f32(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn silu_f64(value: f64) -> f64 {
    value / (1.0 + (-value).exp())
}

fn run_matrix_all_rows(
    matrix: &PackedMatrix,
    input: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f64>) {
    let mut scalar = Vec::with_capacity(matrix.rows);
    let mut simd = Vec::with_capacity(matrix.rows);
    let mut tiled = Vec::with_capacity(matrix.rows);
    let mut reference = Vec::with_capacity(matrix.rows);
    for row in 0..matrix.rows {
        let terms = matrix.terms(row, input);
        scalar.push(scalar_reduce(&terms));
        simd.push(rejected_simdgroup_proxy_reduce(&terms));
        tiled.push(tiled_neumaier_reduce(&terms, matrix.group_size));
        reference.push(fp64_reference(&terms));
    }
    (scalar, simd, tiled, reference)
}

fn max_abs_delta(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
        .fold(0.0, f64::max)
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn expert_chain_report(
    gate: &PackedMatrix,
    up: &PackedMatrix,
    down: &PackedMatrix,
) -> Result<ExpertChainReport, Box<dyn Error>> {
    if gate.cols != up.cols || gate.rows != up.rows || down.cols != gate.rows || down.rows != 2048 {
        return Err(
            "selected expert organs do not form the expected Qwen30 gate/up/down chain".into(),
        );
    }
    let mut input_reports = Vec::new();
    for family in InputFamily::ALL {
        let input = deterministic_input(family, gate.cols);
        let (gate_scalar, gate_simd, gate_tiled, gate_reference) =
            run_matrix_all_rows(gate, &input);
        let (up_scalar, up_simd, up_tiled, up_reference) = run_matrix_all_rows(up, &input);

        let scalar_swiglu = gate_scalar
            .iter()
            .zip(&up_scalar)
            .map(|(&g, &u)| silu_f32(g) * u)
            .collect::<Vec<_>>();
        let simd_swiglu = gate_simd
            .iter()
            .zip(&up_simd)
            .map(|(&g, &u)| silu_f32(g) * u)
            .collect::<Vec<_>>();
        let tiled_swiglu = gate_tiled
            .iter()
            .zip(&up_tiled)
            .map(|(&g, &u)| silu_f32(g) * u)
            .collect::<Vec<_>>();
        let reference_swiglu = gate_reference
            .iter()
            .zip(&up_reference)
            .map(|(&g, &u)| silu_f64(g) * u)
            .collect::<Vec<_>>();

        let (down_scalar, _, _, _) = run_matrix_all_rows(down, &scalar_swiglu);
        let (_, _, _, down_reference) = run_matrix_all_rows_f64_reference(down, &reference_swiglu);
        let (_, down_simd_from_chain, _, _) = run_matrix_all_rows(down, &simd_swiglu);
        let (_, _, down_tiled_from_chain, _) = run_matrix_all_rows(down, &tiled_swiglu);
        let mut scalar_stats = ErrorAccumulator::default();
        let mut simd_stats = ErrorAccumulator::default();
        let mut tiled_stats = ErrorAccumulator::default();
        for index in 0..down.rows {
            scalar_stats.observe(
                down_scalar[index],
                down_scalar[index],
                down_reference[index],
            );
            simd_stats.observe(
                down_simd_from_chain[index],
                down_scalar[index],
                down_reference[index],
            );
            tiled_stats.observe(
                down_tiled_from_chain[index],
                down_scalar[index],
                down_reference[index],
            );
        }
        let scalar_argmax = argmax(&down_scalar);
        let simd_argmax = argmax(&down_simd_from_chain);
        let tiled_argmax = argmax(&down_tiled_from_chain);
        let down_scalar_control = scalar_stats.finish();
        let down_rejected_simdgroup_balanced_tree_proxy = simd_stats.finish();
        let down_tiled_neumaier_candidate = tiled_stats.finish();
        input_reports.push(ChainInputReport {
            input_family: family.name(),
            swiglu_max_abs_delta_simd_vs_scalar: max_abs_delta(&simd_swiglu, &scalar_swiglu),
            swiglu_max_abs_delta_tiled_vs_scalar: max_abs_delta(&tiled_swiglu, &scalar_swiglu),
            tiled_precision_not_worse_than_scalar_control: tiled_not_worse_than_scalar(
                &down_tiled_neumaier_candidate,
                &down_scalar_control,
            ),
            down_scalar_control,
            down_rejected_simdgroup_balanced_tree_proxy,
            down_tiled_neumaier_candidate,
            downstream_projection_argmax_not_lm_head_or_token: BTreeMap::from([
                ("scalar_control", scalar_argmax),
                ("rejected_simdgroup_proxy", simd_argmax),
                ("tiled_neumaier_candidate", tiled_argmax),
            ]),
            same_downstream_projection_argmax_as_scalar: BTreeMap::from([
                ("rejected_simdgroup_proxy", simd_argmax == scalar_argmax),
                ("tiled_neumaier_candidate", tiled_argmax == scalar_argmax),
            ]),
        });
    }
    Ok(ExpertChainReport {
        description: "Real admitted layer-0 expert-0 gate/up/SwiGLU/down chain on deterministic activation families. This is numerical amplification characterization only; it is not a complete layer, lm-head, sampled token, or generation test.",
        source_bound_organs: vec![matrix_binding(gate), matrix_binding(up), matrix_binding(down)],
        input_reports,
    })
}

fn run_matrix_all_rows_f64_reference(
    matrix: &PackedMatrix,
    input: &[f64],
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f64>) {
    let mut reference = Vec::with_capacity(matrix.rows);
    for row in 0..matrix.rows {
        let mut sum = 0.0f64;
        for col in 0..matrix.cols {
            // Decode the exact stored FP16 scale/sign, but retain the supplied
            // predecessor activation and accumulation in FP64 for the
            // independent numerical reference.
            let flat = row * matrix.cols + col;
            let group = flat / matrix.group_size;
            let scale_offset = matrix.scale_offset + group * std::mem::size_of::<u16>();
            let scale = f64::from(
                f16::from_bits(u16::from_le_bytes([
                    matrix.payload[scale_offset],
                    matrix.payload[scale_offset + 1],
                ]))
                .to_f32(),
            );
            let group_bit = flat % matrix.group_size;
            let sign_byte = matrix.payload
                [matrix.sign_offset + group * (matrix.group_size / 8) + group_bit / 8];
            let positive = ((sign_byte >> (group_bit % 8)) & 1) != 0;
            sum += if positive { scale } else { -scale } * input[col];
        }
        reference.push(sum);
    }
    // This helper is used only for its FP64 vector. Keeping the tuple shape
    // aligned with `run_matrix_all_rows` prevents callers from accidentally
    // treating the reference as a runtime candidate.
    (Vec::new(), Vec::new(), Vec::new(), reference)
}

fn candidate_contract() -> Result<CandidateContractReport, Box<dyn Error>> {
    let shader_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("shaders/qwen_binary_precision_safe_tiled_candidate.metal")
        .canonicalize()?;
    let shader_sha256 = sha256_hex(&fs::read(&shader_path)?);
    Ok(CandidateContractReport {
        shader_path: shader_path.display().to_string(),
        shader_sha256,
        shader_entry_point: "qwen_binary_sign_scale_matvec_tiled_neumaier_candidate",
        geometry: "one 32-lane SIMDgroup owns one row; lane N owns group N; eight rows per 256-thread threadgroup; admitted Qwen30 bound: group_size=128 and groups_per_row<=32",
        cpu_algorithm: "serial scalar within each 128-value group plus deterministic group-order Neumaier compensation; not bit-identical to scalar control by construction",
        unregistered: true,
        gpu_execution_performed: false,
    })
}

fn integration_criteria() -> IntegrationCriteria {
    IntegrationCriteria {
        precision_gate: vec![
            "The exact manifest raw digest, manifest seal, every selected payload digest, group_size=128, and expected Qwen30 shapes must match this report before a later GPU trial is comparable.",
            "For every source-bound projection/input family and the real gate/up/down chain, the tiled candidate must stay finite and must not have worse FP64-reference relative-L2 or max-absolute error than the scalar control. A failure rejects the candidate before GPU use.",
            "The rejected SIMD proxy is characterization only: no proxy metric may override the prior real prompt-B exact-token failure.",
        ],
        gpu_component_gate: vec![
            "Under an exclusive Qwen30 GPU lease, compile this unregistered shader in an isolated test library and compare its output to the matching CPU tiled-Neumaier reference for all listed Qwen30 shapes and deterministic vectors.",
            "GPU component parity must be finite for every output and within abs_error <= 5e-5 * max(1, abs(cpu_tiled_output)); any larger discrepancy rejects the shader implementation rather than relaxing the end-to-end gate.",
            "The trial must record the candidate shader SHA, isolated host target SHA, current artifact manifest seal, runtime executable SHA, device name, and command topology.",
        ],
        full_model_gate: vec![
            "A candidate that passes component parity must separately re-run all-layer BOS/control comparison and both source-template prompt A and prompt B feedback loops against the current scalar control.",
            "Exact completion token identity for every tested feedback token is mandatory. Prompt B previously diverged for the rejected SIMD candidate, so matching prompt A is insufficient.",
            "Only after the complete native runtime, HCLI, capability, restart, and clean-performance gates are independently re-earned may any TG/TPS gate consider the candidate. This laboratory cannot satisfy any of those gates.",
        ],
        disallowed_promotions: vec![
            "No edit or registration of the live Qwen30 runtime, watcher, server, registry, canonical runtime receipt, HCLI receipt, TPS receipt, TG receipt, or tournament gate is authorized by this target.",
            "No component timing, CPU timing, proxy argmax, or downstream-projection argmax in this report is BASE_TRUE_TPS or a generation/capability result.",
        ],
    }
}

fn write_report_atomic(path: &Path, report: &Report) -> Result<(), Box<dyn Error>> {
    let parent = path.parent().ok_or("output path has no parent")?;
    if !parent.is_dir() {
        return Err(format!(
            "output parent does not exist or is not a directory: {}",
            parent.display()
        )
        .into());
    }
    let body = serde_json::to_vec_pretty(report)?;
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    fs::write(&temporary, body)?;
    fs::rename(&temporary, path)?;
    Ok(())
}

fn run(args: Args) -> Result<(), Box<dyn Error>> {
    let (source_binding, matrices) = load_selected_matrices(
        &args.manifest,
        &args.expected_manifest_document_sha256,
        &args.expected_manifest_seal_sha256,
    )?;
    let mut projection_reports = Vec::new();
    for spec in PROJECTION_SPECS {
        projection_reports.push(projection_report(
            matrices
                .get(spec.category)
                .ok_or("selected matrix disappeared after binding")?,
            args.sample_rows,
        ));
    }
    let expert_chain = expert_chain_report(
        matrices.get("expert_gate").ok_or("missing expert gate")?,
        matrices.get("expert_up").ok_or("missing expert up")?,
        matrices.get("expert_down").ok_or("missing expert down")?,
    )?;
    let candidate_contract = candidate_contract()?;
    let projection_input_cases = projection_reports
        .iter()
        .map(|report| report.input_reports.len())
        .sum::<usize>();
    let real_expert_chain_cases = expert_chain.input_reports.len();
    let all_cases_passed = projection_reports.iter().all(|report| {
        report
            .input_reports
            .iter()
            .all(|input| input.tiled_precision_not_worse_than_scalar_control)
    }) && expert_chain
        .input_reports
        .iter()
        .all(|input| input.tiled_precision_not_worse_than_scalar_control);
    let cpu_precision_gate = CpuPrecisionGateReport {
        status: if all_cases_passed {
            "CPU_PRECISION_GATE_PASSED_NOT_GPU_OR_RUNTIME_ADMISSION"
        } else {
            "CPU_PRECISION_GATE_REJECTED_TILED_CANDIDATE"
        },
        projection_input_cases,
        real_expert_chain_cases,
        all_cases_passed,
        rule: "All source-bound cases must be finite and tiled-Neumaier max-absolute and relative-L2 error against the FP64 reference must be no worse than scalar control. This numerical CPU gate does not establish Metal parity or full-model token parity.",
    };
    let mut report = Report {
        schema: SCHEMA,
        status: "CPU_ONLY_SOURCE_BOUND_REDUCTION_CHARACTERIZATION_NOT_GPU_OR_RUNTIME_RECEIPT",
        source_binding,
        candidate_contract,
        cpu_precision_gate,
        projection_reports,
        real_expert_chain_error_amplification: expert_chain,
        integration_criteria: integration_criteria(),
        claim_boundary: vec![
            "CPU-only: no Metal context, GPU dispatch, runtime mutation, watcher mutation, or registry mutation occurred.",
            "The complete Qwen30 native runtime remains controlled by its separate canonical receipt and scalar kernel unless a later candidate passes every listed gate.",
            "This report does not claim coherent generation, HCLI, BASE_TRUE_TPS, TG10, TG3, Agent OS qualification, or tournament readiness.",
        ],
        unsealed_preimage_sha256: String::new(),
    };
    let preimage = serde_json::to_vec(&report)?;
    report.unsealed_preimage_sha256 = sha256_hex(&preimage);
    write_report_atomic(&args.out, &report)
}

fn main() {
    match parse_args().and_then(run) {
        Ok(()) => {}
        Err(error) => {
            eprintln!("ascension_qwen30_packed_matvec_exactness: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_control_order_is_repeatable_and_bit_stable() {
        let terms = (0..GROUP_SIZE * 2)
            .map(|index| {
                let sign = if index % 3 == 0 { -1.0 } else { 1.0 };
                sign * ((index % 19) as f32 + 0.125) * 0.03125
            })
            .collect::<Vec<_>>();
        let first = scalar_reduce(&terms);
        for _ in 0..32 {
            assert_eq!(scalar_reduce(&terms).to_bits(), first.to_bits());
        }
    }

    #[test]
    fn strided_simd_proxy_has_a_distinct_associativity_surface() {
        let mut terms = vec![0.0f32; GROUP_SIZE * 2];
        terms[0] = 16_777_216.0;
        terms[1] = 1.0;
        terms[2] = -16_777_216.0;
        terms[32] = 3.0;
        let scalar = scalar_reduce(&terms);
        let simd = rejected_simdgroup_proxy_reduce(&terms);
        assert_ne!(scalar.to_bits(), simd.to_bits());
        assert_eq!(scalar, 3.0);
        assert_eq!(simd, 5.0);
    }

    #[test]
    fn tiled_neumaier_recovers_cancellation_lost_by_scalar_control() {
        let mut terms = vec![0.0f32; GROUP_SIZE];
        terms[0] = 16_777_216.0;
        terms[1] = 1.0;
        terms[2] = -16_777_216.0;
        let scalar = scalar_reduce(&terms);
        let tiled = tiled_neumaier_reduce(&terms, GROUP_SIZE);
        let reference = fp64_reference(&terms);
        assert_eq!(scalar, 0.0);
        assert_eq!(tiled, 1.0);
        assert_eq!(reference, 1.0);
    }

    #[test]
    fn qwen30_projection_shapes_fit_tiled_candidate_contract() {
        for spec in PROJECTION_SPECS {
            assert_eq!(spec.shape[1] % GROUP_SIZE, 0, "{}", spec.tensor_name);
            assert!(
                spec.shape[1] / GROUP_SIZE <= SIMD_LANES,
                "{} exceeds one-lane-per-group candidate bound",
                spec.tensor_name
            );
        }
    }

    #[test]
    fn deterministic_inputs_are_reproducible_and_finite() {
        for family in InputFamily::ALL {
            let left = deterministic_input(family, 4096);
            let right = deterministic_input(family, 4096);
            assert_eq!(left, right);
            assert!(left.iter().all(|value| value.is_finite()));
        }
    }
}
