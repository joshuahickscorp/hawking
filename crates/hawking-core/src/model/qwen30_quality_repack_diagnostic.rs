//! Typed, non-serving construction boundary for the isolated Qwen30 HQ30GR2
//! quality candidate.
//!
//! The normal [`super::qwen30_complete_runtime::Qwen30CompleteNativeRuntime`]
//! deliberately accepts only direct `HQ30G1B1` tensors.  This module does not
//! weaken that control.  Instead it turns an already admitted HQ30GR2 catalog
//! into a typed, immutable payload view which a future bounded all-layer
//! *diagnostic* runtime can consume.  It creates no Metal context, opens no
//! source BF16 weights, serves no endpoint, and executes no model token.

use super::qwen_complete_binary::{
    qwen30_quality_residual_matvec_f64, CompleteBinaryArtifact, CompleteBinaryTensor,
    Qwen30QualityRepackArtifact, Qwen30QualityRepackTensorLayout,
    Qwen30QualityRepackVerifiedTensor, QwenCompleteBinaryModel,
};
use crate::{Error, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

#[cfg(target_os = "macos")]
use crate::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};

pub const QWEN30_QUALITY_REPACK_DIAGNOSTIC_TENSOR_COUNT: usize = 18_867;
pub const QWEN30_QUALITY_REPACK_SPARSE_GATE_UP_KERNEL: &str =
    "qwen30_quality_repack_sparse_gate_up_swiglu";

const L0_E0_GATE: &str = "model.layers.0.mlp.experts.0.gate_proj.weight";
const L0_E0_UP: &str = "model.layers.0.mlp.experts.0.up_proj.weight";
const QWEN30_MOE_ROWS: usize = 768;
const QWEN30_HIDDEN_COLS: usize = 2048;
const QWEN30_GROUP_SIZE: usize = 128;

fn candidate_error(message: impl Into<String>) -> Error {
    Error::Model(format!(
        "qwen30 HQ30GR2 diagnostic catalog: {}",
        message.into()
    ))
}

/// Immutable typed candidate catalog.  Constructing it is CPU-only and
/// requires an already admitted complete HQ30GR2 artifact.  The catalog owns
/// the admission object so its byte snapshots remain alive for any later
/// device-residency construction.
#[derive(Clone, Debug)]
pub struct Qwen30QualityRepackDiagnosticCatalog {
    artifact: Qwen30QualityRepackArtifact,
    typed_tensors: BTreeMap<String, Qwen30QualityRepackVerifiedTensor>,
    direct_tensor_count: usize,
    sparse_residual_tensor_count: usize,
}

/// Exact host-side ABI contract for the separate direct-base-plus-sparse
/// residual gate/up Metal kernel.  It is deliberately data-only: an executor
/// must still allocate device buffers and demonstrate CPU/device parity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen30QualityRepackSparseGateUpDispatch {
    pub kernel_name: &'static str,
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
    pub gate_residual_count: usize,
    pub up_residual_count: usize,
    pub exact_non_fma_scalar_order_required: bool,
    pub direct_fallback_for_sparse_residual_forbidden: bool,
}

impl Qwen30QualityRepackDiagnosticCatalog {
    /// Build a typed view from a fully admitted candidate.  Every one of the
    /// 18,867 immutable snapshots is retyped against its admission header;
    /// selected HQ30GR2 organs cannot become direct tensors by omission.
    pub fn from_admitted(artifact: Qwen30QualityRepackArtifact) -> Result<Self> {
        if artifact.tensors.len() != QWEN30_QUALITY_REPACK_DIAGNOSTIC_TENSOR_COUNT
            || artifact.verified_payload_count() != QWEN30_QUALITY_REPACK_DIAGNOSTIC_TENSOR_COUNT
            || !artifact.has_complete_verified_payload_cache()
        {
            return Err(candidate_error(
                "candidate admission does not retain exactly one immutable payload snapshot for every Qwen30 tensor",
            ));
        }
        let expected_sparse = BTreeSet::from([L0_E0_GATE.to_owned(), L0_E0_UP.to_owned()]);
        let selected_sparse = artifact
            .selected_residual_organs
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        if selected_sparse != expected_sparse || artifact.selected_residual_organs.len() != 2 {
            return Err(candidate_error(
                "candidate selected residual organs are not exactly L0/E0 gate/up",
            ));
        }

        let mut typed_tensors = BTreeMap::new();
        let mut direct_tensor_count = 0usize;
        let mut sparse_residual_tensor_count = 0usize;
        for (name, layout) in &artifact.tensors {
            let typed = artifact.verified_typed_tensor(name)?;
            let selected = expected_sparse.contains(name);
            match (&layout.layout, &typed, selected) {
                (
                    Qwen30QualityRepackTensorLayout::Direct(_),
                    Qwen30QualityRepackVerifiedTensor::Direct { .. },
                    false,
                ) => direct_tensor_count = direct_tensor_count.saturating_add(1),
                (
                    Qwen30QualityRepackTensorLayout::SparseResidual(_),
                    Qwen30QualityRepackVerifiedTensor::SparseResidual { .. },
                    true,
                ) => sparse_residual_tensor_count = sparse_residual_tensor_count.saturating_add(1),
                _ => {
                    return Err(candidate_error(format!(
                        "tensor {name:?} layout/type does not match the sealed selected-residual set"
                    )));
                }
            }
            if typed_tensors.insert(name.clone(), typed).is_some() {
                return Err(candidate_error(format!(
                    "candidate typed catalog repeats tensor {name:?}"
                )));
            }
        }
        if typed_tensors.len() != artifact.tensors.len()
            || direct_tensor_count != QWEN30_QUALITY_REPACK_DIAGNOSTIC_TENSOR_COUNT - 2
            || sparse_residual_tensor_count != 2
        {
            return Err(candidate_error(
                "candidate typed catalog direct/sparse layout counts are incomplete",
            ));
        }
        Ok(Self {
            artifact,
            typed_tensors,
            direct_tensor_count,
            sparse_residual_tensor_count,
        })
    }

    pub fn manifest_seal_sha256(&self) -> &str {
        &self.artifact.manifest_seal_sha256
    }

    pub fn source_revision(&self) -> &str {
        &self.artifact.source_revision
    }

    pub fn direct_tensor_count(&self) -> usize {
        self.direct_tensor_count
    }

    pub fn sparse_residual_tensor_count(&self) -> usize {
        self.sparse_residual_tensor_count
    }

    /// Number of immutable candidate payload snapshots retained by the
    /// admission which constructed this typed catalog.  It is exposed for
    /// diagnostic binding only; individual payloads remain typed-only.
    pub fn verified_payload_count(&self) -> usize {
        self.artifact.verified_payload_count()
    }

    /// Typed immutable payload for a future one-time device-residency build.
    /// There is intentionally no untyped/raw accessor in this diagnostic view.
    pub fn typed_tensor(&self, name: &str) -> Result<&Qwen30QualityRepackVerifiedTensor> {
        self.typed_tensors.get(name).ok_or_else(|| {
            candidate_error(format!("candidate typed catalog has no tensor {name:?}"))
        })
    }

    /// Validate the two changed L0/E0 payloads against the narrow shader ABI.
    /// This does not dispatch Metal; it only makes an accidental direct
    /// fallback or geometry drift impossible to hide before a device lease.
    pub fn sparse_gate_up_dispatch(&self) -> Result<Qwen30QualityRepackSparseGateUpDispatch> {
        let gate = self.typed_tensor(L0_E0_GATE)?;
        let up = self.typed_tensor(L0_E0_UP)?;
        let extract = |name: &str, tensor: &Qwen30QualityRepackVerifiedTensor| match tensor {
            Qwen30QualityRepackVerifiedTensor::SparseResidual { header, .. } => {
                if header.shape != [QWEN30_MOE_ROWS, QWEN30_HIDDEN_COLS]
                    || header.base.group_size != QWEN30_GROUP_SIZE
                    || header.residual_count == 0
                {
                    return Err(candidate_error(format!(
                        "sparse candidate tensor {name:?} has incompatible gate/up geometry"
                    )));
                }
                Ok(header.residual_count)
            }
            Qwen30QualityRepackVerifiedTensor::Direct { .. } => Err(candidate_error(format!(
                "selected sparse candidate tensor {name:?} reached a direct fallback"
            ))),
        };
        Ok(Qwen30QualityRepackSparseGateUpDispatch {
            kernel_name: QWEN30_QUALITY_REPACK_SPARSE_GATE_UP_KERNEL,
            rows: QWEN30_MOE_ROWS,
            cols: QWEN30_HIDDEN_COLS,
            group_size: QWEN30_GROUP_SIZE,
            gate_residual_count: extract(L0_E0_GATE, gate)?,
            up_residual_count: extract(L0_E0_UP, up)?,
            exact_non_fma_scalar_order_required: true,
            direct_fallback_for_sparse_residual_forbidden: true,
        })
    }

    /// Scalar, exact-format CPU oracle for the two selected HQ30GR2 organs.
    ///
    /// This is deliberately limited to a single gate/up/SwiGLU activation
    /// vector.  It consumes the immutable admission snapshots through the
    /// HQ30GR2 parser and never opens a source weight or materializes a dense
    /// matrix.  A future device invocation must use this only as a parity
    /// reference; it is not a host fallback for the all-layer diagnostic.
    pub fn sparse_gate_up_cpu_oracle_f64(&self, input: &[f64]) -> Result<Vec<f64>> {
        let specification = self.sparse_gate_up_dispatch()?;
        if input.len() != specification.cols {
            return Err(candidate_error(format!(
                "sparse gate/up CPU oracle input has {} values, expected {}",
                input.len(),
                specification.cols
            )));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(candidate_error(
                "sparse gate/up CPU oracle input contains a non-finite value",
            ));
        }
        let matvec = |name: &str| -> Result<Vec<f64>> {
            let typed = self.typed_tensor(name)?;
            let Qwen30QualityRepackVerifiedTensor::SparseResidual { header, payload } = typed
            else {
                return Err(candidate_error(format!(
                    "selected sparse candidate tensor {name:?} reached a direct CPU fallback"
                )));
            };
            if header.shape != [specification.rows, specification.cols]
                || header.base.group_size != specification.group_size
            {
                return Err(candidate_error(format!(
                    "selected sparse candidate tensor {name:?} drifted from its dispatch geometry"
                )));
            }
            let (observed_header, output) = qwen30_quality_residual_matvec_f64(payload, input)?;
            if observed_header != *header || output.len() != specification.rows {
                return Err(candidate_error(format!(
                    "selected sparse candidate tensor {name:?} CPU oracle differs from its typed admission snapshot"
                )));
            }
            Ok(output)
        };
        let gate = matvec(L0_E0_GATE)?;
        let up = matvec(L0_E0_UP)?;
        let mut activation = Vec::with_capacity(specification.rows);
        for (row, (gate_value, up_value)) in gate.into_iter().zip(up).enumerate() {
            let silu = gate_value / (1.0 + (-gate_value).exp());
            let value = silu * up_value;
            if !value.is_finite() {
                return Err(candidate_error(format!(
                    "sparse gate/up CPU oracle produced a non-finite activation at row {row}"
                )));
            }
            activation.push(value);
        }
        if activation.len() != specification.rows {
            return Err(candidate_error(
                "sparse gate/up CPU oracle activation length differs from typed geometry",
            ));
        }
        Ok(activation)
    }

    /// Produce a crate-private direct-base view solely for the existing
    /// complete Qwen30 graph machinery. Every ordinary tensor remains the
    /// candidate admission snapshot; the two HQ30GR2 tensors expose their
    /// embedded direct bases here and must be intercepted by the separately
    /// typed sparse gate/up device pair. This is not a direct artifact
    /// admission and is never exposed to a serving path.
    pub(crate) fn direct_base_view_for_diagnostic_runtime(&self) -> Result<CompleteBinaryArtifact> {
        let mut tensors = BTreeMap::new();
        let mut verified_payloads = BTreeMap::new();
        for (name, original) in &self.artifact.tensors {
            let (header, payload): (_, Arc<[u8]>) = match self.typed_tensor(name)? {
                Qwen30QualityRepackVerifiedTensor::Direct { header, payload } => {
                    (header.clone(), payload.clone())
                }
                Qwen30QualityRepackVerifiedTensor::SparseResidual { header, payload } => {
                    let base_end = header
                        .base_offset
                        .checked_add(header.base_payload_bytes)
                        .ok_or_else(|| {
                            candidate_error(format!(
                                "selected candidate tensor {name:?} base range overflows"
                            ))
                        })?;
                    let base = payload.get(header.base_offset..base_end).ok_or_else(|| {
                        candidate_error(format!("selected candidate tensor {name:?} immutable base payload is truncated"))
                    })?;
                    (header.base.clone(), Arc::<[u8]>::from(base.to_vec()))
                }
            };
            let tensor = CompleteBinaryTensor {
                tensor_name: name.clone(),
                source_shard: original.source_shard.clone(),
                source_shard_sha256: original.source_shard_sha256.clone(),
                source_dtype: original.source_dtype.clone(),
                // This still names the physical HQ30GR2 file. Therefore a
                // mistaken file reread fails closed; native execution reads
                // only the immutable payload snapshot below.
                artifact_path: original.artifact_path.clone(),
                artifact_sha256: original.artifact_sha256.clone(),
                header,
            };
            if tensors.insert(name.clone(), tensor).is_some()
                || verified_payloads.insert(name.clone(), payload).is_some()
            {
                return Err(candidate_error(format!(
                    "candidate direct-base view duplicates tensor {name:?}"
                )));
            }
        }
        if tensors.len() != QWEN30_QUALITY_REPACK_DIAGNOSTIC_TENSOR_COUNT
            || verified_payloads.len() != tensors.len()
        {
            return Err(candidate_error(
                "candidate direct-base view does not retain every immutable payload snapshot",
            ));
        }
        Ok(CompleteBinaryArtifact {
            model: QwenCompleteBinaryModel::Qwen30Coder,
            manifest_path: self.artifact.manifest_path.clone(),
            manifest_seal_sha256: self.artifact.manifest_seal_sha256.clone(),
            source_audit_path: self.artifact.source_audit_path.clone(),
            source_audit_seal_sha256: self.artifact.source_audit_seal_sha256.clone(),
            source_revision: self.artifact.source_revision.clone(),
            source_index_path: self.artifact.source_index_path.clone(),
            source_weight_elements: self.artifact.source_weight_elements,
            tensor_payload_bytes: self.artifact.tensor_payload_bytes,
            tensors,
            verified_payloads,
        })
    }
}

// This narrowly mirrors the existing Qwen complete-runtime scalar ABI.  It
// stays local so the generic Metal dispatcher need not expose a broad mutable
// scalar surface merely for this isolated candidate diagnostic.
#[cfg(target_os = "macos")]
trait Qwen30QualityRepackSetScalar {
    fn qwen30_quality_set_u32(&self, index: u64, value: u32);
}

#[cfg(target_os = "macos")]
impl Qwen30QualityRepackSetScalar for ::metal::ComputeCommandEncoderRef {
    #[inline(always)]
    fn qwen30_quality_set_u32(&self, index: u64, value: u32) {
        self.set_bytes(
            index,
            std::mem::size_of::<u32>() as u64,
            &value as *const u32 as *const _,
        );
    }
}

#[cfg(target_os = "macos")]
fn bytes_for_f32(elements: usize, label: &str) -> Result<usize> {
    elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| candidate_error(format!("{label} byte count overflows usize")))
}

#[cfg(target_os = "macos")]
fn u32_checked(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| candidate_error(format!("{label} exceeds the Metal uint ABI")))
}

/// Device-resident direct base plus sparse correction payload for exactly one
/// selected HQ30GR2 organ.  It has no direct-only constructor by design.
#[cfg(target_os = "macos")]
#[derive(Clone)]
struct Qwen30QualityRepackGpuSparseTensor {
    signs: PinnedBuffer,
    scales: PinnedBuffer,
    residual_indices: PinnedBuffer,
    residual_values: PinnedBuffer,
    residual_count: usize,
}

#[cfg(target_os = "macos")]
impl Qwen30QualityRepackGpuSparseTensor {
    fn upload(
        context: &MetalContext,
        name: &str,
        tensor: &Qwen30QualityRepackVerifiedTensor,
    ) -> Result<Self> {
        let Qwen30QualityRepackVerifiedTensor::SparseResidual { header, payload } = tensor else {
            return Err(candidate_error(format!(
                "selected candidate tensor {name:?} must be HQ30GR2, not a direct fallback"
            )));
        };
        if header.shape != [QWEN30_MOE_ROWS, QWEN30_HIDDEN_COLS]
            || header.base.group_size != QWEN30_GROUP_SIZE
            || header.residual_count == 0
        {
            return Err(candidate_error(format!(
                "selected candidate tensor {name:?} has incompatible sparse gate/up geometry"
            )));
        }
        let base_end = header
            .base_offset
            .checked_add(header.base_payload_bytes)
            .ok_or_else(|| {
                candidate_error(format!("selected tensor {name:?} base range overflows"))
            })?;
        let base = payload.get(header.base_offset..base_end).ok_or_else(|| {
            candidate_error(format!(
                "selected tensor {name:?} immutable base payload is truncated"
            ))
        })?;
        let scales = base
            .get(header.base.scale_offset..header.base.sign_offset)
            .ok_or_else(|| {
                candidate_error(format!("selected tensor {name:?} scales are truncated"))
            })?;
        let signs = base
            .get(header.base.sign_offset..header.base.payload_bytes)
            .ok_or_else(|| {
                candidate_error(format!("selected tensor {name:?} signs are truncated"))
            })?;
        let residual_indices = payload
            .get(header.indices_offset..header.values_offset)
            .ok_or_else(|| {
                candidate_error(format!(
                    "selected tensor {name:?} residual indices are truncated"
                ))
            })?;
        let residual_values = payload
            .get(header.values_offset..header.payload_bytes)
            .ok_or_else(|| {
                candidate_error(format!(
                    "selected tensor {name:?} residual values are truncated"
                ))
            })?;
        if residual_indices.len() != header.residual_count * std::mem::size_of::<u32>()
            || residual_values.len() != header.residual_count * std::mem::size_of::<u16>()
        {
            return Err(candidate_error(format!(
                "selected tensor {name:?} sparse residual byte geometry differs from admission header"
            )));
        }
        Ok(Self {
            signs: context.new_buffer_with_bytes_checked(signs)?,
            scales: context.new_buffer_with_bytes_checked(scales)?,
            residual_indices: context.new_buffer_with_bytes_checked(residual_indices)?,
            residual_values: context.new_buffer_with_bytes_checked(residual_values)?,
            residual_count: header.residual_count,
        })
    }
}

/// The only device upload/dispatch surface for the two selected HQ30GR2
/// gate/up tensors.  Constructing it requires an explicit Metal context, but
/// none is created by this module; callers must already hold a diagnostic GPU
/// lease.  It is not used by the live scalar Qwen30 server.
#[cfg(target_os = "macos")]
#[derive(Clone)]
pub struct Qwen30QualityRepackSparseGateUpDevicePair {
    gate: Qwen30QualityRepackGpuSparseTensor,
    up: Qwen30QualityRepackGpuSparseTensor,
    spec: Qwen30QualityRepackSparseGateUpDispatch,
}

#[cfg(target_os = "macos")]
impl Qwen30QualityRepackSparseGateUpDevicePair {
    /// Upload the two typed sparse organs from immutable admission snapshots.
    /// This method cannot accept a raw path or an ordinary direct tensor.
    pub fn upload(
        context: &MetalContext,
        catalog: &Qwen30QualityRepackDiagnosticCatalog,
    ) -> Result<Self> {
        let spec = catalog.sparse_gate_up_dispatch()?;
        let gate = Qwen30QualityRepackGpuSparseTensor::upload(
            context,
            L0_E0_GATE,
            catalog.typed_tensor(L0_E0_GATE)?,
        )?;
        let up = Qwen30QualityRepackGpuSparseTensor::upload(
            context,
            L0_E0_UP,
            catalog.typed_tensor(L0_E0_UP)?,
        )?;
        if gate.residual_count != spec.gate_residual_count
            || up.residual_count != spec.up_residual_count
        {
            return Err(candidate_error(
                "device upload residual counts differ from the typed dispatch specification",
            ));
        }
        Ok(Self { gate, up, spec })
    }

    pub fn specification(&self) -> &Qwen30QualityRepackSparseGateUpDispatch {
        &self.spec
    }

    /// Encode one route-major sparse gate/up/SwiGLU activation.  The caller
    /// supplies the device-produced L0 post-attention RMSNorm buffer and an
    /// output offset inside the normal route-major activation workspace.
    /// It never invokes a CPU projection or reads a source/BF16 weight.
    pub fn encode(
        &self,
        tcb: &mut TokenCommandBuffer<'_>,
        input: &PinnedBuffer,
        output: &PinnedBuffer,
        output_offset_bytes: usize,
    ) -> Result<()> {
        let input_required = bytes_for_f32(self.spec.cols, "sparse gate/up input")?;
        if (input.length() as usize) < input_required {
            return Err(candidate_error(
                "sparse gate/up input buffer is shorter than 2048 F32 values",
            ));
        }
        let output_required = bytes_for_f32(self.spec.rows, "sparse gate/up output")?;
        let output_end = output_offset_bytes
            .checked_add(output_required)
            .ok_or_else(|| candidate_error("sparse gate/up output offset overflows usize"))?;
        if output_end > output.length() as usize {
            return Err(candidate_error(
                "sparse gate/up output range exceeds the route-major activation workspace",
            ));
        }
        let rows = u32_checked(self.spec.rows, "sparse gate/up rows")?;
        let cols = u32_checked(self.spec.cols, "sparse gate/up cols")?;
        let group_size = u32_checked(self.spec.group_size, "sparse gate/up group size")?;
        let gate_residual_count =
            u32_checked(self.gate.residual_count, "gate sparse residual count")?;
        let up_residual_count = u32_checked(self.up.residual_count, "up sparse residual count")?;
        tcb.dispatch_threads(
            QWEN30_QUALITY_REPACK_SPARSE_GATE_UP_KERNEL,
            (rows, 1, 1),
            (rows.min(256).max(1), 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&self.gate.signs), 0);
                encoder.set_buffer(1, Some(&self.gate.scales), 0);
                encoder.set_buffer(2, Some(&self.gate.residual_indices), 0);
                encoder.set_buffer(3, Some(&self.gate.residual_values), 0);
                encoder.set_buffer(4, Some(&self.up.signs), 0);
                encoder.set_buffer(5, Some(&self.up.scales), 0);
                encoder.set_buffer(6, Some(&self.up.residual_indices), 0);
                encoder.set_buffer(7, Some(&self.up.residual_values), 0);
                encoder.set_buffer(8, Some(input), 0);
                encoder.set_buffer(9, Some(output), output_offset_bytes as u64);
                encoder.qwen30_quality_set_u32(10, gate_residual_count);
                encoder.qwen30_quality_set_u32(11, up_residual_count);
                encoder.qwen30_quality_set_u32(12, rows);
                encoder.qwen30_quality_set_u32(13, cols);
                encoder.qwen30_quality_set_u32(14, group_size);
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::qwen_complete_binary::{CompleteBinaryHeader, Qwen30QualityResidualHeader};
    use std::path::PathBuf;

    fn direct_header() -> CompleteBinaryHeader {
        CompleteBinaryHeader {
            version: 1,
            group_size: QWEN30_GROUP_SIZE,
            shape: vec![QWEN30_MOE_ROWS, QWEN30_HIDDEN_COLS],
            elements: QWEN30_MOE_ROWS * QWEN30_HIDDEN_COLS,
            groups: (QWEN30_MOE_ROWS * QWEN30_HIDDEN_COLS) / QWEN30_GROUP_SIZE,
            scale_offset: 0,
            sign_offset: 0,
            payload_bytes: 0,
        }
    }

    fn sparse_header() -> Qwen30QualityResidualHeader {
        Qwen30QualityResidualHeader {
            shape: vec![QWEN30_MOE_ROWS, QWEN30_HIDDEN_COLS],
            base: direct_header(),
            residual_count: 1,
            base_offset: 0,
            base_payload_bytes: 0,
            indices_offset: 0,
            values_offset: 0,
            payload_bytes: 0,
            indices_sha256: "0".repeat(64),
            values_sha256: "0".repeat(64),
        }
    }

    fn placeholder_artifact() -> Qwen30QualityRepackArtifact {
        Qwen30QualityRepackArtifact {
            manifest_path: PathBuf::from("/candidate/manifest.json"),
            manifest_seal_sha256: "0".repeat(64),
            source_audit_path: PathBuf::from("/candidate/source-audit.json"),
            source_audit_seal_sha256: "0".repeat(64),
            source_revision: "fixture".into(),
            source_index_path: PathBuf::from("/candidate/index.json"),
            source_weight_elements: 0,
            tensor_payload_bytes: 0,
            selected_residual_organs: vec![L0_E0_GATE.into(), L0_E0_UP.into()],
            payload_verification_workers: 1,
            tensors: BTreeMap::new(),
            verified_payloads: BTreeMap::new(),
        }
    }

    fn catalog(
        gate: Qwen30QualityRepackVerifiedTensor,
        up: Qwen30QualityRepackVerifiedTensor,
    ) -> Qwen30QualityRepackDiagnosticCatalog {
        Qwen30QualityRepackDiagnosticCatalog {
            artifact: placeholder_artifact(),
            typed_tensors: BTreeMap::from([(L0_E0_GATE.into(), gate), (L0_E0_UP.into(), up)]),
            direct_tensor_count: QWEN30_QUALITY_REPACK_DIAGNOSTIC_TENSOR_COUNT - 2,
            sparse_residual_tensor_count: 2,
        }
    }

    #[test]
    fn sparse_gate_up_kernel_contract_is_narrow_and_not_a_serving_selection() {
        assert_eq!(
            QWEN30_QUALITY_REPACK_SPARSE_GATE_UP_KERNEL,
            "qwen30_quality_repack_sparse_gate_up_swiglu"
        );
        assert_eq!(QWEN30_QUALITY_REPACK_DIAGNOSTIC_TENSOR_COUNT, 18_867);
        let source = crate::metal::SHADER_QWEN30_QUALITY_REPACK_SPARSE_GATE_UP;
        assert!(source.contains("kernel void qwen30_quality_repack_sparse_gate_up_swiglu("));
        assert!(source.contains("#pragma clang fp contract(off)"));
        assert!(!source.contains("fma("));
    }

    #[test]
    fn selected_pair_refuses_direct_fallback_before_any_device_dispatch() {
        let direct = || Qwen30QualityRepackVerifiedTensor::Direct {
            header: direct_header(),
            payload: Arc::from(Vec::<u8>::new()),
        };
        let error = catalog(direct(), direct())
            .sparse_gate_up_dispatch()
            .unwrap_err();
        assert!(error.to_string().contains("direct fallback"), "{error}");
    }

    #[test]
    fn cpu_oracle_rejects_bad_captured_vector_geometry_before_reading_payload() {
        let sparse = || Qwen30QualityRepackVerifiedTensor::SparseResidual {
            header: sparse_header(),
            payload: Arc::from(Vec::<u8>::new()),
        };
        let error = catalog(sparse(), sparse())
            .sparse_gate_up_cpu_oracle_f64(&[0.0])
            .unwrap_err();
        assert!(error.to_string().contains("2048"), "{error}");
    }
}
