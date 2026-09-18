//! Provider-neutral payload comparison for Flash source/native boundary controls.
//!
//! This owner compares already-admitted raw payload bytes.  It has no model,
//! Metal, source-checkpoint, receipt-admission, or promotion authority.  F32
//! metrics are diagnostic until an experiment supplies explicit bounds;
//! ordered I32 route IDs remain exact.

use crate::{Error, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};

pub const SCHEMA: &str = "hawking.flash.boundary_payload_comparison.v1";

/// Model-neutral identity for one ordered source/native diagnostic seam.
///
/// This is intentionally a core-only contract: it names what an adapter
/// observed without loading a source runtime, retaining a payload, or placing
/// diagnostic metadata on Hawking's resident token path.  The historical
/// module name remains because Flash is the first caller, not because the
/// contract is Flash-only.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct DiagnosticSemanticSeam {
    pub ordinal: usize,
    pub layer: Option<usize>,
    pub semantic_id: String,
    pub shape: Vec<usize>,
}

impl DiagnosticSemanticSeam {
    fn validate(&self) -> Result<()> {
        if self.semantic_id.trim().is_empty() {
            return Err(comparison_error("semantic seam id is empty"));
        }
        if self.shape.is_empty() || self.shape.contains(&0) {
            return Err(comparison_error(format!(
                "semantic seam {} has an empty or non-positive shape",
                self.semantic_id
            )));
        }
        Ok(())
    }
}

/// Validate a capture plan before an adapter compares payload bytes.
///
/// IDs may recur at different layers; the ordered `(ordinal, layer,
/// semantic_id, shape)` tuple is the identity.  This keeps generic model
/// organs such as repeated layer outputs from being rejected as duplicates.
pub fn validate_ordered_semantic_seams(seams: &[DiagnosticSemanticSeam]) -> Result<()> {
    if seams.is_empty() {
        return Err(comparison_error("semantic seam sequence is empty"));
    }
    for (expected_ordinal, seam) in seams.iter().enumerate() {
        seam.validate()?;
        if seam.ordinal != expected_ordinal {
            return Err(comparison_error(format!(
                "semantic seam ordinals must be contiguous from zero; expected {expected_ordinal}, got {}",
                seam.ordinal
            )));
        }
    }
    Ok(())
}

/// A declared numerical policy at one semantic seam.
///
/// It is a verification contract, not a kernel implementation or an accepted
/// tolerance.  A future model must opt into its own schedule explicitly.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct DiagnosticPrecisionSchedule {
    pub profile_id: String,
    pub logical_dtype: String,
    pub accumulation_dtype: String,
    pub rounding_point: String,
    pub storage_dtype: String,
    pub scope: String,
    pub reduction_topology: Option<String>,
    pub reduction_threadgroup_size: Option<usize>,
}

impl DiagnosticPrecisionSchedule {
    fn validate(&self) -> Result<()> {
        for (field, value) in [
            ("profile_id", &self.profile_id),
            ("logical_dtype", &self.logical_dtype),
            ("accumulation_dtype", &self.accumulation_dtype),
            ("rounding_point", &self.rounding_point),
            ("storage_dtype", &self.storage_dtype),
            ("scope", &self.scope),
        ] {
            if value.trim().is_empty() {
                return Err(comparison_error(format!(
                    "precision schedule {field} is empty"
                )));
            }
        }
        if self
            .reduction_topology
            .as_deref()
            .is_some_and(|topology| topology.trim().is_empty())
        {
            return Err(comparison_error(
                "precision schedule reduction topology is empty",
            ));
        }
        if self.reduction_threadgroup_size == Some(0) {
            return Err(comparison_error(
                "precision schedule reduction threadgroup size is zero",
            ));
        }
        Ok(())
    }
}

/// Refuse an observed schedule that differs from the predeclared schedule.
pub fn require_precision_schedule(
    expected: &DiagnosticPrecisionSchedule,
    observed: &DiagnosticPrecisionSchedule,
) -> Result<()> {
    expected.validate()?;
    observed.validate()?;
    if expected != observed {
        return Err(comparison_error(
            "observed precision schedule differs from the predeclared contract",
        ));
    }
    Ok(())
}

/// The first exact-payload mismatch in an already ordered diagnostic trace.
///
/// This identifies an observation boundary only.  It deliberately does not
/// assign semantic cause or authorize a numerical acceptance bound.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct DiagnosticFirstObservedDifference {
    pub ordinal: usize,
    pub layer: Option<usize>,
    pub semantic_id: String,
}

pub fn first_observed_payload_difference(
    seams: &[DiagnosticSemanticSeam],
    payloads_exact: &[bool],
) -> Result<Option<DiagnosticFirstObservedDifference>> {
    validate_ordered_semantic_seams(seams)?;
    if seams.len() != payloads_exact.len() {
        return Err(comparison_error(format!(
            "semantic seam count {} differs from payload comparison count {}",
            seams.len(),
            payloads_exact.len()
        )));
    }
    Ok(seams.iter().zip(payloads_exact).find_map(|(seam, exact)| {
        (!*exact).then(|| DiagnosticFirstObservedDifference {
            ordinal: seam.ordinal,
            layer: seam.layer,
            semantic_id: seam.semantic_id.clone(),
        })
    }))
}

fn comparison_error(detail: impl Into<String>) -> Error {
    Error::Model(format!("Flash boundary comparison: {}", detail.into()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut result = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut result, "{byte:02x}").expect("writing to String cannot fail");
    }
    result
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct FlashBoundaryContinuousMetrics {
    pub finite: bool,
    pub elements: usize,
    pub max_abs: f64,
    pub relative_l2: Option<f64>,
    pub cosine: Option<f64>,
    pub reference_l2_zero: bool,
    pub candidate_l2_zero: bool,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct FlashBoundaryPayloadComparison {
    pub schema: &'static str,
    pub dtype: &'static str,
    pub elements: usize,
    pub reference_sha256: String,
    pub candidate_sha256: String,
    pub exact_sha256: bool,
    pub ordered_i32_exact: Option<bool>,
    pub continuous: Option<FlashBoundaryContinuousMetrics>,
    pub numerical_acceptance_bound: Option<f64>,
    pub semantic_verdict_allowed: bool,
}

fn decode_f32(bytes: &[u8], expected_elements: usize, label: &str) -> Result<Vec<f32>> {
    if expected_elements == 0 || bytes.len() != expected_elements.saturating_mul(4) {
        return Err(comparison_error(format!(
            "{label} F32_LE extent differs: bytes={} expected_elements={expected_elements}",
            bytes.len()
        )));
    }
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(comparison_error(format!(
            "{label} F32_LE payload contains a non-finite value"
        )));
    }
    Ok(values)
}

fn decode_i32(bytes: &[u8], expected_elements: usize, label: &str) -> Result<Vec<i32>> {
    if expected_elements == 0 || bytes.len() != expected_elements.saturating_mul(4) {
        return Err(comparison_error(format!(
            "{label} I32_LE extent differs: bytes={} expected_elements={expected_elements}",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn continuous_metrics(reference: &[f32], candidate: &[f32]) -> FlashBoundaryContinuousMetrics {
    let mut max_abs = 0.0_f64;
    let mut error_squared = 0.0_f64;
    let mut reference_squared = 0.0_f64;
    let mut candidate_squared = 0.0_f64;
    let mut dot = 0.0_f64;
    for (&expected, &observed) in reference.iter().zip(candidate.iter()) {
        let expected = f64::from(expected);
        let observed = f64::from(observed);
        let error = observed - expected;
        max_abs = max_abs.max(error.abs());
        error_squared += error * error;
        reference_squared += expected * expected;
        candidate_squared += observed * observed;
        dot += expected * observed;
    }
    let error_l2 = error_squared.sqrt();
    let reference_l2 = reference_squared.sqrt();
    let candidate_l2 = candidate_squared.sqrt();
    let relative_l2 = if reference_l2 == 0.0 {
        (error_l2 == 0.0).then_some(0.0)
    } else {
        Some(error_l2 / reference_l2)
    };
    let cosine =
        (reference_l2 != 0.0 && candidate_l2 != 0.0).then_some(dot / (reference_l2 * candidate_l2));
    FlashBoundaryContinuousMetrics {
        finite: true,
        elements: reference.len(),
        max_abs,
        relative_l2,
        cosine,
        reference_l2_zero: reference_l2 == 0.0,
        candidate_l2_zero: candidate_l2 == 0.0,
    }
}

/// Compare two exact raw payloads.  `dtype` is deliberately closed to the two
/// formats emitted by the Flash boundary capture contract.
pub fn compare_flash_boundary_payloads(
    reference: &[u8],
    candidate: &[u8],
    dtype: &str,
    expected_elements: usize,
) -> Result<FlashBoundaryPayloadComparison> {
    let reference_sha256 = sha256_hex(reference);
    let candidate_sha256 = sha256_hex(candidate);
    let exact_sha256 = reference_sha256 == candidate_sha256;
    match dtype {
        "F32_LE" => {
            let reference_values = decode_f32(reference, expected_elements, "reference")?;
            let candidate_values = decode_f32(candidate, expected_elements, "candidate")?;
            Ok(FlashBoundaryPayloadComparison {
                schema: SCHEMA,
                dtype: "F32_LE",
                elements: expected_elements,
                reference_sha256,
                candidate_sha256,
                exact_sha256,
                ordered_i32_exact: None,
                continuous: Some(continuous_metrics(&reference_values, &candidate_values)),
                numerical_acceptance_bound: None,
                semantic_verdict_allowed: false,
            })
        }
        "I32_LE" => {
            let reference_values = decode_i32(reference, expected_elements, "reference")?;
            let candidate_values = decode_i32(candidate, expected_elements, "candidate")?;
            Ok(FlashBoundaryPayloadComparison {
                schema: SCHEMA,
                dtype: "I32_LE",
                elements: expected_elements,
                reference_sha256,
                candidate_sha256,
                exact_sha256,
                ordered_i32_exact: Some(reference_values == candidate_values),
                continuous: None,
                numerical_acceptance_bound: None,
                semantic_verdict_allowed: false,
            })
        }
        other => Err(comparison_error(format!(
            "unsupported payload dtype {other:?}; expected F32_LE or I32_LE"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn i32_bytes(values: &[i32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn seam(
        ordinal: usize,
        layer: Option<usize>,
        semantic_id: &str,
        shape: &[usize],
    ) -> DiagnosticSemanticSeam {
        DiagnosticSemanticSeam {
            ordinal,
            layer,
            semantic_id: semantic_id.into(),
            shape: shape.into(),
        }
    }

    fn schedule(profile_id: &str) -> DiagnosticPrecisionSchedule {
        DiagnosticPrecisionSchedule {
            profile_id: profile_id.into(),
            logical_dtype: "BF16".into(),
            accumulation_dtype: "F32".into(),
            rounding_point: "after normalization".into(),
            storage_dtype: "F32 carrier".into(),
            scope: "bounded diagnostic only".into(),
            reduction_topology: Some("CONTIGUOUS_FULL_PAIRWISE_F32".into()),
            reduction_threadgroup_size: Some(256),
        }
    }

    #[test]
    fn model_neutral_diagnostic_contracts_preserve_order_and_precision() {
        let seams = vec![
            seam(0, Some(0), "layer_output", &[1, 8]),
            seam(1, Some(1), "layer_output", &[1, 8]),
            seam(2, Some(1), "router_logits", &[1, 4]),
        ];
        validate_ordered_semantic_seams(&seams).unwrap();
        assert_eq!(
            first_observed_payload_difference(&seams, &[true, false, false]).unwrap(),
            Some(DiagnosticFirstObservedDifference {
                ordinal: 1,
                layer: Some(1),
                semantic_id: "layer_output".into(),
            })
        );
        require_precision_schedule(&schedule("EXAMPLE"), &schedule("EXAMPLE")).unwrap();
        assert!(require_precision_schedule(&schedule("EXAMPLE"), &schedule("OTHER")).is_err());
    }

    #[test]
    fn diagnostic_contracts_fail_closed_on_sequence_or_extent_drift() {
        assert!(validate_ordered_semantic_seams(&[
            seam(0, Some(0), "embedding_output", &[1, 8]),
            seam(2, Some(0), "normalization_output", &[1, 8]),
        ])
        .is_err());
        assert!(
            validate_ordered_semantic_seams(&[seam(0, Some(0), "embedding_output", &[],)]).is_err()
        );
        assert!(first_observed_payload_difference(
            &[seam(0, Some(0), "embedding_output", &[1, 8])],
            &[],
        )
        .is_err());
    }

    #[test]
    fn f32_metrics_are_diagnostic_and_zero_safe() {
        let exact = compare_flash_boundary_payloads(
            &f32_bytes(&[0.0, 0.0]),
            &f32_bytes(&[0.0, 0.0]),
            "F32_LE",
            2,
        )
        .unwrap();
        assert!(exact.exact_sha256);
        let metrics = exact.continuous.unwrap();
        assert_eq!(metrics.relative_l2, Some(0.0));
        assert_eq!(metrics.cosine, None);
        assert!(!exact.semantic_verdict_allowed);
        assert_eq!(exact.numerical_acceptance_bound, None);

        let shifted = compare_flash_boundary_payloads(
            &f32_bytes(&[1.0, 2.0]),
            &f32_bytes(&[1.0, 3.0]),
            "F32_LE",
            2,
        )
        .unwrap();
        assert_eq!(shifted.continuous.unwrap().max_abs, 1.0);
    }

    #[test]
    fn ordered_route_ids_are_exact() {
        let reference = i32_bytes(&[4, 9, 2]);
        let same = compare_flash_boundary_payloads(&reference, &reference, "I32_LE", 3).unwrap();
        assert_eq!(same.ordered_i32_exact, Some(true));
        let reversed =
            compare_flash_boundary_payloads(&reference, &i32_bytes(&[2, 9, 4]), "I32_LE", 3)
                .unwrap();
        assert_eq!(reversed.ordered_i32_exact, Some(false));
    }

    #[test]
    fn malformed_extent_nonfinite_and_unknown_dtype_fail_closed() {
        assert!(compare_flash_boundary_payloads(&[0; 4], &[0; 8], "F32_LE", 1).is_err());
        assert!(compare_flash_boundary_payloads(
            &f32_bytes(&[f32::NAN]),
            &f32_bytes(&[0.0]),
            "F32_LE",
            1,
        )
        .is_err());
        assert!(compare_flash_boundary_payloads(&[0; 4], &[0; 4], "BF16_LE", 1).is_err());
    }
}
