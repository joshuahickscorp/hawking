//! **Doctor collapse.** All treatments implement ONE contract —
//! observe/diagnose/enumerate/budget/apply/evaluate — and record ONE treatment-artifact schema. A Doctor
//! pack owns no controller, queue, receipt engine, source identity, resource scheduler, or run lifecycle;
//! only mechanisms executable through the Seed's runtime / execution IR stay active. The reference
//! treatment is the Seed's sparse full-precision residual rescue ([`crate::subbit::doctor_rescue`]), and
//! every treatment must stay inside the same physical byte budget (Gravity's same-rate law).

use super::provider::{Context, Provider, ProviderOutput, ResourceUsage};
use crate::gravity;
use crate::pack::CapabilityKind;
use crate::subbit::{self, SubBitMatrix};
use crate::Result;
use serde::Serialize;

/// A deterministic, genuinely low-rank tensor (rank-8 outer products + tiny noise). Rank-32 ternary
/// approximates it well, so the residual is small and concentrated — exactly the regime where a sparse
/// Doctor correction meaningfully reduces error. Stands in for a compressible source tensor slice.
pub fn synth_lowrank(rows: usize, cols: usize) -> Vec<f32> {
    let rank = 8usize;
    let mut w = vec![0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let mut acc = 0f32;
            for k in 0..rank {
                let u = (((i * 131 + k * 17) % 101) as f32 / 101.0 - 0.5) * 0.5;
                let v = (((j * 197 + k * 31) % 103) as f32 / 103.0 - 0.5) * 0.5;
                acc += u * v;
            }
            let noise = (((i * 48271 + j * 40503) % 211) as f32 / 211.0 - 0.5) * 0.01;
            w[i * cols + j] = acc + noise;
        }
    }
    w
}

/// A stable probe set (many probes → `output_divergence` is a low-variance estimator of the matrix error).
pub fn probes(cols: usize, n: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|s| (0..cols).map(|j| (((j * 2654435761 + s) >> 7) & 0xFF) as f32 / 128.0 - 1.0).collect())
        .collect()
}

/// The one Doctor treatment-artifact schema.
#[derive(Debug, Clone, Serialize)]
pub struct TreatmentArtifact {
    pub mechanism: String,
    pub diagnosis: String,
    pub allocated_bytes: usize,
    /// EXACT rational target rate (bits per weight).
    pub target_rate_num: u32,
    pub target_rate_den: u32,
    pub affected_scope: [usize; 2],
    pub execution_contract: String,
    pub before_divergence: f64,
    pub after_divergence: f64,
    pub before_bpw: f64,
    pub after_bpw: f64,
    pub within_budget: bool,
}

/// A diagnosis: how bad is the untreated artifact, and how many correction entries are worth allocating.
#[derive(Debug, Clone, Serialize)]
pub struct Diagnosis {
    pub divergence: f64,
    pub whole_bpw: f64,
    pub proposed_entries: usize,
}

/// The one Doctor treatment contract.
pub trait Doctor {
    fn mechanism(&self) -> &str;

    /// observe + diagnose: measure the untreated artifact against probes and propose a correction size.
    fn diagnose(&self, w: &[f32], rows: usize, cols: usize, art: &SubBitMatrix, probes: &[Vec<f32>], budget_bpw: f64) -> Diagnosis;

    /// budget: how many bits a `k`-entry correction costs (for the physical-conservation check).
    fn correction_bits(&self, k: usize) -> usize;

    /// apply: produce a treated artifact within `budget_bpw`, or None if it would exceed the budget.
    fn treat(&self, w: &[f32], rows: usize, cols: usize, art: SubBitMatrix, k: usize, budget_bpw: f64) -> Option<SubBitMatrix>;

    /// evaluate: build the sealed treatment artifact from before/after measurements.
    #[allow(clippy::too_many_arguments)]
    fn evaluate(&self, rows: usize, cols: usize, before_div: f64, after_div: f64, before_bpw: f64, after_bpw: f64, budget_bpw: f64) -> TreatmentArtifact;
}

/// Reference treatment: sparse full-precision residual (largest |W - scale·AB| entries as (row,col,f16)).
pub struct SparseResidual;

impl Doctor for SparseResidual {
    fn mechanism(&self) -> &str {
        "sparse_residual"
    }

    fn diagnose(&self, w: &[f32], rows: usize, cols: usize, art: &SubBitMatrix, probes: &[Vec<f32>], budget_bpw: f64) -> Diagnosis {
        let divergence = subbit::output_divergence(w, rows, cols, art, probes);
        let whole_bpw = art.whole_bpw();
        // headroom (in bits) between the current artifact and the budget, in 48-bit correction entries.
        let budget_bits = budget_bpw * (rows * cols) as f64;
        let headroom = (budget_bits - art.bits() as f64).max(0.0);
        let proposed_entries = (headroom / 48.0) as usize;
        Diagnosis { divergence, whole_bpw, proposed_entries }
    }

    fn correction_bits(&self, k: usize) -> usize {
        k * 48 // f16 value (16) + row (16) + col (16)
    }

    fn treat(&self, w: &[f32], rows: usize, cols: usize, art: SubBitMatrix, k: usize, budget_bpw: f64) -> Option<SubBitMatrix> {
        subbit::doctor_rescue(w, rows, cols, art, k, budget_bpw)
    }

    fn evaluate(&self, rows: usize, cols: usize, before_div: f64, after_div: f64, before_bpw: f64, after_bpw: f64, budget_bpw: f64) -> TreatmentArtifact {
        let n_weights = (rows * cols) as u64;
        let after_bits = (after_bpw * (rows * cols) as f64).round() as u64;
        // Gravity same-rate law: the whole treated artifact must remain within the physical budget.
        let within_budget = gravity::doctor_within_budget(after_bits, 0, 0, budget_bpw, n_weights).is_ok();
        TreatmentArtifact {
            mechanism: self.mechanism().into(),
            diagnosis: format!("untreated divergence {before_div:.4} at {before_bpw:.3} BPW"),
            allocated_bytes: ((after_bpw - before_bpw).max(0.0) * (rows * cols) as f64 / 8.0) as usize,
            target_rate_num: after_bits.max(1) as u32,
            target_rate_den: (rows * cols) as u32,
            affected_scope: [rows, cols],
            execution_contract: "direct sparse correction: y[row] += val*x[col]".into(),
            before_divergence: before_div,
            after_divergence: after_div,
            before_bpw,
            after_bpw,
            within_budget,
        }
    }
}

/// A `Provider` over the Doctor contract: forge a synthetic tensor, diagnose, treat within budget, seal.
pub struct DoctorProvider<D: Doctor> {
    pub doctor: D,
    capability: String,
}

impl<D: Doctor> DoctorProvider<D> {
    pub fn new(doctor: D) -> Self {
        let capability = format!("doctor.{}", doctor.mechanism());
        DoctorProvider { doctor, capability }
    }
}

impl<D: Doctor> Provider for DoctorProvider<D> {
    fn capability(&self) -> &str {
        &self.capability
    }
    fn kind(&self) -> CapabilityKind {
        CapabilityKind::DoctorTreatment
    }
    fn run(&self, _ctx: &Context, input: serde_json::Value) -> Result<ProviderOutput> {
        let rows = input["rows"].as_u64().unwrap_or(256) as usize;
        let cols = input["cols"].as_u64().unwrap_or(256) as usize;
        let budget = input["budget_bpw"].as_f64().unwrap_or(0.99);
        let w = synth_lowrank(rows, cols);
        let probes = probes(cols, 32);
        let untreated = subbit::fit(&w, rows, cols, 32.min(rows.min(cols) / 2).max(4));
        let diag = self.doctor.diagnose(&w, rows, cols, &untreated, &probes, budget);
        let before_div = diag.divergence;
        let before_bpw = untreated.whole_bpw();
        let refit = subbit::fit(&w, rows, cols, 32.min(rows.min(cols) / 2).max(4));
        let treated = self.doctor.treat(&w, rows, cols, refit, diag.proposed_entries, budget);
        let (after_div, after_bpw) = match &treated {
            Some(t) => (subbit::output_divergence(&w, rows, cols, t, &probes), t.whole_bpw()),
            None => (before_div, before_bpw),
        };
        let art = self.doctor.evaluate(rows, cols, before_div, after_div, before_bpw, after_bpw, budget);
        let result = serde_json::to_value(&art)?;
        let metrics = serde_json::json!({
            "divergence_before": before_div,
            "divergence_after": after_div,
            "bpw_after": after_bpw,
            "within_budget": art.within_budget,
            "improved": after_div <= before_div,
        });
        Ok(ProviderOutput::sealed(result, metrics, ResourceUsage::default()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_residual_reduces_divergence_within_budget() {
        let (m, n) = (256usize, 256usize);
        let w = synth_lowrank(m, n);
        let pr = probes(n, 32);
        let doc = SparseResidual;
        let untreated = subbit::fit(&w, m, n, 32);
        let diag = doc.diagnose(&w, m, n, &untreated, &pr, 0.99);
        assert!(diag.proposed_entries > 0);
        let before = diag.divergence;
        let treated = doc.treat(&w, m, n, subbit::fit(&w, m, n, 32), diag.proposed_entries, 0.99).expect("within budget");
        let after = subbit::output_divergence(&w, m, n, &treated, &pr);
        let art = doc.evaluate(m, n, before, after, untreated.whole_bpw(), treated.whole_bpw(), 0.99);
        assert!(after < before, "Doctor must reduce divergence: {before} -> {after}");
        assert!(art.within_budget, "treatment must stay within the physical budget");
        assert!(art.after_bpw < 1.0, "still sub-bit after treatment");
    }
}
