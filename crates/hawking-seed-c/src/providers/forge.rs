//! **Forge collapse.** All active representation families implement ONE contract —
//! inspect/fit/pack/measure/execute/validate/repairability — and record ONE artifact schema. There are no
//! per-family fit/pack/evaluate frameworks and no campaign-specific wrappers. Shared source slicing,
//! probes, byte accounting, evaluation, and evidence live here (or in the Seed); a family contributes only
//! its representation-specific mathematics.
//!
//! The reference family, `TernaryLatentFamily`, is the Seed's proven sub-bit mathematics
//! ([`crate::subbit`]) expressed through the contract — not a reimplementation.

use super::provider::{Context, Provider, ProviderOutput, ResourceUsage};
use crate::gravity::Rate;
use crate::pack::CapabilityKind;
use crate::subbit::{self, SubBitMatrix};
use crate::Result;
use serde::Serialize;

/// The one Forge artifact schema.
#[derive(Debug, Clone, Serialize)]
pub struct ForgeArtifact {
    pub family: String,
    /// Parent (source tensor) identity this artifact represents.
    pub parent_identity: String,
    /// Tensor scope [rows, cols].
    pub tensor_scope: [usize; 2],
    /// EXACT rational whole-artifact rate (bits per weight) — floats are never scientific identity.
    pub rate_num: u32,
    pub rate_den: u32,
    pub base_bytes: usize,
    pub metadata_bytes: usize,
    /// Bytes reserved for a Doctor treatment inside the same physical budget.
    pub doctor_reserve_bytes: usize,
    pub execution_requirements: Vec<String>,
    /// Validation evidence: relative output divergence vs the dense reference.
    pub output_divergence: f64,
    pub subbit: bool,
}

impl ForgeArtifact {
    pub fn rate(&self) -> Rate {
        Rate::new(self.rate_num.max(1), self.rate_den.max(1))
    }
}

/// The one representation-family contract.
pub trait Forge {
    fn family(&self) -> &str;

    /// inspect: describe the tensor scope + declared execution requirements without fitting.
    fn inspect(&self, rows: usize, cols: usize) -> Vec<String>;

    /// fit + pack + measure: produce a sealed artifact for the tensor `w` (rows×cols, row-major).
    fn forge(&self, parent_identity: &str, w: &[f32], rows: usize, cols: usize) -> Result<(SubBitMatrix, ForgeArtifact)>;

    /// execute: run the compact operator DIRECTLY (never densifying).
    fn execute(&self, art: &SubBitMatrix, x: &[f32]) -> Vec<f32>;

    /// validate: relative output divergence vs the dense reference over probe inputs.
    fn validate(&self, w: &[f32], rows: usize, cols: usize, art: &SubBitMatrix, probes: &[Vec<f32>]) -> f64;

    /// repairability: whether a Doctor treatment can further reduce divergence within budget.
    fn repairable(&self) -> bool;
}

/// Reference family: ternary latent factorization W ≈ scale·(A·B), executed as y = scale·A·(B·x).
pub struct TernaryLatentFamily {
    pub rank: usize,
}

impl TernaryLatentFamily {
    pub fn new(rank: usize) -> Self {
        TernaryLatentFamily { rank }
    }
    fn effective_rank(&self, rows: usize, cols: usize) -> usize {
        self.rank.min(rows.min(cols) / 2).max(4)
    }
}

impl Forge for TernaryLatentFamily {
    fn family(&self) -> &str {
        "ternary_latent"
    }

    fn inspect(&self, rows: usize, cols: usize) -> Vec<String> {
        vec![
            format!("ternary latent factorization, rank {}", self.effective_rank(rows, cols)),
            "two ternary mat-vecs y = scale*A*(B*x)".into(),
            "5 trits/byte packing (log2(3)~1.6 bits/entry)".into(),
        ]
    }

    fn forge(&self, parent_identity: &str, w: &[f32], rows: usize, cols: usize) -> Result<(SubBitMatrix, ForgeArtifact)> {
        let r = self.effective_rank(rows, cols);
        let sb = subbit::fit(w, rows, cols, r);
        let bits = sb.bits();
        let art = ForgeArtifact {
            family: self.family().into(),
            parent_identity: parent_identity.into(),
            tensor_scope: [rows, cols],
            rate_num: bits as u32,
            rate_den: (rows * cols) as u32,
            base_bytes: bits / 8,
            metadata_bytes: 4, // the f32 scale
            doctor_reserve_bytes: 0,
            execution_requirements: self.inspect(rows, cols),
            output_divergence: f64::NAN, // filled by validate()
            subbit: sb.whole_bpw() < 1.0,
        };
        Ok((sb, art))
    }

    fn execute(&self, art: &SubBitMatrix, x: &[f32]) -> Vec<f32> {
        art.matvec(x)
    }

    fn validate(&self, w: &[f32], rows: usize, cols: usize, art: &SubBitMatrix, probes: &[Vec<f32>]) -> f64 {
        subbit::output_divergence(w, rows, cols, art, probes)
    }

    fn repairable(&self) -> bool {
        true
    }
}

/// A `Provider` over the Forge contract: forge + validate a tensor, seal the artifact as evidence.
pub struct ForgeProvider<F: Forge> {
    pub forge: F,
    capability: String,
}

impl<F: Forge> ForgeProvider<F> {
    pub fn new(forge: F) -> Self {
        let capability = format!("forge.{}", forge.family());
        ForgeProvider { forge, capability }
    }
}

impl<F: Forge> Provider for ForgeProvider<F> {
    fn capability(&self) -> &str {
        &self.capability
    }
    fn kind(&self) -> CapabilityKind {
        CapabilityKind::ForgeFamily
    }
    fn run(&self, _ctx: &Context, input: serde_json::Value) -> Result<ProviderOutput> {
        // input: { rows, cols, seed } — a deterministic synthetic tensor stands in for a source slice.
        let rows = input["rows"].as_u64().unwrap_or(256) as usize;
        let cols = input["cols"].as_u64().unwrap_or(256) as usize;
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| (((i * 48271) % 997) as f32 / 997.0 - 0.5) * 0.1)
            .collect();
        let (sb, mut art) = self.forge.forge("synthetic:probe-tensor", &w, rows, cols)?;
        let probes: Vec<Vec<f32>> = (0..4)
            .map(|s| (0..cols).map(|j| (((j * 2654435761 + s) >> 7) & 0xFF) as f32 / 128.0 - 1.0).collect())
            .collect();
        art.output_divergence = self.forge.validate(&w, rows, cols, &sb, &probes);
        let result = serde_json::to_value(&art)?;
        let metrics = serde_json::json!({
            "whole_bpw": sb.whole_bpw(),
            "subbit": art.subbit,
            "rate": art.rate().label(),
            "output_divergence": art.output_divergence,
        });
        Ok(ProviderOutput::sealed(result, metrics, ResourceUsage { owned_bits: sb.bits(), ..Default::default() }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ternary_family_forges_subbit_artifact_and_executes_directly() {
        let (m, n) = (256usize, 256usize);
        let w: Vec<f32> = (0..m * n).map(|i| (((i * 48271) % 997) as f32 / 997.0 - 0.5) * 0.1).collect();
        let fam = TernaryLatentFamily::new(32);
        let (sb, art) = fam.forge("parent:test", &w, m, n).unwrap();
        assert!(art.subbit, "artifact must be sub-bit");
        assert!(art.rate().is_subbit(), "exact rational rate must be sub-bit: {}", art.rate().label());
        let y = fam.execute(&sb, &vec![0.1f32; n]);
        assert_eq!(y.len(), m);
        let probes: Vec<Vec<f32>> = (0..3).map(|s| (0..n).map(|j| (((j + s) % 7) as f32 - 3.0) * 0.1).collect()).collect();
        let div = fam.validate(&w, m, n, &sb, &probes);
        assert!(div.is_finite() && div >= 0.0);
    }
}
