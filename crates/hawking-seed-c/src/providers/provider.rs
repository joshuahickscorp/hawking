//! **Shared provider traits.** A runtime pack is a *pure provider*: it receives Seed-owned context and
//! returns a result, metrics, an evidence payload, and resource usage. It may **not** independently mutate
//! Seed state — it holds no `&mut Machine`, owns no queue/scheduler/lease/run-state, and writes no receipt
//! of its own. The caller (the Seed's controller) records the returned evidence.
//!
//! Every capability kind (adapter, forge, doctor, metal, speculation, validation) is a specialization of
//! the one [`Provider`] contract; the capability-specific traits live in their own modules and each
//! implements `Provider`.

use super::source_decl::SourceRecord;
use crate::evidence::receipt;
use crate::gravity::Rate;
use crate::pack::CapabilityKind;
use crate::record::Record;
use crate::Result;
use serde::Serialize;

/// The Seed-owned execution context handed to a provider. It is READ-ONLY with respect to Seed state: the
/// provider sees the run id, the persistence root (for locating assets), the verified source record, and
/// the current Gravity rate — but no control handle. Control stays with the Seed.
#[derive(Debug, Clone)]
pub struct Context {
    pub run_id: String,
    pub state_root: String,
    pub source: SourceRecord,
    pub gravity_current: Rate,
}

impl Context {
    pub fn new(run_id: &str, state_root: &str, source: SourceRecord, gravity_current: Rate) -> Self {
        Context { run_id: run_id.into(), state_root: state_root.into(), source, gravity_current }
    }
}

/// Resource usage a provider reports back for accounting (never enforced by the provider itself).
#[derive(Debug, Clone, Default, Serialize)]
pub struct ResourceUsage {
    pub source_bytes_read: usize,
    pub owned_bits: usize,
    pub wall_ms: f64,
}

/// The single provider return shape: result + metrics + a SEALED evidence record + resource usage.
#[derive(Debug, Clone)]
pub struct ProviderOutput {
    pub result: serde_json::Value,
    pub metrics: serde_json::Value,
    pub evidence: Record,
    pub resource_usage: ResourceUsage,
}

impl ProviderOutput {
    /// Build an output, sealing the evidence as an `evaluation` receipt through the Seed's ONE engine.
    pub fn sealed(
        result: serde_json::Value,
        metrics: serde_json::Value,
        usage: ResourceUsage,
    ) -> Self {
        let evidence = receipt(
            "evaluation",
            serde_json::json!({ "result": result, "metrics": metrics, "resource_usage": serde_json::to_value(&usage).unwrap_or(serde_json::json!({})) }),
        );
        ProviderOutput { result, metrics, evidence, resource_usage: usage }
    }
}

/// The one provider contract. Every capability provider implements it.
pub trait Provider {
    /// The capability id this provider realizes (e.g. `adapter.gpt_oss`, `forge.ternary_latent`).
    fn capability(&self) -> &str;
    /// The capability kind — used by the one registry to type its payloads.
    fn kind(&self) -> CapabilityKind;
    /// Execute against Seed-owned context. Returns evidence; never mutates Seed state.
    fn run(&self, ctx: &Context, input: serde_json::Value) -> Result<ProviderOutput>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoopProvider;
    impl Provider for NoopProvider {
        fn capability(&self) -> &str {
            "test.noop"
        }
        fn kind(&self) -> CapabilityKind {
            CapabilityKind::ValidationSuite
        }
        fn run(&self, ctx: &Context, _input: serde_json::Value) -> Result<ProviderOutput> {
            Ok(ProviderOutput::sealed(
                serde_json::json!({"ran_for": ctx.run_id}),
                serde_json::json!({"ok": true}),
                ResourceUsage { source_bytes_read: 0, owned_bits: 0, wall_ms: 0.0 },
            ))
        }
    }

    #[test]
    fn provider_output_is_sealed_evidence() {
        let p = NoopProvider;
        let ctx = Context::new("run-1", "/tmp/root", SourceRecord::local("/tmp"), Rate::new(4, 5));
        let out = p.run(&ctx, serde_json::json!({})).unwrap();
        assert!(out.evidence.verify().is_ok());
        assert_eq!(out.evidence.kind, "evaluation");
    }
}
