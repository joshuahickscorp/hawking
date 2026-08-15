//! Mutually-useful tool sets (bible §16), not isolated tools.

use super::enforce::{EffectBoundary, ToolVersion};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Reference to a concrete tool known to the gateway catalog.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolRef {
    pub id: String,
    pub name: String,
    pub version: ToolVersion,
    pub effects: Vec<EffectBoundary>,
    /// Full schema is held here at registration; progressive disclosure to the
    /// model still only ships compact rows until grant (ToolSearch pattern).
    pub input_schema: Value,
    pub output_schema: Option<Value>,
    /// Optional credential key the session must hold (e.g. `hf_token`).
    pub requires_credential: Option<String>,
}

/// One seat in a bundle with a role name (bible kernel-bundle example).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundleMember {
    pub tool_id: String,
    pub role: String,
    /// If true, unhealthy/missing member fails the whole bundle grant.
    pub required: bool,
}

/// A mutually-useful tool set retrieved as a unit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolBundle {
    pub id: String,
    pub name: String,
    pub members: Vec<BundleMember>,
    /// Prior on how well these tools work together ∈ [0, 1].
    pub mutual_affinity: f32,
}

/// Strongly-typed alias for the bible's kernel-lab bundle example.
pub type KernelBundle = ToolBundle;

/// The §16 example kernel bundle.
pub fn kernel_bundle() -> ToolBundle {
    ToolBundle {
        id: "kernel".into(),
        name: "kernel lab set".into(),
        members: vec![
            BundleMember {
                tool_id: "fs.read".into(),
                role: "source_reader".into(),
                required: true,
            },
            BundleMember {
                tool_id: "profiler.sample".into(),
                role: "profiler".into(),
                required: true,
            },
            BundleMember {
                tool_id: "compiler.invoke".into(),
                role: "compiler".into(),
                required: true,
            },
            BundleMember {
                tool_id: "bench.run".into(),
                role: "benchmark_runner".into(),
                required: true,
            },
            BundleMember {
                tool_id: "receipt.verify".into(),
                role: "receipt_verifier".into(),
                required: true,
            },
            BundleMember {
                tool_id: "artifact.inspect".into(),
                role: "artifact_inspector".into(),
                required: true,
            },
        ],
        mutual_affinity: 0.95,
    }
}
