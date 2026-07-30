//! Deterministic placement simulator + content-addressed sections + KV ownership.
//!
//! ## KV ownership invariant
//!
//! Every KV range `[token_start, token_end)` for every layer has **exactly one**
//! owning node. No range unowned. No range double-owned. This holds for a
//! placement and must be re-asserted after node failure + replan.
//!
//! Placement is pure and seeded: same inputs + same seed → same plan.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use super::node::{BandwidthClass, NodeCapabilities, NodeId};
use super::pipeline::StageId;
use super::qualification::QualificationKind;

/// Blake3 hex digest identifying section bytes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ContentHash(pub String);

impl ContentHash {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A model section (contiguous layer range) identified by content hash.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSection {
    pub name: String,
    /// Inclusive start layer index.
    pub layer_start: u32,
    /// Exclusive end layer index.
    pub layer_end: u32,
    pub bytes: u64,
    /// Content hash over name + layer range + payload bytes.
    pub content_hash: ContentHash,
}

impl ModelSection {
    /// Build a section and content-address it from `payload`.
    pub fn content_addressed(
        name: impl Into<String>,
        layer_start: u32,
        layer_end: u32,
        bytes: u64,
        payload: &[u8],
    ) -> Self {
        let name = name.into();
        let content_hash = hash_section(&name, layer_start, layer_end, payload);
        Self {
            name,
            layer_start,
            layer_end,
            bytes,
            content_hash,
        }
    }

    pub fn layer_count(&self) -> u32 {
        self.layer_end.saturating_sub(self.layer_start)
    }
}

pub fn hash_section(name: &str, layer_start: u32, layer_end: u32, payload: &[u8]) -> ContentHash {
    let mut h = blake3::Hasher::new();
    h.update(b"hawking.fabric.section.v1");
    h.update(name.as_bytes());
    h.update(&layer_start.to_le_bytes());
    h.update(&layer_end.to_le_bytes());
    h.update(payload);
    ContentHash(h.finalize().to_hex().to_string())
}

/// Workload class driving pipeline / cost estimation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkloadClass {
    pub name: String,
    pub seq_len: u32,
    pub microbatch_size: u32,
    pub num_microbatches: u32,
}

impl Default for WorkloadClass {
    fn default() -> Self {
        Self {
            name: "default".into(),
            seq_len: 1024,
            microbatch_size: 1,
            num_microbatches: 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SectionPlacement {
    pub content_hash: ContentHash,
    pub section_name: String,
    pub node_id: NodeId,
    pub layer_start: u32,
    pub layer_end: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageAssignment {
    pub stage_id: StageId,
    pub node_id: NodeId,
    pub layer_start: u32,
    pub layer_end: u32,
    pub section_hash: ContentHash,
}

/// One exclusive KV token range owned by a single node for a layer span.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvRangeOwnership {
    pub layer_start: u32,
    pub layer_end: u32,
    pub token_start: u32,
    pub token_end: u32,
    pub owner: NodeId,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredictedCost {
    /// Abstract cost units (deterministic function of plan + workload).
    pub total: u64,
    pub transfer_bytes: u64,
    pub pipeline_bubbles: u64,
}

/// Full placement plan. Simulated plans **must** be labelled.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlacementPlan {
    pub schema: String,
    pub plan_id: String,
    pub seed: u64,
    pub qualification: QualificationKind,
    /// Required true for any non-physical plan. Schema validation rejects
    /// simulated plans with this false/missing.
    pub not_physical_qualification: bool,
    pub section_placements: Vec<SectionPlacement>,
    pub stage_assignments: Vec<StageAssignment>,
    pub kv_ownership: Vec<KvRangeOwnership>,
    pub predicted_cost: PredictedCost,
    /// Filename-safe label; simulated plans include `simulated` in the name.
    pub artifact_label: String,
}

pub const PLACEMENT_SCHEMA: &str = "hawking.fabric.placement.v1";

#[derive(Debug, Clone)]
pub struct PlacementRequest {
    pub sections: Vec<ModelSection>,
    pub nodes: Vec<NodeCapabilities>,
    pub workload: WorkloadClass,
    pub seed: u64,
    pub qualification: QualificationKind,
}

/// Deterministic, seeded placement simulator. No wall-clock dependence.
#[derive(Debug, Default)]
pub struct PlacementSimulator;

impl PlacementSimulator {
    pub fn new() -> Self {
        Self
    }

    pub fn place(&self, req: &PlacementRequest) -> Result<PlacementPlan, PlacementError> {
        if req.nodes.is_empty() {
            return Err(PlacementError::NoNodes);
        }
        if req.sections.is_empty() {
            return Err(PlacementError::NoSections);
        }

        // Stable sort: nodes by (-memory, -cores, id); sections by content_hash.
        let mut nodes = req.nodes.clone();
        nodes.sort_by(|a, b| {
            b.total_memory_bytes
                .cmp(&a.total_memory_bytes)
                .then(b.physical_cores.cmp(&a.physical_cores))
                .then(a.node_id.cmp(&b.node_id))
        });
        let mut sections = req.sections.clone();
        sections.sort_by(|a, b| a.content_hash.cmp(&b.content_hash));

        // Seeded order perturbation: rotate section list by seed % n (deterministic).
        let rot = (req.seed as usize) % sections.len();
        sections.rotate_left(rot);

        let mut remaining: BTreeMap<NodeId, u64> = nodes
            .iter()
            .map(|n| (n.node_id.clone(), n.total_memory_bytes))
            .collect();

        let mut section_placements = Vec::new();
        let mut stage_assignments = Vec::new();
        let mut transfer_bytes = 0u64;

        for (stage_idx, section) in sections.iter().enumerate() {
            let owner = pick_owner(&nodes, &remaining, section.bytes, req.seed, stage_idx).ok_or(
                PlacementError::InsufficientCapacity {
                    section: section.name.clone(),
                    bytes: section.bytes,
                },
            )?;
            let rem = remaining.get_mut(&owner).expect("owner in map");
            *rem = rem.saturating_sub(section.bytes);

            section_placements.push(SectionPlacement {
                content_hash: section.content_hash.clone(),
                section_name: section.name.clone(),
                node_id: owner.clone(),
                layer_start: section.layer_start,
                layer_end: section.layer_end,
            });
            stage_assignments.push(StageAssignment {
                stage_id: StageId(format!("stage-{stage_idx}")),
                node_id: owner,
                layer_start: section.layer_start,
                layer_end: section.layer_end,
                section_hash: section.content_hash.clone(),
            });
        }

        // Pipeline bubble estimate: |stages - 1| * microbatches abstract units.
        let stages = stage_assignments.len() as u64;
        let bubbles = stages.saturating_sub(1) * req.workload.num_microbatches as u64;

        // Transfer: activation bytes between consecutive stages on different nodes.
        for w in stage_assignments.windows(2) {
            if w[0].node_id != w[1].node_id {
                // Abstract activation size: seq * microbatch * 2 bytes * hidden=4096 proxy.
                let act = (req.workload.seq_len as u64)
                    .saturating_mul(req.workload.microbatch_size as u64)
                    .saturating_mul(4096 * 2);
                transfer_bytes = transfer_bytes.saturating_add(act);
            }
        }

        // Bandwidth-weighted cost (deterministic table).
        let bw_cost = |id: &NodeId| -> u64 {
            nodes
                .iter()
                .find(|n| &n.node_id == id)
                .map(|n| bandwidth_penalty(n.bandwidth_class))
                .unwrap_or(100)
        };
        let mut total = transfer_bytes / 1024; // KB units
        for sp in &section_placements {
            total = total.saturating_add(sp.layer_end.saturating_sub(sp.layer_start) as u64 * 10);
            total = total.saturating_add(bw_cost(&sp.node_id));
        }
        total = total.saturating_add(bubbles * 3);
        // Fold seed lightly so different seeds can change cost when rotation changes owners.
        total = total.saturating_add(req.seed % 97);

        let kv_ownership = build_kv_ownership(&stage_assignments, req.workload.seq_len);

        let not_physical = !req.qualification.is_physical();
        let artifact_label = match req.qualification {
            QualificationKind::Simulated => format!("placement_plan_simulated_seed{}", req.seed),
            QualificationKind::SoftwareFixture => {
                format!("placement_plan_software_fixture_seed{}", req.seed)
            }
            QualificationKind::PhysicalHardware => {
                format!("placement_plan_physical_seed{}", req.seed)
            }
        };

        let plan = PlacementPlan {
            schema: PLACEMENT_SCHEMA.into(),
            plan_id: format!("plan-{:016x}", plan_fingerprint(req)),
            seed: req.seed,
            qualification: req.qualification,
            not_physical_qualification: not_physical,
            section_placements,
            stage_assignments,
            kv_ownership,
            predicted_cost: PredictedCost {
                total,
                transfer_bytes,
                pipeline_bubbles: bubbles,
            },
            artifact_label,
        };

        validate_placement_plan_schema(&plan)?;
        KvOwnershipInvariant::assert_holds(&plan.kv_ownership, req.workload.seq_len)?;
        Ok(plan)
    }

    /// Replan after a failed node: drop the node and place remaining sections.
    pub fn replan_after_failure(
        &self,
        req: &PlacementRequest,
        failed: &NodeId,
    ) -> Result<PlacementPlan, PlacementError> {
        let nodes: Vec<_> = req
            .nodes
            .iter()
            .filter(|n| &n.node_id != failed)
            .cloned()
            .collect();
        if nodes.is_empty() {
            return Err(PlacementError::NoNodes);
        }
        // Bump seed deterministically from failed node id so replan differs stably.
        let mut h = blake3::Hasher::new();
        h.update(&req.seed.to_le_bytes());
        h.update(failed.as_str().as_bytes());
        let digest = h.finalize();
        let mut seed_bytes = [0u8; 8];
        seed_bytes.copy_from_slice(&digest.as_bytes()[..8]);
        let seed = u64::from_le_bytes(seed_bytes);
        let replan_req = PlacementRequest {
            sections: req.sections.clone(),
            nodes,
            workload: req.workload.clone(),
            seed,
            qualification: req.qualification,
        };
        self.place(&replan_req)
    }
}

fn pick_owner(
    nodes: &[NodeCapabilities],
    remaining: &BTreeMap<NodeId, u64>,
    need: u64,
    seed: u64,
    stage_idx: usize,
) -> Option<NodeId> {
    let mut candidates: Vec<&NodeCapabilities> = nodes
        .iter()
        .filter(|n| remaining.get(&n.node_id).copied().unwrap_or(0) >= need)
        .collect();
    if candidates.is_empty() {
        // Fall back to node with most remaining capacity (may oversubscribe abstractly).
        return nodes
            .iter()
            .max_by_key(|n| remaining.get(&n.node_id).copied().unwrap_or(0))
            .map(|n| n.node_id.clone());
    }
    // Deterministic tie-break with seed + stage.
    candidates.sort_by(|a, b| {
        let ra = remaining.get(&a.node_id).copied().unwrap_or(0);
        let rb = remaining.get(&b.node_id).copied().unwrap_or(0);
        rb.cmp(&ra).then(a.node_id.cmp(&b.node_id))
    });
    let idx = ((seed as usize).wrapping_add(stage_idx.wrapping_mul(31))) % candidates.len();
    Some(candidates[idx].node_id.clone())
}

fn bandwidth_penalty(bw: BandwidthClass) -> u64 {
    match bw {
        BandwidthClass::InProcess => 1,
        BandwidthClass::Localhost => 2,
        BandwidthClass::Lan10g => 5,
        BandwidthClass::Lan1g => 20,
        BandwidthClass::Wan => 200,
    }
}

fn build_kv_ownership(stages: &[StageAssignment], seq_len: u32) -> Vec<KvRangeOwnership> {
    // Core invariant construction: each stage owns the full token range for its layers.
    // Exactly one owner per (layer, token) because stages have disjoint layer ranges
    // for a linear pipeline partition (enforced by section construction).
    stages
        .iter()
        .map(|s| KvRangeOwnership {
            layer_start: s.layer_start,
            layer_end: s.layer_end,
            token_start: 0,
            token_end: seq_len,
            owner: s.node_id.clone(),
        })
        .collect()
}

fn plan_fingerprint(req: &PlacementRequest) -> u64 {
    let mut h = blake3::Hasher::new();
    h.update(&req.seed.to_le_bytes());
    h.update(req.qualification.as_str().as_bytes());
    h.update(req.workload.name.as_bytes());
    h.update(&req.workload.seq_len.to_le_bytes());
    for s in &req.sections {
        h.update(s.content_hash.as_str().as_bytes());
    }
    for n in &req.nodes {
        h.update(n.node_id.as_str().as_bytes());
        h.update(&n.total_memory_bytes.to_le_bytes());
    }
    let d = h.finalize();
    let mut b = [0u8; 8];
    b.copy_from_slice(&d.as_bytes()[..8]);
    u64::from_le_bytes(b)
}

// ---------------------------------------------------------------------------
// KV ownership invariant
// ---------------------------------------------------------------------------

/// Invariant: every (layer, token) cell has exactly one owner.
pub struct KvOwnershipInvariant;

impl KvOwnershipInvariant {
    pub fn assert_holds(
        ownership: &[KvRangeOwnership],
        seq_len: u32,
    ) -> Result<(), PlacementError> {
        if ownership.is_empty() {
            return Err(PlacementError::KvInvariant {
                detail: "no KV ranges".into(),
            });
        }

        // Expand layer coverage: for each layer index, collect token coverage map.
        let mut max_layer = 0u32;
        for r in ownership {
            if r.layer_end > max_layer {
                max_layer = r.layer_end;
            }
            if r.token_end > seq_len {
                return Err(PlacementError::KvInvariant {
                    detail: format!("token_end {} exceeds seq_len {}", r.token_end, seq_len),
                });
            }
            if r.token_start >= r.token_end || r.layer_start >= r.layer_end {
                return Err(PlacementError::KvInvariant {
                    detail: "empty layer or token range".into(),
                });
            }
        }

        // For each layer, build owner per token (None / Some / conflict).
        for layer in 0..max_layer {
            let mut owner_at: Vec<Option<&NodeId>> = vec![None; seq_len as usize];
            for r in ownership {
                if layer < r.layer_start || layer >= r.layer_end {
                    continue;
                }
                for t in r.token_start..r.token_end {
                    let slot = &mut owner_at[t as usize];
                    match slot {
                        None => *slot = Some(&r.owner),
                        Some(existing) if *existing == &r.owner => {}
                        Some(existing) => {
                            return Err(PlacementError::KvInvariant {
                                detail: format!(
                                    "double-owned layer={layer} token={t}: {} and {}",
                                    existing, r.owner
                                ),
                            });
                        }
                    }
                }
            }
            for (t, o) in owner_at.iter().enumerate() {
                if o.is_none() {
                    return Err(PlacementError::KvInvariant {
                        detail: format!("unowned layer={layer} token={t}"),
                    });
                }
            }
        }

        // Also check no two ranges double-cover with different owners (already done).
        let _owners: BTreeSet<_> = ownership.iter().map(|r| &r.owner).collect();
        Ok(())
    }

    /// Drop all ranges owned by `failed` and return the set of lost ranges.
    pub fn ranges_lost_on_failure(
        ownership: &[KvRangeOwnership],
        failed: &NodeId,
    ) -> Vec<KvRangeOwnership> {
        ownership
            .iter()
            .filter(|r| &r.owner == failed)
            .cloned()
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Schema validation — rejects unlabelled simulated results
// ---------------------------------------------------------------------------

pub fn validate_placement_plan_schema(plan: &PlacementPlan) -> Result<(), PlacementError> {
    if plan.schema != PLACEMENT_SCHEMA {
        return Err(PlacementError::Schema {
            detail: format!("unexpected schema {}", plan.schema),
        });
    }
    match plan.qualification {
        QualificationKind::Simulated => {
            if !plan.not_physical_qualification {
                return Err(PlacementError::Schema {
                    detail: "simulated placement must set not_physical_qualification=true".into(),
                });
            }
            if !plan.artifact_label.contains("simulated") {
                return Err(PlacementError::Schema {
                    detail: "simulated placement artifact_label must contain 'simulated'".into(),
                });
            }
            if !plan.plan_id.is_empty() && plan.artifact_label.contains("physical") {
                return Err(PlacementError::Schema {
                    detail: "simulated placement must not claim physical in artifact_label".into(),
                });
            }
        }
        QualificationKind::SoftwareFixture => {
            if !plan.not_physical_qualification {
                return Err(PlacementError::Schema {
                    detail: "software_fixture placement must set not_physical_qualification=true"
                        .into(),
                });
            }
        }
        QualificationKind::PhysicalHardware => {
            if plan.not_physical_qualification {
                return Err(PlacementError::Schema {
                    detail: "physical_hardware placement must not set not_physical_qualification"
                        .into(),
                });
            }
        }
    }
    Ok(())
}

/// Reject an unlabelled simulated result (for tests that construct bad plans).
pub fn reject_unlabelled_simulated(plan: &PlacementPlan) -> Result<(), PlacementError> {
    if plan.qualification == QualificationKind::Simulated && !plan.not_physical_qualification {
        return Err(PlacementError::Schema {
            detail: "unlabelled simulated result rejected".into(),
        });
    }
    if plan.qualification == QualificationKind::Simulated
        && !plan.artifact_label.to_ascii_lowercase().contains("sim")
    {
        return Err(PlacementError::Schema {
            detail: "simulated artifact must be labelled simulated".into(),
        });
    }
    validate_placement_plan_schema(plan)
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PlacementError {
    #[error("no nodes available for placement")]
    NoNodes,
    #[error("no sections to place")]
    NoSections,
    #[error("insufficient capacity for section {section} ({bytes} bytes)")]
    InsufficientCapacity { section: String, bytes: u64 },
    #[error("KV ownership invariant violated: {detail}")]
    KvInvariant { detail: String },
    #[error("placement schema error: {detail}")]
    Schema { detail: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fabric::node::SimulatedNodeSet;
    fn sample_sections() -> Vec<ModelSection> {
        vec![
            ModelSection::content_addressed("embed", 0, 2, 4 * GIB, b"embed-payload-v1"),
            ModelSection::content_addressed("mid", 2, 6, 8 * GIB, b"mid-payload-v1"),
            ModelSection::content_addressed("head", 6, 8, 3 * GIB, b"head-payload-v1"),
        ]
    }
    const GIB: u64 = 1024 * 1024 * 1024;
    #[test]
    fn content_hash_stable_across_repacks() {
        let a = ModelSection::content_addressed("mid", 2, 6, 8 * GIB, b"mid-payload-v1");
        let b = ModelSection::content_addressed("mid", 2, 6, 999, b"mid-payload-v1");
        assert_eq!(a.content_hash, b.content_hash);
        let c = ModelSection::content_addressed("mid", 2, 6, 8 * GIB, b"mid-payload-v2");
        assert_ne!(a.content_hash, c.content_hash);
    }
    #[test]
    fn placement_determinism_same_seed_same_plan() {
        let nodes = SimulatedNodeSet::heterogeneous_sim("sim-det-v1").nodes;
        let req = PlacementRequest {
            sections: sample_sections(),
            nodes,
            workload: WorkloadClass {
                name: "decode".into(),
                seq_len: 512,
                microbatch_size: 2,
                num_microbatches: 4,
            },
            seed: 0xC0FFEE,
            qualification: QualificationKind::Simulated,
        };
        let sim = PlacementSimulator::new();
        let p1 = sim.place(&req).unwrap();
        let p2 = sim.place(&req).unwrap();
        assert_eq!(p1, p2);
        assert!(p1.not_physical_qualification);
        assert!(p1.artifact_label.contains("simulated"));
    }
    #[test]
    fn different_seeds_can_differ() {
        let nodes = SimulatedNodeSet::heterogeneous_sim("sim-seed-v1").nodes;
        let base = PlacementRequest {
            sections: sample_sections(),
            nodes,
            workload: WorkloadClass::default(),
            seed: 1,
            qualification: QualificationKind::Simulated,
        };
        let sim = PlacementSimulator::new();
        let p1 = sim.place(&base).unwrap();
        let mut base2 = base.clone();
        base2.seed = 2;
        let p2 = sim.place(&base2).unwrap();
        assert_ne!(p1.plan_id, p2.plan_id);
    }
    #[test]
    fn kv_ownership_invariant_holds() {
        let nodes = SimulatedNodeSet::heterogeneous_sim("sim-kv-v1").nodes;
        let req = PlacementRequest {
            sections: sample_sections(),
            nodes,
            workload: WorkloadClass {
                name: "kv".into(),
                seq_len: 128,
                microbatch_size: 1,
                num_microbatches: 2,
            },
            seed: 7,
            qualification: QualificationKind::Simulated,
        };
        let plan = PlacementSimulator::new().place(&req).unwrap();
        KvOwnershipInvariant::assert_holds(&plan.kv_ownership, req.workload.seq_len).unwrap();
    }
    #[test]
    fn kv_invariant_after_failure_replan() {
        let nodes = SimulatedNodeSet::heterogeneous_sim("sim-kv-fail-v1").nodes;
        let req = PlacementRequest {
            sections: sample_sections(),
            nodes: nodes.clone(),
            workload: WorkloadClass {
                name: "kv".into(),
                seq_len: 64,
                microbatch_size: 1,
                num_microbatches: 2,
            },
            seed: 11,
            qualification: QualificationKind::Simulated,
        };
        let sim = PlacementSimulator::new();
        let plan = sim.place(&req).unwrap();
        let failed = plan.section_placements[0].node_id.clone();
        let lost = KvOwnershipInvariant::ranges_lost_on_failure(&plan.kv_ownership, &failed);
        assert!(!lost.is_empty(), "expected some KV lost on node failure");
        let replan = sim.replan_after_failure(&req, &failed).unwrap();
        assert!(replan
            .section_placements
            .iter()
            .all(|sp| sp.node_id != failed));
        KvOwnershipInvariant::assert_holds(&replan.kv_ownership, req.workload.seq_len).unwrap();
    }
    #[test]
    fn schema_rejects_unlabelled_simulated_result() {
        let mut plan = PlacementPlan {
            schema: PLACEMENT_SCHEMA.into(),
            plan_id: "bad".into(),
            seed: 0,
            qualification: QualificationKind::Simulated,
            not_physical_qualification: false, // ILLEGAL
            section_placements: vec![],
            stage_assignments: vec![],
            kv_ownership: vec![],
            predicted_cost: PredictedCost {
                total: 0,
                transfer_bytes: 0,
                pipeline_bubbles: 0,
            },
            artifact_label: "looks_physical".into(),
        };
        let err = reject_unlabelled_simulated(&plan).unwrap_err();
        assert!(matches!(err, PlacementError::Schema { .. }));
        plan.not_physical_qualification = true;
        plan.artifact_label = "placement_plan_simulated_seed0".into();
        validate_placement_plan_schema(&plan).unwrap();
    }
}
