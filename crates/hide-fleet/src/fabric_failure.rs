//! Heartbeats, failure detection, and failure/replay receipts.
//!
//! Silent loss of work is the failure mode. When a node dies mid-request the
//! system produces a receipt naming what was lost, what was replayed, and from
//! which checkpoint.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::node::NodeId;
use super::pipeline::{PipelineScheduler, StageId};
use super::placement::{KvOwnershipInvariant, KvRangeOwnership, PlacementPlan};
use super::qualification::QualificationKind;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CheckpointId(pub String);

impl CheckpointId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LostWorkSummary {
    pub stages: Vec<StageId>,
    pub kv_ranges: Vec<KvRangeOwnership>,
    pub in_flight_microbatches: u32,
}

/// Receipt produced on node failure + replay. Never silent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FailureReplayReceipt {
    pub schema: String,
    pub request_id: String,
    pub failed_node: NodeId,
    pub lost_work: LostWorkSummary,
    pub replayed_from_checkpoint: CheckpointId,
    pub replayed_stages: Vec<StageId>,
    pub replan_plan_id: String,
    pub qualification: QualificationKind,
    pub not_physical_qualification: bool,
    /// Filename-safe label for this receipt artifact.
    pub artifact_label: String,
}

pub const FAILURE_RECEIPT_SCHEMA: &str = "hawking.fabric.failure_replay_receipt.v1";

/// Tracks last heartbeat seq per node. Detection uses sequence gaps, not wall clock,
/// so tests stay deterministic.
#[derive(Debug, Default)]
pub struct HeartbeatMonitor {
    /// node -> last seq seen
    last_seq: BTreeMap<NodeId, u64>,
    /// nodes declared dead
    dead: BTreeMap<NodeId, u64>,
    /// max missed seqs before death (deterministic threshold)
    pub miss_threshold: u64,
}

impl HeartbeatMonitor {
    pub fn new(miss_threshold: u64) -> Self {
        Self {
            last_seq: BTreeMap::new(),
            dead: BTreeMap::new(),
            miss_threshold: miss_threshold.max(1),
        }
    }

    pub fn observe(&mut self, node: NodeId, seq: u64) {
        self.last_seq.insert(node, seq);
    }

    /// Advance a global "tick" counter for a node that did not heartbeat.
    /// If the gap from last_seq exceeds threshold, mark dead.
    pub fn note_missed(&mut self, node: &NodeId, global_tick: u64) -> bool {
        if self.dead.contains_key(node) {
            return true;
        }
        let last = self.last_seq.get(node).copied().unwrap_or(0);
        if global_tick.saturating_sub(last) >= self.miss_threshold {
            self.dead.insert(node.clone(), global_tick);
            true
        } else {
            false
        }
    }

    pub fn is_dead(&self, node: &NodeId) -> bool {
        self.dead.contains_key(node)
    }

    pub fn dead_nodes(&self) -> Vec<NodeId> {
        self.dead.keys().cloned().collect()
    }
}

#[derive(Debug)]
pub struct FailureDetector {
    pub heartbeats: HeartbeatMonitor,
}

impl FailureDetector {
    pub fn new(miss_threshold: u64) -> Self {
        Self {
            heartbeats: HeartbeatMonitor::new(miss_threshold),
        }
    }

    /// Build a failure/replay receipt after a node dies mid-request.
    pub fn build_receipt(
        &self,
        request_id: impl Into<String>,
        failed: &NodeId,
        plan: &PlacementPlan,
        pipeline: &PipelineScheduler,
        replan: &PlacementPlan,
        checkpoint: CheckpointId,
    ) -> FailureReplayReceipt {
        let lost_stages: Vec<StageId> = plan
            .stage_assignments
            .iter()
            .filter(|s| &s.node_id == failed)
            .map(|s| s.stage_id.clone())
            .collect();
        let lost_kv = KvOwnershipInvariant::ranges_lost_on_failure(&plan.kv_ownership, failed);
        let st = pipeline.status();
        let in_flight = st.in_flight as u32;

        // Replay stages whose layers overlap any lost stage.
        let lost_layers: Vec<(u32, u32)> = plan
            .stage_assignments
            .iter()
            .filter(|s| &s.node_id == failed)
            .map(|s| (s.layer_start, s.layer_end))
            .collect();
        let replayed_stages: Vec<StageId> = replan
            .stage_assignments
            .iter()
            .filter(|s| {
                lost_layers
                    .iter()
                    .any(|(ls, le)| s.layer_start < *le && s.layer_end > *ls)
            })
            .map(|s| s.stage_id.clone())
            .collect();

        let qualification = if plan.qualification == QualificationKind::Simulated {
            QualificationKind::Simulated
        } else {
            QualificationKind::SoftwareFixture
        };

        FailureReplayReceipt {
            schema: FAILURE_RECEIPT_SCHEMA.into(),
            request_id: request_id.into(),
            failed_node: failed.clone(),
            lost_work: LostWorkSummary {
                stages: lost_stages,
                kv_ranges: lost_kv,
                in_flight_microbatches: in_flight,
            },
            replayed_from_checkpoint: checkpoint,
            replayed_stages,
            replan_plan_id: replan.plan_id.clone(),
            qualification,
            not_physical_qualification: true,
            artifact_label: format!(
                "failure_replay_receipt_{}_{}",
                qualification.as_str(),
                failed.as_str()
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fabric::node::SimulatedNodeSet;
    use crate::fabric::placement::{
        ModelSection, PlacementRequest, PlacementSimulator, WorkloadClass,
    };
    const GIB: u64 = 1024 * 1024 * 1024;
    #[test]
    fn heartbeat_marks_dead_after_threshold() {
        let mut mon = HeartbeatMonitor::new(3);
        let n = NodeId::new("n1");
        mon.observe(n.clone(), 1);
        assert!(!mon.note_missed(&n, 2));
        assert!(!mon.note_missed(&n, 3));
        assert!(mon.note_missed(&n, 4)); // 4-1=3 >= 3
        assert!(mon.is_dead(&n));
    }
    #[test]
    fn receipt_names_lost_and_replayed() {
        let nodes = SimulatedNodeSet::heterogeneous_sim("sim-fail-v1").nodes;
        let sections = vec![
            ModelSection::content_addressed("a", 0, 2, 2 * GIB, b"a"),
            ModelSection::content_addressed("b", 2, 4, 2 * GIB, b"b"),
            ModelSection::content_addressed("c", 4, 6, 2 * GIB, b"c"),
        ];
        let workload = WorkloadClass {
            name: "f".into(),
            seq_len: 32,
            microbatch_size: 1,
            num_microbatches: 2,
        };
        let req = PlacementRequest {
            sections,
            nodes,
            workload: workload.clone(),
            seed: 9,
            qualification: QualificationKind::Simulated,
        };
        let sim = PlacementSimulator::new();
        let plan = sim.place(&req).unwrap();
        let failed = plan.stage_assignments[0].node_id.clone();
        let mut pipe = PipelineScheduler::from_plan(&plan, &workload, 2);
        let _ = pipe.tick();
        pipe.mark_node_failed(&failed);
        let replan = sim.replan_after_failure(&req, &failed).unwrap();
        let det = FailureDetector::new(2);
        let receipt = det.build_receipt(
            "req-1",
            &failed,
            &plan,
            &pipe,
            &replan,
            CheckpointId::new("ckpt-after-stage0"),
        );
        assert!(!receipt.lost_work.stages.is_empty());
        assert!(!receipt.lost_work.kv_ranges.is_empty());
        assert_eq!(receipt.replayed_from_checkpoint.0, "ckpt-after-stage0");
        assert!(receipt.not_physical_qualification);
        assert!(
            receipt.artifact_label.contains("simulated")
                || receipt.artifact_label.contains("software")
        );
        assert_ne!(receipt.replan_plan_id, plan.plan_id);
    }
}
