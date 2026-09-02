//! Lane roles, evidence packets, and the MemGate-controlled lane scheduler.
//!
//! The ceiling is a bootstrap maximum (3), NOT a requirement. The MemGate may
//! admit 3/2/1/0 depending on measured pressure.

use std::collections::BTreeSet;
use std::time::Instant;

use super::dag::{HcliDag, NodeId, NodeStatus};
use super::memgate::MemGate;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LaneId(pub String);

/// Task-dependent roles. Do not launch redundant lanes with identical prompts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneRole {
    /// Architecture, decomposition, integration design.
    Architect,
    /// Code implementation, isolated patch/worktree.
    Implementer,
    /// Falsification, regression analysis, simpler alternatives, destructive-edit detection.
    Adversary,
}

impl LaneRole {
    /// Short label for the compact UI.
    pub fn short(self) -> &'static str {
        match self {
            LaneRole::Architect => "ARCH",
            LaneRole::Implementer => "BUILD",
            LaneRole::Adversary => "REDTEAM",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneStatus {
    Idle,
    Thinking,
    Testing,
    QueuedMemGate,
    Done,
    Failed,
}

impl LaneStatus {
    pub fn label(self) -> &'static str {
        match self {
            LaneStatus::Idle => "IDLE",
            LaneStatus::Thinking => "THINKING",
            LaneStatus::Testing => "TESTING",
            LaneStatus::QueuedMemGate => "QUEUED · MEMGATE",
            LaneStatus::Done => "DONE",
            LaneStatus::Failed => "FAILED",
        }
    }
}

/// Per-lane context budget. Do NOT clone the entire parent context into each
/// lane; optimize evidence per token.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ContextBudget {
    pub max_tokens: usize,
}

/// A bounded evidence packet + explicit contract handed to a child lane.
#[derive(Clone, Debug, Default)]
pub struct EvidencePacket {
    pub role: LaneRole,
    pub task: String,
    pub files: Vec<String>,
    pub receipts: Vec<String>,
    pub task_local_map: String,
    pub output_contract: String,
    pub context_budget: ContextBudget,
}

#[derive(Clone, Debug)]
pub struct Lane {
    pub id: LaneId,
    pub role: Option<LaneRole>,
    pub node: Option<NodeId>,
    pub session: Option<String>,
    pub status: LaneStatus,
    pub started_at: Option<Instant>,
    pub last_elapsed_ms: u64,
}

impl Lane {
    /// One line of the compact UI: `A  ARCH        THINKING`.
    pub fn render_line(&self, letter: char) -> String {
        let role = self.role.map(|r| r.short()).unwrap_or("GENERIC");
        format!("{letter}  {:<12}{}", role, self.status.label())
    }
}

/// The MemGate-controlled lane scheduler.
#[derive(Clone, Debug)]
pub struct LaneScheduler {
    pub lanes: Vec<Lane>,
    pub last_admission_reason: Option<String>,
}

impl LaneScheduler {
    pub fn new(ceiling: usize) -> Self {
        let count = ceiling.min(3).max(1);
        let lanes = (0..count)
            .map(|i| Lane {
                id: LaneId(format!("{}", char::from(b'A' + i as u8))),
                role: None,
                node: None,
                session: None,
                status: LaneStatus::Idle,
                started_at: None,
                last_elapsed_ms: 0,
            })
            .collect();
        Self {
            lanes,
            last_admission_reason: None,
        }
    }

    pub fn admitted_count(&self) -> usize {
        self.lanes
            .iter()
            .filter(|l| matches!(l.status, LaneStatus::Thinking | LaneStatus::Testing))
            .count()
    }

    pub fn lane_latencies_ms(&self) -> Vec<u64> {
        self.lanes.iter().map(|l| l.last_elapsed_ms).collect()
    }

    /// Admit as many ready nodes as the MemGate allows, assigning them to idle
    /// lanes. Returns the admitted node ids.
    pub fn admit(&mut self, dag: &mut HcliDag, gate: &dyn MemGate) -> Vec<NodeId> {
        let ready = dag.ready_nodes();
        if ready.is_empty() {
            return Vec::new();
        }
        let decision = gate.admit(ready.len());
        let limit = decision.admitted_lanes.min(gate.ceiling());
        let mut admitted = Vec::new();
        let mut taken = BTreeSet::new();

        let active: Vec<NodeId> = self
            .lanes
            .iter()
            .filter(|l| matches!(l.status, LaneStatus::Thinking | LaneStatus::Testing))
            .filter_map(|l| l.node.clone())
            .collect();

        for node_id in ready {
            if admitted.len() >= limit {
                break;
            }
            let node = match dag.get(&node_id) {
                Some(n) => n,
                None => continue,
            };

            let compatible = active
                .iter()
                .all(|other| dag.can_run_concurrently(other, &node_id))
                && admitted
                    .iter()
                    .all(|other| dag.can_run_concurrently(other, &node_id));
            if !compatible {
                continue;
            }

            let lane_idx = self.lanes.iter().position(|l| l.status == LaneStatus::Idle);
            if let Some(idx) = lane_idx {
                if let Some(lane) = self.lanes.get_mut(idx) {
                    lane.role = Some(node.role);
                    lane.node = Some(node_id.clone());
                    lane.status = LaneStatus::Thinking;
                    lane.started_at = Some(Instant::now());
                    lane.last_elapsed_ms = 0;
                }
                if let Some(n) = dag.get_mut(&node_id) {
                    n.status = NodeStatus::Running;
                }
                taken.insert(node_id.clone());
                admitted.push(node_id);
            }
        }

        // Any ready node not admitted is queued (MemGate refused) — do not fail.
        for node_id in &ready {
            if !taken.contains(node_id) {
                if let Some(n) = dag.get_mut(node_id) {
                    n.status = NodeStatus::Queued;
                }
            }
        }

        self.last_admission_reason = Some(decision.reason);
        admitted
    }

    pub fn complete_lane(&mut self, node: &NodeId) {
        self.finish_lane(node, LaneStatus::Done);
    }

    pub fn fail_lane(&mut self, node: &NodeId) {
        self.finish_lane(node, LaneStatus::Failed);
    }

    pub fn complete_lane_by_label(&mut self, label: &str) {
        if let Some(node) = self
            .lanes
            .iter()
            .find(|l| l.id.0 == label)
            .and_then(|l| l.node.clone())
        {
            self.complete_lane(&node);
        }
    }

    fn finish_lane(&mut self, node: &NodeId, status: LaneStatus) {
        if let Some(lane) = self
            .lanes
            .iter_mut()
            .find(|l| l.node.as_ref() == Some(node))
        {
            if let Some(start) = lane.started_at.take() {
                lane.last_elapsed_ms = start.elapsed().as_millis() as u64;
            }
            lane.status = status;
            lane.node = None;
            lane.role = None;
        }
    }
}
