//! Lane roles, evidence packets, and the MemGate-controlled lane scheduler.
//!
//! The ceiling is a bootstrap maximum (3), NOT a requirement. The MemGate may
//! admit 3/2/1/0 depending on measured pressure.

use std::collections::BTreeSet;

use super::dag::{HaiderDag, NodeId, NodeStatus};
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
    pub role: LaneRole,
    pub node: Option<NodeId>,
    pub session: Option<String>,
    pub status: LaneStatus,
}

impl Lane {
    /// One line of the compact UI: `A  ARCH        THINKING`.
    pub fn render_line(&self, letter: char) -> String {
        format!("{letter}  {:<12}{}", self.role.short(), self.status.label())
    }
}

/// The MemGate-controlled lane scheduler.
#[derive(Clone, Debug)]
pub struct LaneScheduler {
    pub lanes: Vec<Lane>,
}

impl LaneScheduler {
    pub fn new(ceiling: usize) -> Self {
        let roles = [LaneRole::Architect, LaneRole::Implementer, LaneRole::Adversary];
        let lanes = (0..ceiling.min(3).max(1))
            .map(|i| Lane {
                id: LaneId(format!("{}", char::from(b'A' + i as u8))),
                role: roles[i],
                node: None,
                session: None,
                status: LaneStatus::Idle,
            })
            .collect();
        Self { lanes }
    }

    pub fn admitted_count(&self) -> usize {
        self.lanes
            .iter()
            .filter(|l| matches!(l.status, LaneStatus::Thinking | LaneStatus::Testing))
            .count()
    }

    /// Admit as many ready nodes as the MemGate allows, assigning them to idle
    /// lanes with compatible roles. Returns the admitted node ids.
    pub fn admit(&mut self, dag: &mut HaiderDag, gate: &dyn MemGate) -> Vec<NodeId> {
        let ready = dag.ready_nodes();
        if ready.is_empty() {
            return Vec::new();
        }
        let decision = gate.admit(ready.len());
        let ceiling = gate.ceiling();
        let mut admitted = Vec::new();
        let mut taken = BTreeSet::new();
        for node_id in ready {
            if admitted.len() >= decision.admitted_lanes.min(ceiling) {
                break;
            }
            let node = match dag.get(&node_id) {
                Some(n) => n,
                None => continue,
            };
            let lane_idx = self
                .lanes
                .iter()
                .position(|l| l.status == LaneStatus::Idle && l.role == node.role)
                .and_then(|idx| {
                    let ok = admitted
                        .iter()
                        .all(|other| dag.can_run_concurrently(other, &node_id));
                    if ok {
                        Some(idx)
                    } else {
                        None
                    }
                });
            if let Some(idx) = lane_idx {
                if let Some(lane) = self.lanes.get_mut(idx) {
                    lane.node = Some(node_id.clone());
                    lane.status = LaneStatus::Thinking;
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
        admitted
    }
}
