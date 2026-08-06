//! Fabric Agent — per-node process logic.
//!
//! Registers a node, reports real capabilities, heartbeats, accepts placement
//! assignments, and reports failure. Runs as a local OS process on this
//! machine; the ABI does not assume co-location with the coordinator.

use parking_lot::Mutex;
use serde_json::json;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use super::failure::{CheckpointId, FailureDetector, FailureReplayReceipt};
use super::node::{NodeCapabilities, NodeId, OsNodeProbe};
use super::placement::{ContentHash, PlacementPlan};
use super::protocol::{
    AgentRequest, AgentResponse, PlacementAssignment, FABRIC_PLACEHOLDER_EVENT_KIND,
};
use super::qualification::QualificationKind;

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub node_id: NodeId,
    pub listen_addr: String,
}

impl AgentConfig {
    pub fn new(node_id: impl Into<String>, listen_addr: impl Into<String>) -> Self {
        Self {
            node_id: NodeId::new(node_id),
            listen_addr: listen_addr.into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AgentState {
    pub node_id: NodeId,
    pub capabilities: NodeCapabilities,
    pub alive: bool,
    pub heartbeat_seq: u64,
    pub assignment: Option<PlacementAssignment>,
    pub held_section_hashes: BTreeSet<String>,
    pub last_request_id: Option<String>,
    pub last_checkpoint: Option<CheckpointId>,
    pub injected_failure: bool,
}

/// In-process fabric agent. The binary wraps this with a TCP loop.
pub struct FabricAgent {
    config: AgentConfig,
    probe: OsNodeProbe,
    state: Mutex<AgentState>,
    running: AtomicBool,
    hb_seq: AtomicU64,
}

impl FabricAgent {
    pub fn new(config: AgentConfig) -> Self {
        let probe = OsNodeProbe::new(config.node_id.as_str());
        let capabilities = probe.probe_once();
        let state = AgentState {
            node_id: config.node_id.clone(),
            capabilities,
            alive: true,
            heartbeat_seq: 0,
            assignment: None,
            held_section_hashes: BTreeSet::new(),
            last_request_id: None,
            last_checkpoint: None,
            injected_failure: false,
        };
        Self {
            config,
            probe,
            state: Mutex::new(state),
            running: AtomicBool::new(true),
            hb_seq: AtomicU64::new(0),
        }
    }

    pub fn node_id(&self) -> NodeId {
        self.config.node_id.clone()
    }

    pub fn capabilities(&self) -> NodeCapabilities {
        // Refresh free memory on read.
        let mut caps = self.probe.probe_once();
        caps.node_id = self.config.node_id.clone();
        let mut st = self.state.lock();
        st.capabilities = caps.clone();
        caps
    }

    pub fn snapshot(&self) -> AgentState {
        self.state.lock().clone()
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    pub fn handle(&self, req: AgentRequest) -> AgentResponse {
        if !self.running.load(Ordering::SeqCst) {
            return AgentResponse::Error {
                message: "agent stopped".into(),
            };
        }
        match req {
            AgentRequest::Register { .. } => {
                let caps = self.capabilities();
                AgentResponse::Registered { capabilities: caps }
            }
            AgentRequest::Heartbeat { node_id, seq } => {
                let mut st = self.state.lock();
                if st.injected_failure || !st.alive {
                    return AgentResponse::Failed {
                        node_id: st.node_id.clone(),
                        reason: "node dead".into(),
                    };
                }
                st.heartbeat_seq = seq;
                self.hb_seq.store(seq, Ordering::SeqCst);
                AgentResponse::HeartbeatAck { node_id, seq }
            }
            AgentRequest::Assign { assignment } => self.accept_assignment(assignment),
            AgentRequest::RunRequest {
                request_id,
                plan_id,
            } => self.run_request(request_id, plan_id),
            AgentRequest::InjectFailure { node_id } => {
                let mut st = self.state.lock();
                if st.node_id != node_id {
                    return AgentResponse::Error {
                        message: format!("wrong node: {}", st.node_id),
                    };
                }
                st.alive = false;
                st.injected_failure = true;
                // Placeholder event — Bridge lane owns the real event model.
                AgentResponse::PlaceholderEvent {
                    kind: FABRIC_PLACEHOLDER_EVENT_KIND.into(),
                    payload: json!({
                        "event": "node_failure_injected",
                        "node_id": node_id.as_str(),
                    }),
                }
            }
            AgentRequest::GetStatus { node_id } => {
                let st = self.state.lock();
                if st.node_id != node_id {
                    return AgentResponse::Error {
                        message: "node id mismatch".into(),
                    };
                }
                AgentResponse::Status {
                    node_id: st.node_id.clone(),
                    alive: st.alive && !st.injected_failure,
                    assignment_plan_id: st.assignment.as_ref().map(|a| a.plan_id.clone()),
                    held_section_hashes: st.held_section_hashes.iter().cloned().collect(),
                }
            }
            AgentRequest::Shutdown => {
                self.running.store(false, Ordering::SeqCst);
                let mut st = self.state.lock();
                st.alive = false;
                AgentResponse::Ok {
                    node_id: st.node_id.clone(),
                    detail: "shutdown".into(),
                }
            }
        }
    }

    fn accept_assignment(&self, assignment: PlacementAssignment) -> AgentResponse {
        let mut st = self.state.lock();
        if !st.alive || st.injected_failure {
            return AgentResponse::Failed {
                node_id: st.node_id.clone(),
                reason: "cannot accept assignment: node dead".into(),
            };
        }
        if assignment.assigned_node != st.node_id {
            return AgentResponse::Error {
                message: format!(
                    "assignment for {} delivered to {}",
                    assignment.assigned_node, st.node_id
                ),
            };
        }
        // Prove we hold the sections the plan assigns to us (by content hash).
        let mut held = BTreeSet::new();
        for sp in &assignment.plan.section_placements {
            if sp.node_id == st.node_id {
                held.insert(sp.content_hash.0.clone());
            }
        }
        let plan_id = assignment.plan_id.clone();
        st.held_section_hashes = held.clone();
        st.assignment = Some(assignment);
        st.last_checkpoint = Some(CheckpointId::new(format!("ckpt-assigned-{plan_id}")));
        AgentResponse::AssignmentAccepted {
            node_id: st.node_id.clone(),
            plan_id,
            held_section_hashes: held.into_iter().collect(),
        }
    }

    fn run_request(&self, request_id: String, plan_id: String) -> AgentResponse {
        let mut st = self.state.lock();
        if !st.alive || st.injected_failure {
            return AgentResponse::Failed {
                node_id: st.node_id.clone(),
                reason: format!("node dead mid-request {request_id}"),
            };
        }
        let Some(assignment) = st.assignment.as_ref() else {
            return AgentResponse::Error {
                message: "no assignment".into(),
            };
        };
        if assignment.plan_id != plan_id {
            return AgentResponse::Error {
                message: format!("plan mismatch: have {} want {plan_id}", assignment.plan_id),
            };
        }
        let local_stages = assignment
            .plan
            .stage_assignments
            .iter()
            .filter(|s| s.node_id == st.node_id)
            .count() as u32;
        st.last_request_id = Some(request_id.clone());
        st.last_checkpoint = Some(CheckpointId::new(format!(
            "ckpt-req-{request_id}-after-local"
        )));
        AgentResponse::RequestProgress {
            node_id: st.node_id.clone(),
            request_id,
            completed_local_stages: local_stages,
        }
    }

    /// Prove held content hashes match the plan for this node.
    pub fn prove_holds(&self, plan: &PlacementPlan) -> Result<Vec<ContentHash>, String> {
        let st = self.state.lock();
        let mut proofs = Vec::new();
        for sp in &plan.section_placements {
            if sp.node_id != st.node_id {
                continue;
            }
            if !st.held_section_hashes.contains(&sp.content_hash.0) {
                return Err(format!(
                    "node {} missing section hash {}",
                    st.node_id, sp.content_hash.0
                ));
            }
            proofs.push(sp.content_hash.clone());
        }
        Ok(proofs)
    }
}

/// Shared handle for multi-threaded TCP server.
pub type FabricAgentHandle = Arc<FabricAgent>;

/// Build a software-fixture receipt when coordinating failure outside the agent.
pub fn fixture_receipt(
    request_id: &str,
    failed: &NodeId,
    plan: &PlacementPlan,
    replan: &PlacementPlan,
    lost_in_flight: u32,
) -> FailureReplayReceipt {
    use super::failure::{LostWorkSummary, FAILURE_RECEIPT_SCHEMA};
    use super::placement::KvOwnershipInvariant;

    let lost_stages = plan
        .stage_assignments
        .iter()
        .filter(|s| &s.node_id == failed)
        .map(|s| s.stage_id.clone())
        .collect::<Vec<_>>();
    let lost_kv = KvOwnershipInvariant::ranges_lost_on_failure(&plan.kv_ownership, failed);
    let lost_layers: Vec<(u32, u32)> = plan
        .stage_assignments
        .iter()
        .filter(|s| &s.node_id == failed)
        .map(|s| (s.layer_start, s.layer_end))
        .collect();
    let replayed_stages = replan
        .stage_assignments
        .iter()
        .filter(|s| {
            lost_layers
                .iter()
                .any(|(ls, le)| s.layer_start < *le && s.layer_end > *ls)
        })
        .map(|s| s.stage_id.clone())
        .collect();

    FailureReplayReceipt {
        schema: FAILURE_RECEIPT_SCHEMA.into(),
        request_id: request_id.into(),
        failed_node: failed.clone(),
        lost_work: LostWorkSummary {
            stages: lost_stages,
            kv_ranges: lost_kv,
            in_flight_microbatches: lost_in_flight,
        },
        replayed_from_checkpoint: CheckpointId::new(format!("ckpt-before-fail-{request_id}")),
        replayed_stages,
        replan_plan_id: replan.plan_id.clone(),
        qualification: QualificationKind::SoftwareFixture,
        not_physical_qualification: true,
        artifact_label: format!(
            "failure_replay_receipt_software_fixture_{}",
            failed.as_str()
        ),
    }
}

/// Helper used by coordinator logic after detecting death.
pub fn detector() -> FailureDetector {
    FailureDetector::new(3)
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
    fn agent_registers_real_capabilities() {
        let agent = FabricAgent::new(AgentConfig::new("agent-a", "127.0.0.1:0"));
        let caps = agent.capabilities();
        assert_eq!(caps.node_id.as_str(), "agent-a");
        assert!(caps.total_memory_bytes > 0);
        assert_ne!(
            caps.total_memory_bytes,
            crate::fabric::node::FIXED_FAKE_MEMORY_BYTES
        );
    }
    #[test]
    fn agent_accepts_assignment_and_proves_hashes() {
        let nodes = SimulatedNodeSet::homogeneous_pair_sim("sim-agent-v1", 64 * GIB, 8).nodes;
        let sections = vec![
            ModelSection::content_addressed("s0", 0, 2, 4 * GIB, b"s0"),
            ModelSection::content_addressed("s1", 2, 4, 4 * GIB, b"s1"),
        ];
        let req = PlacementRequest {
            sections,
            nodes: nodes.clone(),
            workload: WorkloadClass::default(),
            seed: 1,
            qualification: QualificationKind::Simulated,
        };
        let plan = PlacementSimulator::new().place(&req).unwrap();
        let target = plan.section_placements[0].node_id.clone();
        let agent = FabricAgent::new(AgentConfig::new(target.as_str(), "127.0.0.1:0"));
        let resp = agent.handle(AgentRequest::Assign {
            assignment: PlacementAssignment {
                plan_id: plan.plan_id.clone(),
                plan: plan.clone(),
                assigned_node: target.clone(),
            },
        });
        match resp {
            AgentResponse::AssignmentAccepted {
                held_section_hashes,
                ..
            } => {
                assert!(!held_section_hashes.is_empty());
            }
            other => panic!("unexpected {other:?}"),
        }
        agent.prove_holds(&plan).unwrap();
    }
}
