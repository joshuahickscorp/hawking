//! HCLI P0.5 — MemGate-controlled parent parallelism.
//!
//! One logical PARENT owns the Ultragoal, DAG, frontier, and final synthesis.
//! Child lanes receive bounded evidence packets and explicit contracts, and are
//! admitted only as far as the authoritative MemGate allows (ceiling 3, but the
//! gate may admit 3/2/1/0 under measured pressure).
//!
//! This module does NOT duplicate the existing Hawking memory/resource gate; it
//! adapts it behind [`memgate::MemGate`] and records per-episode feedback so the
//! real optimal concurrency can be learned.

pub mod cli;
pub mod context_governor;
pub mod dag;
pub mod harvest;
pub mod lanes;
pub mod memgate;
pub mod tool_bus;
pub mod ultragoal;

pub use cli::run as run_cli;
pub use context_governor::{ContextBudgets, ContextGovernor, TaskType};
pub use dag::{HcliDag, HcliNode, NodeId, NodeResult, NodeStatus, ResourceClass, Scope};
pub use harvest::{harvest, Contradiction, FrontierPacket, Harvest, LaneOutput, TestResult};
pub use lanes::{
    ContextBudget, EvidencePacket, Lane, LaneId, LaneRole, LaneScheduler, LaneStatus,
};
pub use memgate::{AdmissionDecision, EpisodeFeedback, MemGate, MemoryPressure, SystemMemGate};
pub use tool_bus::{ToolBus, ToolError, ToolRequest, ToolResult};
pub use ultragoal::{IngestSummary, MissionState, Obligation, ObligationStatus, Steer};

use std::time::Instant;

/// The parent orchestrator. It owns the DAG and the lane scheduler, keeps moving
/// while lanes run, and harvests immediately when lanes finish.
pub struct Hcli {
    pub dag: HcliDag,
    pub lanes: LaneScheduler,
    pub gate: std::sync::Arc<dyn MemGate>,
    /// Feedback from completed parallel episodes, used to learn concurrency.
    pub episodes: Vec<EpisodeFeedback>,
    /// Wall-clock start of the current episode (for feedback).
    episode_start: Option<Instant>,
}

impl Hcli {
    pub fn new(gate: std::sync::Arc<dyn MemGate>, ceiling: usize) -> Self {
        Self {
            dag: HcliDag::new(),
            lanes: LaneScheduler::new(ceiling),
            gate,
            episodes: Vec::new(),
            episode_start: None,
        }
    }

    /// Begin a parallel episode.
    pub fn start_episode(&mut self) {
        self.episode_start = Some(Instant::now());
    }

    /// Admit as many ready nodes as the MemGate allows. Returns admitted nodes.
    pub fn admit_ready(&mut self) -> Vec<NodeId> {
        self.lanes.admit(&mut self.dag, self.gate.as_ref())
    }

    /// The parent keeps moving while lanes run: CPU-only work (inspect code,
    /// construct tests, prepare context packets, update DAG state).
    pub fn parent_cpu_work(&mut self) {
        // Hook for CPU-only parent work. Does not block on lanes.
    }

    /// Harvest immediately when lanes finish.
    pub fn harvest_finished(&mut self, outputs: Vec<LaneOutput>) -> Harvest {
        let h = harvest(&outputs);
        if let Some(start) = self.episode_start.take() {
            let p = self.gate.pressure();
            let elapsed = start.elapsed();
            let successful_work = outputs
                .iter()
                .filter(|o| {
                    o.test_results.iter().any(|t| t.passed) || !o.proposed_actions.is_empty()
                })
                .count();
            let fb = EpisodeFeedback {
                admitted_lane_count: self.lanes.admitted_count(),
                peak_wired_bytes: p.wired_bytes,
                peak_compressed_bytes: p.compressed_bytes,
                peak_swap_bytes: p.swap_bytes,
                model_memory_bytes: p.resident_model_bytes,
                context_sizes: h.per_lane_tokens.values().cloned().collect(),
                wall_time_ms: elapsed.as_millis() as u64,
                aggregate_token_throughput: h.per_lane_tokens.values().sum::<usize>() as f64
                    / elapsed.as_secs_f64().max(1e-6),
                per_lane_latency_ms: self.lanes.lane_latencies_ms(),
                successful_work,
            };
            self.episodes.push(fb);
        }
        for o in &outputs {
            self.lanes.complete_lane_by_label(&o.lane);
        }
        h
    }

    /// Mark a node complete and release its lane.
    pub fn complete_node(&mut self, node: &NodeId, result: NodeResult) {
        self.dag.complete(node, result);
        self.lanes.complete_lane(node);
    }

    /// If the MemGate refuses a child, queue it (do not fail the Ultragoal).
    pub fn on_memgate_refuse(&mut self, node: &NodeId) {
        if let Some(n) = self.dag.get_mut(node) {
            n.status = NodeStatus::Queued;
        }
    }

    /// If a lane crashes: preserve output, reap, retry only if policy permits.
    pub fn on_lane_crash(&mut self, node: &NodeId, preserve: NodeResult) {
        if let Some(n) = self.dag.get_mut(node) {
            n.status = NodeStatus::Failed;
            n.result = Some(preserve);
        }
        self.lanes.fail_lane(node);
    }

    /// Compact status block for the UI (section 14).
    pub fn render_status(&self, parent_label: &str) -> String {
        let p = self.gate.pressure();
        let gb = 1024u64 * 1024 * 1024;
        let used_gb = (p.wired_bytes.saturating_add(p.compressed_bytes)) / gb;
        let total_gb = p.total_physical_bytes / gb;
        let admitted = self.lanes.admitted_count();
        let ceiling = self.gate.ceiling();
        let mut out = String::new();
        out.push_str("HCLI\n\n");
        out.push_str(&format!("PARENT   {parent_label}\n"));
        out.push_str(&format!("MEM      {used_gb} / {total_gb} GB\n"));
        out.push_str(&format!("LANES    {admitted} / {ceiling} admitted\n\n"));
        for (i, lane) in self.lanes.lanes.iter().enumerate() {
            let letter = char::from(b'A' + (i % 26) as u8);
            out.push_str(&lane.render_line(letter));
            out.push('\n');
        }
        out
    }
}
