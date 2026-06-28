use crate::govern::{Budget, BudgetLedger};
use crate::plan::schema::{Plan, StepStatus};
use crate::verify::oracle::Verdict;
use hide_core::ids::{RunId, SessionId, StepId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Intake,
    Plan,
    SelectStep,
    Act,
    Observe,
    Verify,
    Repair,
    Replan,
    Finalize,
    Done,
    Aborted,
    Paused,
}

impl Phase {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Phase::Done | Phase::Aborted)
    }

    /// snake_case wire name (matches the serde rename) — used for event payloads
    /// so the projection's snake_case parser round-trips correctly.
    pub fn wire_name(&self) -> &'static str {
        match self {
            Phase::Intake => "intake",
            Phase::Plan => "plan",
            Phase::SelectStep => "select_step",
            Phase::Act => "act",
            Phase::Observe => "observe",
            Phase::Verify => "verify",
            Phase::Repair => "repair",
            Phase::Replan => "replan",
            Phase::Finalize => "finalize",
            Phase::Done => "done",
            Phase::Aborted => "aborted",
            Phase::Paused => "paused",
        }
    }
}

/// A frame on the search/subagent stack (bible §4.2 `stack: Vec<Frame>`). Bounds
/// search-node and subagent recursion depth (K8).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum Frame {
    /// A search node (best-of-N candidate exploration).
    Search { step_id: StepId, tier: String, candidates: u32 },
    /// A nested subagent run.
    Subagent { child_run: RunId, objective: String },
}

/// A pending approval request (typed, §4.3) raised when an effectful step needs
/// human sign-off under suggest-only autonomy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApprovalRequest {
    pub step_id: StepId,
    pub summary: String,
    pub effects: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentState {
    pub session_id: SessionId,
    pub run_id: RunId,
    pub objective: String,
    pub phase: Phase,
    pub plan: Option<Plan>,
    pub cursor: Option<StepId>,
    pub budget: Budget,
    pub ledger: BudgetLedger,
    pub last_verdict: Option<Verdict>,
    /// All verdicts from the most recent verify pass (gate input + repair ctx).
    #[serde(default)]
    pub last_verdicts: Vec<Verdict>,
    pub repair_count: BTreeMap<StepId, u8>,
    /// Replan count (bounded by `Budget.max_replans`).
    #[serde(default)]
    pub replan_count: u8,
    /// Search/subagent frames (bounded-depth recursion).
    #[serde(default)]
    pub stack: Vec<Frame>,
    /// The context manifest hash from the last context compile that grounded a
    /// step (provenance for replay / debugging).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_manifest: Option<String>,
    /// Steering instructions injected mid-run (Interrupt::Steer).
    #[serde(default)]
    pub steer: Vec<String>,
    /// Typed pending approval (set when entering Paused).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pending_approval: Option<ApprovalRequest>,
    /// Lessons carried forward from failures into the next repair/replan (§4.7).
    #[serde(default)]
    pub lessons: Vec<String>,
}

impl AgentState {
    pub fn new(session_id: SessionId, run_id: RunId, objective: String) -> Self {
        Self {
            session_id,
            run_id,
            objective,
            phase: Phase::Intake,
            plan: None,
            cursor: None,
            budget: Budget::default(),
            ledger: BudgetLedger::default(),
            last_verdict: None,
            last_verdicts: Vec::new(),
            repair_count: BTreeMap::new(),
            replan_count: 0,
            stack: Vec::new(),
            context_manifest: None,
            steer: Vec::new(),
            pending_approval: None,
            lessons: Vec::new(),
        }
    }

    pub fn mark_cursor(&mut self, status: StepStatus) {
        if let (Some(plan), Some(cursor)) = (&mut self.plan, &self.cursor) {
            if let Some(step) = plan.steps.iter_mut().find(|step| &step.id == cursor) {
                step.status = status;
            }
        }
    }

    /// Repairs consumed for the current cursor step.
    pub fn cursor_repair_count(&self) -> u8 {
        self.cursor
            .as_ref()
            .and_then(|c| self.repair_count.get(c))
            .copied()
            .unwrap_or(0)
    }
}
