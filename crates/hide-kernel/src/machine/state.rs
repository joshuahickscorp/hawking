use crate::govern::{Budget, BudgetLedger};
use crate::plan::schema::{Plan, StepStatus};
use crate::verify::oracle::Verdict;
use hide_core::ids::{RunId, SessionId, StepId};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

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

/// A typed lesson distilled from a failure, anchored to the decision that
/// produced it (§4.7). Replaces the old free-string list so learnings carry
/// provenance (phase + step) for replay and can be retained with a bound.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Lesson {
    pub text: String,
    pub phase: Phase,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step_id: Option<StepId>,
    /// Reserved monotonic stamp. The driver has no clock; the emitting event
    /// carries the authoritative timestamp. Defaults to 0.
    #[serde(default)]
    pub ts: u64,
}

/// Cap on retained lessons — Reflexion plateaus around 3-5 and an unbounded
/// scratchpad induces confabulation, so `push_lesson` evicts the oldest beyond
/// this.
const MAX_LESSONS: usize = 5;

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
    /// Typed (provenance-anchored) + bounded — see `push_lesson` / `MAX_LESSONS`.
    #[serde(default)]
    pub lessons: Vec<Lesson>,
    /// Rolling fingerprints of the last K verify passes (normalized
    /// oracle/status/first-failure). When the last K are identical the run has
    /// stalled (repair is not converging) and routes to Replan instead of
    /// looping Repair forever (W-F5-1 convergence detection).
    #[serde(default)]
    pub verdict_history: VecDeque<String>,
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
            verdict_history: VecDeque::new(),
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

    /// Record a lesson with bounded retention (oldest evicted past
    /// `MAX_LESSONS`), so the scratchpad cannot grow unboundedly.
    pub fn push_lesson(&mut self, lesson: Lesson) {
        self.lessons.push(lesson);
        while self.lessons.len() > MAX_LESSONS {
            self.lessons.remove(0);
        }
    }
}

#[cfg(test)]
mod lesson_tests {
    use super::*;
    use hide_core::ids::{RunId, SessionId};

    fn lesson(n: usize) -> Lesson {
        Lesson {
            text: format!("L{n}"),
            phase: Phase::Repair,
            step_id: None,
            ts: 0,
        }
    }

    #[test]
    fn push_lesson_is_bounded_and_evicts_oldest() {
        let mut s = AgentState::new(SessionId::new(), RunId::new(), "obj".to_string());
        for i in 0..(MAX_LESSONS + 2) {
            s.push_lesson(lesson(i));
        }
        assert_eq!(s.lessons.len(), MAX_LESSONS);
        // The two oldest were evicted; the newest is retained.
        assert_eq!(s.lessons.first().unwrap().text, "L2");
        assert_eq!(s.lessons.last().unwrap().text, format!("L{}", MAX_LESSONS + 1));
        assert_eq!(s.lessons[0].phase, Phase::Repair);
    }
}
