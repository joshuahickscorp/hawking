//! HCLI Agent OS — Planning Diagnostic scaffold (Ascension Bible §14).
//!
//! **What this is:** typed stage pipeline + per-stage receipts + the kernel-
//! research contract that forbids unbounded “try optimizations” plans.
//! Model-free; no agent-loop execution.
//!
//! **Relationship to existing plan machinery:**
//! * [`hide_kernel::plan::schema::Plan`] — execution DAG (steps + acceptance
//!   oracles). The diagnostic *produces* / *challenges* that plan; it does not
//!   replace it.
//! * [`crate::plan_domain::PlanRecord`] — durable PlanCard projection for the
//!   IDE. Host may map a verified diagnostic plan into a PlanRecord later.
//! * Kernel FSM phases (`Intake`…`Replan`…`Done`) are the *run* driver; this
//!   module is the *planning quality* protocol wrapped around plan synthesis.
//!
//! **Boundary honesty:** stage advancement here records receipts only. Real
//! tool retrieval, model plan synthesis, challenge critique, and execution
//! remain deferred to Agent OS activation (bible §0 / §35 step 9).

use hide_core::ids::{now_ms, SessionId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Schema id for durable planning-diagnostic receipts.
pub const PLANNING_DIAGNOSTIC_SCHEMA: &str = "hcli.planning_diagnostic.v1";

// ---------------------------------------------------------------------------
// Stage pipeline (bible §14)
// ---------------------------------------------------------------------------

/// Ordered stages of the planning diagnostic.
///
/// ```text
/// GOAL INTERPRETATION
/// → TOOL RETRIEVAL
/// → PLAN
/// → PLAN CHALLENGE
/// → EXECUTION
/// → OBSERVATION
/// → REPLAN          (optional loop back toward PLAN / PLAN CHALLENGE)
/// → VERIFICATION
/// → REPORT
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticStage {
    GoalInterpretation,
    ToolRetrieval,
    Plan,
    PlanChallenge,
    Execution,
    Observation,
    Replan,
    Verification,
    Report,
}

impl DiagnosticStage {
    pub const ALL: &'static [DiagnosticStage] = &[
        DiagnosticStage::GoalInterpretation,
        DiagnosticStage::ToolRetrieval,
        DiagnosticStage::Plan,
        DiagnosticStage::PlanChallenge,
        DiagnosticStage::Execution,
        DiagnosticStage::Observation,
        DiagnosticStage::Replan,
        DiagnosticStage::Verification,
        DiagnosticStage::Report,
    ];

    pub fn wire_name(self) -> &'static str {
        match self {
            Self::GoalInterpretation => "goal_interpretation",
            Self::ToolRetrieval => "tool_retrieval",
            Self::Plan => "plan",
            Self::PlanChallenge => "plan_challenge",
            Self::Execution => "execution",
            Self::Observation => "observation",
            Self::Replan => "replan",
            Self::Verification => "verification",
            Self::Report => "report",
        }
    }

    /// Forward edges of the pipeline (not including replan loops).
    pub fn successors(self) -> &'static [DiagnosticStage] {
        use DiagnosticStage::*;
        match self {
            GoalInterpretation => &[ToolRetrieval],
            ToolRetrieval => &[Plan],
            Plan => &[PlanChallenge],
            PlanChallenge => &[Execution, Plan], // challenge may send back to plan
            Execution => &[Observation],
            Observation => &[Replan, Verification],
            Replan => &[Plan, PlanChallenge, Verification],
            Verification => &[Report, Replan],
            Report => &[],
        }
    }

    pub fn is_terminal_stage(self) -> bool {
        matches!(self, DiagnosticStage::Report)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageStatus {
    Pending,
    Running,
    Passed,
    Failed,
    Skipped,
    Blocked,
}

impl StageStatus {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            StageStatus::Passed | StageStatus::Failed | StageStatus::Skipped
        )
    }
}

// ---------------------------------------------------------------------------
// Receipts
// ---------------------------------------------------------------------------

/// One stage's durable receipt. Every stage emits exactly one of these when
/// it leaves `Running`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageReceipt {
    pub stage: DiagnosticStage,
    pub status: StageStatus,
    /// Free-form structured summary (goal text, tool ids, plan hash, …).
    pub summary: String,
    /// Machine-readable payload for host projection / later model integration.
    #[serde(default)]
    pub payload: BTreeMap<String, serde_json::Value>,
    pub started_ms: u64,
    pub finished_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl StageReceipt {
    pub fn open(stage: DiagnosticStage, at_ms: u64) -> Self {
        Self {
            stage,
            status: StageStatus::Running,
            summary: String::new(),
            payload: BTreeMap::new(),
            started_ms: at_ms,
            finished_ms: None,
            error: None,
        }
    }

    pub fn finish(
        mut self,
        status: StageStatus,
        summary: impl Into<String>,
        at_ms: u64,
    ) -> Self {
        self.status = status;
        self.summary = summary.into();
        self.finished_ms = Some(at_ms);
        self
    }

    pub fn with_payload(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.payload.insert(key.into(), value);
        self
    }

    pub fn fail(mut self, error: impl Into<String>, at_ms: u64) -> Self {
        self.status = StageStatus::Failed;
        self.error = Some(error.into());
        self.finished_ms = Some(at_ms);
        self
    }
}

// ---------------------------------------------------------------------------
// Kernel / research planning contract (bible §14 second half)
// ---------------------------------------------------------------------------

/// Required answers for kernel/research planning. A plan that cannot fill
/// these fields is not admissible for research execution.
///
/// Bible questions:
/// * What is the measured bottleneck?
/// * What evidence distinguishes candidate explanations?
/// * What is the cheapest experiment that can disprove the hypothesis?
/// * Which tools are required?
/// * What result causes promotion?
/// * What result causes retirement?
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelResearchContract {
    pub measured_bottleneck: String,
    pub distinguishing_evidence: String,
    pub cheapest_disprove_experiment: String,
    pub required_tools: Vec<String>,
    pub promotion_result: String,
    pub retirement_result: String,
    /// Explicit bound on the experiment (max steps / wall ms / budget units).
    pub experiment_bound: ExperimentBound,
}

/// Hard bound so “try optimizations” cannot mean unbounded work.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentBound {
    pub max_steps: u32,
    pub max_wall_ms: u64,
    /// Optional dollar / compute credit ceiling (0 = unset).
    #[serde(default)]
    pub max_compute_units: u64,
}

impl Default for ExperimentBound {
    fn default() -> Self {
        Self {
            max_steps: 8,
            max_wall_ms: 30 * 60 * 1000,
            max_compute_units: 0,
        }
    }
}

/// Phrases that mark an *unbounded* optimization plan. Presence of any of
/// these (case-insensitive) in the plan body without a filled
/// [`KernelResearchContract`] is a hard reject.
pub const UNBOUNDED_PLAN_MARKERS: &[&str] = &[
    "try optimizations",
    "try some optimizations",
    "try various optimizations",
    "optimize until better",
    "keep optimizing",
    "tune until",
    "improve performance somehow",
    "randomly try",
    "just try things",
    "see what works",
];

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ContractError {
    #[error("missing or empty field: {0}")]
    EmptyField(&'static str),
    #[error("required_tools must be non-empty for kernel research plans")]
    NoTools,
    #[error("experiment bound is unbounded (max_steps=0 or max_wall_ms=0)")]
    UnboundedExperiment,
    #[error("plan body contains unbounded optimization language: {0}")]
    UnboundedLanguage(String),
    #[error("promotion_result and retirement_result must differ")]
    PromotionEqualsRetirement,
}

impl KernelResearchContract {
    /// Validate the six bible questions + experiment bound.
    pub fn validate(&self) -> Result<(), ContractError> {
        for (name, val) in [
            ("measured_bottleneck", self.measured_bottleneck.as_str()),
            (
                "distinguishing_evidence",
                self.distinguishing_evidence.as_str(),
            ),
            (
                "cheapest_disprove_experiment",
                self.cheapest_disprove_experiment.as_str(),
            ),
            ("promotion_result", self.promotion_result.as_str()),
            ("retirement_result", self.retirement_result.as_str()),
        ] {
            if val.trim().is_empty() {
                return Err(ContractError::EmptyField(name));
            }
        }
        if self.required_tools.is_empty()
            || self.required_tools.iter().all(|t| t.trim().is_empty())
        {
            return Err(ContractError::NoTools);
        }
        if self.experiment_bound.max_steps == 0 || self.experiment_bound.max_wall_ms == 0 {
            return Err(ContractError::UnboundedExperiment);
        }
        if self.promotion_result.trim() == self.retirement_result.trim() {
            return Err(ContractError::PromotionEqualsRetirement);
        }
        // Disprove experiment itself must not be unbounded language.
        reject_unbounded_language(&self.cheapest_disprove_experiment)?;
        Ok(())
    }
}

/// Reject plan bodies that use unbounded optimization language.
pub fn reject_unbounded_language(text: &str) -> Result<(), ContractError> {
    let lower = text.to_ascii_lowercase();
    for marker in UNBOUNDED_PLAN_MARKERS {
        if lower.contains(marker) {
            return Err(ContractError::UnboundedLanguage((*marker).to_string()));
        }
    }
    Ok(())
}

/// A proposed plan step that is admissible for research only when bounded
/// and (for kernel research) accompanied by a valid contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticPlanProposal {
    pub title: String,
    pub objective: String,
    pub steps_text: Vec<String>,
    /// Required when [`PlanningMode::KernelResearch`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub research_contract: Option<KernelResearchContract>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanningMode {
    /// General agent planning (still forbids unbounded markers).
    General,
    /// Kernel/lab research — full contract required before PlanChallenge can pass.
    KernelResearch,
}

impl DiagnosticPlanProposal {
    /// Validate proposal shape for the given mode.
    pub fn validate(&self, mode: PlanningMode) -> Result<(), ContractError> {
        if self.title.trim().is_empty() {
            return Err(ContractError::EmptyField("title"));
        }
        if self.objective.trim().is_empty() {
            return Err(ContractError::EmptyField("objective"));
        }
        if self.steps_text.is_empty() {
            return Err(ContractError::EmptyField("steps_text"));
        }
        // Concatenate for language scan.
        let mut body = String::new();
        body.push_str(&self.title);
        body.push('\n');
        body.push_str(&self.objective);
        for s in &self.steps_text {
            body.push('\n');
            body.push_str(s);
        }
        reject_unbounded_language(&body)?;

        match mode {
            PlanningMode::General => Ok(()),
            PlanningMode::KernelResearch => {
                let contract = self
                    .research_contract
                    .as_ref()
                    .ok_or(ContractError::EmptyField("research_contract"))?;
                contract.validate()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pipeline run
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticRunStatus {
    Active,
    Completed,
    Failed,
    Blocked,
}

/// One planning-diagnostic run: ordered stage receipts + current stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanningDiagnosticRun {
    pub id: String,
    pub session_id: SessionId,
    pub mode: PlanningMode,
    pub goal: String,
    pub status: DiagnosticRunStatus,
    pub current_stage: DiagnosticStage,
    pub receipts: Vec<StageReceipt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proposal: Option<DiagnosticPlanProposal>,
    pub replan_count: u32,
    pub max_replans: u32,
    pub created_ms: u64,
    pub schema: String,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DiagnosticError {
    #[error("stage {0:?} is not the current stage")]
    WrongStage(DiagnosticStage),
    #[error("no open receipt for stage {0:?}")]
    NoOpenReceipt(DiagnosticStage),
    #[error("illegal stage transition {from:?} → {to:?}")]
    IllegalTransition {
        from: DiagnosticStage,
        to: DiagnosticStage,
    },
    #[error("run is not active")]
    NotActive,
    #[error("plan challenge blocked: {0}")]
    ChallengeBlocked(String),
    #[error(transparent)]
    Contract(#[from] ContractError),
    #[error("replan budget exhausted")]
    ReplanBudgetExhausted,
    #[error("observation receipt required before replan or verification")]
    MissingObservation,
    #[error("plan challenge must pass before execution in kernel research mode")]
    ChallengeRequired,
}

impl PlanningDiagnosticRun {
    pub fn start(
        session_id: SessionId,
        goal: impl Into<String>,
        mode: PlanningMode,
    ) -> Self {
        let now = now_ms();
        Self {
            id: format!("pdiag_{}", ulid::Ulid::new()),
            session_id,
            mode,
            goal: goal.into(),
            status: DiagnosticRunStatus::Active,
            current_stage: DiagnosticStage::GoalInterpretation,
            receipts: Vec::new(),
            proposal: None,
            replan_count: 0,
            max_replans: 3,
            created_ms: now,
            schema: PLANNING_DIAGNOSTIC_SCHEMA.to_string(),
        }
    }

    /// Open the current stage (Running receipt).
    pub fn begin_stage(&mut self, at_ms: u64) -> Result<&StageReceipt, DiagnosticError> {
        self.require_active()?;
        // Refuse double-open.
        if self
            .receipts
            .last()
            .map(|r| r.stage == self.current_stage && r.status == StageStatus::Running)
            .unwrap_or(false)
        {
            return Ok(self.receipts.last().unwrap());
        }
        self.receipts
            .push(StageReceipt::open(self.current_stage, at_ms));
        Ok(self.receipts.last().unwrap())
    }

    /// Complete the current stage and advance if the transition is legal.
    pub fn complete_stage(
        &mut self,
        status: StageStatus,
        summary: impl Into<String>,
        next: Option<DiagnosticStage>,
        at_ms: u64,
    ) -> Result<(), DiagnosticError> {
        self.require_active()?;
        let stage = self.current_stage;
        let idx = self
            .receipts
            .iter()
            .rposition(|r| r.stage == stage && r.status == StageStatus::Running)
            .ok_or(DiagnosticError::NoOpenReceipt(stage))?;

        let summary = summary.into();
        if status == StageStatus::Failed {
            self.receipts[idx] = self.receipts[idx].clone().fail(summary.clone(), at_ms);
            self.status = DiagnosticRunStatus::Failed;
            return Ok(());
        }

        self.receipts[idx] = self.receipts[idx]
            .clone()
            .finish(status, summary, at_ms);

        if let Some(n) = next {
            self.transition_to(n)?;
        } else if stage.is_terminal_stage() && status == StageStatus::Passed {
            self.status = DiagnosticRunStatus::Completed;
        }
        Ok(())
    }

    /// Attach a plan proposal at the Plan stage; validates contract immediately.
    pub fn attach_proposal(
        &mut self,
        proposal: DiagnosticPlanProposal,
    ) -> Result<(), DiagnosticError> {
        self.require_active()?;
        if self.current_stage != DiagnosticStage::Plan
            && self.current_stage != DiagnosticStage::Replan
        {
            return Err(DiagnosticError::WrongStage(self.current_stage));
        }
        proposal.validate(self.mode)?;
        self.proposal = Some(proposal);
        Ok(())
    }

    /// Plan-challenge gate: kernel research requires a valid contract and a
    /// prior passed Plan receipt. Challenge may send the run back to Plan.
    pub fn resolve_challenge(
        &mut self,
        accepted: bool,
        summary: impl Into<String>,
        at_ms: u64,
    ) -> Result<(), DiagnosticError> {
        self.require_active()?;
        if self.current_stage != DiagnosticStage::PlanChallenge {
            return Err(DiagnosticError::WrongStage(self.current_stage));
        }
        // Ensure we have an open receipt.
        self.begin_stage(at_ms)?;
        if self.mode == PlanningMode::KernelResearch {
            let prop = self
                .proposal
                .as_ref()
                .ok_or_else(|| {
                    DiagnosticError::ChallengeBlocked(
                        "no proposal attached for kernel research".into(),
                    )
                })?;
            prop.validate(PlanningMode::KernelResearch)?;
        }
        if accepted {
            self.complete_stage(
                StageStatus::Passed,
                summary,
                Some(DiagnosticStage::Execution),
                at_ms,
            )?;
        } else {
            // Challenge rejection is not a run failure: mark the stage Blocked
            // and send the pipeline back to Plan for revision.
            let idx = self
                .receipts
                .iter()
                .rposition(|r| {
                    r.stage == DiagnosticStage::PlanChallenge
                        && r.status == StageStatus::Running
                })
                .ok_or(DiagnosticError::NoOpenReceipt(
                    DiagnosticStage::PlanChallenge,
                ))?;
            let mut receipt = self.receipts[idx].clone().finish(
                StageStatus::Blocked,
                summary,
                at_ms,
            );
            receipt.error = Some("plan challenge rejected".into());
            self.receipts[idx] = receipt;
            self.transition_to(DiagnosticStage::Plan)?;
            self.status = DiagnosticRunStatus::Active;
        }
        Ok(())
    }

    /// Enter execution only after a passed PlanChallenge (kernel research).
    pub fn enter_execution(&mut self, at_ms: u64) -> Result<(), DiagnosticError> {
        self.require_active()?;
        if self.current_stage != DiagnosticStage::Execution {
            // Auto-path from challenge if already advanced.
            if self.current_stage == DiagnosticStage::PlanChallenge {
                if self.mode == PlanningMode::KernelResearch
                    && !self.stage_passed(DiagnosticStage::PlanChallenge)
                {
                    return Err(DiagnosticError::ChallengeRequired);
                }
            } else {
                return Err(DiagnosticError::WrongStage(self.current_stage));
            }
        }
        if self.mode == PlanningMode::KernelResearch
            && !self.stage_passed(DiagnosticStage::PlanChallenge)
        {
            return Err(DiagnosticError::ChallengeRequired);
        }
        self.begin_stage(at_ms)?;
        Ok(())
    }

    /// Record observation then branch to Replan or Verification.
    pub fn complete_observation(
        &mut self,
        summary: impl Into<String>,
        replan: bool,
        at_ms: u64,
    ) -> Result<(), DiagnosticError> {
        self.require_active()?;
        if self.current_stage != DiagnosticStage::Observation {
            // Allow completing observation only when we are on that stage.
            // Callers should have advanced Execution → Observation already.
            if self.current_stage == DiagnosticStage::Execution {
                // close execution as passed if still open, then move.
                if self
                    .receipts
                    .last()
                    .map(|r| r.stage == DiagnosticStage::Execution && r.status == StageStatus::Running)
                    .unwrap_or(false)
                {
                    self.complete_stage(
                        StageStatus::Passed,
                        "execution finished",
                        Some(DiagnosticStage::Observation),
                        at_ms,
                    )?;
                } else {
                    self.transition_to(DiagnosticStage::Observation)?;
                }
            } else {
                return Err(DiagnosticError::WrongStage(self.current_stage));
            }
        }
        self.begin_stage(at_ms)?;
        let next = if replan {
            DiagnosticStage::Replan
        } else {
            DiagnosticStage::Verification
        };
        self.complete_stage(StageStatus::Passed, summary, Some(next), at_ms)
    }

    /// Replan: requires a prior Observation receipt; increments replan budget.
    pub fn replan(
        &mut self,
        new_proposal: DiagnosticPlanProposal,
        summary: impl Into<String>,
        at_ms: u64,
    ) -> Result<(), DiagnosticError> {
        self.require_active()?;
        if !self.stage_passed(DiagnosticStage::Observation)
            && !self
                .receipts
                .iter()
                .any(|r| r.stage == DiagnosticStage::Observation)
        {
            return Err(DiagnosticError::MissingObservation);
        }
        if self.replan_count >= self.max_replans {
            return Err(DiagnosticError::ReplanBudgetExhausted);
        }
        if self.current_stage != DiagnosticStage::Replan {
            if self
                .current_stage
                .successors()
                .contains(&DiagnosticStage::Replan)
                || self.current_stage == DiagnosticStage::Observation
            {
                self.transition_to(DiagnosticStage::Replan)?;
            } else {
                return Err(DiagnosticError::WrongStage(self.current_stage));
            }
        }
        self.begin_stage(at_ms)?;
        new_proposal.validate(self.mode)?;
        self.proposal = Some(new_proposal);
        self.replan_count += 1;
        self.complete_stage(
            StageStatus::Passed,
            summary,
            Some(DiagnosticStage::PlanChallenge),
            at_ms,
        )
    }

    /// Finish verification → report (or replan on failure).
    pub fn complete_verification(
        &mut self,
        passed: bool,
        summary: impl Into<String>,
        at_ms: u64,
    ) -> Result<(), DiagnosticError> {
        self.require_active()?;
        if self.current_stage != DiagnosticStage::Verification {
            return Err(DiagnosticError::WrongStage(self.current_stage));
        }
        if !self
            .receipts
            .iter()
            .any(|r| r.stage == DiagnosticStage::Observation)
        {
            return Err(DiagnosticError::MissingObservation);
        }
        self.begin_stage(at_ms)?;
        if passed {
            self.complete_stage(
                StageStatus::Passed,
                summary,
                Some(DiagnosticStage::Report),
                at_ms,
            )
        } else if self.replan_count < self.max_replans {
            self.complete_stage(
                StageStatus::Failed,
                summary,
                Some(DiagnosticStage::Replan),
                at_ms,
            )?;
            self.status = DiagnosticRunStatus::Active;
            Ok(())
        } else {
            self.complete_stage(StageStatus::Failed, summary, None, at_ms)?;
            self.status = DiagnosticRunStatus::Failed;
            Ok(())
        }
    }

    /// Seal the report stage and mark the run completed.
    pub fn emit_report(
        &mut self,
        summary: impl Into<String>,
        at_ms: u64,
    ) -> Result<&StageReceipt, DiagnosticError> {
        self.require_active()?;
        if self.current_stage != DiagnosticStage::Report {
            return Err(DiagnosticError::WrongStage(self.current_stage));
        }
        self.begin_stage(at_ms)?;
        self.complete_stage(StageStatus::Passed, summary, None, at_ms)?;
        self.status = DiagnosticRunStatus::Completed;
        Ok(self.receipts.last().unwrap())
    }

    pub fn stage_passed(&self, stage: DiagnosticStage) -> bool {
        self.receipts
            .iter()
            .any(|r| r.stage == stage && r.status == StageStatus::Passed)
    }

    pub fn receipt_for(&self, stage: DiagnosticStage) -> Option<&StageReceipt> {
        self.receipts.iter().rev().find(|r| r.stage == stage)
    }

    fn require_active(&self) -> Result<(), DiagnosticError> {
        if self.status != DiagnosticRunStatus::Active {
            Err(DiagnosticError::NotActive)
        } else {
            Ok(())
        }
    }

    pub fn transition_to(&mut self, next: DiagnosticStage) -> Result<(), DiagnosticError> {
        let from = self.current_stage;
        if from == next {
            return Ok(());
        }
        // Allow replan loops and challenge→plan without listing every edge twice.
        let legal = from.successors().contains(&next)
            || (from == DiagnosticStage::PlanChallenge && next == DiagnosticStage::Plan)
            || (from == DiagnosticStage::Plan && next == DiagnosticStage::PlanChallenge)
            || (matches!(
                from,
                DiagnosticStage::Observation | DiagnosticStage::Verification
            ) && next == DiagnosticStage::Replan)
            || (from == DiagnosticStage::Replan
                && matches!(
                    next,
                    DiagnosticStage::Plan | DiagnosticStage::PlanChallenge
                ));
        if !legal {
            return Err(DiagnosticError::IllegalTransition { from, to: next });
        }
        self.current_stage = next;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Happy-path helper (scaffold demo / tests)
// ---------------------------------------------------------------------------

/// Drive a kernel-research diagnostic through the full pipeline with a valid
/// contract. Model-free fixture path for tests and host demos.
pub fn fixture_kernel_research_happy_path(
    session_id: SessionId,
    goal: &str,
) -> Result<PlanningDiagnosticRun, DiagnosticError> {
    let mut run = PlanningDiagnosticRun::start(session_id, goal, PlanningMode::KernelResearch);
    let t0 = 1_000u64;

    run.begin_stage(t0)?;
    run.complete_stage(
        StageStatus::Passed,
        "goal: measure prefill bandwidth bottleneck",
        Some(DiagnosticStage::ToolRetrieval),
        t0 + 1,
    )?;

    run.begin_stage(t0 + 2)?;
    run.complete_stage(
        StageStatus::Passed,
        "tools: metal_capture, roofline_probe",
        Some(DiagnosticStage::Plan),
        t0 + 3,
    )?;

    run.begin_stage(t0 + 4)?;
    let proposal = DiagnosticPlanProposal {
        title: "Disprove host-sync bottleneck".into(),
        objective: goal.into(),
        steps_text: vec![
            "Capture one decode step GPU timeline".into(),
            "Compare device vs host bounded region".into(),
        ],
        research_contract: Some(KernelResearchContract {
            measured_bottleneck: "decode step stalls 2.1ms on host sync".into(),
            distinguishing_evidence:
                "gpu timeline shows encoder wait; counters show zero DRAM stall".into(),
            cheapest_disprove_experiment:
                "run 3 decode steps with host sync removed under fixed token budget".into(),
            required_tools: vec!["metal_capture".into(), "roofline_probe".into()],
            promotion_result: "host-sync removal improves p50 decode ≥15% on fixed corpus".into(),
            retirement_result: "no improvement ≥3% after 3 bounded runs".into(),
            experiment_bound: ExperimentBound {
                max_steps: 3,
                max_wall_ms: 120_000,
                max_compute_units: 1,
            },
        }),
    };
    run.attach_proposal(proposal)?;
    run.complete_stage(
        StageStatus::Passed,
        "bounded disprove plan attached",
        Some(DiagnosticStage::PlanChallenge),
        t0 + 5,
    )?;

    run.resolve_challenge(true, "challenge accepted: bound + evidence present", t0 + 6)?;
    run.enter_execution(t0 + 7)?;
    run.complete_stage(
        StageStatus::Passed,
        "three decode runs sealed",
        Some(DiagnosticStage::Observation),
        t0 + 8,
    )?;
    run.complete_observation("p50 +18% decode; promote", false, t0 + 9)?;
    run.complete_verification(true, "promotion threshold met", t0 + 10)?;
    run.emit_report("bottleneck: host sync; promoted removal patch", t0 + 11)?;
    Ok(run)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::ids::SessionId;

    fn valid_contract() -> KernelResearchContract {
        KernelResearchContract {
            measured_bottleneck: "L2 miss on expert weight load".into(),
            distinguishing_evidence: "counter A high, counter B low".into(),
            cheapest_disprove_experiment: "block-load 4 experts once, measure miss rate".into(),
            required_tools: vec!["perf_counters".into()],
            promotion_result: "miss rate drops ≥20%".into(),
            retirement_result: "miss rate change <5%".into(),
            experiment_bound: ExperimentBound::default(),
        }
    }

    #[test]
    fn rejects_unbounded_try_optimizations_language() {
        let err = reject_unbounded_language("we should try optimizations on the kernel").unwrap_err();
        assert!(matches!(err, ContractError::UnboundedLanguage(_)));
    }

    #[test]
    fn rejects_empty_research_contract_fields() {
        let mut c = valid_contract();
        c.measured_bottleneck.clear();
        assert!(matches!(
            c.validate(),
            Err(ContractError::EmptyField("measured_bottleneck"))
        ));
    }

    #[test]
    fn rejects_unbounded_experiment_steps() {
        let mut c = valid_contract();
        c.experiment_bound.max_steps = 0;
        assert!(matches!(
            c.validate(),
            Err(ContractError::UnboundedExperiment)
        ));
    }

    #[test]
    fn rejects_identical_promotion_and_retirement() {
        let mut c = valid_contract();
        c.retirement_result = c.promotion_result.clone();
        assert!(matches!(
            c.validate(),
            Err(ContractError::PromotionEqualsRetirement)
        ));
    }

    #[test]
    fn kernel_research_proposal_requires_contract() {
        let p = DiagnosticPlanProposal {
            title: "x".into(),
            objective: "y".into(),
            steps_text: vec!["step".into()],
            research_contract: None,
        };
        assert!(matches!(
            p.validate(PlanningMode::KernelResearch),
            Err(ContractError::EmptyField("research_contract"))
        ));
        // General mode allows missing contract but still scans language.
        assert!(p.validate(PlanningMode::General).is_ok());
    }

    #[test]
    fn proposal_rejects_unbounded_step_text() {
        let p = DiagnosticPlanProposal {
            title: "speed up".into(),
            objective: "faster".into(),
            steps_text: vec!["try optimizations until it looks good".into()],
            research_contract: Some(valid_contract()),
        };
        assert!(matches!(
            p.validate(PlanningMode::KernelResearch),
            Err(ContractError::UnboundedLanguage(_))
        ));
    }

    #[test]
    fn stage_pipeline_happy_path_emits_all_receipts() {
        let run = fixture_kernel_research_happy_path(
            SessionId::from("ses_diag"),
            "find decode bottleneck",
        )
        .unwrap();
        assert_eq!(run.status, DiagnosticRunStatus::Completed);
        assert_eq!(run.schema, PLANNING_DIAGNOSTIC_SCHEMA);
        // Must have passed through the required stages.
        for stage in [
            DiagnosticStage::GoalInterpretation,
            DiagnosticStage::ToolRetrieval,
            DiagnosticStage::Plan,
            DiagnosticStage::PlanChallenge,
            DiagnosticStage::Execution,
            DiagnosticStage::Observation,
            DiagnosticStage::Verification,
            DiagnosticStage::Report,
        ] {
            assert!(
                run.stage_passed(stage),
                "expected passed receipt for {stage:?}"
            );
        }
        assert!(run.proposal.as_ref().unwrap().research_contract.is_some());
    }

    #[test]
    fn challenge_rejection_returns_to_plan() {
        let mut run = PlanningDiagnosticRun::start(
            SessionId::from("ses_c"),
            "goal",
            PlanningMode::KernelResearch,
        );
        // Fast-forward to PlanChallenge with a valid proposal.
        run.current_stage = DiagnosticStage::Plan;
        run.attach_proposal(DiagnosticPlanProposal {
            title: "t".into(),
            objective: "o".into(),
            steps_text: vec!["bounded step".into()],
            research_contract: Some(valid_contract()),
        })
        .unwrap();
        run.current_stage = DiagnosticStage::PlanChallenge;
        run.resolve_challenge(false, "missing distinguishing evidence detail", 50)
            .unwrap();
        assert_eq!(run.current_stage, DiagnosticStage::Plan);
        assert_eq!(run.status, DiagnosticRunStatus::Active);
        let ch = run.receipt_for(DiagnosticStage::PlanChallenge).unwrap();
        assert_eq!(ch.status, StageStatus::Blocked);
    }

    #[test]
    fn execution_blocked_without_passed_challenge_in_research_mode() {
        let mut run = PlanningDiagnosticRun::start(
            SessionId::from("ses_e"),
            "goal",
            PlanningMode::KernelResearch,
        );
        run.current_stage = DiagnosticStage::Execution;
        // No PlanChallenge receipt at all.
        let err = run.enter_execution(10).unwrap_err();
        assert!(matches!(err, DiagnosticError::ChallengeRequired));
    }

    #[test]
    fn replan_requires_observation_and_respects_budget() {
        let mut run = PlanningDiagnosticRun::start(
            SessionId::from("ses_r"),
            "goal",
            PlanningMode::General,
        );
        run.max_replans = 1;
        let prop = DiagnosticPlanProposal {
            title: "t".into(),
            objective: "o".into(),
            steps_text: vec!["s".into()],
            research_contract: None,
        };
        assert!(matches!(
            run.replan(prop.clone(), "no obs yet", 1),
            Err(DiagnosticError::MissingObservation)
        ));

        // Seed an observation receipt and allow replan.
        run.receipts.push(
            StageReceipt::open(DiagnosticStage::Observation, 2)
                .finish(StageStatus::Passed, "saw failure", 3),
        );
        run.current_stage = DiagnosticStage::Replan;
        run.replan(prop.clone(), "adjust steps", 4).unwrap();
        assert_eq!(run.replan_count, 1);
        assert_eq!(run.current_stage, DiagnosticStage::PlanChallenge);

        // Budget exhausted.
        run.receipts.push(
            StageReceipt::open(DiagnosticStage::Observation, 5)
                .finish(StageStatus::Passed, "still failing", 6),
        );
        run.current_stage = DiagnosticStage::Replan;
        assert!(matches!(
            run.replan(prop, "again", 7),
            Err(DiagnosticError::ReplanBudgetExhausted)
        ));
    }

    #[test]
    fn illegal_stage_skip_rejected() {
        let mut run = PlanningDiagnosticRun::start(
            SessionId::from("ses_i"),
            "goal",
            PlanningMode::General,
        );
        // Cannot jump GoalInterpretation → Execution.
        assert!(matches!(
            run.transition_to(DiagnosticStage::Execution),
            Err(DiagnosticError::IllegalTransition { .. })
        ));
    }

    #[test]
    fn all_stages_have_wire_names() {
        for s in DiagnosticStage::ALL {
            assert!(!s.wire_name().is_empty());
        }
    }
}
