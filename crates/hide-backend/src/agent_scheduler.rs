//! HCLI Agent OS — Agent Scheduler scaffold (Ascension Bible §12–§13).
//!
//! **What this is:** pure data model + state machine for *many logical agents
//! sharing one loaded model weight copy*. No inference, no weight load, no
//! continuous-batch decode loop, no kernel step driver.
//!
//! **What already exists (reused, not reinvented):**
//! * [`hide_fleet::PriorityClass`] / [`hide_fleet::ConcurrencyClass`] —
//!   machine-wide job priority and Model-vs-CpuOnly pools.
//! * [`hide_fleet::FleetGovernor`] / [`hide_fleet::FleetScheduler`] —
//!   RAM/thermal/spawn-rate admission for *jobs* on the box.
//! * [`hide_core::ids::SessionId`] / [`RunId`] — durable session/run identity.
//! * [`hide_kernel::checkpoint::AgentCheckpoint`] — per-run fold snapshot
//!   (referenced by id here; restore stays in the kernel).
//! * `hawking_serve::batch::Scheduler` — token-level continuous batching
//!   (this module only records *batch cohort affinity*; it does not own slots).
//!
//! **Boundary honesty:** transitions here are scheduling policy only. Real
//! agent-loop integration (tool dispatch, kernel `step`, serve decode, weight
//! residency control) is deferred until the Agent OS activation gate in bible §0.
//!
//! Gated by: Proto-Frankenstein sealed/offloaded and Qwen bootstrap path ready.
//! This crate path is model-free and safe to compile/test without that gate.

use hide_core::ids::{now_ms, RunId, SessionId};
use hide_fleet::{ConcurrencyClass, PriorityClass};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

/// Schema id for durable scheduler receipts / projections.
pub const AGENT_SCHEDULER_SCHEMA: &str = "hcli.agent_scheduler.v1";

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

/// One *logical* agent. Many of these share a single [`ModelResidency`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct LogicalAgentId(pub String);

impl LogicalAgentId {
    pub fn new() -> Self {
        Self(format!("lag_{}", ulid::Ulid::new()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for LogicalAgentId {
    fn default() -> Self {
        Self::new()
    }
}

impl From<&str> for LogicalAgentId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Stable id for a loaded model process / weight residency (not a weight copy
/// per agent — one residency, many logical agents).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ModelResidencyId(pub String);

impl ModelResidencyId {
    pub fn new(label: impl Into<String>) -> Self {
        Self(label.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Handle to a kernel checkpoint (seq boundary). The bytes live in the kernel
/// / host checkpoint store; the scheduler only tracks the reference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointRef {
    pub session_id: SessionId,
    pub run_id: RunId,
    /// Event-log seq at which the snapshot was taken.
    pub seq: u64,
}

// ---------------------------------------------------------------------------
// Residency (bible §25 modes, agent-facing)
// ---------------------------------------------------------------------------

/// How models sit in unified memory relative to agent work.
///
/// Distinct from MoE *weight-cache* residency in `hawking-core`. This is the
/// Agent OS notion: which *roles* are co-resident so agents can pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidencyMode {
    /// Executor + reviewer both loaded; pipeline candidate N+1 while reviewing N.
    DualResident,
    /// Only the executor stays loaded; reviews queue until target unloads.
    ExecutorResident,
    /// Fully phase-separated: load → work → checkpoint/unload → next role.
    PhaseSeparated,
}

impl Default for ResidencyMode {
    fn default() -> Self {
        // Conservative default until dual-resident footprint is measured.
        ResidencyMode::PhaseSeparated
    }
}

/// One loaded model supporting many logical agents.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelResidency {
    pub id: ModelResidencyId,
    /// Human / catalog role label (e.g. `qwen3-coder-30b-executor`).
    pub role_label: String,
    pub mode: ResidencyMode,
    /// Max concurrent *generating* agents on this residency (maps toward
    /// serve `max_batch_size`; not the fleet job ceiling).
    pub max_batch_slots: u32,
    /// Agents currently attached (any non-terminal state).
    pub attached: Vec<LogicalAgentId>,
}

impl ModelResidency {
    pub fn new(
        id: impl Into<String>,
        role_label: impl Into<String>,
        max_batch_slots: u32,
    ) -> Self {
        Self {
            id: ModelResidencyId::new(id),
            role_label: role_label.into(),
            mode: ResidencyMode::default(),
            max_batch_slots: max_batch_slots.max(1),
            attached: Vec::new(),
        }
    }

    pub fn generating_capacity_left(&self, agents: &BTreeMap<LogicalAgentId, LogicalAgent>) -> u32 {
        let live = self
            .attached
            .iter()
            .filter(|id| {
                agents
                    .get(id)
                    .map(|a| a.state == AgentSchedState::Generating)
                    .unwrap_or(false)
            })
            .count() as u32;
        self.max_batch_slots.saturating_sub(live)
    }
}

// ---------------------------------------------------------------------------
// Agent lifecycle state machine
// ---------------------------------------------------------------------------

/// Lifecycle of a logical agent on the shared scheduler.
///
/// ```text
/// Registered → Queued → Admitted → Generating ⇄ ToolWaiting
///                              ↘ Checkpointed → Queued (resume)
/// Generating / ToolWaiting → Completed | Failed | Cancelled
/// Queued / Admitted / Generating → Preempted → Checkpointed
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentSchedState {
    /// Known to the scheduler, not yet enqueued for work.
    Registered,
    /// Waiting for a generation slot on a residency.
    Queued,
    /// Slot reserved; not yet generating (prefill / admit handshake).
    Admitted,
    /// Actively consuming a model forward pass / decode slot.
    Generating,
    /// Suspended while a tool runs — must **not** hold a batch slot.
    ToolWaiting,
    /// Yielded a checkpoint (preempt, pressure, phase unload).
    Checkpointed,
    /// Lost its slot to a higher-priority agent; must checkpoint then re-queue.
    Preempted,
    Completed,
    Failed,
    Cancelled,
}

impl AgentSchedState {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            AgentSchedState::Completed | AgentSchedState::Failed | AgentSchedState::Cancelled
        )
    }

    /// Counts against the residency's model batch ceiling.
    pub fn holds_batch_slot(self) -> bool {
        matches!(
            self,
            AgentSchedState::Admitted | AgentSchedState::Generating
        )
    }

    /// Eligible to be selected for a free batch slot.
    pub fn is_runnable(self) -> bool {
        matches!(
            self,
            AgentSchedState::Queued | AgentSchedState::Checkpointed
        )
    }
}

/// Why an agent moved between states (audit / metrics).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchedEvent {
    Register,
    Enqueue,
    Admit,
    BeginGenerate,
    SuspendToolWait { tool_name: String },
    ResumeFromTool,
    Checkpoint { reason: String },
    ResumeFromCheckpoint,
    Preempt { victim_of: String },
    Complete,
    Fail { reason: String },
    Cancel,
    /// Fairness quantum expired; agent re-queued to prevent monopoly.
    FairnessYield,
    /// Starvation boost applied (priority temporarily elevated).
    StarvationBoost { from: String, to: String },
}

// ---------------------------------------------------------------------------
// Logical agent record
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogicalAgent {
    pub id: LogicalAgentId,
    pub session_id: SessionId,
    pub run_id: Option<RunId>,
    pub objective: String,
    pub state: AgentSchedState,
    pub priority: PriorityClass,
    /// Effective priority after starvation boost (may differ from `priority`).
    pub effective_priority: PriorityClass,
    pub concurrency_class: ConcurrencyClass,
    pub residency_id: Option<ModelResidencyId>,
    /// Optional affinity to a continuous-batch cohort (serve-side slot group).
    pub batch_cohort: Option<String>,
    pub checkpoint: Option<CheckpointRef>,
    /// Wall-clock ms when first queued (for queue-time / starvation metrics).
    pub queued_at_ms: Option<u64>,
    /// Wall-clock ms of last state transition.
    pub last_transition_ms: u64,
    /// Cumulative ms spent in `Queued` (accounting only; pure update on ticks).
    pub queue_wait_ms: u64,
    /// Cumulative ms spent in `ToolWaiting`.
    pub tool_wait_ms: u64,
    /// How many times this agent has been admitted (for fairness accounting).
    pub admit_count: u32,
    /// How many tokens of generation quantum remain before fairness yield.
    pub fairness_quantum_remaining: u32,
    pub history: Vec<SchedEvent>,
}

impl LogicalAgent {
    pub fn new(
        session_id: SessionId,
        objective: impl Into<String>,
        priority: PriorityClass,
    ) -> Self {
        let now = now_ms();
        Self {
            id: LogicalAgentId::new(),
            session_id,
            run_id: None,
            objective: objective.into(),
            state: AgentSchedState::Registered,
            priority,
            effective_priority: priority,
            concurrency_class: ConcurrencyClass::Model,
            residency_id: None,
            batch_cohort: None,
            checkpoint: None,
            queued_at_ms: None,
            last_transition_ms: now,
            queue_wait_ms: 0,
            tool_wait_ms: 0,
            admit_count: 0,
            fairness_quantum_remaining: 0,
            history: vec![SchedEvent::Register],
        }
    }

    fn push(&mut self, event: SchedEvent, at_ms: u64) {
        self.history.push(event);
        self.last_transition_ms = at_ms;
    }
}

// ---------------------------------------------------------------------------
// Scheduler policy + core
// ---------------------------------------------------------------------------

/// Fairness / starvation knobs (bible §13 measures: queue time, tool wait,
/// starvation prevention). Pure policy — host supplies the clock via `at_ms`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchedulerPolicy {
    /// After this many ms waiting in queue, boost effective priority one class.
    pub starvation_threshold_ms: u64,
    /// Generation steps (or token quanta) before a generating agent must yield
    /// if others are waiting at equal-or-higher effective priority.
    pub fairness_quantum: u32,
    /// Max agents admitted per `schedule_tick` (batch cohort size upper bound).
    pub max_admit_per_tick: u32,
}

impl Default for SchedulerPolicy {
    fn default() -> Self {
        Self {
            starvation_threshold_ms: 5_000,
            fairness_quantum: 64,
            max_admit_per_tick: 8,
        }
    }
}

/// Metrics the bible lists under §13 (scaffolded counters only).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SchedulerMetrics {
    pub verified_tasks_completed: u64,
    pub total_queue_wait_ms: u64,
    pub total_tool_wait_ms: u64,
    pub admits: u64,
    pub preemptions: u64,
    pub starvation_boosts: u64,
    pub fairness_yields: u64,
    pub checkpoints: u64,
}

/// Decision output of one pure schedule tick (mirrors fleet `TickPlan` shape).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AgentTickPlan {
    pub admit: Vec<LogicalAgentId>,
    pub preempt: Vec<LogicalAgentId>,
    pub fairness_yield: Vec<LogicalAgentId>,
    pub starvation_boost: Vec<LogicalAgentId>,
    /// Free batch slots on the target residency after this plan.
    pub free_batch_slots: u32,
}

/// In-memory Agent OS scheduler: many logical agents, one (or few) residencies.
///
/// **Not** the machine-wide [`hide_fleet::FleetScheduler`] (jobs/resources) and
/// **not** the serve continuous-batch slot table. This layer answers: *which
/// logical agents may hold a generation slot on a shared loaded model right now?*
#[derive(Debug, Clone, Default)]
pub struct AgentScheduler {
    pub policy: SchedulerPolicy,
    pub residencies: BTreeMap<ModelResidencyId, ModelResidency>,
    pub agents: BTreeMap<LogicalAgentId, LogicalAgent>,
    /// Ready queue ordered by (effective_priority, queued_at_ms, id).
    /// Stored as a deque of ids; re-sorted on each tick for determinism.
    ready: VecDeque<LogicalAgentId>,
    pub metrics: SchedulerMetrics,
}

impl AgentScheduler {
    pub fn new(policy: SchedulerPolicy) -> Self {
        Self {
            policy,
            residencies: BTreeMap::new(),
            agents: BTreeMap::new(),
            ready: VecDeque::new(),
            metrics: SchedulerMetrics::default(),
        }
    }

    pub fn register_residency(&mut self, residency: ModelResidency) {
        self.residencies.insert(residency.id.clone(), residency);
    }

    pub fn register_agent(&mut self, agent: LogicalAgent) -> LogicalAgentId {
        let id = agent.id.clone();
        self.agents.insert(id.clone(), agent);
        id
    }

    /// Attach agent to a residency and enqueue for generation.
    pub fn enqueue(
        &mut self,
        agent_id: &LogicalAgentId,
        residency_id: &ModelResidencyId,
        at_ms: u64,
    ) -> Result<(), SchedError> {
        let agent = self
            .agents
            .get_mut(agent_id)
            .ok_or(SchedError::UnknownAgent)?;
        if agent.state.is_terminal() {
            return Err(SchedError::Terminal);
        }
        if !matches!(
            agent.state,
            AgentSchedState::Registered
                | AgentSchedState::Checkpointed
                | AgentSchedState::Queued
        ) {
            return Err(SchedError::IllegalTransition {
                from: agent.state,
                to: AgentSchedState::Queued,
            });
        }
        if !self.residencies.contains_key(residency_id) {
            return Err(SchedError::UnknownResidency);
        }
        agent.residency_id = Some(residency_id.clone());
        agent.state = AgentSchedState::Queued;
        if agent.queued_at_ms.is_none() {
            agent.queued_at_ms = Some(at_ms);
        }
        agent.push(SchedEvent::Enqueue, at_ms);

        let res = self.residencies.get_mut(residency_id).unwrap();
        if !res.attached.contains(agent_id) {
            res.attached.push(agent_id.clone());
        }
        if !self.ready.contains(agent_id) {
            self.ready.push_back(agent_id.clone());
        }
        Ok(())
    }

    /// Pure tick: starvation boost → fairness yield → preempt → admit.
    /// Mutates agent records according to the plan.
    pub fn schedule_tick(
        &mut self,
        residency_id: &ModelResidencyId,
        at_ms: u64,
    ) -> Result<AgentTickPlan, SchedError> {
        if !self.residencies.contains_key(residency_id) {
            return Err(SchedError::UnknownResidency);
        }

        let mut plan = AgentTickPlan::default();

        // 1. Starvation boosts for long-waiting queued agents.
        let waiting: Vec<LogicalAgentId> = self
            .agents
            .iter()
            .filter(|(_, a)| {
                a.state == AgentSchedState::Queued
                    && a.residency_id.as_ref() == Some(residency_id)
            })
            .map(|(id, _)| id.clone())
            .collect();
        for id in waiting {
            if let Some(boosted) = self.maybe_starvation_boost(&id, at_ms) {
                plan.starvation_boost.push(boosted);
            }
        }

        // 2. Fairness yield: generating agents that exhausted quantum while
        //    others wait at equal-or-higher effective priority.
        let generators: Vec<LogicalAgentId> = self
            .agents
            .iter()
            .filter(|(_, a)| {
                a.state == AgentSchedState::Generating
                    && a.residency_id.as_ref() == Some(residency_id)
                    && a.fairness_quantum_remaining == 0
            })
            .map(|(id, _)| id.clone())
            .collect();
        let someone_waiting = self.agents.values().any(|a| {
            a.state == AgentSchedState::Queued && a.residency_id.as_ref() == Some(residency_id)
        });
        if someone_waiting {
            for id in generators {
                self.fairness_yield(&id, at_ms)?;
                plan.fairness_yield.push(id);
            }
        }

        // 3. Preempt lowest effective priority generator if Interactive waits
        //    and batch is full.
        let interactive_waiting = self.agents.values().any(|a| {
            a.state == AgentSchedState::Queued
                && a.residency_id.as_ref() == Some(residency_id)
                && a.effective_priority == PriorityClass::Interactive
        });
        let free = self.free_batch_slots(residency_id);
        if interactive_waiting && free == 0 {
            if let Some(victim) = self.lowest_preemptible(residency_id) {
                self.preempt(&victim, "interactive_waiting", at_ms)?;
                plan.preempt.push(victim);
            }
        }

        // 4. Admit ready agents in priority order up to free slots / tick cap.
        self.sort_ready(residency_id);
        let free = self.free_batch_slots(residency_id);
        plan.free_batch_slots = free;
        let admit_budget = free.min(self.policy.max_admit_per_tick);
        let candidates: Vec<LogicalAgentId> = self
            .ready
            .iter()
            .filter(|id| {
                self.agents.get(id).map(|a| {
                    a.state.is_runnable()
                        && a.residency_id.as_ref() == Some(residency_id)
                        && a.concurrency_class == ConcurrencyClass::Model
                }) == Some(true)
            })
            .take(admit_budget as usize)
            .cloned()
            .collect();
        for id in candidates {
            self.admit(&id, at_ms)?;
            plan.admit.push(id);
        }
        plan.free_batch_slots = self.free_batch_slots(residency_id);
        Ok(plan)
    }

    /// Admitted → Generating (host has begun a model forward for this agent).
    pub fn begin_generate(
        &mut self,
        agent_id: &LogicalAgentId,
        at_ms: u64,
    ) -> Result<(), SchedError> {
        let agent = self
            .agents
            .get_mut(agent_id)
            .ok_or(SchedError::UnknownAgent)?;
        if agent.state != AgentSchedState::Admitted {
            return Err(SchedError::IllegalTransition {
                from: agent.state,
                to: AgentSchedState::Generating,
            });
        }
        agent.state = AgentSchedState::Generating;
        agent.fairness_quantum_remaining = self.policy.fairness_quantum;
        agent.push(SchedEvent::BeginGenerate, at_ms);
        Ok(())
    }

    /// Consume one unit of the fairness quantum while generating.
    pub fn consume_quantum(&mut self, agent_id: &LogicalAgentId) -> Result<(), SchedError> {
        let agent = self
            .agents
            .get_mut(agent_id)
            .ok_or(SchedError::UnknownAgent)?;
        if agent.state != AgentSchedState::Generating {
            return Err(SchedError::IllegalTransition {
                from: agent.state,
                to: AgentSchedState::Generating,
            });
        }
        agent.fairness_quantum_remaining = agent.fairness_quantum_remaining.saturating_sub(1);
        Ok(())
    }

    /// Generating → ToolWaiting. **Releases** the batch slot (critical).
    pub fn suspend_for_tool(
        &mut self,
        agent_id: &LogicalAgentId,
        tool_name: impl Into<String>,
        at_ms: u64,
    ) -> Result<(), SchedError> {
        let agent = self
            .agents
            .get_mut(agent_id)
            .ok_or(SchedError::UnknownAgent)?;
        if agent.state != AgentSchedState::Generating {
            return Err(SchedError::IllegalTransition {
                from: agent.state,
                to: AgentSchedState::ToolWaiting,
            });
        }
        let name = tool_name.into();
        agent.state = AgentSchedState::ToolWaiting;
        agent.push(
            SchedEvent::SuspendToolWait {
                tool_name: name,
            },
            at_ms,
        );
        Ok(())
    }

    /// ToolWaiting → Queued (re-enter ready set for a free slot).
    pub fn resume_from_tool(
        &mut self,
        agent_id: &LogicalAgentId,
        at_ms: u64,
    ) -> Result<(), SchedError> {
        let agent = self
            .agents
            .get_mut(agent_id)
            .ok_or(SchedError::UnknownAgent)?;
        if agent.state != AgentSchedState::ToolWaiting {
            return Err(SchedError::IllegalTransition {
                from: agent.state,
                to: AgentSchedState::Queued,
            });
        }
        // Account tool wait against metrics.
        let waited = at_ms.saturating_sub(agent.last_transition_ms);
        agent.tool_wait_ms = agent.tool_wait_ms.saturating_add(waited);
        self.metrics.total_tool_wait_ms =
            self.metrics.total_tool_wait_ms.saturating_add(waited);
        agent.state = AgentSchedState::Queued;
        agent.queued_at_ms = Some(at_ms);
        agent.push(SchedEvent::ResumeFromTool, at_ms);
        if !self.ready.contains(agent_id) {
            self.ready.push_back(agent_id.clone());
        }
        Ok(())
    }

    /// Checkpoint the agent (preempt, pressure, phase unload). Releases slot.
    pub fn checkpoint(
        &mut self,
        agent_id: &LogicalAgentId,
        checkpoint: CheckpointRef,
        reason: impl Into<String>,
        at_ms: u64,
    ) -> Result<(), SchedError> {
        let agent = self
            .agents
            .get_mut(agent_id)
            .ok_or(SchedError::UnknownAgent)?;
        if agent.state.is_terminal() {
            return Err(SchedError::Terminal);
        }
        if !matches!(
            agent.state,
            AgentSchedState::Generating
                | AgentSchedState::ToolWaiting
                | AgentSchedState::Admitted
                | AgentSchedState::Preempted
        ) {
            return Err(SchedError::IllegalTransition {
                from: agent.state,
                to: AgentSchedState::Checkpointed,
            });
        }
        agent.checkpoint = Some(checkpoint);
        agent.state = AgentSchedState::Checkpointed;
        agent.push(
            SchedEvent::Checkpoint {
                reason: reason.into(),
            },
            at_ms,
        );
        self.metrics.checkpoints += 1;
        Ok(())
    }

    /// Checkpointed → Queued for resume on the same residency.
    pub fn resume_from_checkpoint(
        &mut self,
        agent_id: &LogicalAgentId,
        at_ms: u64,
    ) -> Result<(), SchedError> {
        let agent = self
            .agents
            .get_mut(agent_id)
            .ok_or(SchedError::UnknownAgent)?;
        if agent.state != AgentSchedState::Checkpointed {
            return Err(SchedError::IllegalTransition {
                from: agent.state,
                to: AgentSchedState::Queued,
            });
        }
        if agent.checkpoint.is_none() {
            return Err(SchedError::MissingCheckpoint);
        }
        agent.state = AgentSchedState::Queued;
        agent.queued_at_ms = Some(at_ms);
        agent.push(SchedEvent::ResumeFromCheckpoint, at_ms);
        if !self.ready.contains(agent_id) {
            self.ready.push_back(agent_id.clone());
        }
        Ok(())
    }

    pub fn complete(&mut self, agent_id: &LogicalAgentId, at_ms: u64) -> Result<(), SchedError> {
        self.finish(agent_id, AgentSchedState::Completed, SchedEvent::Complete, at_ms)
    }

    pub fn fail(
        &mut self,
        agent_id: &LogicalAgentId,
        reason: impl Into<String>,
        at_ms: u64,
    ) -> Result<(), SchedError> {
        self.finish(
            agent_id,
            AgentSchedState::Failed,
            SchedEvent::Fail {
                reason: reason.into(),
            },
            at_ms,
        )
    }

    pub fn cancel(&mut self, agent_id: &LogicalAgentId, at_ms: u64) -> Result<(), SchedError> {
        self.finish(agent_id, AgentSchedState::Cancelled, SchedEvent::Cancel, at_ms)
    }

    /// Snapshot of free batch slots on a residency (admitted + generating count).
    pub fn free_batch_slots(&self, residency_id: &ModelResidencyId) -> u32 {
        let Some(res) = self.residencies.get(residency_id) else {
            return 0;
        };
        let held = self
            .agents
            .values()
            .filter(|a| {
                a.residency_id.as_ref() == Some(residency_id) && a.state.holds_batch_slot()
            })
            .count() as u32;
        res.max_batch_slots.saturating_sub(held)
    }

    // ---- internal helpers -------------------------------------------------

    fn admit(&mut self, agent_id: &LogicalAgentId, at_ms: u64) -> Result<(), SchedError> {
        let agent = self
            .agents
            .get_mut(agent_id)
            .ok_or(SchedError::UnknownAgent)?;
        if !agent.state.is_runnable() {
            return Err(SchedError::IllegalTransition {
                from: agent.state,
                to: AgentSchedState::Admitted,
            });
        }
        if let Some(q0) = agent.queued_at_ms {
            let waited = at_ms.saturating_sub(q0);
            agent.queue_wait_ms = agent.queue_wait_ms.saturating_add(waited);
            self.metrics.total_queue_wait_ms =
                self.metrics.total_queue_wait_ms.saturating_add(waited);
        }
        agent.state = AgentSchedState::Admitted;
        agent.admit_count += 1;
        agent.queued_at_ms = None;
        // Reset starvation boost after successful admit.
        agent.effective_priority = agent.priority;
        agent.push(SchedEvent::Admit, at_ms);
        self.metrics.admits += 1;
        self.ready.retain(|id| id != agent_id);
        Ok(())
    }

    fn preempt(
        &mut self,
        agent_id: &LogicalAgentId,
        reason: &str,
        at_ms: u64,
    ) -> Result<(), SchedError> {
        let agent = self
            .agents
            .get_mut(agent_id)
            .ok_or(SchedError::UnknownAgent)?;
        if !agent.state.holds_batch_slot() {
            return Err(SchedError::IllegalTransition {
                from: agent.state,
                to: AgentSchedState::Preempted,
            });
        }
        agent.state = AgentSchedState::Preempted;
        agent.push(
            SchedEvent::Preempt {
                victim_of: reason.to_string(),
            },
            at_ms,
        );
        self.metrics.preemptions += 1;
        // Auto-checkpoint on preempt (scaffold: synthetic seq 0 until host fills).
        let cp = CheckpointRef {
            session_id: agent.session_id.clone(),
            run_id: agent.run_id.clone().unwrap_or_else(RunId::new),
            seq: 0,
        };
        agent.checkpoint = Some(cp.clone());
        agent.state = AgentSchedState::Checkpointed;
        agent.push(
            SchedEvent::Checkpoint {
                reason: format!("preempt:{reason}"),
            },
            at_ms,
        );
        self.metrics.checkpoints += 1;
        Ok(())
    }

    fn fairness_yield(
        &mut self,
        agent_id: &LogicalAgentId,
        at_ms: u64,
    ) -> Result<(), SchedError> {
        let agent = self
            .agents
            .get_mut(agent_id)
            .ok_or(SchedError::UnknownAgent)?;
        if agent.state != AgentSchedState::Generating {
            return Err(SchedError::IllegalTransition {
                from: agent.state,
                to: AgentSchedState::Queued,
            });
        }
        agent.state = AgentSchedState::Queued;
        agent.queued_at_ms = Some(at_ms);
        agent.fairness_quantum_remaining = 0;
        agent.push(SchedEvent::FairnessYield, at_ms);
        self.metrics.fairness_yields += 1;
        if !self.ready.contains(agent_id) {
            self.ready.push_back(agent_id.clone());
        }
        Ok(())
    }

    fn maybe_starvation_boost(
        &mut self,
        agent_id: &LogicalAgentId,
        at_ms: u64,
    ) -> Option<LogicalAgentId> {
        let agent = self.agents.get_mut(agent_id)?;
        if agent.state != AgentSchedState::Queued {
            return None;
        }
        let waited = agent
            .queued_at_ms
            .map(|q| at_ms.saturating_sub(q))
            .unwrap_or(0);
        if waited < self.policy.starvation_threshold_ms {
            return None;
        }
        let boosted = boost_priority(agent.effective_priority);
        if boosted == agent.effective_priority {
            return None;
        }
        let from = format!("{:?}", agent.effective_priority);
        let to = format!("{boosted:?}");
        agent.effective_priority = boosted;
        agent.push(
            SchedEvent::StarvationBoost {
                from: from.clone(),
                to: to.clone(),
            },
            at_ms,
        );
        self.metrics.starvation_boosts += 1;
        Some(agent_id.clone())
    }

    fn lowest_preemptible(&self, residency_id: &ModelResidencyId) -> Option<LogicalAgentId> {
        // Interactive is never a victim; prefer Batch/Idle generators.
        self.agents
            .iter()
            .filter(|(_, a)| {
                a.residency_id.as_ref() == Some(residency_id)
                    && a.state.holds_batch_slot()
                    && a.effective_priority > PriorityClass::High
            })
            .max_by(|(_, a), (_, b)| {
                // Higher PriorityClass ordinal = lower urgency (Idle > Batch > …).
                a.effective_priority
                    .cmp(&b.effective_priority)
                    .then_with(|| a.admit_count.cmp(&b.admit_count))
            })
            .map(|(id, _)| id.clone())
    }

    fn sort_ready(&mut self, residency_id: &ModelResidencyId) {
        let mut ids: Vec<LogicalAgentId> = self
            .ready
            .iter()
            .filter(|id| {
                self.agents
                    .get(id)
                    .map(|a| a.residency_id.as_ref() == Some(residency_id) && a.state.is_runnable())
                    .unwrap_or(false)
            })
            .cloned()
            .collect();
        ids.sort_by(|a, b| {
            let aa = &self.agents[a];
            let bb = &self.agents[b];
            // Interactive (0) before Idle (4): Ord on PriorityClass.
            aa.effective_priority
                .cmp(&bb.effective_priority)
                .then_with(|| {
                    aa.queued_at_ms
                        .unwrap_or(u64::MAX)
                        .cmp(&bb.queued_at_ms.unwrap_or(u64::MAX))
                })
                .then_with(|| a.as_str().cmp(b.as_str()))
        });
        // Keep non-matching ready entries, then append sorted residency ready set.
        let others: VecDeque<LogicalAgentId> = self
            .ready
            .iter()
            .filter(|id| !ids.contains(id))
            .cloned()
            .collect();
        self.ready = others;
        for id in ids {
            self.ready.push_back(id);
        }
    }

    fn finish(
        &mut self,
        agent_id: &LogicalAgentId,
        state: AgentSchedState,
        event: SchedEvent,
        at_ms: u64,
    ) -> Result<(), SchedError> {
        let agent = self
            .agents
            .get_mut(agent_id)
            .ok_or(SchedError::UnknownAgent)?;
        if agent.state.is_terminal() {
            return Err(SchedError::Terminal);
        }
        agent.state = state;
        agent.push(event, at_ms);
        if state == AgentSchedState::Completed {
            self.metrics.verified_tasks_completed += 1;
        }
        self.ready.retain(|id| id != agent_id);
        if let Some(rid) = agent.residency_id.clone() {
            if let Some(res) = self.residencies.get_mut(&rid) {
                res.attached.retain(|id| id != agent_id);
            }
        }
        Ok(())
    }
}

/// Raise priority one class toward Interactive (starvation boost).
fn boost_priority(p: PriorityClass) -> PriorityClass {
    match p {
        PriorityClass::Interactive => PriorityClass::Interactive,
        PriorityClass::High => PriorityClass::Interactive,
        PriorityClass::Normal => PriorityClass::High,
        PriorityClass::Batch => PriorityClass::Normal,
        PriorityClass::Idle => PriorityClass::Batch,
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SchedError {
    #[error("unknown logical agent")]
    UnknownAgent,
    #[error("unknown model residency")]
    UnknownResidency,
    #[error("agent is terminal")]
    Terminal,
    #[error("illegal transition {from:?} → {to:?}")]
    IllegalTransition {
        from: AgentSchedState,
        to: AgentSchedState,
    },
    #[error("resume requires a checkpoint ref")]
    MissingCheckpoint,
}

// ---------------------------------------------------------------------------
// Tests — state transitions, fairness, tool-wait slot release, starvation
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::ids::SessionId;

    fn residency() -> ModelResidency {
        ModelResidency::new("res_exec", "qwen3-coder-30b-executor", 2)
    }

    fn agent(priority: PriorityClass) -> LogicalAgent {
        LogicalAgent::new(SessionId::from("ses_test"), "do work", priority)
    }

    #[test]
    fn one_residency_many_agents_no_weight_copy_per_agent() {
        let mut sched = AgentScheduler::new(SchedulerPolicy::default());
        let res = residency();
        let rid = res.id.clone();
        sched.register_residency(res);

        let a = sched.register_agent(agent(PriorityClass::Normal));
        let b = sched.register_agent(agent(PriorityClass::Normal));
        let c = sched.register_agent(agent(PriorityClass::Batch));
        sched.enqueue(&a, &rid, 1_000).unwrap();
        sched.enqueue(&b, &rid, 1_001).unwrap();
        sched.enqueue(&c, &rid, 1_002).unwrap();

        // Three logical agents, one residency record.
        assert_eq!(sched.residencies.len(), 1);
        assert_eq!(sched.residencies[&rid].attached.len(), 3);
        assert_eq!(sched.free_batch_slots(&rid), 2);
    }

    #[test]
    fn admit_begin_generate_tool_wait_releases_slot() {
        let mut sched = AgentScheduler::new(SchedulerPolicy::default());
        let res = residency();
        let rid = res.id.clone();
        sched.register_residency(res);
        let a = sched.register_agent(agent(PriorityClass::Normal));
        sched.enqueue(&a, &rid, 100).unwrap();

        let plan = sched.schedule_tick(&rid, 200).unwrap();
        assert_eq!(plan.admit, vec![a.clone()]);
        assert_eq!(sched.agents[&a].state, AgentSchedState::Admitted);
        assert_eq!(sched.free_batch_slots(&rid), 1); // max 2, one admitted

        sched.begin_generate(&a, 210).unwrap();
        assert_eq!(sched.agents[&a].state, AgentSchedState::Generating);
        assert!(sched.agents[&a].state.holds_batch_slot());

        sched.suspend_for_tool(&a, "fs.read", 300).unwrap();
        assert_eq!(sched.agents[&a].state, AgentSchedState::ToolWaiting);
        assert!(!sched.agents[&a].state.holds_batch_slot());
        // Slot fully free again — another agent can use the model.
        assert_eq!(sched.free_batch_slots(&rid), 2);
    }

    #[test]
    fn resume_from_tool_requeues_and_accounts_wait() {
        let mut sched = AgentScheduler::new(SchedulerPolicy::default());
        let res = residency();
        let rid = res.id.clone();
        sched.register_residency(res);
        let a = sched.register_agent(agent(PriorityClass::Normal));
        sched.enqueue(&a, &rid, 100).unwrap();
        sched.schedule_tick(&rid, 110).unwrap();
        sched.begin_generate(&a, 120).unwrap();
        sched.suspend_for_tool(&a, "shell.exec", 130).unwrap();
        sched.resume_from_tool(&a, 530).unwrap(); // 400 ms tool wait
        assert_eq!(sched.agents[&a].state, AgentSchedState::Queued);
        assert_eq!(sched.agents[&a].tool_wait_ms, 400);
        assert_eq!(sched.metrics.total_tool_wait_ms, 400);
    }

    #[test]
    fn checkpoint_and_resume_round_trip() {
        let mut sched = AgentScheduler::new(SchedulerPolicy::default());
        let res = residency();
        let rid = res.id.clone();
        sched.register_residency(res);
        let mut ag = agent(PriorityClass::Normal);
        ag.run_id = Some(RunId::from("run_1"));
        let a = sched.register_agent(ag);
        sched.enqueue(&a, &rid, 1).unwrap();
        sched.schedule_tick(&rid, 2).unwrap();
        sched.begin_generate(&a, 3).unwrap();

        let cp = CheckpointRef {
            session_id: SessionId::from("ses_test"),
            run_id: RunId::from("run_1"),
            seq: 42,
        };
        sched
            .checkpoint(&a, cp.clone(), "phase_unload", 4)
            .unwrap();
        assert_eq!(sched.agents[&a].state, AgentSchedState::Checkpointed);
        assert_eq!(sched.agents[&a].checkpoint.as_ref().unwrap().seq, 42);
        assert_eq!(sched.free_batch_slots(&rid), 2);

        sched.resume_from_checkpoint(&a, 5).unwrap();
        assert_eq!(sched.agents[&a].state, AgentSchedState::Queued);
        // Still has checkpoint ref for kernel restore.
        assert!(sched.agents[&a].checkpoint.is_some());
    }

    #[test]
    fn interactive_preempts_batch_when_full() {
        let mut sched = AgentScheduler::new(SchedulerPolicy {
            max_admit_per_tick: 4,
            ..SchedulerPolicy::default()
        });
        let res = ModelResidency::new("res_exec", "executor", 1); // single slot
        let rid = res.id.clone();
        sched.register_residency(res);

        let batch = sched.register_agent(agent(PriorityClass::Batch));
        let interactive = sched.register_agent(agent(PriorityClass::Interactive));
        sched.enqueue(&batch, &rid, 10).unwrap();
        let plan1 = sched.schedule_tick(&rid, 20).unwrap();
        assert_eq!(plan1.admit, vec![batch.clone()]);
        sched.begin_generate(&batch, 21).unwrap();
        assert_eq!(sched.free_batch_slots(&rid), 0);

        sched.enqueue(&interactive, &rid, 30).unwrap();
        let plan2 = sched.schedule_tick(&rid, 40).unwrap();
        assert_eq!(plan2.preempt, vec![batch.clone()]);
        assert_eq!(plan2.admit, vec![interactive.clone()]);
        assert_eq!(
            sched.agents[&batch].state,
            AgentSchedState::Checkpointed
        );
        assert_eq!(
            sched.agents[&interactive].state,
            AgentSchedState::Admitted
        );
        assert_eq!(sched.metrics.preemptions, 1);
    }

    #[test]
    fn starvation_boost_raises_effective_priority() {
        let mut sched = AgentScheduler::new(SchedulerPolicy {
            starvation_threshold_ms: 1_000,
            ..SchedulerPolicy::default()
        });
        // No free slots so the idle agent stays queued and can be boosted.
        let res = ModelResidency::new("res_exec", "executor", 1);
        let rid = res.id.clone();
        sched.register_residency(res);

        let holder = sched.register_agent(agent(PriorityClass::Normal));
        let starved = sched.register_agent(agent(PriorityClass::Idle));
        sched.enqueue(&holder, &rid, 0).unwrap();
        sched.schedule_tick(&rid, 1).unwrap();
        sched.begin_generate(&holder, 2).unwrap();

        sched.enqueue(&starved, &rid, 10).unwrap();
        let plan = sched.schedule_tick(&rid, 2_000).unwrap(); // waited 1990 ms
        assert!(plan.starvation_boost.contains(&starved));
        assert_eq!(
            sched.agents[&starved].effective_priority,
            PriorityClass::Batch
        );
        assert_eq!(sched.metrics.starvation_boosts, 1);
    }

    #[test]
    fn fairness_yield_when_quantum_exhausted_and_others_wait() {
        let mut sched = AgentScheduler::new(SchedulerPolicy {
            fairness_quantum: 3,
            ..SchedulerPolicy::default()
        });
        let res = ModelResidency::new("res_exec", "executor", 1);
        let rid = res.id.clone();
        sched.register_residency(res);

        let monopolist = sched.register_agent(agent(PriorityClass::Normal));
        let waiter = sched.register_agent(agent(PriorityClass::Normal));
        sched.enqueue(&monopolist, &rid, 1).unwrap();
        sched.schedule_tick(&rid, 2).unwrap();
        sched.begin_generate(&monopolist, 3).unwrap();
        // Exhaust quantum.
        for _ in 0..3 {
            sched.consume_quantum(&monopolist).unwrap();
        }
        assert_eq!(sched.agents[&monopolist].fairness_quantum_remaining, 0);

        sched.enqueue(&waiter, &rid, 10).unwrap();
        let plan = sched.schedule_tick(&rid, 20).unwrap();
        assert!(plan.fairness_yield.contains(&monopolist));
        assert_eq!(sched.agents[&monopolist].state, AgentSchedState::Queued);
        // Waiter should be admitted into the freed slot.
        assert!(plan.admit.contains(&waiter));
        assert_eq!(sched.metrics.fairness_yields, 1);
    }

    #[test]
    fn illegal_transition_rejected() {
        let mut sched = AgentScheduler::new(SchedulerPolicy::default());
        let res = residency();
        let rid = res.id.clone();
        sched.register_residency(res);
        let a = sched.register_agent(agent(PriorityClass::Normal));
        // Cannot begin_generate from Registered.
        assert!(matches!(
            sched.begin_generate(&a, 1),
            Err(SchedError::IllegalTransition { .. })
        ));
        // Cannot suspend without generating.
        assert!(matches!(
            sched.suspend_for_tool(&a, "x", 1),
            Err(SchedError::IllegalTransition { .. })
        ));
        sched.enqueue(&a, &rid, 1).unwrap();
        sched.complete(&a, 2).unwrap();
        assert!(matches!(sched.enqueue(&a, &rid, 3), Err(SchedError::Terminal)));
    }

    #[test]
    fn priority_order_interactive_before_batch() {
        let mut sched = AgentScheduler::new(SchedulerPolicy::default());
        let res = ModelResidency::new("res_exec", "executor", 1);
        let rid = res.id.clone();
        sched.register_residency(res);

        let batch = sched.register_agent(agent(PriorityClass::Batch));
        let interactive = sched.register_agent(agent(PriorityClass::Interactive));
        // Enqueue batch first; interactive should still win the single slot.
        sched.enqueue(&batch, &rid, 1).unwrap();
        sched.enqueue(&interactive, &rid, 2).unwrap();
        let plan = sched.schedule_tick(&rid, 3).unwrap();
        assert_eq!(plan.admit, vec![interactive]);
        assert_eq!(sched.agents[&batch].state, AgentSchedState::Queued);
    }

    #[test]
    fn schema_constant_stable() {
        assert_eq!(AGENT_SCHEDULER_SCHEMA, "hcli.agent_scheduler.v1");
    }
}
