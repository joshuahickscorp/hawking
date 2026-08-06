//! HIDE YOU automations: durable, permission-bounded background jobs.
//!
//! An [`Automation`] is a declared standing goal (reminder, recurring brief,
//! connector summary, calendar prep, email triage, project status check, watch
//! condition, research monitor, file ingestion pipeline, or agent job). Every
//! automation carries a closed [`PermissionSet`]. The job it spawns receives a
//! [`JobCapability`] *derived from* that set and structurally cannot widen it.
//!
//! # The property that matters most
//!
//! **A background agent cannot inherit broader authority than the automation
//! grants.** Tool use is gated by the job capability; an attempt to call a tool
//! the automation did not grant fails closed and is recorded on the result.
//!
//! # What this module is (and is not)
//!
//! * **Is:** declaration model, capability derivation, durable store, injected
//!   clock, stop-condition enforcement, schedule-slot idempotency, fixture tool
//!   registry, inspectable result history.
//! * **Is not:** a wall-clock daemon, launchd/cron installer, real connector or
//!   model execution, fleets, Fabric, or Metal. Real tool bodies are fixture
//!   stubs; wall-clock wiring is a later step.
//!
//! Model-free throughout. Deterministic under an injected [`Clock`].

use super::*;
use crate::error::Result;
use crate::persistence::DynKeyValueStore;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeSet;

// ---------------------------------------------------------------------------
// Results and the automation declaration
// ---------------------------------------------------------------------------

/// One recorded tool attempt inside a job.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolAttempt {
    pub tool: String,
    pub ok: bool,
    /// When false, the attempt was blocked by capability (authority containment).
    pub authorized: bool,
    pub detail: String,
    pub output: Option<Value>,
}

/// Outcome of one automation job run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunResult {
    pub job_id: AutomationJobId,
    pub automation_id: AutomationId,
    pub started_ms: u64,
    pub finished_ms: u64,
    pub ok: bool,
    pub schedule_slot: Option<String>,
    pub tool_attempts: Vec<ToolAttempt>,
    pub stop_reason: Option<StopReason>,
    pub summary: String,
    pub tokens_used: u64,
    pub notifications: Vec<String>,
}

/// Full automation declaration + live bookkeeping.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Automation {
    pub id: AutomationId,
    pub kind: AutomationKind,
    pub goal: String,
    pub trigger: TriggerSpec,
    /// Connector names the automation may use (also mirrored into permissions).
    pub connectors: Vec<String>,
    /// Tool names the automation may use (also mirrored into permissions).
    pub tools: Vec<String>,
    /// Closed permission set. Jobs derive capability from this and cannot widen.
    pub permissions: PermissionSet,
    pub budget: ResourceBudget,
    pub usage: BudgetUsage,
    pub notification_policy: NotificationPolicy,
    pub stop_condition: StopCondition,
    pub status: AutomationStatus,
    pub stop_reason: Option<StopReason>,
    pub last_result: Option<RunResult>,
    /// Last N results (newest last). Capped by the engine's `result_history_limit`.
    pub results: Vec<RunResult>,
    pub next_run_ms: Option<u64>,
    /// Schedule slots already fired (idempotent triggers).
    pub fired_slots: BTreeSet<String>,
    pub failure_count: u32,
    pub created_ms: u64,
    pub updated_ms: u64,
}

impl Automation {
    /// Build a new active automation. Permissions are exactly `tools` ∪ `connectors`.
    pub fn declare(
        kind: AutomationKind,
        goal: impl Into<String>,
        trigger: TriggerSpec,
        tools: impl IntoIterator<Item = impl Into<String>>,
        connectors: impl IntoIterator<Item = impl Into<String>>,
        budget: ResourceBudget,
        notification_policy: NotificationPolicy,
        stop_condition: StopCondition,
        now_ms: u64,
    ) -> Self {
        let tools: Vec<String> = tools.into_iter().map(Into::into).collect();
        let connectors: Vec<String> = connectors.into_iter().map(Into::into).collect();
        let permissions = PermissionSet::new(tools.clone(), connectors.clone());
        let next_run_ms = match &trigger {
            TriggerSpec::Once { at_ms } => Some(*at_ms),
            TriggerSpec::Interval { anchor_ms, .. } => Some(*anchor_ms),
            TriggerSpec::CronSlot { at_ms, .. } => Some(*at_ms),
            TriggerSpec::Manual | TriggerSpec::Watch { .. } => None,
        };
        Self {
            id: AutomationId::new(),
            kind,
            goal: goal.into(),
            trigger,
            connectors,
            tools,
            permissions,
            budget,
            usage: BudgetUsage::default(),
            notification_policy,
            stop_condition,
            status: AutomationStatus::Active,
            stop_reason: None,
            last_result: None,
            results: Vec::new(),
            next_run_ms,
            fired_slots: BTreeSet::new(),
            failure_count: 0,
            created_ms: now_ms,
            updated_ms: now_ms,
        }
    }

    /// Readable full declaration (inspectability).
    pub fn declaration(&self) -> Value {
        json!({
            "id": self.id.as_str(),
            "kind": self.kind,
            "goal": self.goal,
            "trigger": self.trigger,
            "connectors": self.connectors,
            "tools": self.tools,
            "permissions": {
                "tools": self.permissions.tools().iter().cloned().collect::<Vec<_>>(),
                "connectors": self.permissions.connectors().iter().cloned().collect::<Vec<_>>(),
            },
            "budget": self.budget,
            "usage": self.usage,
            "notification_policy": self.notification_policy,
            "stop_condition": self.stop_condition,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "next_run_ms": self.next_run_ms,
            "last_result": self.last_result,
            "failure_count": self.failure_count,
            "created_ms": self.created_ms,
            "updated_ms": self.updated_ms,
        })
    }

    /// Last `n` results (newest last). Inspectability surface.
    pub fn last_n_results(&self, n: usize) -> &[RunResult] {
        let len = self.results.len();
        let start = len.saturating_sub(n);
        &self.results[start..]
    }

    pub(super) fn should_stop_after_run(&self) -> Option<StopReason> {
        if let Some(axis) = self.usage.exhausted_axis(&self.budget) {
            return Some(StopReason::BudgetExhausted {
                axis: axis.to_string(),
            });
        }
        match &self.stop_condition {
            StopCondition::Never => None,
            StopCondition::AfterRuns { count } => {
                if self.usage.runs >= *count {
                    Some(StopReason::AfterRuns { count: *count })
                } else {
                    None
                }
            }
            StopCondition::ConditionMet { .. } => None, // evaluated on event
            StopCondition::MaxFailures { count } => {
                if self.failure_count >= *count {
                    Some(StopReason::MaxFailures { count: *count })
                } else {
                    None
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Durable store
// ---------------------------------------------------------------------------

/// Persist automations in a [`KeyValueStore`] so they survive process restart.
pub struct AutomationStore;

impl AutomationStore {
    pub const NAMESPACE: &'static str = "you_automations";

    pub fn put(kv: &DynKeyValueStore, automation: &Automation) -> Result<()> {
        let value = serde_json::to_value(automation)?;
        kv.put(Self::NAMESPACE, automation.id.as_str(), value)
    }

    pub fn get(kv: &DynKeyValueStore, id: &str) -> Result<Option<Automation>> {
        match kv.get(Self::NAMESPACE, id)? {
            Some(value) => Ok(Some(serde_json::from_value(value)?)),
            None => Ok(None),
        }
    }

    pub fn list_all(kv: &DynKeyValueStore) -> Result<Vec<Automation>> {
        let mut out: Vec<Automation> = kv
            .list(Self::NAMESPACE)?
            .into_iter()
            .filter_map(|(_, value)| serde_json::from_value(value).ok())
            .collect();
        out.sort_by(|a, b| {
            a.created_ms
                .cmp(&b.created_ms)
                .then_with(|| a.id.as_str().cmp(b.id.as_str()))
        });
        Ok(out)
    }

    pub fn delete(kv: &DynKeyValueStore, id: &str) -> Result<()> {
        kv.delete(Self::NAMESPACE, id)
    }
}
