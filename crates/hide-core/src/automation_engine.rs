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

use crate::error::{HideError, Result};
use crate::persistence::DynKeyValueStore;
use parking_lot::RwLock;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::Arc;
use super::*;


// ---------------------------------------------------------------------------
// Job + engine
// ---------------------------------------------------------------------------

/// A single in-flight (or completed) job spawned from an automation.
///
/// Holds a [`JobCapability`] derived from the parent automation's
/// [`PermissionSet`]. Tool dispatch goes through [`AutomationJob::use_tool`],
/// which fails closed on unauthorized tools.
#[derive(Debug, Clone)]
pub struct AutomationJob {
    pub id: AutomationJobId,
    pub automation_id: AutomationId,
    pub capability: JobCapability,
    pub schedule_slot: Option<String>,
    pub started_ms: u64,
    pub tool_attempts: Vec<ToolAttempt>,
    pub tokens_used: u64,
    pub halted: Option<StopReason>,
}

impl AutomationJob {
    /// Invoke a fixture tool under this job's capability. Unauthorized tools
    /// fail closed and are recorded; the job is marked halted with
    /// [`StopReason::AuthorityDenied`].
    pub fn use_tool(
        &mut self,
        registry: &FixtureToolRegistry,
        tool: &str,
        args: Value,
    ) -> Result<FixtureToolResult> {
        if let Err(err) = self.capability.require_tool(tool) {
            self.tool_attempts.push(ToolAttempt {
                tool: tool.to_string(),
                ok: false,
                authorized: false,
                detail: err.to_string(),
                output: None,
            });
            self.halted = Some(StopReason::AuthorityDenied {
                tool: tool.to_string(),
            });
            return Err(err);
        }
        let fixture = registry.get(tool).ok_or_else(|| {
            HideError::NotFound(format!("fixture tool '{tool}' not in registry"))
        })?;
        let result = fixture.invoke(&args);
        self.tokens_used += result.tokens_used;
        self.tool_attempts.push(ToolAttempt {
            tool: tool.to_string(),
            ok: result.ok,
            authorized: true,
            detail: "ok".to_string(),
            output: Some(result.output.clone()),
        });
        Ok(result)
    }
}

/// Plan of tool calls a job should attempt (fixture-level "agent body").
#[derive(Debug, Clone, Default)]
pub struct JobPlan {
    pub tool_calls: Vec<(String, Value)>,
}

/// Engine: create, persist, tick, spawn, enforce stop, inspect.
pub struct AutomationEngine {
    kv: DynKeyValueStore,
    clock: Arc<dyn Clock>,
    registry: FixtureToolRegistry,
    result_history_limit: usize,
    /// In-memory index of loaded automations (mirrors durable store).
    live: RwLock<BTreeMap<String, Automation>>,
}

impl AutomationEngine {
    pub fn new(
        kv: DynKeyValueStore,
        clock: Arc<dyn Clock>,
        registry: FixtureToolRegistry,
    ) -> Self {
        Self {
            kv,
            clock,
            registry,
            result_history_limit: 32,
            live: RwLock::new(BTreeMap::new()),
        }
    }

    pub fn with_history_limit(mut self, n: usize) -> Self {
        self.result_history_limit = n.max(1);
        self
    }

    /// Load every durable automation into the live map (restart recovery).
    pub fn recover(&self) -> Result<Vec<AutomationId>> {
        let all = AutomationStore::list_all(&self.kv)?;
        let mut live = self.live.write();
        live.clear();
        let mut ids = Vec::new();
        for a in all {
            ids.push(a.id.clone());
            live.insert(a.id.as_str().to_string(), a);
        }
        Ok(ids)
    }

    /// Register a new automation and durably persist it.
    pub fn create(&self, mut automation: Automation) -> Result<Automation> {
        let now = self.clock.now_ms();
        automation.updated_ms = now;
        if automation.created_ms == 0 {
            automation.created_ms = now;
        }
        AutomationStore::put(&self.kv, &automation)?;
        let id = automation.id.as_str().to_string();
        self.live.write().insert(id, automation.clone());
        Ok(automation)
    }

    pub fn get(&self, id: &str) -> Option<Automation> {
        self.live.read().get(id).cloned()
    }

    pub fn list(&self) -> Vec<Automation> {
        let mut out: Vec<_> = self.live.read().values().cloned().collect();
        out.sort_by(|a, b| {
            a.created_ms
                .cmp(&b.created_ms)
                .then_with(|| a.id.as_str().cmp(b.id.as_str()))
        });
        out
    }

    /// Inspectable declaration + last N results.
    pub fn inspect(&self, id: &str, last_n: usize) -> Result<Value> {
        let a = self
            .get(id)
            .ok_or_else(|| HideError::NotFound(format!("automation '{id}'")))?;
        Ok(json!({
            "declaration": a.declaration(),
            "results": a.last_n_results(last_n),
        }))
    }

    fn persist(&self, automation: &Automation) -> Result<()> {
        AutomationStore::put(&self.kv, automation)?;
        self.live
            .write()
            .insert(automation.id.as_str().to_string(), automation.clone());
        Ok(())
    }

    /// Cancel an automation (terminal).
    pub fn cancel(&self, id: &str) -> Result<Automation> {
        let mut a = self
            .get(id)
            .ok_or_else(|| HideError::NotFound(format!("automation '{id}'")))?;
        a.status = AutomationStatus::Cancelled;
        a.stop_reason = Some(StopReason::Cancelled);
        a.next_run_ms = None;
        a.updated_ms = self.clock.now_ms();
        self.persist(&a)?;
        Ok(a)
    }

    /// Present a named watch/stop condition as met. If it matches the stop
    /// condition, the automation halts. If it matches a watch trigger, fires once.
    pub fn signal_condition(&self, id: &str, name: &str) -> Result<Option<RunResult>> {
        let mut a = self
            .get(id)
            .ok_or_else(|| HideError::NotFound(format!("automation '{id}'")))?;

        if matches!(&a.stop_condition, StopCondition::ConditionMet { name: n } if n == name) {
            a.status = AutomationStatus::Stopped;
            a.stop_reason = Some(StopReason::ConditionMet {
                name: name.to_string(),
            });
            a.next_run_ms = None;
            a.updated_ms = self.clock.now_ms();
            self.persist(&a)?;
            return Ok(None);
        }

        if matches!(&a.trigger, TriggerSpec::Watch { condition } if condition == name)
            && a.status.may_run()
        {
            let slot = format!("watch:{name}");
            return self.run_slot(&mut a, Some(slot), JobPlan::default());
        }

        Ok(None)
    }

    /// Manual wake. Each call mints a unique slot (run count + job mint) so
    /// two manual wakes at the same clock time are not collapsed by schedule
    /// idempotency — that guard applies to schedule slots, not owner-initiated runs.
    pub fn run_manual(&self, id: &str, plan: JobPlan) -> Result<RunResult> {
        let mut a = self
            .get(id)
            .ok_or_else(|| HideError::NotFound(format!("automation '{id}'")))?;
        if !a.status.may_run() {
            return Err(HideError::InvalidState(format!(
                "automation '{id}' is {:?}, cannot run",
                a.status
            )));
        }
        let slot = format!(
            "manual:{}:{}",
            self.clock.now_ms(),
            a.usage.runs.saturating_add(1)
        );
        self.run_slot(&mut a, Some(slot), plan)?
            .ok_or_else(|| HideError::InvalidState("manual run produced no result".into()))
    }

    /// Advance the engine: fire every due, not-yet-fired schedule slot once.
    /// Returns results produced on this tick.
    pub fn tick(&self, default_plan: &JobPlan) -> Result<Vec<RunResult>> {
        let now = self.clock.now_ms();
        let ids: Vec<String> = self.live.read().keys().cloned().collect();
        let mut results = Vec::new();
        for id in ids {
            let Some(mut a) = self.get(&id) else {
                continue;
            };
            if !a.status.may_run() {
                continue;
            }
            // Budget check before spawn.
            if let Some(axis) = a.usage.exhausted_axis(&a.budget) {
                a.status = AutomationStatus::Stopped;
                a.stop_reason = Some(StopReason::BudgetExhausted {
                    axis: axis.to_string(),
                });
                a.next_run_ms = None;
                a.updated_ms = now;
                self.persist(&a)?;
                continue;
            }
            let Some(slot) = a.trigger.slot_if_due(now) else {
                continue;
            };
            if a.fired_slots.contains(&slot) {
                // Idempotent: already ran this slot.
                // Still refresh next_run for intervals.
                a.next_run_ms = a.trigger.next_run_after(now);
                a.updated_ms = now;
                self.persist(&a)?;
                continue;
            }
            if let Some(result) = self.run_slot(&mut a, Some(slot), default_plan.clone())? {
                results.push(result);
            }
        }
        Ok(results)
    }

    /// Fire a specific schedule slot (used by tests and tick). Idempotent: a
    /// second fire of the same slot is a no-op that returns `None`.
    pub fn fire_slot(
        &self,
        id: &str,
        slot: impl Into<String>,
        plan: JobPlan,
    ) -> Result<Option<RunResult>> {
        let mut a = self
            .get(id)
            .ok_or_else(|| HideError::NotFound(format!("automation '{id}'")))?;
        if !a.status.may_run() {
            return Ok(None);
        }
        self.run_slot(&mut a, Some(slot.into()), plan)
    }

    fn run_slot(
        &self,
        a: &mut Automation,
        slot: Option<String>,
        plan: JobPlan,
    ) -> Result<Option<RunResult>> {
        let now = self.clock.now_ms();
        if let Some(ref s) = slot {
            if a.fired_slots.contains(s) {
                return Ok(None);
            }
        }
        if let Some(axis) = a.usage.exhausted_axis(&a.budget) {
            a.status = AutomationStatus::Stopped;
            a.stop_reason = Some(StopReason::BudgetExhausted {
                axis: axis.to_string(),
            });
            a.next_run_ms = None;
            a.updated_ms = now;
            self.persist(a)?;
            return Ok(None);
        }

        // STRUCTURAL: capability is derived from the automation permission set.
        let capability = a.permissions.derive_capability();
        debug_assert!(
            capability.is_within(&a.permissions),
            "derived capability must be within the automation permission set"
        );

        let mut job = AutomationJob {
            id: AutomationJobId::new(),
            automation_id: a.id.clone(),
            capability,
            schedule_slot: slot.clone(),
            started_ms: now,
            tool_attempts: Vec::new(),
            tokens_used: 0,
            halted: None,
        };

        // Execute the plan under the capability gate.
        let mut plan_ok = true;
        for (tool, args) in &plan.tool_calls {
            // Per-call budget: max_tool_calls.
            if let Some(max) = a.budget.max_tool_calls {
                if a.usage.tool_calls >= max {
                    job.halted = Some(StopReason::BudgetExhausted {
                        axis: "max_tool_calls".into(),
                    });
                    plan_ok = false;
                    break;
                }
            }
            match job.use_tool(&self.registry, tool, args.clone()) {
                Ok(_) => {
                    a.usage.tool_calls = a.usage.tool_calls.saturating_add(1);
                }
                Err(_) => {
                    plan_ok = false;
                    break;
                }
            }
            if let Some(max_tokens) = a.budget.max_tokens {
                if a.usage.tokens.saturating_add(job.tokens_used) >= max_tokens {
                    // Will be reflected after usage update.
                }
            }
        }

        let finished = self.clock.now_ms();
        let wall = finished.saturating_sub(now);
        a.usage.wall_ms = a.usage.wall_ms.saturating_add(wall);
        a.usage.tokens = a.usage.tokens.saturating_add(job.tokens_used);
        a.usage.runs = a.usage.runs.saturating_add(1);

        if !plan_ok {
            a.failure_count = a.failure_count.saturating_add(1);
        }

        // Job-level halt (e.g. authority denial) is always recorded on the result.
        // Automation-level halt is only for stop conditions / budget — a single
        // unauthorized tool attempt must not permanently privilege-revoke the
        // standing automation; it fails the job closed and is auditable.
        let job_stop = job.halted.clone();
        let automation_stop = a.should_stop_after_run();
        let result_stop = job_stop.clone().or_else(|| automation_stop.clone());

        let notifications = notifications_for(&a.notification_policy, plan_ok);

        let result = RunResult {
            job_id: job.id.clone(),
            automation_id: a.id.clone(),
            started_ms: now,
            finished_ms: finished,
            ok: plan_ok && job.halted.is_none(),
            schedule_slot: slot.clone(),
            tool_attempts: job.tool_attempts.clone(),
            stop_reason: result_stop,
            summary: if plan_ok {
                format!("completed {} tool call(s)", job.tool_attempts.len())
            } else if matches!(
                job.halted,
                Some(StopReason::AuthorityDenied { .. })
            ) {
                "halted: authority denied".into()
            } else {
                "failed".into()
            },
            tokens_used: job.tokens_used,
            notifications,
        };

        if let Some(ref s) = slot {
            a.fired_slots.insert(s.clone());
        }
        a.last_result = Some(result.clone());
        a.results.push(result.clone());
        if a.results.len() > self.result_history_limit {
            let excess = a.results.len() - self.result_history_limit;
            a.results.drain(0..excess);
        }

        if let Some(reason) = automation_stop {
            a.status = AutomationStatus::Stopped;
            a.stop_reason = Some(reason);
            a.next_run_ms = None;
        } else {
            a.next_run_ms = a.trigger.next_run_after(now);
        }
        a.updated_ms = finished;
        self.persist(a)?;
        Ok(Some(result))
    }

    pub fn registry(&self) -> &FixtureToolRegistry {
        &self.registry
    }

    pub fn clock(&self) -> &dyn Clock {
        self.clock.as_ref()
    }
}

fn notifications_for(policy: &NotificationPolicy, ok: bool) -> Vec<String> {
    match policy {
        NotificationPolicy::Silent => Vec::new(),
        NotificationPolicy::OnFailure if !ok => vec!["failure".into()],
        NotificationPolicy::OnSuccess if ok => vec!["success".into()],
        NotificationPolicy::Always => {
            vec![if ok {
                "success".into()
            } else {
                "failure".into()
            }]
        }
        _ => Vec::new(),
    }
}

