//! The plan domain (Bible sec 14, ch.02 Appendix A.1): a durable, host-owned
//! plan record and its Wire-B `plan` projection.
//!
//! The kernel emits a plan-as-data ([`hide_kernel::plan::schema::Plan`]) into the
//! event log, but that schema is execution-shaped and immutable to the host (it
//! lives in a sibling crate). This module owns the DURABLE, MUTABLE plan record
//! the IDE's PlanCard binds:
//!
//! * [`PlanRecord::from_kernel`] maps a kernel plan into a PlanCard-shaped record,
//!   computing per-step write-blocking from the run's autonomy.
//! * [`PlanRecordStore`] persists it over the KV store (one active plan per
//!   session, mirroring `GoalStore`).
//! * [`store_and_publish`] is the single seam both the live-turn emitter and the
//!   host mutation handlers route through: it writes the durable record AND
//!   publishes it as a `plan` [`UiEventKind::ProjectionPatch`] on the Wire-B bus.
//! * The mutation methods ([`PlanRecord::approve`], [`PlanRecord::edit_step`],
//!   [`PlanRecord::reorder`], [`PlanRecord::skip_step`],
//!   [`PlanRecord::repair_failed_step`]) mutate the record; the host handler
//!   re-persists and republishes.
//!
//! MODEL-FREE: nothing here loads or decides on a model; it reshapes and mutates
//! declared data only.

use crate::ui_bus::UiEventBus;
use hide_core::api::{UiEvent, UiEventKind};
use hide_core::ids::SessionId;
use hide_core::persistence::DynKeyValueStore;
use hide_core::Result;
use hide_kernel::govern::Autonomy;
use hide_kernel::plan::schema::{Plan as KernelPlan, PlanStep as KernelStep, StepKind, StepStatus};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The projection name the frontend PlanCard routes on (already registered in
/// `app/src/wire.ts` `PROJECTION_NAMES`, so nothing new is added here).
pub const PLAN_PROJECTION: &str = "plan";

/// One PlanCard row. Carries everything the card renders per step: the declared
/// contract (acceptance, allowed effects, related files, owner) AND the live
/// state (status, verification, blocker, whether an effectful write is gated).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanStepRecord {
    pub id: String,
    /// The step title/summary the card shows (editable via `edit_plan_step`).
    pub text: String,
    /// Snake_case [`StepStatus`]: pending | ready | running | blocked | completed
    /// | failed | skipped.
    pub status: String,
    pub dependencies: Vec<String>,
    /// The declared acceptance predicate (the up-front verifier contract, K1).
    pub acceptance: String,
    /// The effects this step is allowed to cause (derived from its kind).
    pub effects: Vec<String>,
    pub related_files: Vec<String>,
    /// The agent that owns this step. The kernel plan has no per-step owner, so
    /// the honest default is the root agent.
    pub owner_agent: String,
    /// Verification status: pending | passed | failed | skipped.
    pub verification: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocker: Option<String>,
    /// Approved in the PlanCard (planning-level approval). Note this does NOT
    /// clear `write_blocked`: the per-effect approval gate still owns write
    /// authority under a bounded autonomy.
    pub approved: bool,
    /// This effectful step is gated (write-blocked) under the run's autonomy.
    /// True for an effectful step under any non-`full_auto` autonomy.
    pub write_blocked: bool,
}

/// The durable PlanCard record: one active plan per session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanRecord {
    pub plan_id: String,
    pub title: String,
    pub objective: String,
    /// Snake_case [`hide_kernel::plan::schema::PlanStatus`].
    pub status: String,
    /// Snake_case [`Autonomy`] the plan runs under; carried so a mutation that
    /// appends a step (repair) can gate the new step consistently.
    pub autonomy: String,
    pub approved: bool,
    pub steps: Vec<PlanStepRecord>,
}

/// An effectful step is write-blocked under any autonomy that is not full-auto.
/// `suggest_only` pauses for approval; `read_only` forbids the effect outright.
/// Either way the effectful step stays gated (the plan-mode write-block).
fn write_blocking(autonomy: Autonomy) -> bool {
    !matches!(autonomy, Autonomy::FullAuto)
}

/// Serialize a serde snake_case enum into its wire string (kept in sync with the
/// enum's own `rename_all`, so no hand-maintained mapping drifts).
fn snake<T: Serialize>(v: T) -> String {
    serde_json::to_value(v)
        .ok()
        .and_then(|x| x.as_str().map(str::to_string))
        .unwrap_or_default()
}

/// The effects a step of this kind is allowed to cause.
fn effects_for_kind(kind: StepKind) -> Vec<String> {
    match kind {
        StepKind::Edit => vec!["write_fs".to_string()],
        StepKind::Command => vec!["shell".to_string(), "process".to_string()],
        StepKind::Delegate => vec!["agent_spawn".to_string()],
        StepKind::Investigate => vec!["read_fs".to_string()],
        StepKind::Verify | StepKind::Synthesize | StepKind::Decompose => Vec::new(),
    }
}

/// Verification status derived from a step's status. A step that completed passed
/// its acceptance; a failed/skipped step carries that verdict; anything in flight
/// is `pending`.
fn verification_for_status(status: StepStatus) -> &'static str {
    match status {
        StepStatus::Completed => "passed",
        StepStatus::Failed => "failed",
        StepStatus::Skipped => "skipped",
        _ => "pending",
    }
}

/// Files this step relates to: the artifacts it produces, plus a `path`/`file`
/// named in its concrete tool args (best-effort; the kernel step has no explicit
/// file-scope field).
fn related_files(step: &KernelStep) -> Vec<String> {
    let mut files = step.produced.clone();
    if let Some(args) = &step.tool_args {
        for key in ["path", "file"] {
            if let Some(p) = args.get(key).and_then(|v| v.as_str()) {
                if !files.iter().any(|f| f == p) {
                    files.push(p.to_string());
                }
            }
        }
    }
    files
}

impl PlanRecord {
    /// Build the PlanCard record from a live kernel plan under the run's autonomy.
    pub fn from_kernel(plan: &KernelPlan, autonomy: Autonomy) -> Self {
        let gate = write_blocking(autonomy);
        let steps = plan
            .steps
            .iter()
            .map(|s| PlanStepRecord {
                id: s.id.as_str().to_string(),
                text: s.title.clone(),
                status: snake(s.status),
                dependencies: s.dependencies.iter().map(|d| d.as_str().to_string()).collect(),
                acceptance: s.acceptance.predicate.clone(),
                effects: effects_for_kind(s.kind),
                related_files: related_files(s),
                owner_agent: "root".to_string(),
                verification: verification_for_status(s.status).to_string(),
                blocker: (s.status == StepStatus::Blocked)
                    .then(|| "awaiting dependencies".to_string()),
                approved: false,
                // Only a world-mutating step is write-blocked; a read-only
                // investigate step is not gated even under suggest-only.
                write_blocked: s.is_effectful() && gate,
            })
            .collect();
        Self {
            plan_id: plan.id.as_str().to_string(),
            title: plan.title.clone(),
            objective: plan.objective.clone(),
            status: snake(plan.status),
            autonomy: snake(autonomy),
            approved: false,
            steps,
        }
    }

    fn autonomy_enum(&self) -> Autonomy {
        serde_json::from_value(Value::String(self.autonomy.clone())).unwrap_or(Autonomy::SuggestOnly)
    }

    /// The Wire-B patch body (the record itself).
    pub fn to_patch(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }

    /// The effectful steps that are currently write-blocked (the gated set the
    /// PlanCard highlights).
    pub fn write_blocked_steps(&self) -> Vec<&PlanStepRecord> {
        self.steps.iter().filter(|s| s.write_blocked).collect()
    }

    /// `approve_plan`: `target = None` approves the whole plan (and every step);
    /// `Some(step_id)` approves just that step. Returns `false` if a named step is
    /// unknown. Approval is planning-level: it does NOT clear `write_blocked`.
    pub fn approve(&mut self, target: Option<&str>) -> bool {
        match target {
            None => {
                self.approved = true;
                for s in &mut self.steps {
                    s.approved = true;
                }
                true
            }
            Some(id) => match self.steps.iter_mut().find(|s| s.id == id) {
                Some(s) => {
                    s.approved = true;
                    true
                }
                None => false,
            },
        }
    }

    /// `edit_plan_step`: replace a step's text. Returns `false` if unknown.
    pub fn edit_step(&mut self, step_id: &str, text: impl Into<String>) -> bool {
        match self.steps.iter_mut().find(|s| s.id == step_id) {
            Some(s) => {
                s.text = text.into();
                true
            }
            None => false,
        }
    }

    /// `reorder_plan`: reorder the steps to match `order`, which must be a
    /// permutation of the current step ids (a safe, total reorder). Returns
    /// `false` if it names an unknown id, or omits/duplicates any.
    pub fn reorder(&mut self, order: &[String]) -> bool {
        use std::collections::HashSet;
        let current: HashSet<&str> = self.steps.iter().map(|s| s.id.as_str()).collect();
        let wanted: HashSet<&str> = order.iter().map(|s| s.as_str()).collect();
        if order.len() != self.steps.len() || current != wanted {
            return false;
        }
        let mut by_id: std::collections::HashMap<String, PlanStepRecord> =
            self.steps.drain(..).map(|s| (s.id.clone(), s)).collect();
        self.steps = order.iter().filter_map(|id| by_id.remove(id)).collect();
        true
    }

    /// `skip_step`: mark a step skipped with a reason. A skipped step runs
    /// nothing, so it is no longer write-blocked. Returns `false` if unknown.
    pub fn skip_step(&mut self, step_id: &str, reason: impl Into<String>) -> bool {
        match self.steps.iter_mut().find(|s| s.id == step_id) {
            Some(s) => {
                s.status = "skipped".to_string();
                s.verification = "skipped".to_string();
                s.blocker = Some(reason.into());
                s.write_blocked = false;
                true
            }
            None => false,
        }
    }

    /// Convert a failing verification into a repair step (cheap: one edit step
    /// that depends on the failing step and re-declares its acceptance). Returns
    /// the new step id, or `None` if the step is unknown or not failing.
    /// Idempotent: a second call for the same step returns the existing repair id
    /// without appending a duplicate.
    pub fn repair_failed_step(&mut self, step_id: &str) -> Option<String> {
        let repair_id = format!("{step_id}-repair");
        if self.steps.iter().any(|s| s.id == repair_id) {
            return Some(repair_id);
        }
        let failing = self.steps.iter().find(|s| s.id == step_id)?;
        if failing.verification != "failed" && failing.status != "failed" {
            return None;
        }
        let acceptance = failing.acceptance.clone();
        let related = failing.related_files.clone();
        let text = format!("Repair: {}", failing.text);
        let gate = write_blocking(self.autonomy_enum());
        if let Some(f) = self.steps.iter_mut().find(|s| s.id == step_id) {
            f.blocker = Some(format!("verification failed; repair queued as {repair_id}"));
        }
        self.steps.push(PlanStepRecord {
            id: repair_id.clone(),
            text,
            status: "pending".to_string(),
            dependencies: vec![step_id.to_string()],
            acceptance,
            effects: vec!["write_fs".to_string()],
            related_files: related,
            owner_agent: "root".to_string(),
            verification: "pending".to_string(),
            blocker: None,
            approved: false,
            write_blocked: gate,
        });
        Some(repair_id)
    }
}

/// Durable persistence for [`PlanRecord`]s over the KV store, mirroring
/// `GoalStore`: a stateless facade over the `plans` namespace keyed by session id
/// (one active plan per session).
pub struct PlanRecordStore;

impl PlanRecordStore {
    pub const NAMESPACE: &'static str = "plans";

    pub fn put(kv: &DynKeyValueStore, session: &SessionId, record: &PlanRecord) -> Result<()> {
        kv.put(Self::NAMESPACE, session.as_str(), serde_json::to_value(record)?)
    }

    pub fn get(kv: &DynKeyValueStore, session: &SessionId) -> Option<PlanRecord> {
        kv.get(Self::NAMESPACE, session.as_str())
            .ok()
            .flatten()
            .and_then(|value| serde_json::from_value(value).ok())
    }
}

/// Persist the durable plan record AND publish it on the Wire-B bus as a `plan`
/// ProjectionPatch. The single seam both the live-turn emitter and the host
/// mutation handlers route through, so the durable record and the projection can
/// never diverge.
pub fn store_and_publish(
    kv: &DynKeyValueStore,
    ui_bus: &UiEventBus,
    session: &SessionId,
    seq: u64,
    record: &PlanRecord,
) -> Result<()> {
    PlanRecordStore::put(kv, session, record)?;
    ui_bus.publish(UiEvent {
        seq,
        session_id: Some(session.clone()),
        kind: UiEventKind::ProjectionPatch {
            projection: PLAN_PROJECTION.to_string(),
            patch: record.to_patch(),
        },
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_kernel::plan::schema::{Acceptance, Plan, PlanStep, PlanStatus, StepKind};

    /// A two-step plan: a read-only investigate step and an effectful edit step.
    fn sample_plan() -> Plan {
        let mut investigate = PlanStep::new(
            "look at the failing test",
            StepKind::Investigate,
            Acceptance::predicate("root cause identified"),
        );
        investigate.produced = vec!["notes.md".to_string()];
        let edit = PlanStep::new(
            "fix the bug",
            StepKind::Edit,
            Acceptance::with_oracles("build passes", vec!["build".to_string()]),
        );
        Plan {
            id: hide_core::ids::PlanId::new(),
            title: "fix the bug".to_string(),
            objective: "make the failing test pass".to_string(),
            steps: vec![investigate, edit],
            status: PlanStatus::Active,
            budget: Default::default(),
        }
    }

    #[test]
    fn from_kernel_maps_real_steps_with_declared_fields() {
        let plan = sample_plan();
        let record = PlanRecord::from_kernel(&plan, Autonomy::SuggestOnly);
        assert_eq!(record.steps.len(), 2);
        let inv = &record.steps[0];
        assert_eq!(inv.text, "look at the failing test");
        assert_eq!(inv.effects, vec!["read_fs".to_string()]);
        assert_eq!(inv.related_files, vec!["notes.md".to_string()]);
        assert_eq!(inv.owner_agent, "root");
        assert_eq!(inv.status, "pending");
        // A read-only step is NOT write-blocked, even under suggest-only.
        assert!(!inv.write_blocked);
        let edit = &record.steps[1];
        assert_eq!(edit.acceptance, "build passes");
        assert_eq!(edit.effects, vec!["write_fs".to_string()]);
        // The effectful step IS gated under suggest-only.
        assert!(edit.write_blocked);
    }

    #[test]
    fn write_block_holds_in_suggest_only_and_read_only_but_not_full_auto() {
        let plan = sample_plan();
        for autonomy in [Autonomy::SuggestOnly, Autonomy::ReadOnly] {
            let record = PlanRecord::from_kernel(&plan, autonomy);
            assert_eq!(
                record.write_blocked_steps().len(),
                1,
                "the one effectful step must stay gated under {autonomy:?}"
            );
        }
        let full = PlanRecord::from_kernel(&plan, Autonomy::FullAuto);
        assert!(
            full.write_blocked_steps().is_empty(),
            "full-auto gates nothing"
        );
    }

    #[test]
    fn approve_does_not_clear_the_write_block() {
        let plan = sample_plan();
        let mut record = PlanRecord::from_kernel(&plan, Autonomy::SuggestOnly);
        let edit_id = record.steps[1].id.clone();
        assert!(record.approve(None));
        assert!(record.approved);
        assert!(record.steps.iter().all(|s| s.approved));
        // Planning approval must NOT grant write authority under a bounded autonomy.
        let edit = record.steps.iter().find(|s| s.id == edit_id).unwrap();
        assert!(edit.write_blocked, "approve_plan must not clear the effect gate");
    }

    #[test]
    fn edit_and_reorder_mutate_the_record() {
        let plan = sample_plan();
        let mut record = PlanRecord::from_kernel(&plan, Autonomy::SuggestOnly);
        let a = record.steps[0].id.clone();
        let b = record.steps[1].id.clone();

        assert!(record.edit_step(&a, "look harder"));
        assert_eq!(record.steps[0].text, "look harder");
        assert!(!record.edit_step("missing", "x"));

        assert!(record.reorder(&[b.clone(), a.clone()]));
        assert_eq!(record.steps[0].id, b);
        assert_eq!(record.steps[1].id, a);
        // A non-permutation is rejected.
        assert!(!record.reorder(&[a.clone()]));
        assert!(!record.reorder(&[a.clone(), "ghost".to_string()]));
    }

    #[test]
    fn skip_step_records_reason_and_ungates() {
        let plan = sample_plan();
        let mut record = PlanRecord::from_kernel(&plan, Autonomy::SuggestOnly);
        let edit_id = record.steps[1].id.clone();
        assert!(record.skip_step(&edit_id, "not needed"));
        let edit = record.steps.iter().find(|s| s.id == edit_id).unwrap();
        assert_eq!(edit.status, "skipped");
        assert_eq!(edit.blocker.as_deref(), Some("not needed"));
        assert!(!edit.write_blocked);
    }

    #[test]
    fn repair_failed_step_appends_a_dependent_repair_step() {
        let plan = sample_plan();
        let mut record = PlanRecord::from_kernel(&plan, Autonomy::SuggestOnly);
        let edit_id = record.steps[1].id.clone();
        // Not failing yet -> no repair.
        assert!(record.repair_failed_step(&edit_id).is_none());
        // Mark it failed, then repair.
        record.steps[1].verification = "failed".to_string();
        let repair_id = record.repair_failed_step(&edit_id).expect("repair step created");
        let repair = record.steps.iter().find(|s| s.id == repair_id).unwrap();
        assert_eq!(repair.dependencies, vec![edit_id.clone()]);
        assert_eq!(repair.acceptance, "build passes");
        assert!(repair.write_blocked, "the repair step writes, so it is gated too");
        // Idempotent: a second call does not duplicate.
        let again = record.repair_failed_step(&edit_id).unwrap();
        assert_eq!(again, repair_id);
        assert_eq!(record.steps.iter().filter(|s| s.id == repair_id).count(), 1);
    }
}
