//! Replanning (bible ch.02 §4.5.3 / §4.7.3).
//!
//! Replan when repeated repairs fail (the *approach* is wrong) or the failure
//! reveals the *plan* was wrong (a missing dependency / wrong decomposition).
//! **Localized first** (revise from the failure point, carry a lesson forward),
//! **full** only if needed.

use crate::plan::schema::{Acceptance, Plan, PlanStatus, PlanStep, StepKind, StepStatus};
use hide_core::ids::StepId;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplanRequest {
    pub failed_step: Option<StepId>,
    pub reason: String,
    /// A lesson distilled from the failure, prepended to the revised step.
    #[serde(default)]
    pub lesson: Option<String>,
    /// Localized (revise from the failure point) vs full (resynthesize).
    pub local_only: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplanResult {
    pub plan: Plan,
    pub changed_steps: Vec<StepId>,
}

/// Mark a plan superseded (used when a full replan replaces it entirely).
pub fn supersede(mut plan: Plan) -> Plan {
    plan.status = PlanStatus::Superseded;
    plan
}

/// Localized replan: revise the failed step in place (reset it to pending with
/// the lesson recorded in its rationale, and insert a remediation step before
/// it if the failure points at a missing dependency). The downstream steps are
/// left intact (their deps still reference the revised step's id).
///
/// Returns the revised plan + the ids of the steps that changed. Bounded by the
/// caller against `Budget.max_replans`.
pub fn localized_replan(plan: &Plan, request: &ReplanRequest) -> ReplanResult {
    let mut plan = plan.clone();
    let mut changed = Vec::new();

    if let Some(failed_id) = &request.failed_step {
        if let Some(idx) = plan.steps.iter().position(|s| &s.id == failed_id) {
            // Reset the failed step and fold the lesson into its rationale so the
            // next attempt's repair context carries it (§4.7.2).
            let failed = &mut plan.steps[idx];
            failed.status = StepStatus::Pending;
            failed.attempts = 0;
            failed.repairs = 0;
            if let Some(lesson) = &request.lesson {
                failed.rationale = format!("{} | lesson: {lesson}", failed.rationale);
            }
            changed.push(failed.id.clone());

            // Insert an investigation step *before* the failed one to gather the
            // missing context the approach was lacking. The failed step gains a
            // dependency on it (localized graph surgery, not a full rebuild).
            let mut probe = PlanStep::new(
                format!("Re-investigate before retrying: {}", request.reason),
                StepKind::Investigate,
                Acceptance::predicate("root cause of the prior failure understood"),
            );
            probe.rationale = request
                .lesson
                .clone()
                .unwrap_or_else(|| request.reason.clone());
            let probe_id = probe.id.clone();
            plan.steps[idx].dependencies.push(probe_id.clone());
            plan.steps.insert(idx, probe);
            changed.push(probe_id);
        }
    }
    ReplanResult { plan, changed_steps: changed }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::dag::PlanDag;
    use crate::plan::planner::RuntimePlanner;

    #[test]
    fn localized_replan_resets_failed_and_inserts_probe() {
        let plan = RuntimePlanner::default_dag("obj");
        let edit_id = plan.steps[1].id.clone();
        let req = ReplanRequest {
            failed_step: Some(edit_id.clone()),
            reason: "build kept failing".to_string(),
            lesson: Some("exp must be i64".to_string()),
            local_only: true,
        };
        let result = localized_replan(&plan, &req);
        // The probe + the reset edit changed.
        assert_eq!(result.changed_steps.len(), 2);
        // Still a DAG.
        assert!(PlanDag::acyclic(&result.plan));
        // One more step than before.
        assert_eq!(result.plan.steps.len(), plan.steps.len() + 1);
        // The failed step is pending again and carries the lesson.
        let edit = result.plan.step(&edit_id).unwrap();
        assert_eq!(edit.status, StepStatus::Pending);
        assert!(edit.rationale.contains("i64"));
    }
}
