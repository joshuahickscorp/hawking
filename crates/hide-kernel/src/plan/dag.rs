use crate::plan::schema::{Plan, StepStatus};
use hide_core::ids::StepId;
use std::collections::{BTreeMap, BTreeSet};

pub struct PlanDag;

impl PlanDag {
    pub fn ready_steps(plan: &Plan) -> Vec<StepId> {
        let completed: BTreeSet<_> = plan
            .steps
            .iter()
            .filter(|step| step.status == StepStatus::Completed)
            .map(|step| step.id.clone())
            .collect();
        plan.steps
            .iter()
            .filter(|step| step.status == StepStatus::Pending)
            .filter(|step| step.dependencies.iter().all(|dep| completed.contains(dep)))
            .map(|step| step.id.clone())
            .collect()
    }

    pub fn has_cycle(plan: &Plan) -> bool {
        let deps: BTreeMap<_, _> = plan
            .steps
            .iter()
            .map(|step| (step.id.clone(), step.dependencies.clone()))
            .collect();
        for step in deps.keys() {
            let mut visiting = BTreeSet::new();
            if visit(step, &deps, &mut visiting) {
                return true;
            }
        }
        false
    }
}

fn visit(
    step: &StepId,
    deps: &BTreeMap<StepId, Vec<StepId>>,
    visiting: &mut BTreeSet<StepId>,
) -> bool {
    if !visiting.insert(step.clone()) {
        return true;
    }
    for dep in deps.get(step).into_iter().flatten() {
        if visit(dep, deps, visiting) {
            return true;
        }
    }
    visiting.remove(step);
    false
}
