//! Subagent delegation (bible ch.02 §4.10 / Appendix A.4).
//!
//! A subagent **is just a nested `AgentState`** on the stack — same state
//! machine, same governor (with a derived child budget), same event envelope
//! (child events carry the child `run_id` and `parent` = the spawning step).
//! The parent only ever ingests the *summary* — never the child's raw
//! transcript (the clean-window discipline, §4.10).

use crate::govern::Budget;
use crate::machine::state::{AgentState, Frame, Phase};
use crate::AgentKernel;
use hide_core::ids::{RunId, SessionId};
use hide_core::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IsolationMode {
    None,
    Context,
    Worktree,
    FreshContext,
    MicroVm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubagentKind {
    Research,
    Implement,
    Verify,
    Review,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubagentSpec {
    pub name: String,
    pub objective: String,
    pub kind: SubagentKind,
    pub isolation: IsolationMode,
    /// The child budget (A.4). If absent, derived from the parent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget: Option<Budget>,
    /// Max steps the child may run before its run is joined.
    pub max_steps: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubagentHandle {
    pub session_id: SessionId,
    pub run_id: RunId,
    pub spec: SubagentSpec,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubagentStatus {
    Ok,
    Partial,
    Failed,
    Aborted,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubagentReturn {
    pub handle: SubagentHandle,
    pub status: SubagentStatus,
    /// The ONLY thing entering parent context (§4.10).
    pub summary: String,
    pub lessons: Vec<String>,
    /// Budget used by the child — rolled up into the parent's ledger.
    pub steps_used: u32,
    pub tool_calls_used: u32,
}

/// Spawn a subagent: push a frame on the parent's stack (bounds depth), run a
/// nested `AgentState` to terminal under a child budget, and join — folding the
/// child's budget usage into the parent's ledger and returning only a summary.
pub async fn spawn_and_join(
    kernel: &AgentKernel,
    parent: &mut AgentState,
    spec: SubagentSpec,
) -> Result<SubagentReturn> {
    // Bound recursion: refuse to spawn beyond the stack-depth cap (K8).
    if parent.ledger.stack_depth >= parent.budget.max_stack_depth {
        return Ok(SubagentReturn {
            handle: SubagentHandle {
                session_id: parent.session_id.clone(),
                run_id: parent.run_id.clone(),
                spec,
            },
            status: SubagentStatus::Aborted,
            summary: "stack-depth cap reached; not spawned".to_string(),
            lessons: Vec::new(),
            steps_used: 0,
            tool_calls_used: 0,
        });
    }

    let child_budget = spec.budget.clone().unwrap_or_else(|| parent.budget.child());
    let mut child = kernel
        .start_run(parent.session_id.clone(), spec.objective.clone())
        .await?;
    child.budget = child_budget;
    child.ledger.stack_depth = parent.ledger.stack_depth + 1;

    // Record the spawn on the parent stack + ledger.
    parent.stack.push(Frame::Subagent {
        child_run: child.run_id.clone(),
        objective: spec.objective.clone(),
    });
    parent.ledger.subagents_total += 1;
    parent.ledger.subagents_live += 1;

    // Drive the child to terminal, bounded by its own step cap.
    for _ in 0..spec.max_steps.max(1) {
        if child.phase.is_terminal() {
            break;
        }
        kernel.step(&mut child).await?;
    }

    let status = match child.phase {
        Phase::Done => SubagentStatus::Ok,
        Phase::Aborted => SubagentStatus::Aborted,
        _ => SubagentStatus::Partial,
    };
    let summary = format!(
        "subagent '{}' finished in phase {:?} after {} steps",
        spec.name, child.phase, child.ledger.steps
    );

    // Join: roll the child's budget usage up, pop the frame.
    parent.ledger.steps += child.ledger.steps;
    parent.ledger.tool_calls += child.ledger.tool_calls;
    parent.ledger.subagents_live = parent.ledger.subagents_live.saturating_sub(1);
    parent.stack.pop();

    Ok(SubagentReturn {
        handle: SubagentHandle {
            session_id: child.session_id.clone(),
            run_id: child.run_id.clone(),
            spec,
        },
        status,
        summary,
        lessons: child.lessons.clone(),
        steps_used: child.ledger.steps,
        tool_calls_used: child.ledger.tool_calls,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::event::InMemoryEventLog;
    use hide_core::ids::SessionId;
    use std::sync::Arc;

    #[tokio::test]
    async fn subagent_runs_and_rolls_budget_up() {
        let log = Arc::new(InMemoryEventLog::new());
        let kernel = AgentKernel::new(log);
        let mut parent = kernel
            .start_run(SessionId::new(), "parent objective")
            .await
            .unwrap();
        let before = parent.ledger.steps;
        let ret = spawn_and_join(
            &kernel,
            &mut parent,
            SubagentSpec {
                name: "child".into(),
                objective: "do a small thing".into(),
                kind: SubagentKind::Research,
                isolation: IsolationMode::Context,
                budget: None,
                max_steps: 40,
            },
        )
        .await
        .unwrap();
        assert_eq!(ret.status, SubagentStatus::Ok);
        assert!(ret.steps_used > 0);
        assert!(parent.ledger.steps >= before + ret.steps_used);
        assert_eq!(parent.ledger.subagents_total, 1);
        assert_eq!(parent.ledger.subagents_live, 0);
        assert!(parent.stack.is_empty());
    }
}
