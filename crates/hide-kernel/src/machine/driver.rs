//! The real FSM driver (bible ch.02 §4.4) — the agent loop.
//!
//! No state advances on faith (K1): every step declares its acceptance oracle,
//! `Act` actually performs the step (dispatches a tool or calls the model),
//! `Verify` runs those oracles and the gate decides, and `Repair`/`Replan`/
//! `Paused` execute their budgeted loops. The Governor (K8) gates every
//! transition. In `Replay` mode (K5) effects do not run — recorded outcomes are
//! folded instead.

use crate::govern::{AbortReason, EffectAuthorization, GovernDecision, Governor};
use crate::machine::effects::{action_event, observation_event, state_event, Mode};
use crate::machine::guards::{
    cursor_is_effectful, cursor_step, plan_has_ready_step, plan_is_acyclic, repairs_remaining,
};
use crate::machine::state::{AgentState, ApprovalRequest, Phase};
use crate::plan::dag::PlanDag;
use crate::plan::planner::Planner;
use crate::plan::replan::{localized_replan, supersede, ReplanRequest};
use crate::plan::schema::{PlanStep, StepKind, StepStatus};
use crate::verify::gate::{GateDecision, VerificationGate};
use crate::verify::oracle::{Failure, VerdictStatus, VerificationInput};
use crate::verify::OracleSuite;
use crate::Grounding;
use crate::runtime_client::KernelRuntimeClient;
use hide_core::event::{NewEvent, PlanEvent};
use hide_core::ids::now_ms;
use hide_core::persistence::DynEventLog;
use hide_core::runtime::{InferenceRequest, StreamChunk};
use hide_core::tool::{ToolCall, ToolDispatcher};
use hide_core::{HideError, Result};
use serde_json::json;
use std::collections::BTreeMap;

/// The driver borrows the kernel's long-lived components for one transition.
pub struct AgentDriver<'a> {
    pub events: DynEventLog,
    pub planner: &'a dyn Planner,
    pub suite: &'a OracleSuite,
    pub gate: &'a VerificationGate,
    pub governor: &'a mut Governor,
    pub runtime: Option<&'a KernelRuntimeClient>,
    pub dispatcher: Option<&'a ToolDispatcher>,
    pub grounding: Option<&'a Grounding>,
    pub workspace_root: String,
    pub mode: Mode,
}

impl<'a> AgentDriver<'a> {
    /// Advance the machine one transition.
    pub async fn step(&mut self, state: &mut AgentState) -> Result<()> {
        // K8: the Governor gates every transition first.
        match self.governor.check(state, now_ms()) {
            GovernDecision::Proceed => {}
            GovernDecision::Abort(reason) => {
                self.abort(state, reason).await?;
                return Ok(());
            }
            GovernDecision::Pause(detail) => {
                state.phase = Phase::Paused;
                self.emit_phase(state, detail).await?;
                return Ok(());
            }
        }
        state.ledger.consume_step();

        match state.phase {
            Phase::Intake => {
                state.phase = Phase::Plan;
                self.emit_phase(state, "intake complete").await?;
            }
            Phase::Plan => self.do_plan(state).await?,
            Phase::SelectStep => self.do_select(state).await?,
            Phase::Act => self.do_act(state).await?,
            Phase::Observe => {
                self.emit_phase(state, "observation recorded as data").await?;
                state.phase = Phase::Verify;
            }
            Phase::Verify => self.do_verify(state).await?,
            Phase::Repair => self.do_repair(state).await?,
            Phase::Replan => self.do_replan(state).await?,
            Phase::Paused => self.do_paused(state).await?,
            Phase::Finalize => {
                state.phase = Phase::Done;
                self.emit_phase(state, "run finalized").await?;
            }
            Phase::Done | Phase::Aborted => {
                // Terminal — nothing to do.
            }
        }
        Ok(())
    }

    // --- PLAN: call the planner, gate on dag.acyclic() ----------------------

    async fn do_plan(&mut self, state: &mut AgentState) -> Result<()> {
        let mut plan = self.planner.synthesize(&state.objective).await?;
        plan.budget = state.budget.clone();
        // §4.5.2: a cyclic plan is invalid — replan instead of executing it.
        if !PlanDag::acyclic(&plan) {
            self.events
                .append(NewEvent::plan(
                    state.session_id.clone(),
                    state.run_id.clone(),
                    PlanEvent {
                        action: "rejected_cyclic".to_string(),
                        step_id: None,
                        plan: Some(serde_json::to_value(&plan)?),
                    },
                ))
                .await?;
            state.phase = Phase::Replan;
            self.emit_phase(state, "plan is cyclic; replanning").await?;
            return Ok(());
        }
        self.events
            .append(NewEvent::plan(
                state.session_id.clone(),
                state.run_id.clone(),
                PlanEvent {
                    action: "created".to_string(),
                    step_id: None,
                    plan: Some(serde_json::to_value(&plan)?),
                },
            ))
            .await?;
        state.plan = Some(plan);
        state.phase = Phase::SelectStep;
        self.emit_phase(state, "plan synthesized").await?;
        Ok(())
    }

    // --- SELECT_STEP: next ready step + guards -------------------------------

    async fn do_select(&mut self, state: &mut AgentState) -> Result<()> {
        if !plan_is_acyclic(state) {
            state.phase = Phase::Replan;
            self.emit_phase(state, "plan became cyclic").await?;
            return Ok(());
        }
        if !plan_has_ready_step(state) {
            // No ready steps. If anything failed, finalize honestly; else done.
            state.phase = Phase::Finalize;
            self.emit_phase(state, "no ready steps remain").await?;
            return Ok(());
        }
        let plan = state
            .plan
            .as_ref()
            .ok_or_else(|| HideError::InvalidState("select without plan".to_string()))?;
        let next = PlanDag::ready_steps(plan)
            .into_iter()
            .next()
            .ok_or_else(|| HideError::InvalidState("ready set vanished".to_string()))?;
        state.cursor = Some(next);
        state.mark_cursor(StepStatus::Running);

        // Autonomy gate: an effectful step under suggest-only/read-only must
        // pause for approval (§4.3) before it can act.
        if cursor_is_effectful(state) {
            match self.governor.may_run_effect() {
                EffectAuthorization::Allow => {}
                EffectAuthorization::NeedsApproval => {
                    let step = cursor_step(state).cloned();
                    state.pending_approval = step.as_ref().map(|s| ApprovalRequest {
                        step_id: s.id.clone(),
                        summary: s.title.clone(),
                        effects: vec![format!("{:?}", s.kind)],
                    });
                    state.phase = Phase::Paused;
                    self.emit_phase(state, "effectful step awaits approval").await?;
                    return Ok(());
                }
                EffectAuthorization::Forbidden => {
                    // read-only: skip the effectful step, mark it skipped.
                    state.mark_cursor(StepStatus::Skipped);
                    state.cursor = None;
                    state.phase = Phase::SelectStep;
                    self.emit_phase(state, "effectful step skipped (read-only)").await?;
                    return Ok(());
                }
            }
        }

        // Ground the step's context (uses the index/context seam if present).
        self.ground_cursor(state).await?;
        state.phase = Phase::Act;
        self.emit_phase(state, "selected ready step").await?;
        Ok(())
    }

    /// Ground the current step with codebase context (imports the
    /// context/index crates the audit flagged as declared-but-unused).
    async fn ground_cursor(&mut self, state: &mut AgentState) -> Result<()> {
        let Some(grounding) = self.grounding else {
            return Ok(());
        };
        let task = cursor_step(state)
            .map(|s| s.title.clone())
            .unwrap_or_else(|| state.objective.clone());
        if let Ok(Some(manifest_hash)) = grounding.compile(&task).await {
            state.context_manifest = Some(manifest_hash);
        }
        Ok(())
    }

    // --- ACT: actually do the step ------------------------------------------

    async fn do_act(&mut self, state: &mut AgentState) -> Result<()> {
        let step = cursor_step(state)
            .cloned()
            .ok_or_else(|| HideError::InvalidState("act without cursor".to_string()))?;

        // Bump attempt count on the live plan.
        if let (Some(plan), Some(cursor)) = (state.plan.as_mut(), state.cursor.as_ref()) {
            if let Some(s) = plan.step_mut(cursor) {
                s.attempts += 1;
            }
        }

        // Replay: do not run effects — fold the recorded Observation outcome.
        if self.mode.is_replay() {
            self.emit_phase(state, "replay: folding recorded outcome").await?;
            state.phase = Phase::Observe;
            return Ok(());
        }

        // Emit the Action-class event; its outcome will be an Observation
        // carrying `cause` = this action's id (replay pairing, T3).
        let action = self
            .events
            .append(action_event(
                state.session_id.clone(),
                state.run_id.clone(),
                "agent.action",
                json!({ "step_id": step.id, "kind": format!("{:?}", step.kind) }),
            ))
            .await?;

        let outcome = match step.kind {
            StepKind::Edit | StepKind::Command => {
                let dispatched = self.act_tool(&step).await;
                // K4/K8: count the tool-call against the budget when (and only
                // when) a tool was actually dispatched, so `max_tool_calls` can
                // trip. The Governor's check on the next transition reads this.
                if let Ok((_, true)) = &dispatched {
                    state.ledger.consume_tool_call();
                }
                dispatched.map(|(value, _)| value)
            }
            StepKind::Investigate | StepKind::Synthesize | StepKind::Verify => {
                self.act_model(&step).await
            }
            StepKind::Decompose | StepKind::Delegate => {
                // Decompose/delegate are model-driven boundaries here.
                self.act_model(&step).await
            }
        };

        let outcome_json = match outcome {
            Ok(value) => value,
            Err(err) => json!({ "error": err.to_string() }),
        };
        self.events
            .append(observation_event(
                state.session_id.clone(),
                state.run_id.clone(),
                "agent.observation",
                action.id.clone(),
                outcome_json,
            ))
            .await?;
        state.phase = Phase::Observe;
        Ok(())
    }

    /// Effectful step: dispatch the declared tool through the permission-gated
    /// dispatcher. EXEC_NONZERO is data, so a failing build is still a normal
    /// observation (the Verify gate, not Act, judges correctness).
    ///
    /// Returns `(outcome, dispatched)`. `dispatched` is `true` only when a real
    /// tool was actually sent through the dispatcher — the caller consumes a
    /// tool-call against the budget exactly then (so `max_tool_calls` trips), and
    /// not for the no-dispatcher / model-authored-edit fallbacks.
    async fn act_tool(&self, step: &PlanStep) -> Result<(serde_json::Value, bool)> {
        let Some(dispatcher) = self.dispatcher else {
            return Ok((json!({ "note": "no dispatcher; step recorded without effect" }), false));
        };
        let tool = match &step.tool_hint {
            Some(t) => t.clone(),
            // No explicit tool: an edit step with no tool is a model-authored
            // change recorded as an observation (the oracles verify the result).
            None => return self.act_model(step).await.map(|v| (v, false)),
        };
        let mut args = step.tool_args.clone().unwrap_or_else(|| json!({}));
        if args.get("cwd").is_none() {
            args["cwd"] = json!(self.workspace_root);
        }
        let result = dispatcher.dispatch(ToolCall::new(tool.clone(), args)).await?;
        Ok((
            json!({
                "tool": tool,
                "ok": result.ok,
                "exit_code": result.exit_code,
                "structured": result.structured_content,
            }),
            true,
        ))
    }

    /// Model step: call the runtime to generate (Investigate/Synthesize/Verify).
    async fn act_model(&self, step: &PlanStep) -> Result<serde_json::Value> {
        let Some(runtime) = self.runtime else {
            return Ok(json!({ "note": "no runtime; step recorded without generation" }));
        };
        let prompt = format!(
            "Step: {}\nGoal: {}\n{}",
            step.title, step.acceptance.predicate, step.rationale
        );
        let request = InferenceRequest {
            task_kind: "code".to_string(),
            prompt,
            messages: Vec::new(),
            max_output_tokens: 512,
            sampler: None,
            grammar: None,
            want_logprobs: false,
            metadata: BTreeMap::new(),
        };
        let mut buf = String::new();
        let mut sink = |chunk: StreamChunk| {
            if let StreamChunk::Token { text, .. } = chunk {
                buf.push_str(&text);
            }
            Ok(())
        };
        let stats = runtime.generate(request, &mut sink).await?;
        Ok(json!({
            "generated": buf,
            "input_tokens": stats.input_tokens,
            "output_tokens": stats.output_tokens,
        }))
    }

    // --- VERIFY: run the step's oracles + the gate --------------------------

    async fn do_verify(&mut self, state: &mut AgentState) -> Result<()> {
        let step = cursor_step(state)
            .cloned()
            .ok_or_else(|| HideError::InvalidState("verify without cursor".to_string()))?;
        // Mark entry into VERIFY (so the phase is observable in the event log even
        // when the gate's decision immediately transitions onward).
        self.emit_phase(state, "running acceptance oracles").await?;

        let mut input = VerificationInput::new(self.workspace_root.clone());
        input.step_id = Some(step.id.to_string());
        input.tests = step.acceptance.tests.clone();

        let verdicts = self.suite.run(&step.acceptance.oracles, &input).await?;

        // Emit each verdict as a verify.result event.
        for v in &verdicts {
            self.events
                .append(crate::machine::effects::custom_agent_event(
                    state.session_id.clone(),
                    state.run_id.clone(),
                    "verify.result",
                    serde_json::to_value(v)?,
                ))
                .await?;
        }
        state.last_verdict = verdicts.last().cloned();
        state.last_verdicts = verdicts.clone();

        // Soft step (the escape hatch — semantics, read carefully):
        //
        // This branch accepts a step *without any machine verification*. It fires
        // ONLY when ALL of:
        //   1. the step declared no oracle ids,
        //   2. no verdict ran at all (no probabilistic oracle was wired — the
        //      unknown-id markers from `OracleSuite::run` would land here too, so
        //      an empty set really does mean "nothing to check"), AND
        //   3. the step is NON-effectful (investigate/synthesize/verify) — it
        //      produced output but mutated nothing.
        //
        // K1 ("no state advances on faith") binds *effectful* steps with declared
        // verifiers; a read-only step that wrote no artifact has nothing to verify,
        // so accepting it is not faith — there is no claim to check. The default
        // `StubPlanner` emits exactly such a step, so the minimal kernel can reach
        // `Done` through here; we record an auditable `verify.soft_accept` event so
        // that "verified nothing" is never invisible in the log.
        //
        // An EFFECTFUL step with no declared oracle must NOT reach this branch:
        // the `!step.is_effectful()` guard sends it to the gate, which returns
        // Inconclusive on an empty verdict set (never Accept) — so it repairs or
        // replans rather than silently passing.
        if step.acceptance.oracles.is_empty() && verdicts.is_empty() && !step.is_effectful() {
            self.events
                .append(crate::machine::effects::custom_agent_event(
                    state.session_id.clone(),
                    state.run_id.clone(),
                    "verify.soft_accept",
                    json!({
                        "step_id": step.id,
                        "kind": format!("{:?}", step.kind),
                        "reason": "non-effectful step with no declared oracle and no oracle ran",
                    }),
                ))
                .await?;
            state.mark_cursor(StepStatus::Completed);
            state.cursor = None;
            state.phase = Phase::SelectStep;
            self.emit_phase(state, "soft step accepted (no oracle applies)").await?;
            return Ok(());
        }

        match self.gate.decide(&verdicts) {
            GateDecision::Accept => {
                state.mark_cursor(StepStatus::Completed);
                state.cursor = None;
                state.phase = Phase::SelectStep;
                self.emit_phase(state, "verification passed").await?;
            }
            GateDecision::Repair | GateDecision::Inconclusive => {
                if repairs_remaining(state) {
                    state.phase = Phase::Repair;
                    self.emit_phase(state, "verification failed; repairing").await?;
                } else {
                    // Repairs exhausted → replan (the approach may be wrong).
                    state.phase = Phase::Replan;
                    self.emit_phase(state, "repairs exhausted; replanning").await?;
                }
            }
            GateDecision::Replan => {
                state.phase = Phase::Replan;
                self.emit_phase(state, "gate requested replan").await?;
            }
            GateDecision::Abort => {
                self.abort(state, AbortReason::Steps("gate aborted".to_string())).await?;
            }
        }
        Ok(())
    }

    // --- REPAIR: minimal-context re-attempt ---------------------------------

    async fn do_repair(&mut self, state: &mut AgentState) -> Result<()> {
        // Record the repair attempt + distill a lesson from the structured
        // failures (the minimal-repair context, §4.7.1).
        let failures: Vec<Failure> = state
            .last_verdicts
            .iter()
            .filter(|v| v.status == VerdictStatus::Fail)
            .flat_map(|v| v.failures.clone())
            .collect();
        let lesson = lesson_from_failures(&failures);
        if let Some(l) = &lesson {
            state.lessons.push(l.clone());
        }

        // Bump the repair count for the cursor step.
        if let Some(cursor) = state.cursor.clone() {
            let n = state.repair_count.entry(cursor.clone()).or_insert(0);
            *n += 1;
            if let Some(plan) = state.plan.as_mut() {
                if let Some(s) = plan.step_mut(&cursor) {
                    s.repairs += 1;
                    s.status = StepStatus::Running;
                }
            }
        }

        self.events
            .append(crate::machine::effects::custom_agent_event(
                state.session_id.clone(),
                state.run_id.clone(),
                "repair.attempt",
                json!({
                    "step_id": state.cursor,
                    "failures": failures,
                    "lesson": lesson,
                }),
            ))
            .await?;

        // Re-attempt the same step (Act re-runs with the lesson now in state).
        state.phase = Phase::Act;
        self.emit_phase(state, "re-attempting step with failure context").await?;
        Ok(())
    }

    // --- REPLAN: localized vs full ------------------------------------------

    async fn do_replan(&mut self, state: &mut AgentState) -> Result<()> {
        state.replan_count = state.replan_count.saturating_add(1);
        state.ledger.consume_replan();

        // Bounded by the replan budget — the Governor would abort on the next
        // check, but we finalize honestly here rather than spin.
        if state.replan_count > state.budget.max_replans {
            state.phase = Phase::Finalize;
            self.emit_phase(state, "replan budget exhausted; finalizing honestly").await?;
            return Ok(());
        }

        let reason = state
            .last_verdict
            .as_ref()
            .map(|v| v.detail.clone())
            .unwrap_or_else(|| "verification could not pass".to_string());
        let lesson = state.lessons.last().cloned();

        let new_plan = match &state.plan {
            Some(plan) if state.replan_count <= 1 => {
                // Localized first: revise from the failure point.
                let req = ReplanRequest {
                    failed_step: state.cursor.clone(),
                    reason: reason.clone(),
                    lesson: lesson.clone(),
                    local_only: true,
                };
                let result = localized_replan(plan, &req);
                self.events
                    .append(crate::machine::effects::custom_agent_event(
                        state.session_id.clone(),
                        state.run_id.clone(),
                        "plan.replanned",
                        json!({ "mode": "localized", "changed": result.changed_steps, "reason": reason }),
                    ))
                    .await?;
                result.plan
            }
            _ => {
                // Full replan: supersede the old plan and resynthesize, carrying
                // the lesson into the objective.
                if let Some(old) = state.plan.take() {
                    let superseded = supersede(old);
                    self.events
                        .append(NewEvent::plan(
                            state.session_id.clone(),
                            state.run_id.clone(),
                            PlanEvent {
                                action: "superseded".to_string(),
                                step_id: None,
                                plan: Some(serde_json::to_value(&superseded)?),
                            },
                        ))
                        .await?;
                }
                let objective = match &lesson {
                    Some(l) => format!("{}\n(lesson from prior attempt: {l})", state.objective),
                    None => state.objective.clone(),
                };
                let mut plan = self.planner.synthesize(&objective).await?;
                plan.budget = state.budget.clone();
                self.events
                    .append(crate::machine::effects::custom_agent_event(
                        state.session_id.clone(),
                        state.run_id.clone(),
                        "plan.replanned",
                        json!({ "mode": "full", "reason": reason }),
                    ))
                    .await?;
                plan
            }
        };

        // A replanned plan must still be acyclic.
        if !PlanDag::acyclic(&new_plan) {
            self.abort(state, AbortReason::Steps("replan produced a cyclic plan".to_string()))
                .await?;
            return Ok(());
        }
        state.plan = Some(new_plan);
        state.cursor = None;
        state.phase = Phase::SelectStep;
        self.emit_phase(state, "replanned; reselecting").await?;
        Ok(())
    }

    // --- PAUSED: approval gate + interrupt polling --------------------------

    async fn do_paused(&mut self, state: &mut AgentState) -> Result<()> {
        // The Governor already consumed any pending interrupt in `check`. If the
        // approval was granted out-of-band (pending_approval cleared by the host)
        // resume into Act; otherwise stay paused (idempotent).
        if state.pending_approval.is_none() {
            state.phase = Phase::Act;
            self.emit_phase(state, "approval granted; resuming").await?;
        } else {
            self.emit_phase(state, "awaiting approval").await?;
        }
        Ok(())
    }

    // --- helpers ------------------------------------------------------------

    async fn abort(&mut self, state: &mut AgentState, reason: AbortReason) -> Result<()> {
        state.phase = Phase::Aborted;
        self.events
            .append(crate::machine::effects::custom_agent_event(
                state.session_id.clone(),
                state.run_id.clone(),
                "run.aborted",
                serde_json::to_value(&reason)?,
            ))
            .await?;
        self.emit_phase(state, reason.message().to_string()).await?;
        Ok(())
    }

    async fn emit_phase(&self, state: &AgentState, detail: impl Into<String>) -> Result<()> {
        self.events
            .append(state_event(
                state.session_id.clone(),
                state.run_id.clone(),
                state.phase.wire_name().to_string(),
                detail,
            ))
            .await?;
        Ok(())
    }
}

/// Distill a 1–3 sentence lesson from structured failures (§4.7.2).
fn lesson_from_failures(failures: &[Failure]) -> Option<String> {
    let first = failures.first()?;
    let loc = match (&first.file, first.line) {
        (Some(f), Some(l)) => format!(" at {f}:{l}"),
        (Some(f), None) => format!(" in {f}"),
        _ => String::new(),
    };
    let code = first
        .code
        .as_ref()
        .map(|c| format!(" [{c}]"))
        .unwrap_or_default();
    Some(format!(
        "Prior attempt failed{loc}{code}: {} (category: {}).",
        first.message.lines().next().unwrap_or(&first.message),
        first.category
    ))
}
