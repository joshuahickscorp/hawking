//! Headless HIDE agent-run audit.
//!
//! This is intentionally a thin driver over the real [`BackendHost`] and
//! [`hide_kernel::AgentKernel`], not a second agent implementation. It creates a
//! sealed JSON receipt containing the goal, wall time, runtime decode metrics,
//! token ledger, tool/verification activity, context facts, storage capability,
//! and event-chain integrity. Missing runtime capabilities remain explicit
//! `blocked`/`unavailable` facts rather than synthetic benchmark values.

use crate::{
    hcli_profile::HcliProfile, hcli_sources::HcliSourceContext, host::turn_kernel_autonomy,
    BackendHost, HttpModelProvider,
};
use hide_core::event::Event;
use hide_core::ids::{now_ms, SessionId};
use hide_core::objects::{StorageBudget, BOUND_STATEMENT};
use hide_core::Result;
use hide_kernel::machine::state::{AgentState, Phase};
use serde_json::{json, Value};
use std::path::Path;
use std::time::{Duration, Instant};

pub const HEADLESS_RECEIPT_SCHEMA: &str = "hide.headless.audit.v1";

/// Inputs to one auditable headless agent run.
#[derive(Debug, Clone)]
pub struct HeadlessRunConfig {
    /// The literal goal submitted to the real agent kernel.
    pub goal: String,
    /// A running `hawking serve` base URL. The runner will not silently fall
    /// back to a stub when it is absent or unhealthy.
    pub model_url: Option<String>,
    /// Optional durable session selected by an external controller.  When
    /// absent, the headless driver mints one fresh session per run so parallel
    /// swarm lanes cannot accidentally share transcript state.
    pub session_id: Option<SessionId>,
    /// Outer transition cap for the driver. The kernel's own governor remains
    /// authoritative for steps, wall-clock, effects, and tool calls.
    pub max_transitions: u32,
    /// Named, finite HCLI compute profile. This is applied to the freshly
    /// created [`AgentState`] before its first transition, so the receipt's
    /// budget is a statement about the run that actually happened rather than
    /// a display-only preset.
    pub profile: HcliProfile,
    /// Explicit bounded local evidence derivatives for this run. They are
    /// injected only into actual agent act-model prompts as untrusted reference
    /// material; they do not alter the durable user objective or grant tools.
    pub source_context: Option<HcliSourceContext>,
}

impl Default for HeadlessRunConfig {
    fn default() -> Self {
        Self {
            goal: String::new(),
            model_url: None,
            session_id: None,
            max_transitions: 200,
            profile: HcliProfile::Balanced,
            source_context: None,
        }
    }
}

/// Terminal classification for the headless command. `Completed` is the only
/// successful agent outcome; `Paused` and `StepLimit` are useful receipts but
/// deliberately do not masquerade as a completed task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadlessRunStatus {
    Completed,
    Paused,
    StepLimit,
    BlockedNoModelUrl,
    BlockedRuntimeUnreachable,
    Failed,
}

impl HeadlessRunStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Paused => "paused",
            Self::StepLimit => "step_limit",
            Self::BlockedNoModelUrl => "blocked_no_model_url",
            Self::BlockedRuntimeUnreachable => "blocked_runtime_unreachable",
            Self::Failed => "failed",
        }
    }

    pub fn is_complete(self) -> bool {
        matches!(self, Self::Completed)
    }
}

/// Result returned to a CLI or a lab harness. The JSON has already been sealed
/// with `content_blake3`; callers may persist it with [`write_sealed_receipt`].
#[derive(Debug, Clone)]
pub struct HeadlessRunResult {
    pub status: HeadlessRunStatus,
    pub receipt: Value,
}

/// Drive the real headless kernel if—and only if—the requested runtime passes a
/// local `/healthz` preflight. A missing or unavailable model produces a useful,
/// sealed blocked receipt and performs no synthetic agent work.
pub async fn run_headless_audit(
    host: &BackendHost,
    config: HeadlessRunConfig,
) -> Result<HeadlessRunResult> {
    let started_ms = now_ms();
    let started = Instant::now();
    let max_transitions = config.max_transitions.clamp(1, 2_000);
    let profile_spec = config.profile.spec();
    let kernel_autonomy = turn_kernel_autonomy();
    let requested_autonomy = std::env::var("HIDE_KERNEL_AUTONOMY").ok();
    let host_status = host.status().await;
    let model_url = config
        .model_url
        .as_deref()
        .map(str::trim)
        .filter(|url| !url.is_empty())
        .map(str::to_string);

    let mut runtime = json!({
        "requested_url": model_url,
        "health": { "status": "not_checked" },
        "context_before": { "status": "not_checked" },
        "context_after": { "status": "not_run" },
        "model_tps_claim": "not_measured",
    });
    let mut state: Option<AgentState> = None;
    let mut session_id: Option<SessionId> = None;
    let mut transitions = 0u32;
    let mut failure: Option<String> = None;
    let mut runtime_output_cap: Option<usize> = None;
    let mut compact_model_prompts = false;
    let mut source_context_omitted_for_window = false;

    let status = match model_url.as_deref() {
        None => HeadlessRunStatus::BlockedNoModelUrl,
        Some(url) => {
            let health = probe_health(url).await;
            runtime["health"] = health.clone();
            let provider = HttpModelProvider::new(url);
            let context_before = provider.get_context_info().await;
            runtime_output_cap = context_before
                .as_ref()
                .and_then(|info| info.max_output_tokens)
                .filter(|cap| *cap > 0);
            compact_model_prompts = context_before.as_ref().is_some_and(|info| {
                info.ctx_len_effective
                    .or(info.ctx_len_native)
                    .is_some_and(|window| window <= 128)
            });
            runtime["context_before"] = context_snapshot(context_before);
            if health.get("ready").and_then(Value::as_bool) != Some(true) {
                HeadlessRunStatus::BlockedRuntimeUnreachable
            } else {
                let headless_session = config.session_id.clone().unwrap_or_default();
                let kernel = host.build_headless_kernel(
                    url.to_string(),
                    headless_session.clone(),
                    runtime_output_cap,
                    compact_model_prompts,
                );
                match kernel
                    .start_run(headless_session.clone(), config.goal.clone())
                    .await
                {
                    Ok(mut live_state) => {
                        if let Some(source_context) = config.source_context.as_ref() {
                            if compact_model_prompts {
                                source_context_omitted_for_window = true;
                            } else {
                                live_state.set_supplemental_reference_context(
                                    source_context.model_prompt().to_string(),
                                    source_context.model_prompt_tokens(),
                                );
                            }
                        }
                        // The profile must land before the first `step`: the
                        // state governor, not a CLI label, is authoritative.
                        config.profile.apply_to_state(&mut live_state);
                        session_id = Some(headless_session);
                        let mut terminal = false;
                        for _ in 0..max_transitions {
                            match kernel.step(&mut live_state).await {
                                Ok(()) => {
                                    transitions = transitions.saturating_add(1);
                                    if live_state.phase.is_terminal()
                                        || matches!(live_state.phase, Phase::Paused)
                                    {
                                        terminal = true;
                                        break;
                                    }
                                }
                                Err(error) => {
                                    failure = Some(error.to_string());
                                    break;
                                }
                            }
                        }
                        runtime["context_after"] =
                            context_snapshot(provider.get_context_info().await);
                        let final_status = if failure.is_some() {
                            HeadlessRunStatus::Failed
                        } else if matches!(live_state.phase, Phase::Done) {
                            HeadlessRunStatus::Completed
                        } else if matches!(live_state.phase, Phase::Paused) {
                            HeadlessRunStatus::Paused
                        } else if terminal || live_state.phase.is_terminal() {
                            HeadlessRunStatus::Failed
                        } else {
                            HeadlessRunStatus::StepLimit
                        };
                        state = Some(live_state);
                        final_status
                    }
                    Err(error) => {
                        failure = Some(error.to_string());
                        HeadlessRunStatus::Failed
                    }
                }
            }
        }
    };

    let events = host.services.event_log.scan(None, None, None).await?;
    let session_events: Vec<Event> = match &session_id {
        Some(id) => events
            .iter()
            .filter(|event| &event.session_id == id)
            .cloned()
            .collect(),
        None => Vec::new(),
    };
    let integrity = host.services.event_integrity.verify_chain(&events)?;
    let duration_ms = started.elapsed().as_millis() as u64;
    let agent = state
        .as_ref()
        .map(|live_state| agent_summary(live_state, &session_events, transitions));
    let source_context = source_context_receipt(
        config.source_context.as_ref(),
        state.as_ref(),
        &session_events,
        source_context_omitted_for_window,
    );

    let storage = StorageBudget::default();
    let mut receipt = json!({
        "schema": HEADLESS_RECEIPT_SCHEMA,
        "status": status.as_str(),
        "started_ms": started_ms,
        "finished_ms": now_ms(),
        "wall_elapsed_ms": duration_ms,
        "driver": {
            "max_transitions_requested": config.max_transitions,
            "max_transitions_effective": max_transitions,
            "compute_profile": profile_spec,
            "kernel_autonomy": kernel_autonomy,
            "kernel_autonomy_environment": requested_autonomy,
            "note": "Autonomy controls effect approval only. It does not make search_breadth or self_consistency_k into an implemented multi-agent swarm.",
        },
        "goal": {
            "text": config.goal,
            "blake3": blake3::hash(config.goal.as_bytes()).to_hex().to_string(),
            "utf8_bytes": config.goal.len(),
        },
        "runtime": runtime,
        "host": {
            "workspace_root": host.services.config.workspace_root,
            "crate_version": env!("CARGO_PKG_VERSION"),
            "status": host_status,
        },
        "agent": agent,
        "event_window": {
            "session_id": session_id,
            "event_count": session_events.len(),
            "first_seq": session_events.first().map(|event| event.seq),
            "last_seq": session_events.last().map(|event| event.seq),
            "note": "The event-chain audit covers the full workspace log. This session window includes direct planned-tool events whose run_id can be null by design.",
        },
        "context": {
            "normal_turn_configured_input_tokens": host.services.config.context.max_input_tokens,
            "normal_turn_reserved_output_tokens": host.services.config.context.reserve_output_tokens,
            "kernel_grounding": {
                "descriptor_context_tokens": 8192,
                "manifest_hash_recorded": state.as_ref().and_then(|s| s.context_manifest.clone()),
                "compiled_context_injected_into_model_prompt": state.as_ref().is_some_and(|s| s.context_prompt.is_some()),
                "compiled_context_used_tokens": state.as_ref().and_then(|s| s.context_used_tokens),
                "compiled_context_retained_span_count": state.as_ref().and_then(|s| s.context_retained_span_count),
                "status": if state.as_ref().is_some_and(|s| s.context_prompt.is_some()) { "injected" } else { "not_compiled_or_not_reached" },
                "note": "When a selected step compiles a non-empty pack, the exact packed context is injected into that step's act-model prompt as reference material. The receipt stores its manifest and counts, not the raw context text.",
            },
            "output_limits": {
                "planner_requested_max_output_tokens": 256,
                "agent_act_requested_max_output_tokens": 512,
                "endpoint_max_output_tokens": runtime_output_cap,
                "effective_model_output_cap": runtime_output_cap,
            },
            "live_low_context_mode": {
                "enabled": compact_model_prompts,
                "reason": if compact_model_prompts {
                    "endpoint reported a native/effective context at or below 128 tokens"
                } else {
                    "not_required_by_observed_endpoint_context"
                },
                "grounding": if compact_model_prompts { "omitted_whole_block" } else { "normal_policy" },
                "tool_catalog": if compact_model_prompts { "omitted_whole_block" } else { "normal_policy" },
                "selected_source_context": if source_context_omitted_for_window { "omitted_whole_block" } else { "normal_policy_or_not_requested" },
            },
            "explicit_source_context": source_context,
        },
        "uploads": {
            "http_upload_to_context": false,
            "explicit_local_evidence_attachment": config.source_context.is_some(),
            "literal_unlimited": false,
            "status": if config.source_context.is_some() { "bounded_local_source_attached_or_pending_runtime" } else { "not_requested" },
            "note": "HCLI can attach an explicit bounded local object-store derivative pack to agent act-model prompts. It has no HTTP attachment upload route, no implicit cross-turn retention, and no unlimited-upload claim.",
            "declared_storage_budget": {
                "max_local_bytes": storage.max_local_bytes,
                "max_cloud_bytes": storage.max_cloud_bytes,
                "max_object_bytes": storage.max_object_bytes,
                "policy_note": storage.policy_note,
                "bound_statement": BOUND_STATEMENT,
            },
        },
        "event_chain": integrity,
        "failure": failure,
        "limitations": [
            "A V4 TPS result is valid only when the runtime context identifies a loadable deepseek_v4 artifact and its per-call decode metrics include completed_decode_forwards plus decode_ms.",
            "Agent wall time includes planning, verification, tool dispatch, filesystem work, and scheduling; it is not model decode TPS.",
            "Kernel packed-context injection is evidenced only when kernel_grounding.status is injected; a missing/empty pack is not treated as a long-context retention result.",
            "Only explicitly selected local evidence derivatives can reach this agent's act-model prompts; HTTP uploads, arbitrary URLs, and unlimited capacity are not implemented.",
        ],
    });
    seal(&mut receipt)?;
    Ok(HeadlessRunResult { status, receipt })
}

/// Persist a receipt atomically. The caller owns the target path; this helper
/// never writes a model artifact or changes the workspace outside that path.
pub fn write_sealed_receipt(path: impl AsRef<Path>, receipt: &Value) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(receipt)?)?;
    std::fs::rename(tmp, path)?;
    Ok(())
}

fn seal(receipt: &mut Value) -> Result<()> {
    let bytes = serde_json::to_vec(receipt)?;
    receipt["content_blake3"] = json!(blake3::hash(&bytes).to_hex().to_string());
    Ok(())
}

async fn probe_health(base_url: &str) -> Value {
    let url = format!("{}/healthz", base_url.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap_or_default();
    match client.get(&url).send().await {
        Ok(response) => {
            let status = response.status();
            json!({
                "status": if status.is_success() { "ready" } else { "unhealthy" },
                "ready": status.is_success(),
                "http_status": status.as_u16(),
                "url": url,
            })
        }
        Err(error) => json!({
            "status": "unreachable",
            "ready": false,
            "url": url,
            "error": error.to_string(),
        }),
    }
}

fn context_snapshot(info: Option<crate::model_provider::ContextInfo>) -> Value {
    match info {
        Some(info)
            if !info.model_id.is_empty()
                || !info.arch.is_empty()
                || info.ctx_len_native.is_some()
                || info.ctx_len_effective.is_some()
                || info.recurrent_state_bytes.is_some() =>
        {
            json!({
                "status": "available",
                "model_id": info.model_id,
                "arch": info.arch,
                "ctx_len_native": info.ctx_len_native,
                "ctx_len_effective": info.ctx_len_effective,
                "tq_multiplier": info.tq_multiplier,
                "tq_estimated": info.tq_estimated,
                "recurrent_state_bytes": info.recurrent_state_bytes,
                "active_slots": info.active_slots,
                "free_slots": info.free_slots,
                "max_batch": info.max_batch,
                "max_output_tokens": info.max_output_tokens,
                "artifact_seal_sha256": info.artifact_seal_sha256,
                "capability_status": info.capability_status,
                "metal_dispatches": info.metal_dispatches,
                "chat_template": info.chat_template,
            })
        }
        Some(_) => json!({
            "status": "unavailable",
            "note": "runtime answered /v1/hawking/context without identifiable context fields; no context ceiling is claimed",
        }),
        None => json!({
            "status": "unavailable",
            "note": "runtime did not expose /v1/hawking/context; no context ceiling is claimed",
        }),
    }
}

/// Metadata-only evidence receipt for an agent run. A selected pack is not
/// reported as injected merely because it was parsed: an act-model observation
/// must exist after the live state received the supplemental context.
fn source_context_receipt(
    source_context: Option<&HcliSourceContext>,
    state: Option<&AgentState>,
    events: &[Event],
    omitted_for_live_window: bool,
) -> Value {
    let Some(source_context) = source_context else {
        return json!({
            "status": "not_requested",
            "injected_into_agent_act_model_prompt": false,
        });
    };
    let configured = state.is_some_and(|state| {
        state
            .supplemental_reference_context
            .as_deref()
            .is_some_and(|context| !context.trim().is_empty())
    });
    let act_model_observed = events
        .iter()
        .any(|event| event.kind == "agent.observation" && event.payload.get("generated").is_some());
    let mut receipt = source_context.receipt_json();
    receipt["injection"] = json!({
        "configured_for": "agent_act_model_prompt",
        "configured_in_live_state": configured,
        "act_model_observation_recorded": act_model_observed,
        "injected_into_agent_act_model_prompt": configured && act_model_observed,
        "omitted_for_live_window": omitted_for_live_window,
        "persisted_as_user_objective": false,
        "note": if omitted_for_live_window {
            "The observed live endpoint has a compact context window, so the entire selected evidence block was omitted rather than silently truncated."
        } else if configured && act_model_observed {
            "At least one real agent act-model observation was recorded after the selected evidence pack was installed."
        } else if configured {
            "The pack was installed for act-model prompts, but this run did not record an act-model observation; it is not claimed as consumed."
        } else {
            "The run did not reach a live agent state, so selected evidence was not injected."
        },
    });
    receipt
}

fn agent_summary(state: &AgentState, events: &[Event], transitions: u32) -> Value {
    let mut model_calls = Vec::new();
    let mut input_tokens = 0u64;
    let mut output_tokens = 0u64;
    let mut decode_ms = 0.0f64;
    let mut completed_decode_forwards = 0u64;
    let mut complete_metric_calls = 0u64;
    let mut parsed_model_tool_calls = 0u64;
    let mut dispatched_model_tool_calls = 0u64;

    for event in events {
        let (stage, payload) = match event.kind.as_str() {
            "agent.model_metrics" => (
                event
                    .payload
                    .get("stage")
                    .and_then(Value::as_str)
                    .unwrap_or("planner"),
                &event.payload,
            ),
            "agent.observation" if event.payload.get("generated").is_some() => {
                ("act", &event.payload)
            }
            _ => continue,
        };
        let input = payload
            .get("input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let output = payload
            .get("output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let call_decode_ms = payload.get("decode_ms").and_then(Value::as_f64);
        let call_forwards = payload
            .get("completed_decode_forwards")
            .and_then(Value::as_u64);
        let call_tps = payload.get("decode_tps").and_then(Value::as_f64);
        input_tokens = input_tokens.saturating_add(input);
        output_tokens = output_tokens.saturating_add(output);
        if let (Some(ms), Some(forwards)) = (call_decode_ms, call_forwards) {
            if ms > 0.0 && forwards > 0 {
                decode_ms += ms;
                completed_decode_forwards = completed_decode_forwards.saturating_add(forwards);
                complete_metric_calls = complete_metric_calls.saturating_add(1);
            }
        }
        if let Some(calls) = payload.get("tool_calls").and_then(Value::as_array) {
            parsed_model_tool_calls = parsed_model_tool_calls.saturating_add(calls.len() as u64);
            dispatched_model_tool_calls = dispatched_model_tool_calls.saturating_add(
                calls
                    .iter()
                    .filter(|call| call.get("dispatched").and_then(Value::as_bool) == Some(true))
                    .count() as u64,
            );
        }
        model_calls.push(json!({
            "stage": stage,
            "input_tokens": input,
            "output_tokens": output,
            "decode_ms": call_decode_ms,
            "completed_decode_forwards": call_forwards,
            "decode_tps": call_tps,
        }));
    }

    let aggregate_decode_tps = (decode_ms > 0.0 && completed_decode_forwards > 0)
        .then(|| completed_decode_forwards as f64 / (decode_ms / 1_000.0));
    let plan = state.plan.as_ref().map(|plan| {
        json!({
            "id": plan.id,
            "title": plan.title,
            "status": plan.status,
            "steps": plan.steps.iter().map(|step| json!({
                "id": step.id,
                "title": step.title,
                "kind": format!("{:?}", step.kind),
                "status": step.status,
                "attempts": step.attempts,
                "dependencies": step.dependencies,
                "acceptance_oracles": step.acceptance.oracles,
            })).collect::<Vec<_>>(),
        })
    });

    json!({
        "run_id": state.run_id,
        "session_id": state.session_id,
        "phase": state.phase,
        "transitions_executed": transitions,
        "budget": state.budget,
        "ledger": state.ledger,
        "agent_topology": {
            "configured_max_subagents": state.budget.max_subagents,
            "configured_search_breadth": state.budget.search_breadth,
            "configured_self_consistency_k": state.budget.self_consistency_k,
            "actual_subagents_total": state.ledger.subagents_total,
            "actual_subagents_live": state.ledger.subagents_live,
            "note": "These budget values are not evidence of a realized swarm. Delegate remains model-driven today and does not automatically spawn subagents.",
        },
        "plan": plan,
        "model_calls": model_calls,
        "model_metrics": {
            "recorded_call_count": model_calls.len(),
            "recorded_input_tokens": input_tokens,
            "recorded_output_tokens": output_tokens,
            "ledger_input_tokens": state.ledger.input_tokens,
            "ledger_output_tokens": state.ledger.output_tokens,
            "complete_forward_metric_call_count": complete_metric_calls,
            "completed_decode_forwards": completed_decode_forwards,
            "decode_ms": if complete_metric_calls > 0 { Some(decode_ms) } else { None },
            "aggregate_complete_forward_tps": aggregate_decode_tps,
            "tps_authority": if aggregate_decode_tps.is_some() {
                "sum(completed_decode_forwards) / sum(decode_ms)"
            } else {
                "unavailable: runtime did not supply both completed_decode_forwards and decode_ms for one or more claims"
            },
        },
        "tool_activity": {
            "durable_tool_call_events": events.iter().filter(|event| event.kind == "tool.call").count(),
            "parsed_model_tool_calls": parsed_model_tool_calls,
            "dispatched_model_tool_calls": dispatched_model_tool_calls,
        },
        "verification": {
            "verify_result_events": events.iter().filter(|event| event.kind == "verify.result").count(),
            "last_verdict": state.last_verdict,
            "last_verdict_count": state.last_verdicts.len(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::BackendServices;
    use hide_core::config::HideConfig;
    use hide_core::event::InMemoryEventLog;
    use std::sync::Arc;

    #[test]
    fn seals_a_receipt_without_claiming_tps() {
        let mut receipt = json!({ "schema": HEADLESS_RECEIPT_SCHEMA, "status": "blocked" });
        seal(&mut receipt).unwrap();
        assert!(receipt
            .get("content_blake3")
            .and_then(Value::as_str)
            .is_some_and(|hash| hash.len() == 64));
    }

    #[tokio::test]
    async fn missing_model_url_produces_a_sealed_blocked_receipt_without_a_fake_run() {
        let temp = tempfile::tempdir().unwrap();
        let config = HideConfig::for_workspace(temp.path());
        let services = BackendServices::new(config, Arc::new(InMemoryEventLog::new()));
        let host = BackendHost::from_services(services).unwrap();
        let result = run_headless_audit(
            &host,
            HeadlessRunConfig {
                goal: "prove the runner does not invent a local model".to_string(),
                model_url: None,
                max_transitions: 1,
                ..HeadlessRunConfig::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(result.status, HeadlessRunStatus::BlockedNoModelUrl);
        assert_eq!(
            result.receipt.get("status").and_then(Value::as_str),
            Some("blocked_no_model_url")
        );
        assert!(result.receipt.get("agent").unwrap().is_null());
        assert_eq!(
            result
                .receipt
                .pointer("/runtime/model_tps_claim")
                .and_then(Value::as_str),
            Some("not_measured")
        );
        assert!(result
            .receipt
            .get("content_blake3")
            .and_then(Value::as_str)
            .is_some());
    }

    #[tokio::test]
    async fn fake_runtime_records_planner_and_action_metrics_without_claiming_complete_forward_tps()
    {
        let temp = tempfile::tempdir().unwrap();
        // The receipt audits the durable JSONL chain. An in-memory log has no
        // chain hashes, so this host-bound test uses the real workspace event
        // log rather than the lightweight test double above.
        let host = BackendHost::open_workspace(temp.path()).unwrap();
        let fake = crate::supervisor::testkit::FakeRuntime::spawn().await;
        let result = run_headless_audit(
            &host,
            HeadlessRunConfig {
                goal: "exercise a model-backed headless receipt".to_string(),
                model_url: Some(fake.base_url()),
                max_transitions: 10,
                ..HeadlessRunConfig::default()
            },
        )
        .await
        .unwrap();
        fake.stop();

        let metrics = result.receipt.pointer("/agent/model_metrics").unwrap();
        assert!(
            metrics
                .get("recorded_call_count")
                .and_then(Value::as_u64)
                .unwrap_or(0)
                >= 1,
            "the RuntimePlanner call must be included in the receipt"
        );
        assert!(
            metrics
                .get("ledger_output_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0)
                >= 1,
            "model tokens must be added to the terminal ledger"
        );
        assert!(metrics
            .get("aggregate_complete_forward_tps")
            .unwrap()
            .is_null());
        assert_eq!(
            result
                .receipt
                .pointer("/runtime/context_before/status")
                .and_then(Value::as_str),
            Some("unavailable"),
            "the sparse fake context response must not be mistaken for a live context ceiling"
        );
        assert_eq!(
            result
                .receipt
                .pointer("/event_chain/ok")
                .and_then(Value::as_bool),
            Some(true)
        );
    }

    #[tokio::test]
    async fn selected_local_evidence_is_receipted_only_after_an_agent_act_model_call() {
        let temp = tempfile::tempdir().unwrap();
        let source_path = temp.path().join("agent-evidence.txt");
        let selected_fact = "HCLI_AGENT_SELECTED_FACT_41f2";
        std::fs::write(&source_path, selected_fact).unwrap();
        let sources = crate::hcli_sources::HcliSourceStore::open(temp.path()).unwrap();
        let ingested = sources.ingest_file(&source_path, None, None).unwrap();
        let source_context = sources
            .select_context(&[ingested.reference.id.as_str().to_string()])
            .unwrap();
        let host = BackendHost::open_workspace(temp.path()).unwrap();
        let fake = crate::supervisor::testkit::FakeRuntime::spawn().await;
        let result = run_headless_audit(
            &host,
            HeadlessRunConfig {
                goal: "use the selected evidence in an agent step".to_string(),
                model_url: Some(fake.base_url()),
                max_transitions: 10,
                source_context: Some(source_context),
                ..HeadlessRunConfig::default()
            },
        )
        .await
        .unwrap();
        fake.stop();

        let evidence = result
            .receipt
            .pointer("/context/explicit_source_context")
            .expect("agent evidence receipt");
        assert_eq!(evidence["status"], "selected");
        assert_eq!(
            evidence["selected_sources"][0]["reference_id"],
            ingested.reference.id.as_str()
        );
        assert_eq!(
            evidence["injection"]["configured_for"],
            "agent_act_model_prompt"
        );
        assert_eq!(
            evidence["injection"]["injected_into_agent_act_model_prompt"],
            Value::Bool(true),
            "a real act-model observation must exist before the receipt calls it injected"
        );
        assert!(
            !evidence.to_string().contains(selected_fact),
            "receipt must retain source identity/counts but not selected derivative text"
        );
    }
}
