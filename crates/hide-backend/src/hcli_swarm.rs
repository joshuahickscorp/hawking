//! Parallel, read-safe HCLI analysis swarms.
//!
//! This is intentionally distinct from the experimental `hide-fleet` worktree
//! scheduler.  Each lane here is a real, independently driven [`AgentKernel`]
//! with its own durable session and model calls, but lanes share the supplied
//! workspace.  The normal HCLI profile therefore remains `SuggestOnly`: this
//! is a genuine parallel research/analysis swarm, not a claim of isolated
//! concurrent write agents.

use crate::hcli_profile::HcliProfile;
use crate::headless::{run_headless_audit, HeadlessRunConfig, HeadlessRunStatus};
use crate::model_provider::ContextInfo;
use crate::{BackendHost, HttpModelProvider};
use futures::stream::{self, StreamExt};
use hide_core::ids::now_ms;
use hide_core::runtime::{InferenceRequest, ModelProvider, StreamChunk};
use hide_core::Result;
use serde_json::{json, Value};
use std::time::Instant;

pub const HCLI_SWARM_RECEIPT_SCHEMA: &str = "hcli.parallel_analysis_swarm.v1";

/// A reducer is useful only when it can receive the whole bounded receipt
/// summary.  Sending a prefix would make an apparently complete synthesis omit
/// material evidence from later lanes.
const MAX_SYNTHESIS_INPUT_BYTES: usize = 96 * 1024;
const DEFAULT_SYNTHESIS_MAX_OUTPUT_TOKENS: usize = 2_048;

/// Configuration for a set of independently driven HCLI agent lanes.
#[derive(Debug, Clone)]
pub struct HcliSwarmConfig {
    pub goal: String,
    pub model_url: Option<String>,
    pub profile: HcliProfile,
    /// Number of outer agent lanes. This is separate from the kernel's
    /// configured `search_breadth` and is realized by this executor.
    pub lanes: usize,
    /// Maximum agent kernels allowed to send work at once. This is an actual
    /// outer scheduler bound, distinct from the number of requested lanes.
    pub max_concurrency: usize,
    /// Per-lane driver cap. The profile governor remains authoritative.
    pub max_transitions: u32,
    /// Produce one bounded model synthesis over sealed lane receipts after the
    /// independent lanes finish. The synthesis is explicitly unverified and is
    /// never presented as source evidence by itself.
    pub synthesize: bool,
}

impl Default for HcliSwarmConfig {
    fn default() -> Self {
        let profile = HcliProfile::Power;
        Self {
            goal: String::new(),
            model_url: None,
            profile,
            lanes: profile.budget().search_breadth as usize,
            max_concurrency: profile.budget().search_breadth as usize,
            max_transitions: profile.budget().max_steps,
            synthesize: true,
        }
    }
}

/// Result of a complete outer swarm command. Individual lane receipts remain
/// nested so the aggregate never hides a blocked or failed lane.
#[derive(Debug, Clone)]
pub struct HcliSwarmResult {
    pub complete: bool,
    /// Optional bounded model reduction of the lane receipts. It is a useful
    /// operator-facing report, not a verification verdict or evidence source.
    pub synthesis: Option<String>,
    pub receipt: Value,
}

#[derive(Debug)]
struct SynthesisOutcome {
    text: String,
    stats: hide_core::runtime::GenerationStats,
    /// The value actually sent to the endpoint after applying its advertised
    /// hard output ceiling, when one was available.
    requested_max_output_tokens: usize,
    endpoint_max_output_tokens: Option<usize>,
}

/// Run real agent lanes in parallel, one session per lane. `SuggestOnly` is
/// enforced by the normal profile/kernel configuration; callers seeking
/// destructive multi-worktree execution must use a future worktree-bound fleet
/// path rather than reading this as an isolation guarantee.
pub async fn run_parallel_analysis_swarm(
    host: &BackendHost,
    config: HcliSwarmConfig,
) -> Result<HcliSwarmResult> {
    let started_ms = now_ms();
    let started = Instant::now();
    let lanes = config.lanes.clamp(1, 32);
    let requested_concurrency = config.max_concurrency.clamp(1, lanes);
    let profile_spec = config.profile.spec();
    let goal = config.goal.trim().to_string();
    let (effective_concurrency, compact_lane_goals, admission) =
        admit_runtime_capacity(config.model_url.as_deref(), requested_concurrency).await;

    // This outer queue is a real scheduling limit. Every lane gets a distinct
    // durable session, but only `effective_concurrency` kernels can issue work
    // at a time; this avoids blindly flooding a local runtime whose slot count
    // is known to be lower than the requested breadth.
    let lane_results = stream::iter(0..lanes)
        .map(|index| {
            let lane_goal = lane_goal(&goal, index, lanes, compact_lane_goals);
            let lane_config = HeadlessRunConfig {
                goal: lane_goal,
                model_url: config.model_url.clone(),
                session_id: None,
                max_transitions: config.max_transitions,
                profile: config.profile,
                source_context: None,
            };
            async move { (index, run_headless_audit(host, lane_config).await) }
        })
        .buffer_unordered(effective_concurrency)
        .collect::<Vec<_>>()
        .await;
    let mut lane_results = lane_results;
    lane_results.sort_by_key(|(index, _)| *index);

    let mut complete = true;
    let mut started_agents = 0usize;
    let mut total_model_calls = 0u64;
    let mut completed_decode_forwards = 0u64;
    let mut decode_ms = 0.0f64;
    let mut complete_metric_calls = 0u64;
    let mut lanes_json = Vec::with_capacity(lanes);
    let mut artifact_manifest = Vec::with_capacity(lanes);

    for (index, result) in lane_results {
        match result {
            Ok(result) => {
                complete &= result.status == HeadlessRunStatus::Completed;
                if result
                    .receipt
                    .get("agent")
                    .is_some_and(|agent| agent.is_object())
                {
                    started_agents += 1;
                }
                total_model_calls = total_model_calls.saturating_add(
                    result
                        .receipt
                        .pointer("/agent/model_metrics/recorded_call_count")
                        .and_then(Value::as_u64)
                        .unwrap_or(0),
                );
                complete_metric_calls = complete_metric_calls.saturating_add(
                    result
                        .receipt
                        .pointer("/agent/model_metrics/complete_forward_metric_call_count")
                        .and_then(Value::as_u64)
                        .unwrap_or(0),
                );
                let lane_decode_ms = result
                    .receipt
                    .pointer("/agent/model_metrics/decode_ms")
                    .and_then(Value::as_f64);
                let lane_forwards = result
                    .receipt
                    .pointer("/agent/model_metrics/completed_decode_forwards")
                    .and_then(Value::as_u64);
                if let (Some(ms), Some(forwards)) = (lane_decode_ms, lane_forwards) {
                    if ms > 0.0 && forwards > 0 {
                        decode_ms += ms;
                        completed_decode_forwards =
                            completed_decode_forwards.saturating_add(forwards);
                    }
                }
                let role = lane_role(index, lanes);
                let receipt_hash = result
                    .receipt
                    .get("content_blake3")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                let session_id = result
                    .receipt
                    .pointer("/event_window/session_id")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                artifact_manifest.push(json!({
                    "lane": index + 1,
                    "role": role.name,
                    "session_id": session_id,
                    "sealed_receipt_blake3": receipt_hash,
                    "status": result.status.as_str(),
                    "consumable_by_later_lanes": false,
                    "note": "This immutable receipt is an audit artifact. Cross-lane evidence retrieval and shared-memory context injection are not implemented yet.",
                }));
                lanes_json.push(json!({
                    "lane": index + 1,
                    "role": role.name,
                    "role_focus": role.focus,
                    "status": result.status.as_str(),
                    "receipt": result.receipt,
                }));
            }
            Err(error) => {
                complete = false;
                let role = lane_role(index, lanes);
                artifact_manifest.push(json!({
                    "lane": index + 1,
                    "role": role.name,
                    "status": "driver_error",
                    "consumable_by_later_lanes": false,
                }));
                lanes_json.push(json!({
                    "lane": index + 1,
                    "role": role.name,
                    "role_focus": role.focus,
                    "status": "driver_error",
                    "error": error.to_string(),
                }));
            }
        }
    }

    let all_agent_calls_have_complete_metrics = total_model_calls > 0
        && complete_metric_calls == total_model_calls
        && decode_ms > 0.0
        && completed_decode_forwards > 0;
    let aggregate_complete_forward_tps = all_agent_calls_have_complete_metrics
        .then(|| completed_decode_forwards as f64 / (decode_ms / 1_000.0));
    let synthesis_attempt = match (
        config.synthesize,
        config.model_url.as_deref(),
        total_model_calls,
    ) {
        (true, Some(_), _) if compact_lane_goals => Err(
            "not attempted: observed endpoint context is at or below 128 tokens; use --no-synthesis because the receipt reducer cannot fit without truncation"
                .to_string(),
        ),
        (true, Some(url), calls) if calls > 0 => {
            match build_synthesis_input(&goal, &lanes_json) {
                Ok(synthesis_input) => match synthesize_lane_receipts(url, synthesis_input).await {
                    Ok(outcome) => Ok(outcome),
                    Err(error) => {
                        complete = false;
                        Err(error.to_string())
                    }
                },
                Err(error) => {
                    complete = false;
                    Err(error.to_string())
                }
            }
        }
        (true, Some(_), _) => {
            Err("not attempted: no lane recorded an actual model call".to_string())
        }
        (true, None, _) => Err("not attempted: no local model URL was supplied".to_string()),
        (false, _, _) => Err("disabled by caller".to_string()),
    };
    let synthesis = synthesis_attempt
        .as_ref()
        .ok()
        .map(|outcome| outcome.text.clone());
    let synthesis_metrics = synthesis_attempt.as_ref().ok().map(|outcome| {
        json!({
            "input_tokens": outcome.stats.input_tokens,
            "output_tokens": outcome.stats.output_tokens,
            "requested_max_output_tokens": outcome.requested_max_output_tokens,
            "endpoint_max_output_tokens": outcome.endpoint_max_output_tokens,
            "decode_ms": outcome.stats.decode_ms,
            "completed_decode_forwards": outcome.stats.completed_decode_forwards,
            "complete_forward_tps": outcome.stats.decode_ms.zip(outcome.stats.completed_decode_forwards)
                .and_then(|(milliseconds, forwards)| (milliseconds > 0.0 && forwards > 0).then(|| forwards as f64 / (milliseconds / 1_000.0))),
        })
    });
    let mut receipt = json!({
        "schema": HCLI_SWARM_RECEIPT_SCHEMA,
        "status": if complete { "completed" } else { "incomplete" },
        "started_ms": started_ms,
        "finished_ms": now_ms(),
        "wall_elapsed_ms": started.elapsed().as_millis() as u64,
        "goal": {
            "text": config.goal,
            "blake3": blake3::hash(config.goal.as_bytes()).to_hex().to_string(),
        },
        "execution": {
            "kind": "bounded_parallel_independent_agent_kernels",
            "requested_lanes": config.lanes,
            "effective_lanes": lanes,
            "requested_max_concurrency": config.max_concurrency,
            "effective_max_concurrency": effective_concurrency,
            "runtime_admission": admission,
            "compact_lane_goals": compact_lane_goals,
            "compact_lane_goal_note": if compact_lane_goals {
                "Observed a <=128-token endpoint context; whole verbose role blocks were omitted from lane objectives rather than truncated."
            } else {
                "Normal lane objective format was used."
            },
            "actual_agent_runs_started": started_agents,
            "workspace_isolation": "shared_workspace",
            "effect_safety": "Profiles default to suggest_only. This command is suitable for concurrent analysis and planning, not a claim of isolated concurrent filesystem mutation.",
            "note": "Outer lanes are real and realized. Kernel search_breadth/self_consistency_k remain separate configured budgets and are reported inside each lane receipt. The runtime-capacity observation only bounds launch concurrency; it is not a reservation.",
        },
        "profile": profile_spec,
        "runtime": {
            "requested_url": config.model_url,
            "model_call_count": total_model_calls,
            "complete_forward_metric_call_count": complete_metric_calls,
            "completed_decode_forwards": completed_decode_forwards,
            "decode_ms": all_agent_calls_have_complete_metrics.then_some(decode_ms),
            "aggregate_complete_forward_tps": aggregate_complete_forward_tps,
            "tps_authority": if aggregate_complete_forward_tps.is_some() {
                "all recorded agent calls exposed completed_decode_forwards plus decode_ms; reported value is sum(forwards) / sum(decode_ms)"
            } else {
                "unavailable: every recorded agent model call must expose both completed_decode_forwards and decode_ms"
            },
        },
        "artifact_manifest": {
            "schema": "hcli.swarm.artifact_manifest.v1",
            "contract": "Each lane emits a sealed receipt with a role, session, status, and hash. The manifest is immutable audit metadata, not a shared retrieval context.",
            "artifacts": artifact_manifest,
        },
        "reduction": {
            "requested": config.synthesize,
            "kind": if synthesis.is_some() { "bounded_model_synthesis" } else { "none" },
            "status": if synthesis.is_some() { "completed" } else { "not_completed" },
            "semantic_synthesis": synthesis,
            "runtime_metrics": synthesis_metrics,
            "failure_or_reason": synthesis_attempt.as_ref().err(),
            "verification": "The reduction is target-model output over lane receipts. It is not independently verified evidence and must retain the lane receipt citations/uncertainty it reports.",
        },
        "lanes": lanes_json,
        "limitations": [
            "This is an actual parallel analysis swarm: one independent AgentKernel and durable session per lane, with bounded outer concurrency and differentiated roles.",
            "It is not a worktree-isolated write swarm. Default SuggestOnly autonomy prevents the high-compute profile from granting raw effects.",
            "Lane receipts are sealed audit artifacts but are not automatically available to later lanes as context. A future coordinator must add shared evidence manifests, source partitioning, barriers, and verifier/judge contracts.",
            "Agent wall time includes model planning, verification, tools, and scheduling. It is not decode TPS.",
            "A complete-forward TPS claim requires runtime-reported completed_decode_forwards plus decode_ms for every recorded model call.",
        ],
    });
    seal(&mut receipt)?;
    Ok(HcliSwarmResult {
        complete,
        synthesis,
        receipt,
    })
}

#[derive(Clone, Copy)]
struct LaneRole {
    name: &'static str,
    focus: &'static str,
}

fn lane_role(index: usize, _total: usize) -> LaneRole {
    const ROLES: [LaneRole; 6] = [
        LaneRole {
            name: "scope_and_repository_recon",
            focus: "Map the objective, repository surfaces, constraints, and unknowns before proposing work.",
        },
        LaneRole {
            name: "evidence_and_data_procurement",
            focus: "Identify needed primary evidence, local files, commands, and data gaps; distinguish observed facts from requests for procurement.",
        },
        LaneRole {
            name: "architecture_and_dependency_analysis",
            focus: "Trace component boundaries, data flow, invariants, and integration risks.",
        },
        LaneRole {
            name: "adversarial_verification",
            focus: "Try to falsify assumptions, identify safety/correctness regressions, and specify objective verification gates.",
        },
        LaneRole {
            name: "tooling_and_execution_planning",
            focus: "Design the least-ambiguous tool/workflow sequence, with permissions, artifacts, and rollback points.",
        },
        LaneRole {
            name: "judge_and_delivery_criteria",
            focus: "Define what a complete result must prove, rank unresolved uncertainty, and assess competing approaches.",
        },
    ];
    ROLES[index % ROLES.len()]
}

fn lane_goal(goal: &str, index: usize, total: usize, compact: bool) -> String {
    let role = lane_role(index, total);
    if compact {
        return format!(
            "Read-only lane {}/{} ({}) . Objective: {}",
            index + 1,
            total,
            role.name,
            goal
        );
    }
    format!(
        "HCLI parallel analysis lane {}/{} — role: {}.\n\nRole focus:\n{}\n\nWork independently. Produce evidence with concrete file paths, commands, source references, or observed outputs; label every inference and unresolved data gap. Produce a concrete plan and objective verification criteria. Do not assume other lanes will share your reasoning, and do not claim access to evidence you have not actually inspected.\n\nObjective:\n{}",
        index + 1, total, role.name, role.focus, goal
    )
}

async fn admit_runtime_capacity(
    model_url: Option<&str>,
    requested_concurrency: usize,
) -> (usize, bool, Value) {
    let Some(url) = model_url.filter(|url| !url.trim().is_empty()) else {
        return (
            requested_concurrency,
            false,
            json!({
                "status": "not_checked",
                "reason": "no model URL; lanes will produce explicit blocked receipts rather than synthetic work",
            }),
        );
    };
    let provider = HttpModelProvider::new(url);
    match provider.get_context_info().await {
        Some(info) if info.max_batch > 0 || info.active_slots > 0 || info.free_slots > 0 => {
            let effective = requested_concurrency.min(info.free_slots.max(1));
            (
                effective,
                is_compact_endpoint_context(&info),
                json!({
                    "status": "observed",
                    "active_slots": info.active_slots,
                    "free_slots": info.free_slots,
                    "max_batch": info.max_batch,
                    "ctx_len_native": info.ctx_len_native,
                    "ctx_len_effective": info.ctx_len_effective,
                    "max_output_tokens": info.max_output_tokens,
                    "effective_max_concurrency": effective,
                    "note": "free_slots is a point-in-time runtime observation, not an exclusive reservation",
                }),
            )
        }
        Some(info) => (
            requested_concurrency,
            is_compact_endpoint_context(&info),
            json!({
                "status": "unavailable",
                "reason": "runtime context endpoint did not expose usable slot capacity; requested outer bound retained",
                "ctx_len_native": info.ctx_len_native,
                "ctx_len_effective": info.ctx_len_effective,
                "max_output_tokens": info.max_output_tokens,
            }),
        ),
        None => (
            requested_concurrency,
            false,
            json!({
                "status": "unavailable",
                "reason": "runtime did not expose context/slot capacity; requested outer bound retained",
            }),
        ),
    }
}

/// The <=128-token diagnostic endpoint cannot safely host the ordinary
/// multi-receipt reducer.  Keep this independent from slot admission: a
/// runtime can truthfully expose its context cap while omitting capacity
/// telemetry.
fn is_compact_endpoint_context(info: &ContextInfo) -> bool {
    info.ctx_len_effective
        .or(info.ctx_len_native)
        .is_some_and(|window| window <= 128)
}

fn synthesis_output_cap(endpoint_max_output_tokens: Option<usize>) -> usize {
    endpoint_max_output_tokens
        .filter(|cap| *cap > 0)
        .map(|cap| cap.min(DEFAULT_SYNTHESIS_MAX_OUTPUT_TOKENS))
        .unwrap_or(DEFAULT_SYNTHESIS_MAX_OUTPUT_TOKENS)
}

fn build_synthesis_input(goal: &str, lanes: &[Value]) -> Result<String> {
    let lane_summaries: Vec<Value> = lanes
        .iter()
        .map(|lane| {
            json!({
                "lane": lane.get("lane"),
                "role": lane.get("role"),
                "role_focus": lane.get("role_focus"),
                "status": lane.get("status"),
                "agent": lane.pointer("/receipt/agent").cloned(),
                "context": lane.pointer("/receipt/context/kernel_grounding").cloned(),
                "limitations": lane.pointer("/receipt/limitations").cloned(),
                "sealed_receipt_blake3": lane.pointer("/receipt/content_blake3").cloned(),
            })
        })
        .collect();
    let raw = serde_json::to_string_pretty(&json!({
        "objective": goal,
        "lane_receipt_summaries": lane_summaries,
    }))
    .unwrap_or_else(|_| "{\"lane_receipt_summaries\":[]}".to_string());
    if raw.len() > MAX_SYNTHESIS_INPUT_BYTES {
        return Err(hide_core::error::HideError::Config(format!(
            "swarm reducer was not started: the complete lane-receipt summary is {} bytes, above the {}-byte bounded reducer input; no partial receipt prefix was sent",
            raw.len(), MAX_SYNTHESIS_INPUT_BYTES
        )));
    }
    Ok(format!(
        "You are the bounded HCLI swarm reducer. Synthesize the independent lane receipt summaries below. Only state claims with an explicit lane/evidence attribution. Preserve uncertainty, conflicts, and gaps. Do not invent results, pretend a receipt is external evidence, or authorize effects. Return: (1) evidence-grounded findings, (2) disagreements/uncertainty, (3) prioritized next actions, (4) verification gates.\n\n{}",
        raw
    ))
}

async fn synthesize_lane_receipts(base_url: &str, prompt: String) -> Result<SynthesisOutcome> {
    let provider = HttpModelProvider::new(base_url);
    let context = provider.get_context_info().await;
    if context.as_ref().is_some_and(is_compact_endpoint_context) {
        return Err(hide_core::error::HideError::Config(
            "swarm reducer was not started: endpoint reported a context at or below 128 tokens; whole reducer prompt omitted rather than truncated".to_string(),
        ));
    }
    let endpoint_max_output_tokens = context
        .as_ref()
        .and_then(|info| info.max_output_tokens)
        .filter(|cap| *cap > 0);
    let requested_max_output_tokens = synthesis_output_cap(endpoint_max_output_tokens);
    let request = InferenceRequest {
        task_kind: "hcli.swarm.reducer".to_string(),
        prompt,
        messages: Vec::new(),
        max_output_tokens: requested_max_output_tokens,
        sampler: None,
        grammar: None,
        want_logprobs: false,
        metadata: Default::default(),
    };
    let mut text = String::new();
    let mut sink = |chunk: StreamChunk| match chunk {
        StreamChunk::Token { text: token, .. } => {
            text.push_str(&token);
            Ok(())
        }
        StreamChunk::Done { .. } => Ok(()),
        StreamChunk::Error { message } => {
            Err(hide_core::error::HideError::RuntimeUnavailable(message))
        }
    };
    let stats = provider.generate(request, &mut sink).await?;
    Ok(SynthesisOutcome {
        text,
        stats,
        requested_max_output_tokens,
        endpoint_max_output_tokens,
    })
}

fn seal(receipt: &mut Value) -> Result<()> {
    let bytes = serde_json::to_vec(receipt)?;
    receipt["content_blake3"] = json!(blake3::hash(&bytes).to_hex().to_string());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::BackendServices;
    use hide_core::config::HideConfig;
    use hide_core::event::InMemoryEventLog;
    use std::sync::Arc;

    #[test]
    fn lane_goals_are_independent_and_identifiable() {
        let first = lane_goal("audit this repository", 0, 3, false);
        let second = lane_goal("audit this repository", 1, 3, false);
        assert!(first.contains("lane 1/3"));
        assert!(second.contains("lane 2/3"));
        assert_ne!(first, second);
    }

    #[test]
    fn compact_lane_goal_omits_whole_verbose_role_blocks() {
        let goal = "inspect the endpoint";
        let compact = lane_goal(goal, 0, 2, true);
        assert!(compact.contains(goal));
        assert!(compact.contains("scope_and_repository_recon"));
        assert!(!compact.contains("Role focus:"));
        assert!(!compact.contains("Work independently."));
    }

    #[test]
    fn compact_context_is_detected_without_slot_telemetry() {
        let info = ContextInfo {
            ctx_len_native: Some(128),
            ctx_len_effective: Some(128),
            max_output_tokens: Some(4),
            ..ContextInfo::default()
        };
        assert!(is_compact_endpoint_context(&info));
        assert_eq!(synthesis_output_cap(info.max_output_tokens), 4);
    }

    #[tokio::test]
    async fn missing_runtime_yields_a_sealed_incomplete_swarm_without_fake_agents() {
        let temp = tempfile::tempdir().unwrap();
        let services = BackendServices::new(
            HideConfig::for_workspace(temp.path()),
            Arc::new(InMemoryEventLog::new()),
        );
        let host = BackendHost::from_services(services).unwrap();
        let result = run_parallel_analysis_swarm(
            &host,
            HcliSwarmConfig {
                goal: "do not invent runtime work".to_string(),
                model_url: None,
                profile: HcliProfile::Balanced,
                lanes: 3,
                max_concurrency: 2,
                max_transitions: 1,
                synthesize: true,
            },
        )
        .await
        .unwrap();

        assert!(!result.complete);
        assert_eq!(
            result
                .receipt
                .pointer("/execution/effective_lanes")
                .and_then(Value::as_u64),
            Some(3)
        );
        assert_eq!(
            result
                .receipt
                .pointer("/execution/actual_agent_runs_started")
                .and_then(Value::as_u64),
            Some(0)
        );
        assert_eq!(
            result
                .receipt
                .pointer("/execution/effective_max_concurrency")
                .and_then(Value::as_u64),
            Some(2)
        );
        assert_eq!(
            result
                .receipt
                .get("content_blake3")
                .and_then(Value::as_str)
                .map(str::len),
            Some(64)
        );
    }

    #[test]
    fn differentiated_lanes_carry_distinct_data_and_verification_roles() {
        let evidence = lane_goal("audit the pipeline", 1, 6, false);
        let verifier = lane_goal("audit the pipeline", 3, 6, false);
        assert!(evidence.contains("evidence_and_data_procurement"));
        assert!(verifier.contains("adversarial_verification"));
        assert_ne!(evidence, verifier);
    }

    #[test]
    fn oversized_synthesis_input_is_rejected_without_a_partial_prefix() {
        let oversized = "é".repeat(80_000);
        let source = json!([{
            "lane": 1,
            "role": "evidence",
            "status": "completed",
            "receipt": { "agent": { "plan": oversized } }
        }]);
        let error = build_synthesis_input("goal", source.as_array().unwrap()).unwrap_err();
        assert!(error
            .to_string()
            .contains("no partial receipt prefix was sent"));
    }
}
