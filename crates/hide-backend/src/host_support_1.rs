use crate::approval::{ApprovalDecision, ApprovalHub};
use crate::commands::CommandRouter;
use crate::connectors::{register_backend_connectors, ConnectorRegistry, ConnectorStatus};
use crate::initialize::{ClientCapabilities, ClientInfo, ConnectionRegistry, InitializeResponse};
use crate::interrupt::InterruptHub;
use crate::live_thread::LiveThread;
use crate::memory::{
    MemoryDraft, MemoryLedger, MemoryRecord, MemoryRevalidation, MemoryScope, MemoryStatus,
    PrivacyClass, RevalidateTarget,
};
use crate::policy::{
    derive_policy_decision, tool_declared_effects, PolicyDecision, PolicyDecisionRecord,
};
use crate::process::{ProcessState, ProcessSupervisor, StartSpec};
use crate::replay::BackendReplayService;
use crate::rewind::{self, CheckpointCoverage, FileChange, ForkPoint, RewindTarget, StateRef};
use crate::security::SecurityServices;
use crate::services::{
    BackendCapabilities, BackendServices, Budget, CheckpointRecord, CheckpointStore,
    EnvironmentNode, EnvironmentSwitch, GoalOutcome, GoalRecord, GoalStatus, GoalStore,
    GoalVerdict, JobRecord, JobStatus, JobStore, RepoNode, SharedBackend, Trigger, TriggerEvent,
    TrustState, WorkspaceEdge, WorkspaceEdgeKind, WorkspaceGraph, WorkspaceStore,
};
use crate::supervisor::{RuntimeSupervisor, SupervisorConfig};
use crate::surfaces::SurfaceGraphService;
use crate::tools::{build_default_tool_dispatcher, build_default_tool_registry};
use crate::ui_bus::UiEventBus;
use hide_core::api::{Intent, IntentAck, UiEvent, UiEventKind};
use hide_core::event::{Event, NewEvent, ToolCallEvent, ToolResultEvent};
use hide_core::ids::{EventId, RunId, SessionId, StepId};
use hide_core::observability::{HealthCheck, HealthReport, HealthStatus};
use hide_core::runtime::{ModelRole, RuntimeSupervisorState};
use hide_core::tool::{ToolCall, ToolDispatcher, ToolRegistry, ToolResult, ToolSpec, ToolStatus};
use hide_core::Result;
use hide_fleet::manager::KernelRunLauncher;
use hide_fleet::{
    AgentJob, ConcurrencyClass, FleetConfig, FleetGovernor, FleetManager, OsResourceProbe,
    PriorityClass,
};
use hide_kernel::govern::{Autonomy, Interrupt};
use hide_kernel::machine::state::{AgentState, ApprovalRequest, Phase};
use hide_kernel::session::SessionProjection;
use hide_kernel::{AgentKernel, Grounding};
// Bible Book IX sec 28-29 / sec 78.1 #6: the deterministic verification plane.
// The colliding names (`Verdict`, `VerificationInput`, `Oracle`) are qualified
// as `hide_kernel::verify_plane::*` at their (few) use sites so the function-local
// `hide_kernel::verify::oracle::*` imports in the goal path and the tests keep
// their meaning; only the non-colliding types are imported here.
use super::*;
use hide_kernel::verify_plane::{
    Finding, GateDecision, ReviewRole, ReviewRoleProfile, SourceFile, StaticAnalysisOracle,
    TieredVerdict, VerificationReceipt, VerificationTier,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// What [`run_turn_core`] returns to its callers: the full completion plus the
/// two bits the live [`generate_submit_turn`] path needs to publish its post-turn
/// `context_manifest` (the stream's seq, and the folded-prompt char length for
/// the used-token estimate).
pub(crate) struct TurnOutcome {
    pub(crate) completion: String,
    pub(crate) stream_seq: u64,
    pub(crate) prompt_chars: usize,
}

/// The SOLE product generation core. Both entry points
/// ([`BackendHost::generate_and_publish`] and the spawnable
/// [`generate_submit_turn`]) funnel here so live path and headless tests
/// exercise ONE code path and can never drift. There is no second product
/// SubmitTurn branch (`HIDE_KERNEL_TURN` removed).
///
/// It fixes the Phase-1b facade: instead of a raw prompt with an empty history
/// and a hard `max_output_tokens: 256`, it (1) compiles a REAL `ContextPack`
/// from the code index, (2) rebuilds REAL message history from the event log,
/// (3) folds compiled context + history + the user prompt into `prompt` (the
/// native generate route ignores `messages`), (4) derives the output budget from
/// the model window minus what the context consumed, and (5) persists a
/// `context.compiled` marker before streaming and an `agent.message` assistant
/// event after - so the NEXT turn sees this turn in its history.
///
/// `live_ceiling` (the pre-streaming `/v1/hawking/context` snapshot) is `Some`
/// only on the live path; when set, the token sink emits a throttled per-step
/// occupancy patch. `run_id_label` tags the `runtime.generation` event.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_turn_core(
    inference: Arc<dyn hawking_orch::inference::InferenceClient>,
    event_log: hide_core::persistence::DynEventLog,
    role_registry: Arc<hawking_orch::RoleRegistry>,
    code_index: Arc<dyn hawking_index::CodeIndex>,
    memory: crate::services::DynMemoryStore,
    classed_memory: hawking_context::DynClassedMemory,
    ui_bus: Arc<UiEventBus>,
    session_id: SessionId,
    prompt: String,
    live_ceiling: Option<(Option<usize>, Option<usize>, usize)>,
    run_id_label: Option<String>,
    repo_instructions: Arc<crate::compat_instructions::ResolvedInstructions>,
) -> Result<TurnOutcome> {
    use crate::connectors::choose_context_role;
    use hawking_context::compiler::CompileInput;
    use hawking_context::profiles::ContextProfile;
    use hawking_context::sources::{ClassedMemoryContextSource, CodeIndexContextSource};
    use hawking_context::{ClassBudgets, ContextCompiler, InMemoryMemoryStore, MemoryKind};
    use hawking_orch::router::SimpleRouter;
    use hawking_speculate::TargetVerification;
    use hide_core::runtime::{InferenceMessage, InferenceRequest, StreamChunk};
    use hide_core::types::Provenance;
    use hide_kernel::runtime_client::KernelRuntimeClient;

    // Working memory (turn-local): sole TurnWriteCap mint is inside
    // WorkingTurnGuard::begin; Drop clears the row on every exit path.
    let turn_id = run_id_label
        .clone()
        .unwrap_or_else(|| format!("turn-{}", session_id.as_str()));
    let _working_guard = crate::classed_writers::WorkingTurnGuard::begin(
        classed_memory.clone(),
        turn_id.clone(),
        session_id.as_str(),
        run_id_label.as_deref(),
        &prompt,
    );

    // --- (S3) Compile a REAL ContextPack (bible §4.2). Mirrors the `context`
    // connector so both share one recipe: pick the coding role, size the window
    // to its model, and let the code-index + classed memory compete for budget. ---
    // §7.3 honesty: prefer live-measured native over the role/config default;
    // never pack against an inflated effective ceiling *as if* it were native.
    let role = choose_context_role(&role_registry, None)?;
    let config_native = role.model.context_tokens.max(4096);
    let live_native = live_ceiling.and_then(|(_, n, _)| n).filter(|n| *n > 0);
    let live_effective = live_ceiling.map(|(_, _, c)| c).filter(|c| *c > 0);
    let capability =
        declare_turn_capability(config_native, live_native, live_effective, None, false);
    // Pack against measured/config native (conservative). Effective is reported
    // separately and is never treated as a larger native window.
    let max_input = capability.pack_budget_tokens(false).max(256);
    let mut model = role.model.clone();
    model.context_tokens = max_input;
    // Tokenizer-true packing when a real tokenizer is discoverable (bible §4.2).
    let counter = hawking_context::TokenCounter::discover_from_env()
        .unwrap_or_else(hawking_context::TokenCounter::heuristic);
    let mut compiler = ContextCompiler::new().with_counter(counter);
    compiler.add_source(CodeIndexContextSource::new(code_index, 16));
    // Six memory classes: independent per-class budgets (not one kind filter).
    let class_budgets = ClassBudgets::from_total((max_input / 8).max(64));
    compiler.add_source(
        ClassedMemoryContextSource::new(classed_memory.clone(), class_budgets)
            .with_session(session_id.as_str())
            .with_turn(turn_id.clone()),
    );
    // Bible sec 20 / sec 78.1 #11: fold the repo's resolved Claude Code migration
    // instructions (CLAUDE.md tree + un-scoped rules) into the compiled context as
    // a pinned instruction/system source, honoring precedence (read-last-wins).
    // Added only when the repo actually carries them (an un-migrated repo resolves
    // empty and this is a no-op).
    if !repo_instructions.is_empty() {
        compiler.add_source(repo_instructions.as_source());
    }
    let mut compiled = compiler
        .compile(CompileInput {
            profile: ContextProfile::coding_default(max_input),
            model,
            task: prompt.clone(),
        })
        .await?;
    // Pre-stream live reading (when the ceiling was snapshotted) so rot/meter
    // can include occupancy before generation advances.
    let pre_live = live_ceiling.map(|(state_bytes, native, ceiling)| {
        build_live_manifest(state_bytes, native, ceiling, compiled.manifest.used_tokens)
    });
    seal_compiled_manifest(
        &mut compiled.manifest,
        capability,
        pre_live.as_ref(),
        compiled.tokens_estimated,
    );
    // Surface per-class memory budgets on the context meter.
    if let (Some(meter), Some(ret)) = (
        compiled.manifest.meter.as_mut(),
        classed_memory.last_retrieval(),
    ) {
        meter.explanations.extend(ret.budget_explanations());
    }
    // Spine B (best-effort): accrue the Project Brain with this compile. A brain
    // write must never fail a turn.
    let brain = InMemoryMemoryStore::record(
        MemoryKind::Project,
        format!(
            "task: {prompt}\nretained {} spans, {} tokens used",
            compiled.manifest.retained.len(),
            compiled.manifest.used_tokens
        ),
        Provenance::trusted("submit_turn.compile"),
    );
    let _ = memory.upsert(brain).await;

    // --- (S2) Rebuild REAL message history from the durable event log, then
    // ensure the current user prompt is the final user message (the live path's
    // `user.intent.submit_turn` is already logged, so it is usually present
    // already; `generate_and_publish` may pass an explicit prompt that is not). ---
    let mut messages = rebuild_history(&event_log, &session_id).await?;
    if messages
        .last()
        .map(|m| m.role != "user" || m.content != prompt)
        .unwrap_or(true)
    {
        messages.push(InferenceMessage {
            role: "user".to_string(),
            content: prompt.clone(),
        });
    }
    let history_block = messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");
    // The native `/v1/hawking/generate` route sends only `prompt` (it drops
    // `messages`), so FOLD compiled context + rendered history into `prompt`.
    // `messages` is still populated for a future Chat-route switch.
    let folded_prompt = if compiled.prompt.trim().is_empty() {
        history_block
    } else {
        format!("{}\n\n{}", compiled.prompt, history_block)
    };
    let prompt_chars = folded_prompt.len();

    // --- (S2) Derive the output budget from the window minus what context ate,
    // clamped to a sane band - replacing the hard-coded 256 facade. ---
    // `HIDE_MAX_OUTPUT_TOKENS` (positive int) is an optional hard cap for live
    // smoke / small-model turns; it never *raises* the derived budget.
    let derived = max_input
        .saturating_sub(compiled.manifest.used_tokens)
        .clamp(256, 2048);
    let out_budget = std::env::var("HIDE_MAX_OUTPUT_TOKENS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
        .map(|cap| derived.min(cap))
        .unwrap_or(derived);

    // Durable marker: compile stats + honest capability / rot / meter.
    // The compile receipt lives on the event log (not a pre-token Wire-B patch)
    // so token-first consumers (flagship boot path) are not starved of TokenBatch.
    // Post-turn generate_submit_turn re-emits capability+rot+meter on the live
    // context_manifest projection.
    event_log
        .append(NewEvent::system(
            session_id.clone(),
            "context.compiled",
            context_compiled_payload(
                &compiled.manifest,
                Some(out_budget),
                "single_shot",
                run_id_label.as_deref(),
            ),
        ))
        .await?;

    // Context receipt: which repo instruction files (CLAUDE.md tree + un-scoped
    // rules) folded into this turn's context, in launch order. Logged only when
    // the repo carried migration instructions.
    if !repo_instructions.is_empty() {
        event_log
            .append(NewEvent::system(
                session_id.clone(),
                "context.instructions",
                repo_instructions.receipt_json(),
            ))
            .await?;
    }

    let request = InferenceRequest {
        task_kind: "code".to_string(),
        prompt: folded_prompt,
        messages,
        max_output_tokens: out_budget,
        sampler: None,
        grammar: None,
        want_logprobs: false,
        metadata: Default::default(),
    };

    // Route through the kernel runtime-client seam (router + inference client).
    let router = Arc::new(SimpleRouter::new(role_registry));
    let runtime = KernelRuntimeClient::new(router, inference);

    // A stable seq to key the published UiEvent stream off of.
    let status_event = event_log
        .append(NewEvent::system(
            session_id.clone(),
            "runtime.generation",
            json!({ "task": "code", "run_id": run_id_label }),
        ))
        .await?;
    let stream_id = status_event.seq.to_string();

    let mut buf = String::new();
    // This path is the target runtime itself, not a speculative draft source.
    // Keep its completion authority opaque until the one durable assistant
    // history event is constructed below; raw chunk JSON never writes history.
    let target_gate = TargetVerification::gate();
    let mut target_completion_ordinal = 0u32;
    {
        let bus = ui_bus.clone();
        let sess = session_id.clone();
        let sid = stream_id.clone();
        let seq = status_event.seq;
        let mut tok_count = 0usize;
        let mut sink = |chunk: StreamChunk| {
            match chunk {
                StreamChunk::Token { text, .. } => {
                    buf.push_str(&text);
                    target_completion_ordinal =
                        target_completion_ordinal.checked_add(1).ok_or_else(|| {
                            hide_core::error::HideError::PolicyDenied(
                                "target completion exceeded verified-token ordinal range".into(),
                            )
                        })?;
                    bus.publish_token(seq, Some(sess.clone()), &sid, &text);
                    // Throttled per-step occupancy (every 32 tokens), partial patch
                    // - only when the live ceiling was snapshotted (live path).
                    tok_count += 1;
                    if tok_count % 32 == 0 {
                        if let Some((state_bytes, native, ceiling)) = live_ceiling {
                            let used_est = (prompt_chars + buf.len()) / 4;
                            let live = build_live_manifest(state_bytes, native, ceiling, used_est);
                            if let Ok(mut lj) = serde_json::to_value(&live) {
                                if let Some(o) = lj.as_object_mut() {
                                    o.insert("used_tokens_estimate".to_string(), json!(used_est));
                                    o.insert("estimated".to_string(), json!(true));
                                    o.insert("partial".to_string(), json!(true));
                                }
                                bus.publish(UiEvent {
                                    seq,
                                    session_id: Some(sess.clone()),
                                    kind: UiEventKind::ProjectionPatch {
                                        projection: "context_manifest".to_string(),
                                        patch: json!({ "live": lj }),
                                    },
                                });
                            }
                        }
                    }
                }
                StreamChunk::Done { .. } => {
                    bus.flush(Some(sess.clone()));
                }
                StreamChunk::Error { message } => {
                    bus.publish(UiEvent {
                        seq,
                        session_id: Some(sess.clone()),
                        kind: UiEventKind::Error {
                            code: "generation".to_string(),
                            message,
                        },
                    });
                    return Err(hide_core::error::HideError::PolicyDenied(
                        "target runtime emitted an error chunk; refusing to persist a partial completion"
                            .into(),
                    ));
                }
            }
            Ok(())
        };
        runtime.generate(request, &mut sink).await?;
    }

    // (S2) Persist the assistant turn through the sole target-verified output
    // authority so the NEXT turn's `rebuild_history` cannot consume a raw or
    // provisional model completion.
    crate::classed_writers::VerifiedTokenEventLog::authority(event_log.clone())
        .append_target_verified_assistant(
            session_id.clone(),
            target_gate.emit_target(target_completion_ordinal),
            &buf,
        )
        .await?;

    // Working memory must not outlive the turn — `_working_guard` Drop clears it.
    Ok(TurnOutcome {
        completion: buf,
        stream_seq: status_event.seq,
        prompt_chars,
    })
}

/// Rebuild the prior conversation as `InferenceMessage`s from the durable event
/// log: a `user.intent.submit_turn` becomes a `user` message (its `args.text`),
/// and an `agent.message` with `role == "assistant"` becomes an `assistant`
/// message (its `text`). Everything else is ignored. Ordered by seq (scan order).
pub(crate) async fn rebuild_history(
    event_log: &hide_core::persistence::DynEventLog,
    session_id: &SessionId,
) -> Result<Vec<hide_core::runtime::InferenceMessage>> {
    use hide_core::runtime::InferenceMessage;
    let events = event_log.scan(Some(session_id.clone()), None, None).await?;
    let mut out = Vec::new();
    for ev in events {
        match ev.kind.as_str() {
            "user.intent.submit_turn" => {
                if let Some(text) = ev
                    .payload
                    .get("args")
                    .and_then(|a| a.get("text"))
                    .and_then(|t| t.as_str())
                {
                    if !text.is_empty() {
                        out.push(InferenceMessage {
                            role: "user".to_string(),
                            content: text.to_string(),
                        });
                    }
                }
            }
            "agent.message" => {
                let role = ev
                    .payload
                    .get("role")
                    .and_then(|r| r.as_str())
                    .unwrap_or("assistant");
                if role == "assistant" {
                    if let Some(text) = ev.payload.get("text").and_then(|t| t.as_str()) {
                        out.push(InferenceMessage {
                            role: "assistant".to_string(),
                            content: text.to_string(),
                        });
                    }
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

/// Spine A (W-F2-1): pick the live-context regime. An SSM (a model reporting a
/// constant recurrent-state footprint) surfaces recall FIDELITY from the
/// calibratable probe; a transformer surfaces KV occupancy. The probe is the
/// swap point for a measured boot-needle curve later.
pub(crate) fn build_live_manifest(
    recurrent_state_bytes: Option<usize>,
    ctx_len_native: Option<usize>,
    ceiling: usize,
    state_age_tokens: usize,
) -> hawking_context::manifest::ManifestLive {
    use hawking_context::fidelity::{LinearFidelity, RecallFidelityProbe};
    use hawking_context::manifest::ManifestLive;
    if let Some(state_bytes) = recurrent_state_bytes {
        let probe = LinearFidelity::new(ctx_len_native.unwrap_or(0));
        let fidelity = probe.fidelity(state_age_tokens);
        ManifestLive::ssm(state_bytes, state_age_tokens, fidelity, ceiling)
    } else {
        ManifestLive::transformer(state_age_tokens, ceiling)
    }
}

/// Honest §7.3 capability declaration for one turn.
///
/// Prefer a live-measured native window over the role/config default. Never
/// promote effective (`.tq` / position-scaled) or retrieval-usable context into
/// `native_maximum`. Validated quality/agentic and curves stay unmeasured until
/// a real calibration produces them.
pub(crate) fn declare_turn_capability(
    config_native: usize,
    live_native: Option<usize>,
    live_effective: Option<usize>,
    tq_multiplier: Option<f32>,
    tq_estimated: bool,
) -> hawking_context::ContextCapability {
    use hawking_context::{CompactionMode, ContextCapability, RetrievalMode};
    ContextCapability::declare(
        config_native,
        live_native,
        live_effective,
        tq_multiplier,
        tq_estimated,
        // Code-index + memory retrieval feed the packer; still capped by the window.
        RetrievalMode::RetrieveThenPack,
        // Compiler degrade ladder with recall-gated rollback (Spine B).
        CompactionMode::DegradeWithRecallGate,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::future::BoxFuture;
    use hawking_events::{CanonicalEvent, ContentVerification};
    use hawking_orch::inference::{InferenceClient, StubInferenceClient};
    use hide_core::config::HideConfig;
    use hide_core::event::InMemoryEventLog;
    use hide_core::runtime::{GenerationStats, InferenceRequest, StreamChunk, TokenSink};

    struct ErrorAfterPartialInference;

    impl InferenceClient for ErrorAfterPartialInference {
        fn generate<'a>(
            &'a self,
            _request: InferenceRequest,
            sink: TokenSink<'a>,
        ) -> BoxFuture<'a, Result<GenerationStats>> {
            Box::pin(async move {
                sink(StreamChunk::Token {
                    token_id: None,
                    text: "partial".to_string(),
                })?;
                sink(StreamChunk::Error {
                    message: "transport failed".to_string(),
                })?;
                Ok(GenerationStats {
                    input_tokens: 0,
                    output_tokens: 1,
                    decode_tokens_per_second: None,
                })
            })
        }

        fn embed<'a>(&'a self, _text: &'a str) -> BoxFuture<'a, Result<Vec<f32>>> {
            Box::pin(async move { Ok(Vec::new()) })
        }
    }

    #[tokio::test]
    async fn direct_target_turn_persists_only_target_verified_assistant_history() {
        let root = std::env::temp_dir().join(format!(
            "hawking_hide_direct_target_turn_{}_{}",
            hide_core::ids::now_ms(),
            std::process::id()
        ));
        let raw = Arc::new(InMemoryEventLog::new());
        let services = BackendServices::new(HideConfig::for_workspace(root), raw);
        let session = services.session();

        let outcome = run_turn_core(
            Arc::new(StubInferenceClient::new("target response")),
            services.event_log.clone(),
            services.role_registry.clone(),
            services.code_index.clone(),
            services.memory_store.clone(),
            services.classed_memory.clone(),
            Arc::new(UiEventBus::new(4)),
            session.clone(),
            "user request".to_string(),
            None,
            Some("offline-direct-target-test".to_string()),
            services.repo_instructions.clone(),
        )
        .await
        .expect("direct target turn");
        assert_eq!(outcome.completion, "target response");

        let events = services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await
            .expect("scan durable event log");
        let assistant = events
            .iter()
            .find(|event| event.kind == "agent.message")
            .expect("one durable assistant history event");
        let canonical =
            CanonicalEvent::from_sequenced(assistant.clone()).expect("canonical assistant event");
        assert_eq!(canonical.verification, ContentVerification::TargetVerified);
        assert_eq!(assistant.payload["stream_id"], "target_direct");
        assert_eq!(assistant.payload["completion_id"], 1);
        assert_eq!(assistant.payload["role"], "assistant");
        assert_eq!(assistant.payload["text"], "target response");
        assert_eq!(assistant.payload["verified"], true);

        let history = rebuild_history(&services.event_log, &session)
            .await
            .expect("rebuild verified history");
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, "assistant");
        assert_eq!(history[0].content, "target response");
    }

    #[tokio::test]
    async fn error_after_partial_target_output_never_enters_durable_history() {
        let root = std::env::temp_dir().join(format!(
            "hawking_hide_partial_target_turn_{}_{}",
            hide_core::ids::now_ms(),
            std::process::id()
        ));
        let raw = Arc::new(InMemoryEventLog::new());
        let services = BackendServices::new(HideConfig::for_workspace(root), raw);
        let session = services.session();

        let result = run_turn_core(
            Arc::new(ErrorAfterPartialInference),
            services.event_log.clone(),
            services.role_registry.clone(),
            services.code_index.clone(),
            services.memory_store.clone(),
            services.classed_memory.clone(),
            Arc::new(UiEventBus::new(4)),
            session.clone(),
            "user request".to_string(),
            None,
            Some("offline-partial-target-test".to_string()),
            services.repo_instructions.clone(),
        )
        .await;
        assert!(result.is_err(), "error chunk must fail the target turn");
        let events = services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await
            .expect("scan durable event log");
        assert!(events.iter().all(|event| event.kind != "agent.message"));
        assert!(
            rebuild_history(&services.event_log, &session)
                .await
                .expect("rebuild history")
                .is_empty(),
            "no partial target text may become next-turn context"
        );
    }
}
