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
    /// The only durable assistant event produced by this turn.  Keeping this
    /// identifier next to the completion lets non-UI callers prove exactly
    /// which target-verified history record they received.
    pub(crate) assistant_event_id: EventId,
    /// Metrics returned by the actual HTTP inference call.  They remain
    /// optional at the field level because a runtime can legitimately omit
    /// decode timing or completed-forward counters.
    pub(crate) generation_stats: hide_core::runtime::GenerationStats,
    /// The disposition of an explicit source pack for this actual request.
    /// A compact endpoint either admits the complete block after a full-prompt
    /// fit check or records a whole-block omission; it never partially injects
    /// evidence or reports an inaccurate "injected" receipt.
    pub(crate) source_context_disposition: SourceContextDisposition,
}

/// What happened to an explicitly selected HCLI source pack on this concrete
/// request. A compact diagnostic endpoint can inject only a complete pack that
/// fits alongside its actual native prompt and response reserve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SourceContextDisposition {
    NotRequested,
    Injected,
    OmittedWholeBlockForLiveWindow,
}

impl SourceContextDisposition {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::NotRequested => "not_requested",
            Self::Injected => "injected",
            Self::OmittedWholeBlockForLiveWindow => "omitted_whole_block_for_live_window",
        }
    }
}

/// Build the sole native prompt sent by the HCLI turn path. Keeping this
/// formatting in one helper makes compact evidence admission count the exact
/// string that is subsequently handed to the HTTP provider.
fn fold_native_turn_prompt(
    compiled_prompt: &str,
    source_context_block: Option<&str>,
    history_block: &str,
) -> String {
    match (compiled_prompt.trim().is_empty(), source_context_block) {
        (true, Some(evidence)) => format!("{evidence}\n\n{history_block}"),
        (false, Some(evidence)) => {
            format!("{compiled_prompt}\n\n{evidence}\n\n{history_block}")
        }
        (true, None) => history_block.to_string(),
        (false, None) => format!("{compiled_prompt}\n\n{history_block}"),
    }
}

// This is deliberately an opt-in, trace-only forensic hook.  The ordinary HCLI
// path never writes a raw folded prompt or selected context bodies.  When the
// two environment controls below are explicitly set, it records the exact
// pre-provider compilation boundary and then *refuses* before a model request
// can be constructed.  That lets a bounded diagnostic prove what an HCLI turn
// would send without claiming a generation, HCLI pass, or performance result.
const HCLI_COMPILER_TRACE_PATH_ENV: &str = "HAWKING_HCLI_COMPILER_TRACE_PATH";
const HCLI_COMPILER_TRACE_MODE_ENV: &str = "HAWKING_HCLI_COMPILER_TRACE_MODE";
const HCLI_COMPILER_TRACE_MODE: &str = "NEW_DIAGNOSTIC_NOT_HISTORICAL";

fn hcli_compiler_trace_document(
    manifest: &hawking_context::manifest::ContextManifest,
    compiled_prompt: &str,
    folded_prompt: &str,
    history_message_count: usize,
    requested_output_cap: usize,
) -> Value {
    let selected_spans = manifest
        .retained
        .iter()
        .map(|span| {
            json!({
                "content_id": span.id,
                "source": format!("{:?}", span.source).to_lowercase(),
                "title": span.title,
                "text": span.text,
                "token_count": span.token_count,
                "order_index": span.order_index,
                "compacted_from": span.compacted_from,
            })
        })
        .collect::<Vec<_>>();
    json!({
        "schema": "hawking.ascension.hcli_compiler_pre_execution_trace.v1",
        "status": HCLI_COMPILER_TRACE_MODE,
        "capture_timing": "AFTER_CONTEXT_COMPILATION_BEFORE_PROVIDER_OR_MODEL_EXECUTION",
        "model_execution_started": false,
        "process_id": std::process::id(),
        "compiled_context_manifest": manifest,
        "selected_context_spans": selected_spans,
        "compiled_context_prompt_utf8": compiled_prompt,
        "compiled_context_prompt_blake3": blake3::hash(compiled_prompt.as_bytes()).to_hex().to_string(),
        "folded_native_prompt_utf8": folded_prompt,
        "folded_native_prompt_blake3": blake3::hash(folded_prompt.as_bytes()).to_hex().to_string(),
        "history_message_count": history_message_count,
        "requested_output_cap": requested_output_cap,
        "claim_boundary": {
            "new_diagnostic_not_historical": true,
            "does_not_contact_provider_or_execute_a_model": true,
            "does_not_claim_generation_hcli_tps_tg_capability_or_tournament": true,
        },
    })
}

fn maybe_capture_hcli_compiler_pre_execution_trace(
    manifest: &hawking_context::manifest::ContextManifest,
    compiled_prompt: &str,
    folded_prompt: &str,
    history_message_count: usize,
    requested_output_cap: usize,
) -> Result<bool> {
    let Some(path) = std::env::var_os(HCLI_COMPILER_TRACE_PATH_ENV) else {
        return Ok(false);
    };
    if std::env::var(HCLI_COMPILER_TRACE_MODE_ENV).ok().as_deref() != Some(HCLI_COMPILER_TRACE_MODE)
    {
        return Err(hide_core::error::HideError::Config(format!(
            "{HCLI_COMPILER_TRACE_PATH_ENV} requires {HCLI_COMPILER_TRACE_MODE_ENV}={HCLI_COMPILER_TRACE_MODE}"
        )));
    }
    let destination = PathBuf::from(path);
    if !destination.is_absolute() {
        return Err(hide_core::error::HideError::Config(
            "HCLI compiler trace destination must be an absolute path".into(),
        ));
    }
    let parent = destination.parent().ok_or_else(|| {
        hide_core::error::HideError::Config("HCLI compiler trace destination has no parent".into())
    })?;
    if !parent.is_dir() {
        return Err(hide_core::error::HideError::Config(format!(
            "HCLI compiler trace parent does not exist: {}",
            parent.display()
        )));
    }
    if destination.exists() {
        return Err(hide_core::error::HideError::Config(format!(
            "refusing to overwrite existing HCLI compiler trace: {}",
            destination.display()
        )));
    }
    let raw = serde_json::to_vec_pretty(&hcli_compiler_trace_document(
        manifest,
        compiled_prompt,
        folded_prompt,
        history_message_count,
        requested_output_cap,
    ))
    .map_err(|error| {
        hide_core::error::HideError::Config(format!("serialize HCLI compiler trace: {error}"))
    })?;
    let file_name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            hide_core::error::HideError::Config(
                "HCLI compiler trace destination has no UTF-8 filename".into(),
            )
        })?;
    let temporary = parent.join(format!(".{file_name}.{}.tmp", std::process::id()));
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| {
            hide_core::error::HideError::Config(format!(
                "create HCLI compiler trace temporary file {}: {error}",
                temporary.display()
            ))
        })?;
    use std::io::Write;
    let write_result = (|| -> std::io::Result<()> {
        output.write_all(&raw)?;
        output.write_all(b"\n")?;
        output.sync_all()
    })();
    if let Err(error) = write_result {
        let _ = std::fs::remove_file(&temporary);
        return Err(hide_core::error::HideError::Config(format!(
            "write HCLI compiler trace {}: {error}",
            temporary.display()
        )));
    }
    if let Err(error) = std::fs::rename(&temporary, &destination) {
        let _ = std::fs::remove_file(&temporary);
        return Err(hide_core::error::HideError::Config(format!(
            "publish HCLI compiler trace {}: {error}",
            destination.display()
        )));
    }
    Ok(true)
}

/// Accounting for a complete explicit-evidence block against a compact live
/// native prompt. This remains metadata-only: it proves the admission decision
/// without serializing any derivative text into a durable receipt.
#[derive(Debug, Clone, Copy)]
struct CompactSourceWindowFit {
    native_prompt_without_evidence_tokens: usize,
    native_prompt_with_complete_evidence_tokens: usize,
    reserved_output_tokens: usize,
    native_context_budget_tokens: usize,
    fits: bool,
    tokens_estimated: bool,
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
    live_ceiling: Option<(Option<usize>, Option<usize>, usize, Option<usize>)>,
    run_id_label: Option<String>,
    repo_instructions: Arc<crate::compat_instructions::ResolvedInstructions>,
    // Optional caller-requested upper bound. It can only reduce the
    // context-derived cap; the compiled prompt remains authoritative.
    requested_output_cap: Option<usize>,
    // An explicit, bounded pack of local object-store derivatives selected by
    // HCLI. The type deliberately exposes a model-facing derivative prompt,
    // never raw object bytes. It is injected for this invocation only and is
    // not folded into durable user history.
    source_context: Option<&crate::hcli_sources::HcliSourceContext>,
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
    let live_native = live_ceiling.and_then(|(_, n, _, _)| n).filter(|n| *n > 0);
    let live_effective = live_ceiling.map(|(_, _, c, _)| c).filter(|c| *c > 0);
    let capability =
        declare_turn_capability(config_native, live_native, live_effective, None, false);
    // Pack against measured/config native (conservative). Effective is reported
    // separately and is never treated as a larger native window.
    let max_input = capability.pack_budget_tokens(false);
    if max_input == 0 {
        return Err(hide_core::error::HideError::Config(
            "the selected runtime did not expose a positive native context window".to_string(),
        ));
    }
    // A live endpoint can legitimately report a diagnostic-scale window below
    // HCLI's ordinary repository-context floor.  In that case keep the durable
    // turn, but do not fill its entire native budget with optional repository
    // or memory material and force the endpoint to reject a silently
    // over-window prompt. A selected attachment is also an optional reference
    // block: admit it only after the complete native prompt plus its output
    // reserve fits, otherwise preserve its selection receipt and omit the
    // whole block.
    let low_context_mode = max_input < 256;
    // Tokenizer-true packing when a real tokenizer is discoverable (bible §4.2).
    let counter = hawking_context::TokenCounter::discover_from_env()
        .unwrap_or_else(hawking_context::TokenCounter::heuristic);
    // Reserve the full selected-evidence block *before* compiling repository
    // context on normal endpoints. A compact endpoint has no optional
    // repository/memory pack; after reconstructing the actual native prompt
    // below, it can admit the complete selected block only when that full
    // prompt plus the response reserve fits its observed context. It never
    // sends an evidence prefix merely to make it fit.
    let selected_source_context =
        source_context.filter(|context| !context.model_prompt().trim().is_empty());
    let mut source_context_disposition = match (selected_source_context.is_some(), low_context_mode)
    {
        (false, _) => SourceContextDisposition::NotRequested,
        (true, true) => SourceContextDisposition::OmittedWholeBlockForLiveWindow,
        (true, false) => SourceContextDisposition::Injected,
    };
    let mut source_context = (!low_context_mode)
        .then_some(selected_source_context)
        .flatten();
    let mut omitted_source_context = low_context_mode
        .then_some(selected_source_context)
        .flatten();
    let mut source_context_tokens = source_context
        .map(|context| counter.count(context.model_prompt()))
        .unwrap_or(0);
    let live_output_cap = live_ceiling
        .and_then(|(_, _, _, cap)| cap)
        .filter(|cap| *cap > 0);
    let requested_output_cap = match (requested_output_cap, live_output_cap) {
        (Some(requested), Some(live)) => Some(requested.min(live)),
        (Some(requested), None) => Some(requested),
        (None, Some(live)) => Some(live),
        (None, None) => None,
    };
    let min_output_reserve = requested_output_cap
        .unwrap_or(256)
        .min(max_input.saturating_sub(1))
        .max(1);
    let compiler_input_budget = if source_context_tokens == 0 {
        max_input
    } else {
        max_input
            .checked_sub(source_context_tokens.saturating_add(min_output_reserve))
            .filter(|budget| *budget > 0)
            .ok_or_else(|| {
                hide_core::error::HideError::Config(format!(
                    "selected local evidence requires {source_context_tokens} prompt tokens, leaving less than the {min_output_reserve}-token response reserve in this {max_input}-token HCLI context budget"
                ))
            })?
    };
    let mut model = role.model.clone();
    model.context_tokens = compiler_input_budget;
    let mut compiler = ContextCompiler::new().with_counter(counter.clone());
    if !low_context_mode {
        compiler.add_source(CodeIndexContextSource::new(code_index, 16));
        // Six memory classes: independent per-class budgets (not one kind filter).
        let class_budgets = ClassBudgets::from_total((max_input / 8).max(64));
        compiler.add_source(
            ClassedMemoryContextSource::new(classed_memory.clone(), class_budgets)
                .with_session(session_id.as_str())
                .with_turn(turn_id.clone()),
        );
    }
    // Bible sec 20 / sec 78.1 #11: fold the repo's resolved Claude Code migration
    // instructions (CLAUDE.md tree + un-scoped rules) into the compiled context as
    // a pinned instruction/system source, honoring precedence (read-last-wins).
    // Added only when the repo actually carries them (an un-migrated repo resolves
    // empty and this is a no-op).
    if !low_context_mode && !repo_instructions.is_empty() {
        compiler.add_source(repo_instructions.as_source());
    }
    let mut compiled = compiler
        .compile(CompileInput {
            profile: ContextProfile::coding_default(compiler_input_budget),
            model,
            task: prompt.clone(),
        })
        .await?;
    // Pre-stream live reading (when the ceiling was snapshotted) so rot/meter
    // can include occupancy before generation advances.
    let pre_live = live_ceiling.map(|(state_bytes, native, ceiling, _)| {
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
    // `messages`), so this is the actual attachment injection boundary.
    // Evidence remains separate from reconstructed history: it applies once,
    // is labelled untrusted reference material by the source selector, and
    // cannot make a later session turn appear to have received attachments it
    // did not explicitly select.
    // The compact-window decision is made after history reconstruction and
    // after the real compiler output is known. That is the exact native prompt
    // shape that `HttpModelProvider` will send, rather than an estimate made
    // from the evidence block in isolation.
    let mut compact_source_fit = None;
    if low_context_mode {
        if let Some(candidate) = selected_source_context {
            let prompt_without_evidence =
                fold_native_turn_prompt(&compiled.prompt, None, &history_block);
            let prompt_with_complete_evidence = fold_native_turn_prompt(
                &compiled.prompt,
                Some(candidate.model_prompt()),
                &history_block,
            );
            let native_prompt_without_evidence_tokens = counter.count(&prompt_without_evidence);
            let native_prompt_with_complete_evidence_tokens =
                counter.count(&prompt_with_complete_evidence);
            let fits = native_prompt_with_complete_evidence_tokens
                .saturating_add(min_output_reserve)
                <= max_input;
            compact_source_fit = Some(CompactSourceWindowFit {
                native_prompt_without_evidence_tokens,
                native_prompt_with_complete_evidence_tokens,
                reserved_output_tokens: min_output_reserve,
                native_context_budget_tokens: max_input,
                fits,
                tokens_estimated: !counter.is_accurate(),
            });
            if fits {
                source_context = Some(candidate);
                omitted_source_context = None;
                source_context_disposition = SourceContextDisposition::Injected;
            }
        }
    }
    source_context_tokens = source_context
        .map(|context| counter.count(context.model_prompt()))
        .unwrap_or(0);
    let folded_prompt = fold_native_turn_prompt(
        &compiled.prompt,
        source_context.map(|context| context.model_prompt()),
        &history_block,
    );
    let folded_prompt_tokens = counter.count(&folded_prompt);
    let prompt_chars = folded_prompt.len();

    // --- (S2) Derive the output budget from the window minus what context ate,
    // clamped to a sane band - replacing the hard-coded 256 facade. ---
    // `HIDE_MAX_OUTPUT_TOKENS` (positive int) is an optional hard cap for live
    // smoke / small-model turns; it never *raises* the derived budget.
    let derived = if low_context_mode {
        // Compact admission above proved this prompt plus the reserve fits
        // when evidence was selected. Count the actual final native prompt so
        // the output budget cannot quietly reclaim evidence/headroom.
        max_input
            .saturating_sub(folded_prompt_tokens)
            .clamp(min_output_reserve, 2048)
    } else {
        max_input
            .saturating_sub(compiled.manifest.used_tokens)
            .saturating_sub(source_context_tokens)
            .clamp(min_output_reserve, 2048)
    };
    let out_budget = std::env::var("HIDE_MAX_OUTPUT_TOKENS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
        .map(|cap| derived.min(cap))
        .unwrap_or(derived);
    let out_budget = requested_output_cap
        .filter(|cap| *cap > 0)
        .map(|cap| out_budget.min(cap))
        .unwrap_or(out_budget);

    if maybe_capture_hcli_compiler_pre_execution_trace(
        &compiled.manifest,
        &compiled.prompt,
        &folded_prompt,
        messages.len(),
        out_budget,
    )? {
        return Err(hide_core::error::HideError::PolicyDenied(
            "HCLI compiler trace captured before provider/model execution; trace-only mode intentionally refuses generation".into(),
        ));
    }

    // Durable marker: compile stats + honest capability / rot / meter.
    // The compile receipt lives on the event log (not a pre-token Wire-B patch)
    // so token-first consumers (flagship boot path) are not starved of TokenBatch.
    // Post-turn generate_submit_turn re-emits capability+rot+meter on the live
    // context_manifest projection.
    let mut compiled_payload = context_compiled_payload(
        &compiled.manifest,
        Some(out_budget),
        "single_shot",
        run_id_label.as_deref(),
    );
    match (source_context, omitted_source_context) {
        (Some(context), _) => {
            compiled_payload["explicit_source_context"] = json!({
                "status": source_context_disposition.as_str(),
                "reserved_prompt_tokens": source_context_tokens,
                "token_counting_matches_compiler": context.tokens_estimated() == !counter.is_accurate(),
                "selection": context.receipt_json(),
                "persisted_as_user_history": false,
            });
        }
        (_, Some(context)) => {
            compiled_payload["explicit_source_context"] = json!({
                "status": source_context_disposition.as_str(),
                "selection": context.receipt_json(),
                "model_prompt_omitted": true,
                "persisted_as_user_history": false,
                "reason": "the complete selected evidence block plus the reconstructed native prompt and reserved output did not fit the observed compact context, so the whole block was omitted rather than truncated",
            });
        }
        (None, None) => {}
    }
    if let Some(fit) = compact_source_fit {
        compiled_payload["explicit_source_context"]["compact_window_fit"] = json!({
            "native_prompt_without_evidence_tokens": fit.native_prompt_without_evidence_tokens,
            "native_prompt_with_complete_evidence_tokens": fit.native_prompt_with_complete_evidence_tokens,
            "reserved_output_tokens": fit.reserved_output_tokens,
            "native_context_budget_tokens": fit.native_context_budget_tokens,
            "combined_prompt_and_reserve_tokens": fit
                .native_prompt_with_complete_evidence_tokens
                .saturating_add(fit.reserved_output_tokens),
            "fits": fit.fits,
            "tokens_estimated": fit.tokens_estimated,
            "counting_method": if fit.tokens_estimated {
                "deterministic chars/4 estimate; no tokenizer was discovered"
            } else {
                "tokenizer-backed count from the locally discovered tokenizer"
            },
        });
    }
    event_log
        .append(NewEvent::system(
            session_id.clone(),
            "context.compiled",
            compiled_payload,
        ))
        .await?;

    // Context receipt: which repo instruction files (CLAUDE.md tree + un-scoped
    // rules) folded into this turn's context, in launch order. Logged only when
    // the repo carried migration instructions.
    if !low_context_mode && !repo_instructions.is_empty() {
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
    let generation_stats = {
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
                        if let Some((state_bytes, native, ceiling, _)) = live_ceiling {
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
        runtime.generate(request, &mut sink).await?
    };

    // (S2) Persist the assistant turn through the sole target-verified output
    // authority so the NEXT turn's `rebuild_history` cannot consume a raw or
    // provisional model completion.
    let assistant_event =
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
        assistant_event_id: assistant_event.id,
        generation_stats,
        source_context_disposition,
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
    use std::sync::Mutex;

    struct ErrorAfterPartialInference;

    #[derive(Default)]
    struct RecordingInference {
        requests: Mutex<Vec<InferenceRequest>>,
    }

    impl InferenceClient for RecordingInference {
        fn generate<'a>(
            &'a self,
            request: InferenceRequest,
            sink: TokenSink<'a>,
        ) -> BoxFuture<'a, Result<GenerationStats>> {
            Box::pin(async move {
                self.requests.lock().unwrap().push(request);
                sink(StreamChunk::Token {
                    token_id: None,
                    text: "captured completion".to_string(),
                })?;
                Ok(GenerationStats {
                    input_tokens: 1,
                    output_tokens: 2,
                    decode_ms: None,
                    completed_decode_forwards: None,
                    decode_tokens_per_second: None,
                })
            })
        }

        fn embed<'a>(&'a self, _text: &'a str) -> BoxFuture<'a, Result<Vec<f32>>> {
            Box::pin(async move { Ok(Vec::new()) })
        }
    }

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
                    decode_ms: None,
                    completed_decode_forwards: None,
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
            None,
            None,
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
            None,
            None,
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

    #[tokio::test]
    async fn selected_local_source_is_injected_once_without_entering_durable_history() {
        let workspace = tempfile::tempdir().unwrap();
        let source_path = workspace.path().join("evidence.txt");
        let selected_fact = "HCLI_SELECTED_FACT_9d6e0b";
        std::fs::write(&source_path, selected_fact).unwrap();
        let sources = crate::hcli_sources::HcliSourceStore::open(workspace.path()).unwrap();
        let ingested = sources.ingest_file(&source_path, None, None).unwrap();
        let source_context = sources
            .select_context(&[ingested.reference.id.as_str().to_string()])
            .unwrap();

        let services = BackendServices::new(
            HideConfig::for_workspace(workspace.path()),
            Arc::new(InMemoryEventLog::new()),
        );
        let session = services.session();
        let inference = Arc::new(RecordingInference::default());
        let outcome = run_turn_core(
            inference.clone(),
            services.event_log.clone(),
            services.role_registry.clone(),
            services.code_index.clone(),
            services.memory_store.clone(),
            services.classed_memory.clone(),
            Arc::new(UiEventBus::new(4)),
            session.clone(),
            "answer from the evidence".to_string(),
            None,
            Some("source-context-test".to_string()),
            services.repo_instructions.clone(),
            Some(512),
            Some(&source_context),
        )
        .await
        .unwrap();
        assert_eq!(outcome.completion, "captured completion");

        let request = inference
            .requests
            .lock()
            .unwrap()
            .pop()
            .expect("one real inference request");
        assert!(request.prompt.contains(selected_fact));
        assert!(request.prompt.contains("<source untrusted=\"true\">"));
        assert!(request
            .messages
            .iter()
            .all(|message| !message.content.contains(selected_fact)));

        let events = services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await
            .unwrap();
        let compiled = events
            .iter()
            .find(|event| event.kind == "context.compiled")
            .expect("durable context receipt");
        assert_eq!(
            compiled.payload["explicit_source_context"]["status"],
            "injected"
        );
        assert_eq!(
            compiled.payload["explicit_source_context"]["selection"]["selected_sources"][0]
                ["reference_id"],
            ingested.reference.id.as_str()
        );
        assert_eq!(
            compiled.payload["explicit_source_context"]["selection"]["selected_sources"][0]
                ["content_hash"],
            ingested.record.content_hash.as_str()
        );
        assert!(
            !compiled.payload.to_string().contains(selected_fact),
            "durable receipt must prove selection without echoing derivative text"
        );
        let history = rebuild_history(&services.event_log, &session)
            .await
            .unwrap();
        assert!(history
            .iter()
            .all(|message| !message.content.contains(selected_fact)));
    }

    #[tokio::test]
    async fn compact_live_window_omits_the_entire_selected_source_block() {
        let workspace = tempfile::tempdir().unwrap();
        let source_path = workspace.path().join("evidence.txt");
        let selected_fact = "HCLI_COMPACT_SELECTED_FACT_7f3a";
        // Make the selected derivative nontrivial: this is intentionally the
        // kind of attachment that must not be sliced down to fit a 128-token
        // diagnostic endpoint.
        std::fs::write(
            &source_path,
            format!("{selected_fact}\n{}", "evidence ".repeat(128)),
        )
        .unwrap();
        let sources = crate::hcli_sources::HcliSourceStore::open(workspace.path()).unwrap();
        let ingested = sources.ingest_file(&source_path, None, None).unwrap();
        let source_context = sources
            .select_context(&[ingested.reference.id.as_str().to_string()])
            .unwrap();

        let services = BackendServices::new(
            HideConfig::for_workspace(workspace.path()),
            Arc::new(InMemoryEventLog::new()),
        );
        let session = services.session();
        let inference = Arc::new(RecordingInference::default());
        let outcome = run_turn_core(
            inference.clone(),
            services.event_log.clone(),
            services.role_registry.clone(),
            services.code_index.clone(),
            services.memory_store.clone(),
            services.classed_memory.clone(),
            Arc::new(UiEventBus::new(4)),
            session.clone(),
            "answer the bounded user request".to_string(),
            Some((None, Some(128), 128, Some(4))),
            Some("compact-source-context-test".to_string()),
            services.repo_instructions.clone(),
            Some(4),
            Some(&source_context),
        )
        .await
        .unwrap();
        assert_eq!(
            outcome.source_context_disposition,
            SourceContextDisposition::OmittedWholeBlockForLiveWindow
        );

        let request = inference
            .requests
            .lock()
            .unwrap()
            .pop()
            .expect("one real inference request");
        assert_eq!(request.max_output_tokens, 4);
        assert!(
            !request.prompt.contains(selected_fact),
            "the evidence must be omitted as a whole, not partially injected"
        );
        assert!(!request.prompt.contains("<source untrusted=\"true\">"));

        let events = services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await
            .unwrap();
        let compiled = events
            .iter()
            .find(|event| event.kind == "context.compiled")
            .expect("durable context receipt");
        assert_eq!(
            compiled.payload["explicit_source_context"]["status"],
            "omitted_whole_block_for_live_window"
        );
        assert_eq!(
            compiled.payload["explicit_source_context"]["compact_window_fit"]["fits"],
            false
        );
        assert!(
            compiled.payload["explicit_source_context"]["compact_window_fit"]
                ["combined_prompt_and_reserve_tokens"]
                .as_u64()
                .unwrap()
                > compiled.payload["explicit_source_context"]["compact_window_fit"]
                    ["native_context_budget_tokens"]
                    .as_u64()
                    .unwrap()
        );
        assert_eq!(
            compiled.payload["explicit_source_context"]["selection"]["selected_sources"][0]
                ["reference_id"],
            ingested.reference.id.as_str()
        );
        assert!(
            !compiled.payload.to_string().contains(selected_fact),
            "the receipt records selection metadata, never derivative text"
        );
    }

    #[tokio::test]
    async fn compact_live_window_injects_complete_selected_source_when_full_prompt_fits() {
        let workspace = tempfile::tempdir().unwrap();
        let source_path = workspace.path().join("tiny-evidence.txt");
        let selected_fact = "HCLI_COMPACT_TINY_FACT_f81b";
        std::fs::write(&source_path, selected_fact).unwrap();
        let sources = crate::hcli_sources::HcliSourceStore::open(workspace.path()).unwrap();
        let ingested = sources.ingest_file(&source_path, None, None).unwrap();
        let source_context = sources
            .select_context(&[ingested.reference.id.as_str().to_string()])
            .unwrap();

        let services = BackendServices::new(
            HideConfig::for_workspace(workspace.path()),
            Arc::new(InMemoryEventLog::new()),
        );
        let session = services.session();
        services
            .event_log
            .append(NewEvent::system(
                session.clone(),
                "user.intent.submit_turn",
                json!({ "args": { "text": "Earlier durable context must remain visible." } }),
            ))
            .await
            .unwrap();
        let inference = Arc::new(RecordingInference::default());
        let outcome = run_turn_core(
            inference.clone(),
            services.event_log.clone(),
            services.role_registry.clone(),
            services.code_index.clone(),
            services.memory_store.clone(),
            services.classed_memory.clone(),
            Arc::new(UiEventBus::new(4)),
            session.clone(),
            "What local fact was selected?".to_string(),
            Some((None, Some(128), 128, Some(4))),
            Some("compact-source-context-fit-test".to_string()),
            services.repo_instructions.clone(),
            Some(4),
            Some(&source_context),
        )
        .await
        .unwrap();
        assert_eq!(
            outcome.source_context_disposition,
            SourceContextDisposition::Injected
        );

        let request = inference
            .requests
            .lock()
            .unwrap()
            .pop()
            .expect("one real inference request");
        assert_eq!(request.max_output_tokens, 4);
        assert!(
            request.prompt.contains(selected_fact),
            "the complete selected evidence block must be present when it fits"
        );
        assert!(request.prompt.contains("<source untrusted=\"true\">"));
        assert!(
            request
                .prompt
                .contains("Earlier durable context must remain visible."),
            "compact admission must count the reconstructed native history, not only the current prompt"
        );

        let events = services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await
            .unwrap();
        let compiled = events
            .iter()
            .find(|event| event.kind == "context.compiled")
            .expect("durable context receipt");
        assert_eq!(
            compiled.payload["explicit_source_context"]["status"],
            "injected"
        );
        let fit = &compiled.payload["explicit_source_context"]["compact_window_fit"];
        assert_eq!(fit["fits"], true);
        assert_eq!(fit["native_context_budget_tokens"], 128);
        assert_eq!(fit["reserved_output_tokens"], 4);
        let counter = hawking_context::TokenCounter::discover_from_env()
            .unwrap_or_else(hawking_context::TokenCounter::heuristic);
        assert_eq!(
            fit["native_prompt_with_complete_evidence_tokens"],
            counter.count(&request.prompt)
        );
        assert!(
            fit["combined_prompt_and_reserve_tokens"].as_u64().unwrap()
                <= fit["native_context_budget_tokens"].as_u64().unwrap()
        );
        assert!(
            !compiled.payload.to_string().contains(selected_fact),
            "the receipt records admission metadata, never derivative text"
        );
    }

    #[test]
    fn compiler_trace_document_is_explicitly_pre_execution_and_marks_raw_prompt_scope() {
        let manifest = hawking_context::manifest::ContextManifest::new(256);
        let trace = hcli_compiler_trace_document(
            &manifest,
            "selected context",
            "selected context\n\nuser: diagnostic prompt",
            1,
            8,
        );
        assert_eq!(trace["status"], HCLI_COMPILER_TRACE_MODE);
        assert_eq!(trace["model_execution_started"], false);
        assert_eq!(
            trace["folded_native_prompt_utf8"],
            "selected context\n\nuser: diagnostic prompt"
        );
        assert_eq!(
            trace["claim_boundary"]["does_not_contact_provider_or_execute_a_model"],
            true
        );
    }
}
