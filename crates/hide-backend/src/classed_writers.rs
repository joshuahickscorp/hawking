//! Production writers for the six HIDE memory classes.
//!
//! Each write-capability type is **minted at exactly one production site** in
//! this module (or, for [`TurnWriteCap`], at the turn-core entry). Callers
//! outside this module must not mint caps — that keeps a second mint obvious
//! in any diff.
//!
//! | class            | mint site                              | trigger                                      |
//! |------------------|----------------------------------------|----------------------------------------------|
//! | working          | [`WorkingTurnGuard::begin`]            | turn core / kernel turn begins (RAII clear)  |
//! | episodic         | [`EpisodicEventMirror::append`]        | durable event of turn/tool/edit/verdict kind |
//! | semantic_project | [`write_semantic_project_explicit`] + [`maybe_distill_project_from_procedural`] | explicit `memory_add` (repo/session); distill on successful build/test recipe |
//! | procedural       | [`write_procedural_from_receipt`]      | successful command/tool receipt              |
//! | user             | [`write_user_explicit`]                | explicit user-scoped `memory_add` only       |
//! | verification     | [`write_verification_from_receipt`]    | verifier path (`run_static_analysis`) only   |
//!
//! Model-turn generation never holds [`UserWriteCap`] or [`VerifierWriteCap`].

use futures::future::BoxFuture;
use hawking_context::{
    ClassMemoryDraft, ClassedMemorySystem, DynClassedMemory, EpisodicWriteCap, MemoryClass,
    ProceduralWriteCap, ProjectWriteCap, TurnWriteCap, UserWriteCap, VerifierWriteCap,
};
use hide_core::event::{Event, EventLog, NewEvent};
use hide_core::tool::{ToolCall, ToolResult, ToolStatus};
use hide_core::Result;
use hide_kernel::verify_plane::{VerificationReceipt, VerificationTier};
use std::sync::Arc;

/// Hard cap on episodic rows per session. Without this, a long session that
/// never calls [`ClassedMemorySystem::evict_session`] would grow without bound.
/// Oldest rows (ULID order) are pruned after each episodic write that exceeds it.
pub const EPISODIC_SESSION_CAP: usize = 2_048;

// ---------------------------------------------------------------------------
// Working — mint: TurnWriteCap::new at turn-core entry; lifetime: WorkingTurnGuard
// ---------------------------------------------------------------------------

/// Seed working (turn-local) scratch at the start of a production turn.
///
/// **Sole production mint site for [`TurnWriteCap`]** (constructed here via
/// `TurnWriteCap::new`). Prefer [`WorkingTurnGuard::begin`] so the row is
/// cleared on every exit path (Ok / Err / panic), not only the success return.
pub fn write_working_at_turn_start(
    classed: &ClassedMemorySystem,
    turn_id: &str,
    session_id: &str,
    run_id: Option<&str>,
    prompt: &str,
) {
    let cap = TurnWriteCap::new(turn_id);
    let mut draft = ClassMemoryDraft::new(format!("turn_prompt: {prompt}"))
        .with_importance(0.6)
        .with_session(session_id)
        .with_turn(turn_id);
    if let Some(run) = run_id {
        draft = draft.with_run(run);
    }
    let _ = classed.write_working(&cap, "run_turn_core", draft);
}

/// Drop working memory for a finished turn.
///
/// Production turn paths should not call this manually: hold a
/// [`WorkingTurnGuard`] and let `Drop` clear the row on every exit path.
pub fn end_working_turn(classed: &ClassedMemorySystem, turn_id: &str) {
    classed.end_turn(turn_id);
}

/// Fail-safe lifetime for turn-local working memory.
///
/// Owns the minimum needed to clear the row: an [`Arc<ClassedMemorySystem>`]
/// and the turn id. On `Drop`, calls [`ClassedMemorySystem::end_turn`] so the
/// retention boundary holds for normal returns, early `?` / `Err`, cancellation,
/// and panic unwind — not only the success path.
///
/// No disarm: the guard always clears on scope exit. Production
/// `run_turn_core` (sole product SubmitTurn) binds one guard after seeding and never
/// double-clear with a manual [`end_working_turn`].
pub struct WorkingTurnGuard {
    classed: Arc<ClassedMemorySystem>,
    turn_id: String,
}

impl WorkingTurnGuard {
    /// Seed working scratch and return a guard that clears it on drop.
    ///
    /// **Sole production construction path** for the fail-safe working-memory
    /// lifetime. Mints [`TurnWriteCap`] only through
    /// [`write_working_at_turn_start`].
    pub fn begin(
        classed: Arc<ClassedMemorySystem>,
        turn_id: impl Into<String>,
        session_id: &str,
        run_id: Option<&str>,
        prompt: &str,
    ) -> Self {
        let turn_id = turn_id.into();
        write_working_at_turn_start(&classed, &turn_id, session_id, run_id, prompt);
        Self { classed, turn_id }
    }

    /// Turn id this guard will clear.
    pub fn turn_id(&self) -> &str {
        &self.turn_id
    }
}

impl Drop for WorkingTurnGuard {
    fn drop(&mut self) {
        self.classed.end_turn(&self.turn_id);
    }
}

// ---------------------------------------------------------------------------
// Episodic — mint: EpisodicWriteCap::mint inside EpisodicEventMirror::append
// ---------------------------------------------------------------------------

/// Event kinds that count as real session episodes (not every log line).
///
/// One episodic record per such event: turns, tool invocations, edits, verdicts.
pub fn is_episodic_kind(kind: &str) -> bool {
    matches!(
        kind,
        "tool.call"
            | "tool.result"
            | "agent.message"
            | "verify.result"
            | "user.intent.submit_turn"
            | "plan.created"
            | "agent.action"
            | "agent.observation"
    ) || kind.starts_with("diff.")
}

/// Summarize a durable event into a single episodic line (bounded volume).
fn episodic_text(event: &Event) -> String {
    let summary = match event.kind.as_str() {
        "user.intent.submit_turn" => event
            .payload
            .get("args")
            .and_then(|a| a.get("text"))
            .and_then(|t| t.as_str())
            .or_else(|| event.payload.get("text").and_then(|t| t.as_str()))
            .unwrap_or("(submit_turn)")
            .to_string(),
        "tool.call" => {
            let tool = event
                .payload
                .get("tool_name")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            format!("invoke {tool}")
        }
        "tool.result" => {
            let ok = event
                .payload
                .get("ok")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let summary = event
                .payload
                .get("summary")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            format!("result ok={ok} {summary}")
        }
        "agent.message" => {
            let text = event
                .payload
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let clipped: String = text.chars().take(240).collect();
            format!("assistant: {clipped}")
        }
        "verify.result" => {
            let oracle = event
                .payload
                .pointer("/receipt/oracle")
                .or_else(|| event.payload.get("oracle"))
                .and_then(|v| v.as_str())
                .unwrap_or("verify");
            let status = event
                .payload
                .pointer("/receipt/verdict/status")
                .or_else(|| event.payload.pointer("/verdict/status"))
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            format!("verdict {oracle} status={status}")
        }
        kind if kind.starts_with("diff.") => {
            let path = event
                .payload
                .get("path")
                .or_else(|| event.payload.pointer("/proposal/hunks/0/file"))
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            format!("edit {kind} path={path}")
        }
        other => {
            let blob = serde_json::to_string(&event.payload).unwrap_or_default();
            let clipped: String = blob.chars().take(160).collect();
            format!("{other}: {clipped}")
        }
    };
    format!("{} | {}", event.kind, summary)
}

/// Write one episodic record for a durable event that clients also see.
///
/// **Sole production mint site for [`EpisodicWriteCap`].**
pub fn write_episodic_from_event(classed: &ClassedMemorySystem, event: &Event) {
    if !is_episodic_kind(&event.kind) {
        return;
    }
    let cap = EpisodicWriteCap::mint();
    let session = event.session_id.as_str().to_string();
    let mut draft = ClassMemoryDraft::new(episodic_text(event))
        .with_importance(episodic_importance(&event.kind))
        .with_session(&session)
        .with_evidence(vec![
            format!("event_id:{}", event.id.as_str()),
            format!("event_seq:{}", event.seq),
            format!("event_kind:{}", event.kind),
        ]);
    if let Some(run) = event.run_id.as_ref() {
        draft = draft.with_run(run.as_str());
    }
    // Best-effort: never fail the durable event path because memory is full/broken.
    if classed.write_episodic(&cap, "event_stream", draft).is_ok() {
        let _ = classed.prune_episodic_session(&session, EPISODIC_SESSION_CAP);
    }
}

fn episodic_importance(kind: &str) -> f32 {
    match kind {
        "user.intent.submit_turn" | "verify.result" => 0.85,
        "tool.call" | "tool.result" | "agent.message" => 0.7,
        k if k.starts_with("diff.") => 0.75,
        _ => 0.55,
    }
}

/// Event log decorator: every durable append that clients can read also feeds
/// episodic memory (for the filtered kind set). The mirror never mutates the
/// event, never fails an append because of a memory write, and is the sole
/// production mint site for [`EpisodicWriteCap`].
pub struct EpisodicEventMirror {
    inner: hide_core::persistence::DynEventLog,
    classed: DynClassedMemory,
}

impl EpisodicEventMirror {
    pub fn new(inner: hide_core::persistence::DynEventLog, classed: DynClassedMemory) -> Self {
        Self { inner, classed }
    }

    /// Wrap an existing log so production services always mirror episodes.
    pub fn wrap(
        inner: hide_core::persistence::DynEventLog,
        classed: DynClassedMemory,
    ) -> hide_core::persistence::DynEventLog {
        Arc::new(Self::new(inner, classed))
    }
}

impl EventLog for EpisodicEventMirror {
    fn append<'a>(&'a self, event: NewEvent) -> BoxFuture<'a, Result<Event>> {
        Box::pin(async move {
            let recorded = self.inner.append(event).await?;
            write_episodic_from_event(&self.classed, &recorded);
            Ok(recorded)
        })
    }

    fn scan<'a>(
        &'a self,
        session_id: Option<hide_core::ids::SessionId>,
        after_seq: Option<u64>,
        limit: Option<usize>,
    ) -> BoxFuture<'a, Result<Vec<Event>>> {
        self.inner.scan(session_id, after_seq, limit)
    }

    fn compact_before<'a>(&'a self, before_seq: u64) -> BoxFuture<'a, Result<usize>> {
        self.inner.compact_before(before_seq)
    }
}

// ---------------------------------------------------------------------------
// Procedural — mint: ProceduralWriteCap::mint in write_procedural_from_receipt
// ---------------------------------------------------------------------------

/// Whether this tool is a command/build/test receipt worth remembering as a recipe.
pub fn is_procedural_tool(tool_name: &str) -> bool {
    tool_name == "shell.run"
        || tool_name.starts_with("shell.")
        || tool_name.contains("test")
        || tool_name.contains("build")
        || tool_name.contains("cargo")
}

/// A receipt only becomes a recipe when the tool succeeded *and* (for process
/// tools) the exit code is zero / absent.
pub fn receipt_succeeded(result: &ToolResult) -> bool {
    result.status == ToolStatus::Ok && result.exit_code.unwrap_or(0) == 0
}

/// Write a procedural memory record from a successful tool receipt.
///
/// **Sole production mint site for [`ProceduralWriteCap`].**
/// Returns `true` when a record was written.
pub fn write_procedural_from_receipt(
    classed: &ClassedMemorySystem,
    call: &ToolCall,
    result: &ToolResult,
    session_id: &str,
    run_id: Option<&str>,
) -> bool {
    if !is_procedural_tool(&call.tool) || !receipt_succeeded(result) {
        return false;
    }
    let cap = ProceduralWriteCap::mint();
    let summary = procedural_summary(call, result);
    let mut draft = ClassMemoryDraft::new(summary)
        .with_importance(0.8)
        .with_session(session_id)
        .with_evidence(vec![
            format!("tool:{}", call.tool),
            format!("call_id:{}", call.call_id.as_str()),
            format!("status:{:?}", result.status),
        ]);
    if let Some(run) = run_id {
        draft = draft.with_run(run);
    }
    match classed.write_procedural(&cap, "tool_receipt", draft) {
        Ok(rec) => {
            maybe_distill_project_from_procedural(classed, call, &rec.text, session_id, run_id);
            true
        }
        Err(_) => false,
    }
}

fn procedural_summary(call: &ToolCall, result: &ToolResult) -> String {
    let argv = call
        .args
        .get("argv")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .or_else(|| {
            call.args
                .get("command")
                .and_then(|v| v.as_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| call.tool.clone());
    let out = result
        .structured_content
        .as_ref()
        .and_then(|v| v.get("stdout").or_else(|| v.get("summary")))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .chars()
        .take(120)
        .collect::<String>();
    if out.is_empty() {
        format!("recipe ok: {argv}")
    } else {
        format!("recipe ok: {argv} → {out}")
    }
}

// ---------------------------------------------------------------------------
// Semantic project — sole mint: mint_project_cap()
// ---------------------------------------------------------------------------

/// **Sole production mint site for [`ProjectWriteCap`].** Both explicit and
/// distillation writers call this; a second mint would be an obvious new helper.
fn mint_project_cap() -> ProjectWriteCap {
    ProjectWriteCap::mint()
}

/// Explicit project fact write (repo/session-scoped durable memory).
pub fn write_semantic_project_explicit(
    classed: &ClassedMemorySystem,
    claim: &str,
    source: &str,
    author: &str,
    citations: &[String],
    session_id: Option<&str>,
) {
    let cap = mint_project_cap();
    let mut draft = ClassMemoryDraft::new(claim)
        .with_importance(0.85)
        .with_evidence({
            let mut e = vec![
                format!("source:{source}"),
                format!("author:{author}"),
                "write:explicit".to_string(),
            ];
            e.extend(citations.iter().map(|c| format!("citation:{c}")));
            e
        });
    if let Some(sid) = session_id {
        draft = draft.with_session(sid);
    }
    let _ = classed.write_semantic_project(&cap, "project_explicit", draft);
}

/// Distillation rule (stated):
///
/// When a **successful procedural recipe** is a recognized build/test command
/// (`cargo test`/`cargo build`, `npm test`/`npm run build`, `pytest`, `go test`,
/// `make test`/`make`, `bazel test`), promote a one-line durable project fact
/// about the toolchain. Never distill from the model turn or from user intent.
///
/// Uses [`mint_project_cap`] (sole ProjectWriteCap mint).
fn maybe_distill_project_from_procedural(
    classed: &ClassedMemorySystem,
    call: &ToolCall,
    recipe_text: &str,
    session_id: &str,
    run_id: Option<&str>,
) {
    let cmd = call
        .args
        .get("argv")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str())
                .collect::<Vec<_>>()
                .join(" ")
                .to_lowercase()
        })
        .unwrap_or_default();
    let fact = if cmd.contains("cargo test") {
        Some("project toolchain: cargo test succeeds in this workspace")
    } else if cmd.contains("cargo build") {
        Some("project toolchain: cargo build succeeds in this workspace")
    } else if cmd.contains("npm test") || cmd.contains("pnpm test") || cmd.contains("yarn test") {
        Some("project toolchain: JS package test script succeeds in this workspace")
    } else if cmd.contains("npm run build") || cmd.contains("pnpm build") {
        Some("project toolchain: JS package build script succeeds in this workspace")
    } else if cmd.contains("pytest") || cmd.contains("python -m pytest") {
        Some("project toolchain: pytest succeeds in this workspace")
    } else if cmd.contains("go test") {
        Some("project toolchain: go test succeeds in this workspace")
    } else if cmd.contains("make test") || cmd == "make" || cmd.starts_with("make ") {
        Some("project toolchain: make succeeds in this workspace")
    } else if cmd.contains("bazel test") {
        Some("project toolchain: bazel test succeeds in this workspace")
    } else {
        None
    };
    let Some(fact) = fact else {
        return;
    };
    // Avoid flooding: skip if an identical fact already exists.
    if let Ok(existing) = classed.list_class(MemoryClass::SemanticProject) {
        if existing.iter().any(|r| r.text == fact) {
            return;
        }
    }
    let cap = mint_project_cap();
    let mut draft = ClassMemoryDraft::new(fact)
        .with_importance(0.7)
        .with_session(session_id)
        .with_evidence(vec![
            format!("distilled_from:{recipe_text}"),
            format!("tool:{}", call.tool),
            "write:distill_from_procedural".to_string(),
        ]);
    if let Some(run) = run_id {
        draft = draft.with_run(run);
    }
    let _ = classed.write_semantic_project(&cap, "project_distill", draft);
}

// ---------------------------------------------------------------------------
// User — mint: UserWriteCap::mint in write_user_explicit only
// ---------------------------------------------------------------------------

/// Explicit user-scoped preference write.
///
/// **Sole production mint site for [`UserWriteCap`].** Never call from the model
/// turn or distillation path.
pub fn write_user_explicit(
    classed: &ClassedMemorySystem,
    claim: &str,
    source: &str,
    author: &str,
) {
    let cap = UserWriteCap::mint();
    let draft = ClassMemoryDraft::new(claim)
        .with_importance(0.9)
        .with_evidence(vec![
            format!("source:{source}"),
            format!("author:{author}"),
            "write:user_explicit".to_string(),
        ]);
    let _ = classed.write_user(&cap, "user_intent", draft);
}

// ---------------------------------------------------------------------------
// Verification — mint: VerifierWriteCap::mint in write_verification_from_receipt
// ---------------------------------------------------------------------------

/// Map a hide-verify tier + pass/fail into the class evidence_tier vocabulary
/// (`asserted` / `tested` / `proven`).
pub fn evidence_tier_for(tier: VerificationTier, pass: bool) -> &'static str {
    match (tier, pass) {
        (VerificationTier::Tier0Structural | VerificationTier::Tier1Deterministic, true) => {
            "proven"
        }
        (VerificationTier::Tier0Structural | VerificationTier::Tier1Deterministic, false) => {
            "tested"
        }
        (VerificationTier::Tier2Reproduction | VerificationTier::Tier3Environment, _) => "tested",
        (VerificationTier::Tier4Review, _) => "asserted",
    }
}

/// Write verification memory from a durable verifier receipt.
///
/// **Sole production mint site for [`VerifierWriteCap`].** Call only from the
/// verifier path (`BackendHost::run_static_analysis`), never from model turn.
pub fn write_verification_from_receipt(
    classed: &ClassedMemorySystem,
    receipt: &VerificationReceipt,
    findings_summary: &str,
    pass: bool,
    session_id: &str,
    run_id: Option<&str>,
) {
    let cap = VerifierWriteCap::mint();
    let tier = evidence_tier_for(receipt.tier, pass);
    let scope = receipt.scope.join(", ");
    let status = if pass { "pass" } else { "fail" };
    let text = format!(
        "claim checked by {}: {} (tier={:?}, scope=[{scope}])",
        receipt.oracle, status, receipt.tier
    );
    let mut evidence = vec![
        format!("verification_id:{}", receipt.verification_id),
        format!("oracle:{}", receipt.oracle),
        format!("source_hash:{}", receipt.source_hash),
        format!("findings:{findings_summary}"),
    ];
    evidence.extend(receipt.scope.iter().map(|s| format!("scope:{s}")));
    let mut draft = ClassMemoryDraft::new(text)
        .with_importance(if pass { 0.9 } else { 0.85 })
        .with_session(session_id)
        .with_evidence(evidence)
        .with_evidence_tier(tier);
    if let Some(run) = run_id {
        draft = draft.with_run(run);
    }
    let _ = classed.write_verification(&cap, "verifier", draft);
}

/// Route a durable `MemoryDraft` (bible ledger) into the matching classed store.
/// User scope → user class only. Repo/Session → semantic_project. Never writes
/// verification.
pub fn mirror_memory_ledger_to_classes(
    classed: &ClassedMemorySystem,
    scope_kind: &str,
    claim: &str,
    source: &str,
    author: &str,
    citations: &[String],
    session_id: Option<&str>,
) {
    match scope_kind {
        "user" => write_user_explicit(classed, claim, source, author),
        "repo" | "session" => {
            write_semantic_project_explicit(classed, claim, source, author, citations, session_id)
        }
        _ => {}
    }
}

/// Helper for tests / diagnostics: count class rows.
pub fn count_class(classed: &ClassedMemorySystem, class: MemoryClass) -> usize {
    classed.count(class).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::event::{InMemoryEventLog, ToolCallEvent, ToolResultEvent};
    use hide_core::ids::{SessionId, ToolCallId};
    use hide_core::tool::{ToolCall, ToolError, ToolResult, ToolStatus};
    use hide_core::types::EffectSet;
    use hide_kernel::verify_plane::{source_hash, Verdict, VerificationReceipt, VerificationTier};
    use serde_json::json;
    fn mem() -> ClassedMemorySystem {
        ClassedMemorySystem::open_in_memory("ws-writers").unwrap()
    }
    #[test]
    fn procedural_success_writes_record_failure_does_not() {
        let sys = mem();
        let call = ToolCall::new(
            "shell.run",
            json!({ "argv": ["cargo", "test", "-p", "hide-core"] }),
        );
        let ok = ToolResult::ok(
            call.call_id.clone(),
            Some(json!({ "stdout": "test result: ok" })),
            EffectSet::default(),
        );
 assert!(write_procedural_from_receipt( &sys, &call, &ok, "sess-1", Some("run-1"), ));
        let rows = sys.list_class(MemoryClass::Procedural).unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].text.contains("cargo test"));
        assert_eq!(rows[0].provenance.writer, "tool_receipt");
        assert_eq!(rows[0].provenance.run_id.as_deref(), Some("run-1"));
        let proj = sys.list_class(MemoryClass::SemanticProject).unwrap();
        assert!(proj.iter().any(|r| r.text.contains("cargo test")));
        assert_eq!(sys.count(MemoryClass::User).unwrap(), 0);
        assert_eq!(sys.count(MemoryClass::Verification).unwrap(), 0);
        let fail_call = ToolCall::new("shell.run", json!({ "argv": ["false"] }));
        let mut fail = ToolResult::ok(fail_call.call_id.clone(), None, EffectSet::default());
        fail.status = ToolStatus::ToolError;
        fail.ok = false;
        fail.error = Some(ToolError::new("EXEC_FAILED", "exit 1", false));
        let proj_before = sys.count(MemoryClass::SemanticProject).unwrap();
 assert!(!write_procedural_from_receipt( &sys, &fail_call, &fail, "sess-1", Some("run-2"), ));
        assert_eq!(sys.list_class(MemoryClass::Procedural).unwrap().len(), 1);
        assert_eq!(sys.count(MemoryClass::SemanticProject).unwrap(), proj_before);
    }
    #[test]
    fn nonzero_exit_does_not_write_procedural() {
        let sys = mem();
        let call = ToolCall::new("shell.run", json!({ "argv": ["cargo", "test"] }));
        let mut result = ToolResult::ok(call.call_id.clone(), None, EffectSet::default());
        result.exit_code = Some(1);
 assert!(!write_procedural_from_receipt( &sys, &call, &result, "s", None ));
        assert_eq!(sys.count(MemoryClass::Procedural).unwrap(), 0);
        assert_eq!(sys.count(MemoryClass::SemanticProject).unwrap(), 0);
    }
    #[test]
    fn working_cleared_at_turn_end() {
        let sys = mem();
        write_working_at_turn_start(&sys, "turn-1", "sess", Some("run-1"), "scratch me");
        assert_eq!(sys.count(MemoryClass::Working).unwrap(), 1);
        end_working_turn(&sys, "turn-1");
        assert_eq!(sys.count(MemoryClass::Working).unwrap(), 0);
    }
    #[test]
    fn working_guard_clears_on_normal_scope_exit() {
        let sys = Arc::new(mem());
        {
            let guard = WorkingTurnGuard::begin(
                sys.clone(),
                "turn-ok",
                "sess",
                Some("run-ok"),
                "scratch me",
            );
            assert_eq!(guard.turn_id(), "turn-ok");
            assert_eq!(sys.count(MemoryClass::Working).unwrap(), 1);
        }
        assert_eq!(sys.count(MemoryClass::Working).unwrap(), 0);
    }
    #[test]
    fn working_guard_clears_on_early_err() {
        let sys = Arc::new(mem());
        let result: std::result::Result<(), &'static str> = (|| {
            let _guard =
                WorkingTurnGuard::begin(sys.clone(), "turn-err", "sess", Some("run-err"), "x");
            assert_eq!(sys.count(MemoryClass::Working).unwrap(), 1);
            Err("injected early failure after seed")
        })();
        assert!(result.is_err());
        assert_eq!(sys.count(MemoryClass::Working).unwrap(), 0);
    }
    #[test]
    fn working_guard_clears_on_panic_unwind() {
        let sys = Arc::new(mem());
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard =
                WorkingTurnGuard::begin(sys.clone(), "turn-panic", "sess", Some("run-p"), "x");
            assert_eq!(sys.count(MemoryClass::Working).unwrap(), 1);
            panic!("forced unwind after seed");
        }));
        assert!(result.is_err(), "panic must propagate");
        assert_eq!(sys.count(MemoryClass::Working).unwrap(), 0);
    }
    #[test]
    fn episodic_session_cap_prunes_oldest() {
        let sys = mem();
        let cap = EpisodicWriteCap::mint();
        for i in 0..5 {
            let draft = ClassMemoryDraft::new(format!("episode-{i}"))
                .with_session("sess-cap")
                .with_evidence(vec![format!("i:{i}")]);
            sys.write_episodic(&cap, "test", draft).unwrap();
        }
        assert_eq!(sys.count(MemoryClass::Episodic).unwrap(), 5);
        let deleted = sys.prune_episodic_session("sess-cap", 3).unwrap();
        assert_eq!(deleted, 2);
        let rows = sys.list_class(MemoryClass::Episodic).unwrap();
        let sess: Vec<_> = rows
            .into_iter()
            .filter(|r| r.session_id.as_deref() == Some("sess-cap"))
            .collect();
        assert_eq!(sess.len(), 3);
        assert_eq!(sys.prune_episodic_session("sess-cap", 3).unwrap(), 0);
        assert_eq!(sys.count(MemoryClass::Episodic).unwrap(), 3);
    }
    #[tokio::test]
    async fn episodic_mirror_writes_on_submit_turn_tool_and_verdict() {
        let sys = Arc::new(mem());
        let log = EpisodicEventMirror::wrap(Arc::new(InMemoryEventLog::new()), sys.clone());
        let session = SessionId::new();
        log.append(NewEvent::user_intent(
            session.clone(),
            hide_core::event::UserIntentEvent {
                intent: "submit_turn".into(),
                args: json!({ "text": "fix the flaky auth test" }),
            },
        ))
        .await
        .unwrap();
        let mut submit = NewEvent::user_intent(
            session.clone(),
            hide_core::event::UserIntentEvent {
                intent: "submit_turn".into(),
                args: json!({ "text": "unique-episode-marker-xyz" }),
            },
        );
        submit.kind = "user.intent.submit_turn".into();
        log.append(submit).await.unwrap();
        let call_id = ToolCallId::new();
        log.append(NewEvent::tool_call(
            session.clone(),
            ToolCallEvent {
                call_id: call_id.clone(),
                tool_name: "shell.run".into(),
                capability_grant_id: None,
                args: json!({ "argv": ["echo", "hi"] }),
                predicted_effects: EffectSet::default(),
            },
        ))
        .await
        .unwrap();
        log.append(NewEvent::tool_result(
            session.clone(),
            ToolResultEvent {
                call_id,
                ok: true,
                summary: "echoed".into(),
                output: None,
                bytes_ref: None,
            },
        ))
        .await
        .unwrap();
        log.append(NewEvent::system(
            session.clone(),
            "verify.result",
            json!({
                "receipt": { "oracle": "static_analysis", "verdict": { "status": "pass" } },
                "oracle": "static_analysis",
            }),
        ))
        .await
        .unwrap();
        let before = sys.count(MemoryClass::Episodic).unwrap();
        log.append(NewEvent::system(
            session.clone(),
            "runtime.generation",
            json!({ "task": "code" }),
        ))
        .await
        .unwrap();
        assert_eq!(sys.count(MemoryClass::Episodic).unwrap(), before);
        let episodes = sys.list_class(MemoryClass::Episodic).unwrap();
        assert!(episodes .iter() .any(|r| r.text.contains("unique-episode-marker-xyz")));
        assert!(episodes.iter().any(|r| r.text.contains("tool.call")));
        assert!(episodes.iter().any(|r| r.text.contains("tool.result")));
        assert!(episodes.iter().any(|r| r.text.contains("verify.result")));
        for r in &episodes {
            assert_eq!(r.provenance.writer, "event_stream");
            assert_eq!(r.session_id.as_deref(), Some(session.as_str()));
            assert!(!r.provenance.evidence.is_empty());
            assert!(r.provenance.written_at_ms > 0);
            assert_eq!(r.class, MemoryClass::Episodic);
        }
        assert_eq!(sys.count(MemoryClass::User).unwrap(), 0);
        assert_eq!(sys.count(MemoryClass::Verification).unwrap(), 0);
        assert_eq!(sys.count(MemoryClass::Procedural).unwrap(), 0);
    }
    #[test]
    fn verification_write_from_receipt_and_user_explicit() {
        let sys = mem();
        let receipt = VerificationReceipt::new(
            "va-test-1",
            VerificationTier::Tier1Deterministic,
            "static_analysis",
            None,
            vec!["src/lib.rs".into()],
            source_hash(b"fn main() {}"),
            Verdict::Pass,
            1_700_000_000_000,
            12,
        );
        write_verification_from_receipt(&sys, &receipt, "no findings", true, "sess", Some("run-v"));
        let v = sys.list_class(MemoryClass::Verification).unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].provenance.writer, "verifier");
        assert_eq!(v[0].evidence_tier.as_deref(), Some("proven"));
        assert_eq!(v[0].provenance.run_id.as_deref(), Some("run-v"));
        assert_eq!(sys.count(MemoryClass::User).unwrap(), 0);
        assert_eq!(sys.count(MemoryClass::Procedural).unwrap(), 0);
        write_user_explicit(&sys, "prefer terse answers", "settings", "user");
        let u = sys.list_class(MemoryClass::User).unwrap();
        assert_eq!(u.len(), 1);
        assert_eq!(u[0].provenance.writer, "user_intent");
        assert!(u[0].workspace_id.is_none());
        assert_eq!(sys.count(MemoryClass::Verification).unwrap(), 1);
    }
    #[test]
    fn project_explicit_write() {
        let sys = mem();
        write_semantic_project_explicit(
            &sys,
            "crate layout is workspace-of-crates under crates/",
            "memory_add",
            "user",
            &["ARCHITECTURE.md".into()],
            None,
        );
        let rows = sys.list_class(MemoryClass::SemanticProject).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].provenance.writer, "project_explicit");
        assert_eq!(sys.count(MemoryClass::User).unwrap(), 0);
        assert_eq!(sys.count(MemoryClass::Verification).unwrap(), 0);
    }
    #[tokio::test]
    async fn round_trip_write_then_compile_retrieves() {
        use hawking_context::compiler::{CompileInput, ContextCompiler};
        use hawking_context::profiles::ContextProfile;
        use hawking_context::sources::ClassedMemoryContextSource;
        use hawking_context::ClassBudgets;
        use hide_core::ids::ModelId;
        use hide_core::runtime::{ModelArchitecture, ModelDescriptor};
        let sys = Arc::new(mem());
        let log = EpisodicEventMirror::wrap(Arc::new(InMemoryEventLog::new()), sys.clone());
        let session = SessionId::new();
        let mut submit = NewEvent::user_intent(
            session.clone(),
            hide_core::event::UserIntentEvent {
                intent: "submit_turn".into(),
                args: json!({ "text": "roundtrip-marker-alpha-omega" }),
            },
        );
        submit.kind = "user.intent.submit_turn".into();
        log.append(submit).await.unwrap();
        let call = ToolCall::new("shell.run", json!({ "argv": ["cargo", "test"] }));
        let ok = ToolResult::ok(
            call.call_id.clone(),
            Some(json!({ "stdout": "ok" })),
            EffectSet::default(),
        );
 assert!(write_procedural_from_receipt( &sys, &call, &ok, session.as_str(), None, ));
        let budgets = ClassBudgets::default_small();
        let mut compiler = ContextCompiler::new();
        compiler.add_source(
            ClassedMemoryContextSource::new(sys.clone(), budgets).with_session(session.as_str()),
        );
        let model = ModelDescriptor {
            id: ModelId::new(),
            name: "test".into(),
            architecture: ModelArchitecture::Transformer,
            context_tokens: 2048,
            tokenizer_signature: "test".into(),
            footprint_mb: 1,
        };
        let compiled = compiler
            .compile(CompileInput {
                profile: ContextProfile::coding_default(2048),
                model,
                task: "roundtrip-marker-alpha-omega cargo test".into(),
            })
            .await
            .unwrap();
        assert!(compiled.prompt.contains("roundtrip-marker-alpha-omega"));
        let ret = sys.last_retrieval().expect("retrieve_for_compile ran");
        let epi = ret.slice(MemoryClass::Episodic).unwrap();
        assert!(!epi.hits.is_empty(), "episodic slice must have hits");
        let proc = ret.slice(MemoryClass::Procedural).unwrap();
        assert!(!proc.hits.is_empty(), "procedural slice must have hits");
 assert!(epi .hits .iter() .any(|h| h.provenance.writer == "event_stream"));
 assert!(proc .hits .iter() .any(|h| h.provenance.writer == "tool_receipt"));
    }
}
