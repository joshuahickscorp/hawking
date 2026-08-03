use crate::security::SecurityServices;
use crate::speculation_safety::VerifiedEffectPermit;
use hide_core::config::HideConfig;
use hide_core::permission::{PermissionEngine, PermissionRequest, PermissionVerdict};
use hide_core::tool::{ToolCall, ToolDispatcher, ToolRegistry, ToolResult};
use hide_core::types::{Decision, EffectKind};
use serde::{Deserialize, Serialize};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

pub fn build_default_tool_registry() -> ToolRegistry {
    let registry = ToolRegistry::default();
    // Unit tests assert EXEC behaviour; nested sandbox-exec is flaky under
    // agent/CI seats. Profile/SBPL coverage lives in hide-kernel::security tests.
    let shell = hide_kernel::tooling::ShellConfig {
        disable_sandbox: cfg!(test),
        ..Default::default()
    };
    hide_kernel::tooling::register_builtin_tools_with(&registry, shell);
    registry
}

tokio::task_local! {
    /// Set for exactly the span of a released security gate. Task-scoped on purpose: a concurrent
    /// turn on another task never sees another task's approval.
    static GATE_RELEASED: ();
}

tokio::task_local! {
    /// Who a dispatch is FOR: the session it belongs to and the run it groups under. Task-scoped,
    /// like the gate scope above, so two concurrent turns never read each other's attribution.
    static DISPATCH_CTX: DispatchContext;
}

/// The session (and optional run) a tool dispatch is attributed to. The recorder reads it to key
/// the durable `tool.call`/`tool.result` pair and the diff a write is reviewable through; the write
/// lease reads it to refuse a write from a session the grant did not name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchContext {
    pub session_id: hide_core::ids::SessionId,
    pub run_id: Option<hide_core::ids::RunId>,
    pub origin: DispatchOrigin,
}

/// Causal class for a tool-dispatch span. Model-origin dispatch is denied for
/// any effect unless a preceding durable target-verified token supplied the
/// opaque permit below. Human/system dispatch stays distinct: user approval is
/// represented by the existing permission and gate machinery, not by a model
/// token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchOrigin {
    HumanOrSystem,
    UnverifiedModel,
    TargetVerifiedModel {
        token_id: u32,
        canonical_event_id: hide_core::ids::EventId,
    },
}

impl DispatchContext {
    pub fn human_or_system(
        session_id: hide_core::ids::SessionId,
        run_id: Option<hide_core::ids::RunId>,
    ) -> Self {
        Self {
            session_id,
            run_id,
            origin: DispatchOrigin::HumanOrSystem,
        }
    }

    pub fn unverified_model(
        session_id: hide_core::ids::SessionId,
        run_id: Option<hide_core::ids::RunId>,
    ) -> Self {
        Self {
            session_id,
            run_id,
            origin: DispatchOrigin::UnverifiedModel,
        }
    }

    fn target_verified_model(
        session_id: hide_core::ids::SessionId,
        run_id: Option<hide_core::ids::RunId>,
        permit: VerifiedEffectPermit,
        call: &ToolCall,
    ) -> hide_core::Result<Self> {
        let (token_id, canonical_event_id) = permit.consume_for(&session_id, &run_id, call)?;
        Ok(Self {
            session_id,
            run_id,
            origin: DispatchOrigin::TargetVerifiedModel {
                token_id,
                canonical_event_id,
            },
        })
    }
}

/// Attribute every dispatch inside `fut` to this session and run.
pub async fn with_dispatch_context<T>(
    session_id: hide_core::ids::SessionId,
    run_id: Option<hide_core::ids::RunId>,
    fut: impl std::future::Future<Output = T>,
) -> T {
    DISPATCH_CTX
        .scope(DispatchContext::human_or_system(session_id, run_id), fut)
        .await
}

/// Run one model-origin tool effect with an opaque permit minted after a
/// durable target-verified token event. The permit is checked against, and
/// consumed by, this exact call before the dispatcher can observe it.
pub async fn dispatch_verified_model_tool_effect(
    session_id: hide_core::ids::SessionId,
    run_id: Option<hide_core::ids::RunId>,
    permit: VerifiedEffectPermit,
    call: ToolCall,
    dispatcher: &ToolDispatcher,
) -> hide_core::Result<ToolResult> {
    let context = DispatchContext::target_verified_model(session_id, run_id, permit, &call)?;
    Ok(DISPATCH_CTX.scope(context, dispatcher.dispatch(call)).await?)
}

/// The attribution in force on this task, if any.
pub fn dispatch_context() -> Option<DispatchContext> {
    DISPATCH_CTX.try_with(|c| c.clone()).ok()
}

/// Run `fut` as the effect of a security gate the user just approved.
///
/// This is THE approved-write path. The shipped `SecurityConfig` default is
/// `workspace_write_default = Decision::Ask`, and [`ToolDispatcher::dispatch`] turns any non-Allow
/// verdict into `PolicyDenied`, so without this every write a released gate performs is refused:
/// the whole-diff revert, the per-hunk revert a code rewind peels off, the editor save. The
/// relaxation lives HERE, around the one release entry point, rather than in a dispatcher cloned
/// per release arm, because a per-arm clone only ever fixes the arm somebody noticed.
pub async fn with_approved_writes<T>(fut: impl std::future::Future<Output = T>) -> T {
    GATE_RELEASED.scope((), fut).await
}

/// Whether the current task is running the effect of a gate the user just released.
///
/// The permission engine below reads this for WRITES; the host reads it for the approval-gated
/// EFFECTS themselves, so a command the catalog marks `Ask` is refused on every channel that can
/// reach it (`/v1/hide/rpc`, an in-process caller) instead of only on the one transport the gate
/// was bolted to.
pub fn gate_released() -> bool {
    GATE_RELEASED.try_with(|_| ()).is_ok()
}

// --- The task-scoped transactional write lease -----------------------------------------------
//
// [`with_approved_writes`] above widens the policy for the duration of ONE released intent. That is
// the right span for a single approved effect and the wrong one for a task: an agent implementing a
// change writes many files across many steps, and on the shipped `workspace_write_default = Ask`
// every one of those writes is refused, so the diff store stays empty and the product cannot do its
// job. The lease widens the SAME wrapper from one intent to one approved task, bounded by declared
// paths, instead of standing up a second permission system.
//
// What the lease relaxes is exactly `fs.write` to a target inside a declared scope. Everything else
// keeps its existing verdict, which is what keeps the non-permitted list out without naming it:
// `shell.exec` (destructive repo operations, force push, deployment, package publishing) and
// `git.write` are different capability kinds; network effects are forced to `Ask` by the inner
// engine before this ever sees them; and the approval-gated EFFECTS are guarded by
// [`gate_released`], which a lease deliberately does not set, so `/v1/hide/rpc` and the connector
// route cannot reach a gated effect with a lease any more than they could without one.

/// A write lease over one approved task.
///
/// Held in process memory ONLY (see [`LEASE`]), so a restart INVALIDATES it: the policy is
/// fail-closed, the user re-approves the task after a restart rather than inheriting an
/// authorization nobody is watching any more.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WriteLease {
    pub lease_id: String,
    /// The workspace-graph repo the scope was read from. It was TRUSTED at grant time; losing that
    /// trust revokes.
    pub repo_id: String,
    /// The session whose task this is. Closing, forking or switching it revokes.
    pub session_id: Option<String>,
    /// The run this task is, when the grant named one. Its completion revokes.
    pub run_id: Option<String>,
    /// The declared scope: absolute, normalized roots. A write outside every one of them is not
    /// covered and stays on the gate path.
    pub scopes: Vec<PathBuf>,
    pub granted_ms: u64,
}

/// How long a lease stays in force without being re-granted (30 minutes). A task the user walked
/// away from stops being authorized; re-approving is one click, an unbounded standing write grant
/// is not recoverable.
pub const LEASE_TTL_MS: u64 = 30 * 60 * 1000;

impl WriteLease {
    /// Whether `target` lies inside a declared scope. Containment is checked on the REAL path (the
    /// deepest existing ancestor is canonicalized first, both sides), so a symlink inside the scope
    /// pointing outside it does not carry the lease out of the repo; the lexical normalization is
    /// the fallback for the create case, where the target does not exist yet.
    pub fn covers(&self, target: &str) -> bool {
        let target = real_path(Path::new(target));
        target.is_absolute()
            && self
                .scopes
                .iter()
                .any(|scope| target.starts_with(real_path(scope)))
    }

    /// Whether this lease is still in force for a write attributed to `session`.
    ///
    /// A lease authorizes ONE task: the session the grant named (an unattributed dispatch, e.g. a
    /// headless in-process caller, is not that session and is not covered), and only until
    /// [`LEASE_TTL_MS`] has passed.
    pub fn in_force_for(&self, session: Option<&str>, now_ms: u64) -> bool {
        let same_session = match (&self.session_id, session) {
            (Some(granted), Some(actual)) => granted == actual,
            // A grant that named no session is the pre-session shape; it stays path-scoped.
            (None, _) => true,
            (Some(_), None) => false,
        };
        same_session && self.within_ttl(now_ms)
    }

    /// Whether the grant is still inside [`LEASE_TTL_MS`]. Shared by the enforcement check above and
    /// by the status read a fresh client makes, so a lapsed lease cannot be enforced by one and
    /// reported as active by the other.
    pub fn within_ttl(&self, now_ms: u64) -> bool {
        now_ms.saturating_sub(self.granted_ms) <= LEASE_TTL_MS
    }
}

/// Resolve `path` to a real path: canonicalize the deepest ancestor that exists (so a symlinked
/// directory resolves to its target) and re-attach the components below it, lexically normalized.
/// Filesystem-truthful where the filesystem has an answer, lexical where it does not.
pub(crate) fn real_path(path: &Path) -> PathBuf {
    let path = normalize(path);
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    let mut head = path.as_path();
    loop {
        if let Ok(real) = head.canonicalize() {
            let mut out = real;
            for part in tail.iter().rev() {
                out.push(part);
            }
            return out;
        }
        match (head.file_name(), head.parent()) {
            (Some(name), Some(parent)) => {
                tail.push(name.to_os_string());
                head = parent;
            }
            _ => return path,
        }
    }
}

/// Lexically normalize a path: drop `.`, pop `..`. No filesystem access, so a file that does not
/// exist yet (the create case the lease has to permit) normalizes the same as one that does.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// The one active lease. Process memory, never persisted: restart invalidates.
///
/// ponytail: one lease at a time. Two concurrent approved tasks in one process would need a map
/// keyed by session; the permission request carries no session id, so that upgrade needs a task
/// context on the request first.
///
/// Production (`cfg(not(test))`): process-global — one slot for the whole process. That is the
/// product rule: a single standing write grant, fail-closed on restart.
///
/// Under `cfg(test)` the slot is **thread-local**. libtest runs each unit test on its own worker
/// thread, and every lease-touching free function (`install` / `revoke*` / `active`) is reached
/// from that thread (the default `#[tokio::test]` runtime is current-thread). Parallel tests
/// therefore cannot clobber each other's grant via a shared static, which is what made
/// `every_write_lease_revocation_trigger_revokes` order-dependent when another test's
/// `CancelRun` / `new_session` / etc. revoked the one process-wide slot mid-assertion.
/// Downstream crates that depend on this one still see the process-global production shape,
/// because they compile this crate without `cfg(test)`.
#[cfg(not(test))]
static LEASE: std::sync::RwLock<Option<WriteLease>> = std::sync::RwLock::new(None);

#[cfg(test)]
std::thread_local! {
    static LEASE: std::sync::RwLock<Option<WriteLease>> = const { std::sync::RwLock::new(None) };
}

/// Run `f` against the lease slot for this process (production) or this test thread (`cfg(test)`).
fn with_lease_slot<R>(f: impl FnOnce(&std::sync::RwLock<Option<WriteLease>>) -> R) -> R {
    #[cfg(not(test))]
    {
        f(&LEASE)
    }
    #[cfg(test)]
    {
        LEASE.with(|slot| f(slot))
    }
}

/// The lease currently in force, if any.
pub fn active_write_lease() -> Option<WriteLease> {
    with_lease_slot(|slot| slot.read().ok().and_then(|l| l.clone()))
}

/// Install a lease, replacing any lease already held (a scope change is a new grant, and the old
/// scope stops being permitted the moment the new one lands).
pub fn install_write_lease(lease: WriteLease) -> WriteLease {
    with_lease_slot(|slot| {
        if let Ok(mut guard) = slot.write() {
            *guard = Some(lease.clone());
        }
    });
    lease
}

/// Revoke unconditionally. Returns the lease that was revoked, or `None` if there was none.
pub fn revoke_write_lease(_reason: &str) -> Option<WriteLease> {
    revoke_write_lease_if(|_| true)
}

/// Revoke on task completion / cancellation.
///
/// Matches the run when the grant named one, and otherwise the SESSION the task ran in: the grant
/// shape the app actually sends carries a session and no run, so keying revocation on the run alone
/// meant the completion trigger could never fire for a real grant.
pub fn revoke_write_lease_for_run(run_id: &str, session_id: Option<&str>) -> Option<WriteLease> {
    revoke_write_lease_if(|l| match &l.run_id {
        Some(run) => run == run_id,
        None => l.session_id.is_some() && l.session_id.as_deref() == session_id,
    })
}

/// Revoke only if the lease is scoped to this repo (repository trust loss).
pub fn revoke_write_lease_for_repo(repo_id: &str) -> Option<WriteLease> {
    revoke_write_lease_if(|l| l.repo_id == repo_id)
}

fn revoke_write_lease_if(pred: impl Fn(&WriteLease) -> bool) -> Option<WriteLease> {
    with_lease_slot(|slot| {
        let mut guard = slot.write().ok()?;
        match guard.as_ref().filter(|l| pred(l)) {
            Some(_) => guard.take(),
            None => None,
        }
    })
}

/// The ONE lease check, read by the ONE permission wrapper below.
///
/// Deliberately narrow: the capability must be `fs.write`, the call must have PREDICTED its writes
/// (an empty effect set means the target is unknown, and an unknown target is not provably in
/// scope), every predicted effect must be a write, and every one of them must land inside a
/// declared scope. That is ordinary source editing: create, modify, move, delete, formatter output,
/// declared generated files, fixtures, transactional agent patches. Anything else keeps the verdict
/// the policy gave it.
/// It is also bound to the TASK it was granted for, not just to the paths: the write must be
/// attributed ([`with_dispatch_context`]) to the session the grant named, and the grant must not
/// have expired. An unattributed write - anything reaching the dispatcher without a task context,
/// which is what an unauthenticated caller on the loopback transport produces - is not covered.
fn lease_covering(request: &PermissionRequest, session: Option<&str>) -> Option<WriteLease> {
    if request.capability_kind != "fs.write" || request.effects.is_empty() {
        return None;
    }
    let lease = active_write_lease()?;
    if !lease.in_force_for(session, hide_core::ids::now_ms()) {
        return None;
    }
    let covered = lease.covers(&request.target)
        && request
            .effects
            .iter()
            .all(|e| e.kind == EffectKind::Write && lease.covers(&e.target));
    covered.then_some(lease)
}

/// The config permission engine, plus two rules that both read `Ask` (the policy wants a human) as
/// `Allow` (the human said yes): inside [`with_approved_writes`], for the span of one released
/// gate; and inside a declared scope, for the span of an approved task's [`WriteLease`]. `Deny` is
/// never relaxed, and with neither in force this is exactly the inner engine.
struct GateReleaseAware<E> {
    inner: E,
    /// The task this engine's dispatcher belongs to, when it was built for one (the turn kernel's).
    /// Otherwise the attribution is ambient ([`with_dispatch_context`]).
    bound: Option<DispatchContext>,
}

impl<E: PermissionEngine> PermissionEngine for GateReleaseAware<E> {
    fn evaluate(&self, request: &PermissionRequest) -> PermissionVerdict {
        let context = self.bound.clone().or_else(dispatch_context);
        if matches!(
            context.as_ref().map(|item| &item.origin),
            Some(DispatchOrigin::UnverifiedModel)
        ) && (!request.effects.is_empty() || request.risk == hide_core::types::RiskLevel::High)
        {
            return PermissionVerdict {
                decision: Decision::Deny,
                reason: "effectful model-origin dispatch refused: no durable target-verified token cause"
                    .to_string(),
                grant_id: None,
            };
        }
        let verdict = self.inner.evaluate(request);
        if verdict.decision != Decision::Ask {
            return verdict;
        }
        let session = context.map(|c| c.session_id.as_str().to_string());
        let granted_by = if GATE_RELEASED.try_with(|_| ()).is_ok() {
            "approved at the security gate".to_string()
        } else if let Some(lease) = lease_covering(request, session.as_deref()) {
            format!("write lease {} covers this path", lease.lease_id)
        } else {
            return verdict;
        };
        PermissionVerdict {
            decision: Decision::Allow,
            reason: format!("{granted_by} ({})", verdict.reason),
            grant_id: verdict.grant_id,
        }
    }
}

/// Clears this test thread's lease slot on entry (and on drop). Isolation between parallel tests
/// comes from the thread-local slot itself — see [`LEASE`] — not from a process-wide mutex. A
/// mutex cannot fix the class: any test that fires `CancelRun` / `new_session` / etc. revokes
/// through the free functions without opting into a guard, so serialization only ever covered
/// the tests that remembered to take it. The guard remains as hygiene for sequential reuse of
/// a libtest worker thread that may still hold a leftover grant from a prior case.
#[cfg(test)]
pub(crate) fn lease_test_guard() -> LeaseTestGuard {
    revoke_write_lease("test setup");
    LeaseTestGuard
}

/// Held for the duration of a lease-touching test so teardown always clears the thread slot.
#[cfg(test)]
#[must_use]
pub(crate) struct LeaseTestGuard;

#[cfg(test)]
impl Drop for LeaseTestGuard {
    fn drop(&mut self) {
        revoke_write_lease("test teardown");
    }
}

pub fn build_default_tool_dispatcher(
    config: &HideConfig,
    registry: Arc<ToolRegistry>,
) -> ToolDispatcher {
    build_task_tool_dispatcher(config, registry, None)
}

/// The dispatcher for ONE task: the same permission engine, told which task it is serving, so the
/// write lease can refuse a write that is not the task the grant named. `None` leaves the
/// attribution ambient ([`with_dispatch_context`]), which is what the host's shared dispatcher
/// needs since it serves every session.
pub fn build_task_tool_dispatcher(
    config: &HideConfig,
    registry: Arc<ToolRegistry>,
    bound: Option<DispatchContext>,
) -> ToolDispatcher {
    ToolDispatcher::new(
        registry,
        Arc::new(GateReleaseAware {
            inner: SecurityServices::permission_engine(config),
            bound,
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::BackendServices;
    use hide_core::event::InMemoryEventLog;
    use hide_core::tool::ToolCall;
    use hide_core::types::Decision;
    use serde_json::json;
    #[test]
    fn default_registry_contains_builtin_tools() {
        let registry = build_default_tool_registry();
        let names: Vec<_> = registry.specs().into_iter().map(|spec| spec.name).collect();
        assert!(names.contains(&"fs.read".to_string()));
        assert!(names.contains(&"fs.write".to_string()));
        assert!(names.contains(&"shell.plan".to_string()));
    }
    #[tokio::test]
    async fn dispatcher_uses_workspace_policy_for_writes() {
        let dir =
            std::env::temp_dir().join(format!("hide_backend_tools_{}", hide_core::ids::now_ms()));
        let mut config = HideConfig::for_workspace(&dir);
        config.security.workspace_write_default = Decision::Allow;
        let registry = Arc::new(build_default_tool_registry());
        let dispatcher = build_default_tool_dispatcher(&config, registry);
        let file = dir.join("allowed.txt");
        let result = dispatcher
            .dispatch(ToolCall::new(
                "fs.write",
                json!({
                    "path": file.to_string_lossy(),
                    "content": "allowed",
                    "create_dirs": true
                }),
            ))
            .await
            .unwrap();
        assert_eq!(result.status, hide_core::tool::ToolStatus::Ok);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "allowed");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn unverified_model_dispatch_is_refused_even_under_an_allow_policy() {
        let dir = std::env::temp_dir().join(format!(
            "hide_unverified_model_{}",
            hide_core::ids::now_ms()
        ));
        let mut config = HideConfig::for_workspace(&dir);
        config.security.workspace_write_default = Decision::Allow;
        let dispatcher = build_task_tool_dispatcher(
            &config,
            Arc::new(build_default_tool_registry()),
            Some(DispatchContext::unverified_model(
                hide_core::ids::SessionId::from("model-session"),
                None,
            )),
        );
        let denied = dispatcher
            .dispatch(ToolCall::new(
                "fs.write",
                json!({
                    "path": dir.join("blocked.txt").to_string_lossy(),
                    "content": "must not write",
                    "create_dirs": true,
                }),
            ))
            .await;
        assert!(denied.is_err());
        assert!(!dir.join("blocked.txt").exists());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn target_verified_model_permit_allows_a_causally_bound_effect() {
        let dir =
            std::env::temp_dir().join(format!("hide_verified_model_{}", hide_core::ids::now_ms()));
        let mut config = HideConfig::for_workspace(&dir);
        config.security.workspace_write_default = Decision::Allow;
        let services = BackendServices::new(config.clone(), Arc::new(InMemoryEventLog::new()));
        let session = hide_core::ids::SessionId::from("verified-model-session");
        let mut sinks = services.verified_token_sinks(session.clone());
        let target = dir.join("allowed.txt");
        let call = ToolCall::new(
            "fs.write",
            json!({
                "path": target.to_string_lossy(),
                "content": "target verified",
                "create_dirs": true,
            }),
        );
        let permit = sinks
            .authorize_verified_tool_effect(
                crate::speculation_safety::HostDurableSinks::target_gate().emit_target(9),
                None,
                &call,
            )
            .await
            .unwrap();
        let dispatcher =
            build_default_tool_dispatcher(&config, Arc::new(build_default_tool_registry()));
        let result = dispatch_verified_model_tool_effect(
            session,
            None,
            permit,
            call,
            &dispatcher,
        )
        .await
        .unwrap();
        assert_eq!(result.status, hide_core::tool::ToolStatus::Ok);
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "target verified");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn verified_model_permit_rejects_a_substituted_call() {
        let dir = std::env::temp_dir().join(format!(
            "hide_verified_model_substitution_{}",
            hide_core::ids::now_ms()
        ));
        let mut config = HideConfig::for_workspace(&dir);
        config.security.workspace_write_default = Decision::Allow;
        let services = BackendServices::new(config.clone(), Arc::new(InMemoryEventLog::new()));
        let session = hide_core::ids::SessionId::from("verified-model-substitution");
        let mut sinks = services.verified_token_sinks(session.clone());
        let admitted = ToolCall::new("fs.read", json!({"path": "safe.txt"}));
        let permit = sinks
            .authorize_verified_tool_effect(
                crate::speculation_safety::HostDurableSinks::target_gate().emit_target(10),
                None,
                &admitted,
            )
            .await
            .unwrap();
        let substituted = ToolCall::new(
            "fs.write",
            json!({
                "path": dir.join("blocked.txt").to_string_lossy(),
                "content": "substitution must fail",
                "create_dirs": true,
            }),
        );
        let dispatcher =
            build_default_tool_dispatcher(&config, Arc::new(build_default_tool_registry()));
        assert!(dispatch_verified_model_tool_effect(session, None, permit, substituted, &dispatcher)
            .await
            .is_err());
        assert!(!dir.join("blocked.txt").exists());
        let _ = std::fs::remove_dir_all(dir);
    }
    fn lease(scopes: Vec<PathBuf>) -> WriteLease {
        WriteLease {
            lease_id: "lease-test".to_string(),
            repo_id: "repo".to_string(),
            session_id: Some("sess".to_string()),
            run_id: Some("run".to_string()),
            scopes,
            granted_ms: hide_core::ids::now_ms(),
        }
    }
    fn lease_covering_here(request: &PermissionRequest) -> Option<WriteLease> {
        let session = dispatch_context().map(|c| c.session_id.as_str().to_string());
        lease_covering(request, session.as_deref())
    }
    fn write_request(target: &str) -> PermissionRequest {
        PermissionRequest {
            capability_kind: "fs.write".to_string(),
            target: target.to_string(),
            risk: hide_core::types::RiskLevel::High,
            effects: vec![hide_core::types::Effect {
                kind: EffectKind::Write,
                target: target.to_string(),
                bytes_hash: None,
                risk: hide_core::types::RiskLevel::High,
                metadata: Default::default(),
            }],
            grant: None,
        }
    }
    #[test]
    fn lease_covers_only_paths_inside_a_declared_scope() {
        let l = lease(vec![PathBuf::from("/repo/app")]);
        assert!(l.covers("/repo/app/src/main.rs"), "inside the scope");
        assert!(
            l.covers("/repo/app/./new/file.rs"),
            "a file that does not exist yet"
        );
        assert!(
            !l.covers("/repo/other/x.rs"),
            "a sibling directory is outside"
        );
        assert!(
            !l.covers("/repo/application/x.rs"),
            "a name prefix is not containment"
        );
        assert!(
            !l.covers("/repo/app/../../etc/passwd"),
            "a parent walk cannot escape"
        );
        assert!(
            !l.covers("relative/path.rs"),
            "a relative target is not provably in scope"
        );
    }
    async fn as_session<T>(session: &str, fut: impl std::future::Future<Output = T>) -> T {
        with_dispatch_context(hide_core::ids::SessionId::from(session), None, fut).await
    }
    #[tokio::test]
    async fn lease_relaxes_only_fs_write_with_predicted_in_scope_effects() {
        let _guard = lease_test_guard();
        install_write_lease(lease(vec![PathBuf::from("/repo")]));
        as_session("sess", async {
            assert!(lease_covering_here(&write_request("/repo/a.rs")).is_some());
            assert!(lease_covering_here(&write_request("/elsewhere/a.rs")).is_none());
            let mut shell = write_request("/repo/a.rs");
            shell.capability_kind = "shell.exec".to_string();
            assert!(lease_covering_here(&shell).is_none());
            let mut git = write_request("/repo/a.rs");
            git.capability_kind = "git.write".to_string();
            assert!(lease_covering_here(&git).is_none());
            let mut blind = write_request("/repo/a.rs");
            blind.effects.clear();
            assert!(lease_covering_here(&blind).is_none());
            let mut mixed = write_request("/repo/a.rs");
            mixed
                .effects
                .push(write_request("/elsewhere/b.rs").effects.remove(0));
            assert!(lease_covering_here(&mixed).is_none());
        })
        .await;
        revoke_write_lease("end of test");
    }
    #[tokio::test]
    async fn a_lease_is_bound_to_the_granting_session_and_expires() {
        let _guard = lease_test_guard();
        install_write_lease(lease(vec![PathBuf::from("/repo")]));
        assert!(
            as_session("sess", async {
                lease_covering_here(&write_request("/repo/a.rs")).is_some()
            })
            .await
        );
        assert!(
            as_session("another-session", async {
                lease_covering_here(&write_request("/repo/a.rs")).is_none()
            })
            .await
        );
        assert!(lease_covering_here(&write_request("/repo/a.rs")).is_none());
        let expired = WriteLease {
            granted_ms: hide_core::ids::now_ms() - LEASE_TTL_MS - 1,
            ..lease(vec![PathBuf::from("/repo")])
        };
        install_write_lease(expired);
        assert!(
            as_session("sess", async {
                lease_covering_here(&write_request("/repo/a.rs")).is_none()
            })
            .await
        );
        revoke_write_lease("end of test");
    }
    #[test]
    fn task_completion_revokes_a_session_only_grant() {
        let _guard = lease_test_guard();
        install_write_lease(WriteLease {
            run_id: None,
            ..lease(vec![PathBuf::from("/repo")])
        });
        assert!(revoke_write_lease_for_run("any-run", Some("other-session")).is_none());
        assert!(revoke_write_lease_for_run("any-run", Some("sess")).is_some());
        assert_eq!(active_write_lease(), None);
    }
    #[test]
    fn a_symlink_inside_the_scope_does_not_escape_it() {
        let dir =
            std::env::temp_dir().join(format!("hide_lease_link_{}", hide_core::ids::now_ms()));
        let scope = dir.join("repo");
        let outside = dir.join("outside");
        std::fs::create_dir_all(&scope).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(&outside, scope.join("link")).unwrap();
        let l = lease(vec![scope.clone()]);
        assert!(
            l.covers(&scope.join("in.rs").to_string_lossy()),
            "an ordinary path inside"
        );
        #[cfg(unix)]
        assert!(!l.covers(&scope.join("link/escaped.rs").to_string_lossy()));
        let _ = std::fs::remove_dir_all(dir);
    }
    #[test]
    fn lease_never_relaxes_a_deny_and_never_grants_the_gate() {
        let _guard = lease_test_guard();
        install_write_lease(lease(vec![PathBuf::from("/repo")]));
        struct AlwaysDeny;
        impl PermissionEngine for AlwaysDeny {
            fn evaluate(&self, _: &PermissionRequest) -> PermissionVerdict {
                PermissionVerdict {
                    decision: Decision::Deny,
                    reason: "denied".to_string(),
                    grant_id: None,
                }
            }
        }
        let engine = GateReleaseAware {
            inner: AlwaysDeny,
            bound: None,
        };
        assert_eq!(
            engine.evaluate(&write_request("/repo/a.rs")).decision,
            Decision::Deny
        );
        assert!(!gate_released(), "a lease is not a released gate");
        revoke_write_lease("end of test");
    }
    #[test]
    fn a_lease_is_process_memory_only_so_restart_invalidates_it() {
        let _guard = lease_test_guard();
        let granted = install_write_lease(lease(vec![PathBuf::from("/repo")]));
        assert_eq!(active_write_lease().as_ref(), Some(&granted));
        assert!(!serde_json::to_string(&granted).unwrap().is_empty());
        revoke_write_lease("restart");
        assert_eq!(
            active_write_lease(),
            None,
            "a restart leaves no lease behind"
        );
    }
    #[test]
    fn scoped_revokes_only_fire_for_their_own_lease() {
        let _guard = lease_test_guard();
        install_write_lease(lease(vec![PathBuf::from("/repo")]));
        assert!(revoke_write_lease_for_run("other-run", None).is_none());
        assert!(revoke_write_lease_for_repo("other-repo").is_none());
        assert!(
            active_write_lease().is_some(),
            "another task's end is not this one's"
        );
        assert!(revoke_write_lease_for_run("run", None).is_some());
        assert_eq!(active_write_lease(), None);
        install_write_lease(lease(vec![PathBuf::from("/repo")]));
        assert!(revoke_write_lease_for_repo("repo").is_some());
        assert_eq!(active_write_lease(), None);
    }
    #[tokio::test]
    async fn a_leased_write_lands_and_an_out_of_scope_write_is_still_refused() {
        let _guard = lease_test_guard();
        let dir = std::env::temp_dir().join(format!("hide_lease_{}", hide_core::ids::now_ms()));
        let scope = dir.join("in");
        std::fs::create_dir_all(&scope).unwrap();
        let config = HideConfig::for_workspace(&dir);
        assert_eq!(config.security.workspace_write_default, Decision::Ask);
        let dispatcher =
            build_default_tool_dispatcher(&config, Arc::new(build_default_tool_registry()));
        let write = |path: PathBuf| {
            with_dispatch_context(
                hide_core::ids::SessionId::from("sess"),
                None,
                dispatcher.dispatch(ToolCall::new(
                    "edit.write_file",
                    json!({ "path": path.to_string_lossy(), "content": "x" }),
                )),
            )
        };
        assert!(
            write(scope.join("a.rs")).await.is_err(),
            "with no lease the write is refused"
        );
        install_write_lease(lease(vec![scope.clone()]));
        assert_eq!(
            write(scope.join("a.rs")).await.unwrap().status,
            hide_core::tool::ToolStatus::Ok
        );
        assert_eq!(std::fs::read_to_string(scope.join("a.rs")).unwrap(), "x");
        assert!(write(dir.join("out.rs")).await.is_err());
        revoke_write_lease("end of test");
        assert!(
            write(scope.join("b.rs")).await.is_err(),
            "after revocation the write asks again"
        );
        let _ = std::fs::remove_dir_all(dir);
    }
}
