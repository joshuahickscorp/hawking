use crate::digest::compute_home_and_sessions;
use crate::services::{BackendServices, DynMemoryStore};
use futures::future::BoxFuture;
use hawking_context::compiler::CompileInput;
use hawking_context::profiles::ContextProfile;
use hawking_context::sources::CodeIndexContextSource;
use hawking_context::{ContextCompiler, InMemoryMemoryStore, MemoryKind};
use hawking_index::{CodeIndex, InMemoryCodeIndex, SearchQuery};
use hawking_orch::{RoleRegistry, Router, SimpleRouter};
use hawking_research::{DynResearchLedger, ResearchRun, ResearchState};
use hide_core::error::{HideError, Result};
use hide_core::persistence::{DynEventLog, DynProjectionStore};
use hide_core::plugin::ExtensionContribution;
use hide_core::runtime::{InferenceRequest, RolePurpose};
use hide_core::types::Provenance;
use hide_personalize::{DynPersonalizationStore, PersonalizationRecord, TaskClass};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectorStatus {
    pub id: String,
    pub healthy: bool,
    pub detail: String,
    pub contributions: Vec<ExtensionContribution>,
}

impl ConnectorStatus {
    fn new(id: impl Into<String>, healthy: bool, detail: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            healthy,
            detail: detail.into(),
            contributions: Vec::new(),
        }
    }

    fn with_contributions(
        id: impl Into<String>,
        healthy: bool,
        detail: impl Into<String>,
        contributions: Vec<ExtensionContribution>,
    ) -> Self {
        Self {
            id: id.into(),
            healthy,
            detail: detail.into(),
            contributions,
        }
    }
}

pub trait Connector: Send + Sync {
    fn id(&self) -> &str;
    fn status<'a>(&'a self) -> BoxFuture<'a, Result<ConnectorStatus>>;
    fn call<'a>(&'a self, method: &'a str, params: Value) -> BoxFuture<'a, Result<Value>>;
}

#[derive(Default)]
pub struct ConnectorRegistry {
    connectors: RwLock<BTreeMap<String, Arc<dyn Connector>>>,
}

impl ConnectorRegistry {
    pub fn register<C: Connector + 'static>(&self, connector: C) {
        self.connectors
            .write()
            .insert(connector.id().to_string(), Arc::new(connector));
    }

    pub fn get(&self, id: &str) -> Option<Arc<dyn Connector>> {
        self.connectors.read().get(id).cloned()
    }

    pub fn ids(&self) -> Vec<String> {
        self.connectors.read().keys().cloned().collect()
    }

    pub async fn call(&self, id: &str, method: &str, params: Value) -> Result<Value> {
        let connector = self
            .get(id)
            .ok_or_else(|| HideError::NotFound(format!("connector {id}")))?;
        connector.call(method, params).await
    }

    pub async fn statuses(&self) -> Vec<ConnectorStatus> {
        let connectors: Vec<_> = self.connectors.read().values().cloned().collect();
        let mut statuses = Vec::new();
        for connector in connectors {
            let status = connector
                .status()
                .await
                .unwrap_or_else(|err| ConnectorStatus::new(connector.id(), false, err.to_string()));
            statuses.push(status);
        }
        statuses
    }
}

#[derive(Clone)]
pub struct PersonalizationConnector {
    store: DynPersonalizationStore,
}

impl PersonalizationConnector {
    pub fn new(store: DynPersonalizationStore) -> Self {
        Self { store }
    }
}

impl Connector for PersonalizationConnector {
    fn id(&self) -> &str {
        "personalization"
    }

    fn status<'a>(&'a self) -> BoxFuture<'a, Result<ConnectorStatus>> {
        Box::pin(async move {
            let count = self.store.load_all()?.len();
            Ok(ConnectorStatus::new(
                self.id(),
                true,
                format!("{count} personalization records available"),
            ))
        })
    }

    fn call<'a>(&'a self, method: &'a str, params: Value) -> BoxFuture<'a, Result<Value>> {
        Box::pin(async move {
            match method {
                "records.list" => {
                    let records = match limit_param(&params) {
                        Some(limit) => self.store.load_recent(limit)?,
                        None => self.store.load_all()?,
                    };
                    Ok(json!({ "records": records }))
                }
                "records.append" => {
                    let record: PersonalizationRecord =
                        serde_json::from_value(payload_or_self(params, "record"))?;
                    self.store.append(&record)?;
                    Ok(json!({ "ok": true }))
                }
                "records.by_task" => {
                    let task_class: TaskClass = serde_json::from_value(
                        params
                            .get("task_class")
                            .cloned()
                            .ok_or_else(|| HideError::Config("missing task_class".to_string()))?,
                    )?;
                    let records = self
                        .store
                        .load_by_task(task_class, limit_param(&params).unwrap_or(100))?;
                    Ok(json!({ "records": records }))
                }
                other => Err(HideError::NotFound(format!(
                    "personalization connector method {other}"
                ))),
            }
        })
    }
}

#[derive(Clone)]
pub struct ResearchConnector {
    ledger: DynResearchLedger,
}

impl ResearchConnector {
    pub fn new(ledger: DynResearchLedger) -> Self {
        Self { ledger }
    }
}

impl Connector for ResearchConnector {
    fn id(&self) -> &str {
        "research"
    }

    fn status<'a>(&'a self) -> BoxFuture<'a, Result<ConnectorStatus>> {
        Box::pin(async move {
            let count = self.ledger.load_runs()?.len();
            Ok(ConnectorStatus::new(
                self.id(),
                true,
                format!("{count} research runs available"),
            ))
        })
    }

    fn call<'a>(&'a self, method: &'a str, params: Value) -> BoxFuture<'a, Result<Value>> {
        Box::pin(async move {
            match method {
                "runs.list" => {
                    let mut runs = self.ledger.load_runs()?;
                    if let Some(limit) = limit_param(&params) {
                        if runs.len() > limit {
                            runs.drain(..runs.len() - limit);
                        }
                    }
                    Ok(json!({ "runs": runs }))
                }
                "runs.latest" => Ok(json!({ "run": self.ledger.latest()? })),
                "runs.append" => {
                    let run: ResearchRun = serde_json::from_value(payload_or_self(params, "run"))?;
                    self.ledger.append_run(&run)?;
                    Ok(json!({ "ok": true }))
                }
                "runs.by_state" => {
                    let state: ResearchState = serde_json::from_value(
                        params
                            .get("state")
                            .cloned()
                            .ok_or_else(|| HideError::Config("missing state".to_string()))?,
                    )?;
                    Ok(json!({ "runs": self.ledger.load_by_state(state)? }))
                }
                other => Err(HideError::NotFound(format!(
                    "research connector method {other}"
                ))),
            }
        })
    }
}

#[derive(Clone)]
pub struct RuntimeConnector {
    roles: Arc<RoleRegistry>,
    /// The supervised engine, when one is configured. This is the ONLY honest source of readiness:
    /// the role registry is a static list of descriptors that is non-empty even with no engine at
    /// all, so a frontend that read `roles.list` as readiness reported ready on every host.
    runtime: Option<Arc<crate::supervisor::RuntimeSupervisor>>,
}

impl RuntimeConnector {
    pub fn new(roles: Arc<RoleRegistry>) -> Self {
        Self {
            roles,
            runtime: None,
        }
    }

    /// Attach the supervised runtime so `state` reports the real supervisor state.
    pub fn with_runtime(
        mut self,
        runtime: Option<Arc<crate::supervisor::RuntimeSupervisor>>,
    ) -> Self {
        self.runtime = runtime;
        self
    }
}

impl Connector for RuntimeConnector {
    fn id(&self) -> &str {
        "runtime"
    }

    fn status<'a>(&'a self) -> BoxFuture<'a, Result<ConnectorStatus>> {
        Box::pin(async move {
            let count = self.roles.all().len();
            Ok(ConnectorStatus::new(
                self.id(),
                count > 0,
                format!("{count} model roles registered"),
            ))
        })
    }

    fn call<'a>(&'a self, method: &'a str, params: Value) -> BoxFuture<'a, Result<Value>> {
        Box::pin(async move {
            match method {
                "roles.list" => Ok(json!({ "roles": self.roles.all() })),
                // The real supervisor state, or "down" with the reason when no engine is
                // configured. Never inferred from the role registry.
                "state" => Ok(match self.runtime.as_ref() {
                    Some(sup) => json!({ "state": sup.state(), "detail": sup.base_url() }),
                    None => json!({ "state": "down", "detail": "no model configured" }),
                }),
                "route" => {
                    let request: InferenceRequest =
                        serde_json::from_value(payload_or_self(params, "request"))?;
                    let router = SimpleRouter::new(self.roles.clone());
                    let decision = router.route(&request)?;
                    Ok(json!({ "decision": decision }))
                }
                other => Err(HideError::NotFound(format!(
                    "runtime connector method {other}"
                ))),
            }
        })
    }
}

/// The courtyard (Home) connector: serves the retrospective digest + session list folded live from the
/// event log. The FE pulls this on connect (like `roles.list`) since the launcher data is not a replayed
/// event stream. Read-only; nothing is written.
pub struct HomeConnector {
    event_log: DynEventLog,
    projection_store: DynProjectionStore,
    workspace_root: PathBuf,
}

impl HomeConnector {
    pub fn new(
        event_log: DynEventLog,
        projection_store: DynProjectionStore,
        workspace_root: PathBuf,
    ) -> Self {
        Self {
            event_log,
            projection_store,
            workspace_root,
        }
    }
}

impl Connector for HomeConnector {
    fn id(&self) -> &str {
        "home"
    }

    fn status<'a>(&'a self) -> BoxFuture<'a, Result<ConnectorStatus>> {
        Box::pin(async move { Ok(ConnectorStatus::new(self.id(), true, "courtyard digest")) })
    }

    fn call<'a>(&'a self, method: &'a str, _params: Value) -> BoxFuture<'a, Result<Value>> {
        Box::pin(async move {
            match method {
                // Fold the event log into the launcher's home + sessions patches.
                "digest" => {
                    let (home, sessions) = compute_home_and_sessions(
                        &self.event_log,
                        &self.projection_store,
                        &self.workspace_root,
                    )
                    .await?;
                    // The write lease rides the SAME connect-time read. It is a process-global
                    // static, published live and never durable, so a reloaded tab was honouring a
                    // lease it could neither see nor revoke. This is the one call every fresh
                    // client already makes, so the indicator comes back with no second fetch and no
                    // second code path; an expired grant reads as no lease, matching enforcement.
                    let lease = crate::tools::active_write_lease()
                        .filter(|l| l.within_ttl(hide_core::ids::now_ms()));
                    let note = if lease.is_some() { "granted" } else { "no lease" };
                    Ok(json!({
                        "home": home,
                        "sessions": sessions,
                        "status": crate::host::write_lease_patch(lease.as_ref(), note),
                    }))
                }
                other => Err(HideError::NotFound(format!(
                    "home connector method {other}"
                ))),
            }
        })
    }
}

#[derive(Clone)]
pub struct CodeIndexConnector {
    index: Arc<InMemoryCodeIndex>,
}

impl CodeIndexConnector {
    pub fn new(index: Arc<InMemoryCodeIndex>) -> Self {
        Self { index }
    }
}

impl Connector for CodeIndexConnector {
    fn id(&self) -> &str {
        "code_index"
    }

    fn status<'a>(&'a self) -> BoxFuture<'a, Result<ConnectorStatus>> {
        Box::pin(async move {
            let health = self.index.health().await?;
            Ok(ConnectorStatus::new(
                self.id(),
                health.degraded.is_empty(),
                format!(
                    "generation={}, indexed_files={}, stale_files={}",
                    health.generation, health.indexed_files, health.stale_files
                ),
            ))
        })
    }

    fn call<'a>(&'a self, method: &'a str, params: Value) -> BoxFuture<'a, Result<Value>> {
        Box::pin(async move {
            match method {
                "file.add_text" => {
                    let path = params
                        .get("path")
                        .and_then(|value| value.as_str())
                        .ok_or_else(|| HideError::Config("missing path".to_string()))?;
                    let content = params
                        .get("content")
                        .and_then(|value| value.as_str())
                        .ok_or_else(|| HideError::Config("missing content".to_string()))?;
                    let content_hash = params
                        .get("content_hash")
                        .and_then(|value| value.as_str())
                        .map(ToOwned::to_owned);
                    self.index.add_text_file(path, content, content_hash);
                    Ok(json!({ "ok": true }))
                }
                "file.index" => {
                    let path = params
                        .get("path")
                        .and_then(|value| value.as_str())
                        .ok_or_else(|| HideError::Config("missing path".to_string()))?;
                    self.index.index_path(path)?;
                    Ok(json!({ "ok": true }))
                }
                "search" => {
                    let query: SearchQuery =
                        serde_json::from_value(payload_or_self(params, "query"))?;
                    let results = self.index.search(query).await?;
                    Ok(json!({ "results": results }))
                }
                "definition" => {
                    let symbol = params
                        .get("symbol")
                        .and_then(|value| value.as_str())
                        .ok_or_else(|| HideError::Config("missing symbol".to_string()))?;
                    Ok(json!({ "occurrences": self.index.definition(symbol).await? }))
                }
                "references" => {
                    let symbol = params
                        .get("symbol")
                        .and_then(|value| value.as_str())
                        .ok_or_else(|| HideError::Config("missing symbol".to_string()))?;
                    Ok(json!({ "occurrences": self.index.references(symbol).await? }))
                }
                "health" => Ok(json!({ "health": self.index.health().await? })),
                other => Err(HideError::NotFound(format!(
                    "code index connector method {other}"
                ))),
            }
        })
    }
}

#[derive(Clone)]
pub struct ContextConnector {
    index: Arc<InMemoryCodeIndex>,
    roles: Arc<RoleRegistry>,
    /// Spine B: the persistent Project Brain — each compile upserts a record so
    /// the agent's working memory of this project accrues across turns/sessions.
    memory: DynMemoryStore,
}

impl ContextConnector {
    pub fn new(
        index: Arc<InMemoryCodeIndex>,
        roles: Arc<RoleRegistry>,
        memory: DynMemoryStore,
    ) -> Self {
        Self {
            index,
            roles,
            memory,
        }
    }
}

impl Connector for ContextConnector {
    fn id(&self) -> &str {
        "context"
    }

    fn status<'a>(&'a self) -> BoxFuture<'a, Result<ConnectorStatus>> {
        Box::pin(async move {
            let index_health = self.index.health().await?;
            Ok(ConnectorStatus::new(
                self.id(),
                !self.roles.all().is_empty(),
                format!(
                    "roles={}, indexed_files={}",
                    self.roles.all().len(),
                    index_health.indexed_files
                ),
            ))
        })
    }

    fn call<'a>(&'a self, method: &'a str, params: Value) -> BoxFuture<'a, Result<Value>> {
        Box::pin(async move {
            match method {
                "compile" => {
                    let task = params
                        .get("task")
                        .and_then(|value| value.as_str())
                        .ok_or_else(|| HideError::Config("missing task".to_string()))?;
                    let max_input_tokens = params
                        .get("max_input_tokens")
                        .and_then(|value| value.as_u64())
                        .and_then(|value| usize::try_from(value).ok())
                        .unwrap_or(4096);
                    let search_limit = params
                        .get("search_limit")
                        .and_then(|value| value.as_u64())
                        .and_then(|value| usize::try_from(value).ok())
                        .unwrap_or(16);
                    let role_name = params.get("role").and_then(|value| value.as_str());
                    let role = choose_context_role(&self.roles, role_name)?;
                    let mut compiler = ContextCompiler::new();
                    let index: Arc<dyn CodeIndex> = self.index.clone();
                    compiler.add_source(CodeIndexContextSource::new(index, search_limit));
                    let compiled = compiler
                        .compile(CompileInput {
                            profile: ContextProfile::coding_default(max_input_tokens),
                            model: role.model,
                            task: task.to_string(),
                        })
                        .await?;
                    // Spine B: accrue the Project Brain — record this compile (task +
                    // what the window retained) as a Project memory. Best-effort: a
                    // brain write must never fail the compile.
                    let brain = InMemoryMemoryStore::record(
                        MemoryKind::Project,
                        format!(
                            "task: {task}\nretained {} spans, {} tokens used",
                            compiled.manifest.retained.len(),
                            compiled.manifest.used_tokens
                        ),
                        Provenance::trusted("context.compile"),
                    );
                    let _ = self.memory.upsert(brain).await;
                    Ok(json!({
                        "prompt": compiled.prompt,
                        "manifest": compiled.manifest
                    }))
                }
                other => Err(HideError::NotFound(format!(
                    "context connector method {other}"
                ))),
            }
        })
    }
}

// ---- Real workspace file I/O (ship item: the editor + Explorer operate on real files) ----

const FS_IGNORE: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    "dist",
    ".hide",
    ".next",
    "build",
    ".turbo",
];
const FS_MAX_DEPTH: usize = 4;
const FS_MAX_NODES: usize = 4000;
const FS_MAX_READ_BYTES: u64 = 2_000_000;

fn lang_for(path: &str) -> &'static str {
    match path.rsplit('.').next().unwrap_or("") {
        "rs" => "rust",
        "ts" | "tsx" => "typescript",
        "js" | "jsx" | "mjs" | "cjs" => "javascript",
        "py" => "python",
        "go" => "go",
        "json" => "json",
        "toml" => "toml",
        "yaml" | "yml" => "yaml",
        "md" | "markdown" => "markdown",
        "css" => "css",
        "html" | "htm" => "html",
        "sh" | "bash" | "zsh" => "shell",
        "c" | "h" => "c",
        "cpp" | "cc" | "hpp" | "cxx" => "cpp",
        "java" => "java",
        "rb" => "ruby",
        "sql" => "sql",
        _ => "plaintext",
    }
}

/// Join a workspace-relative path onto `root`, rejecting anything that could escape it (`..`,
/// absolute, prefix components). The single confinement helper for workspace-relative paths the
/// FRONTEND supplies; the host reuses it for the same reason.
///
/// Containment is checked on the REAL path as well as lexically, so a symlink checked into the
/// repo cannot point a confined write outside it. Both sides are resolved, since the root itself
/// is often reached through a symlink (`/tmp` on macOS).
pub(crate) fn workspace_resolve(root: &std::path::Path, rel: &str) -> Result<std::path::PathBuf> {
    use std::path::Component;
    let rel = rel.trim_start_matches('/');
    if std::path::Path::new(rel).components().any(|c| {
        matches!(
            c,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    }) {
        return Err(HideError::PolicyDenied(format!(
            "path escapes workspace: {rel}"
        )));
    }
    let joined = root.join(rel);
    if !crate::tools::real_path(&joined).starts_with(crate::tools::real_path(root)) {
        return Err(HideError::PolicyDenied(format!(
            "path escapes workspace through a link: {rel}"
        )));
    }
    Ok(joined)
}

/// Real file I/O confined to the project root. Workspace-relative paths only; `..`, absolute, and
/// prefix components are rejected so a path can never escape the root.
///
/// READS ONLY. There is no `write_file` arm: this connector used to carry one, which made it a
/// SECOND write channel next to the agent's, and the one the app actually used. It reached the
/// verifying applier straight off the dispatcher, so the bytes landed while `BackendHost::
/// dispatch_tool` (where the tool events and the diff capture live) was never entered, and no
/// consumer downstream of it could see, review or undo an app write. The single write path is
/// `BackendHost::save_file_effect`.
#[derive(Clone)]
pub struct FsConnector {
    root: std::path::PathBuf,
}

impl FsConnector {
    pub fn new(root: impl Into<std::path::PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn resolve(&self, rel: &str) -> Result<std::path::PathBuf> {
        workspace_resolve(&self.root, rel)
    }
}

fn walk_tree(
    dir: &std::path::Path,
    root: &std::path::Path,
    depth: usize,
    budget: &mut usize,
) -> Vec<Value> {
    if depth > FS_MAX_DEPTH || *budget == 0 {
        return Vec::new();
    }
    let mut entries: Vec<_> = match std::fs::read_dir(dir) {
        Ok(rd) => rd.flatten().collect(),
        Err(_) => return Vec::new(),
    };
    entries.sort_by_key(|e| {
        let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
        (!is_dir, e.file_name().to_string_lossy().to_lowercase())
    });
    let mut out = Vec::new();
    for e in entries {
        if *budget == 0 {
            break;
        }
        let name = e.file_name().to_string_lossy().to_string();
        let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
        if is_dir && FS_IGNORE.contains(&name.as_str()) {
            continue;
        }
        let abs = e.path();
        let rel = abs
            .strip_prefix(root)
            .unwrap_or(&abs)
            .to_string_lossy()
            .replace('\\', "/");
        *budget -= 1;
        if is_dir {
            let children = walk_tree(&abs, root, depth + 1, budget);
            out.push(json!({ "path": rel, "name": name, "dir": true, "children": children }));
        } else {
            out.push(json!({ "path": rel, "name": name, "dir": false }));
        }
    }
    out
}

impl Connector for FsConnector {
    fn id(&self) -> &str {
        "fs"
    }

    fn status<'a>(&'a self) -> BoxFuture<'a, Result<ConnectorStatus>> {
        Box::pin(async move {
            let healthy = self.root.is_dir();
            Ok(ConnectorStatus::new(
                self.id(),
                healthy,
                format!("root={}", self.root.display()),
            ))
        })
    }

    fn call<'a>(&'a self, method: &'a str, params: Value) -> BoxFuture<'a, Result<Value>> {
        Box::pin(async move {
            match method {
                "tree" => {
                    let mut budget = FS_MAX_NODES;
                    let tree = walk_tree(&self.root, &self.root, 0, &mut budget);
                    Ok(json!({ "tree": tree, "root": self.root.display().to_string() }))
                }
                "read_file" => {
                    let path = params
                        .get("path")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| HideError::Config("missing path".to_string()))?;
                    let abs = self.resolve(path)?;
                    let meta =
                        std::fs::metadata(&abs).map_err(|e| HideError::Storage(e.to_string()))?;
                    if meta.len() > FS_MAX_READ_BYTES {
                        return Err(HideError::PolicyDenied(format!(
                            "file too large to open ({} bytes)",
                            meta.len()
                        )));
                    }
                    let text = std::fs::read_to_string(&abs)
                        .map_err(|e| HideError::Storage(e.to_string()))?;
                    // The hash of what the caller is about to edit, so the save can send it back as
                    // `base_hash` and the concurrency guard below has something to compare.
                    let hash = blake3::hash(text.as_bytes()).to_hex().to_string();
                    Ok(json!({ "text": text, "lang": lang_for(path), "path": path, "hash": hash }))
                }
                other => Err(HideError::NotFound(format!("fs connector method {other}"))),
            }
        })
    }
}

/// The connector methods `/v1/hide/connector` may serve: the READS, named one at a time.
///
/// An ALLOWLIST, not a blocklist of the writes somebody happened to classify. The route is a read
/// channel and every mutation reaches the host over `/v1/hide/intent`, where the approval gate and
/// the command catalog are, so an arm that is not on this list is refused whether or not anyone
/// remembered to think about it: a new arm is closed by default rather than open by default. The
/// blocklist this replaces named `records.append`, `runs.append` and `write_file` and missed three
/// more that were reachable with no gate at all: `code_index` `file.add_text` (mutates the shared
/// index), `code_index` `file.index` (`std::fs::read_to_string` of an ABSOLUTE path, none of the
/// workspace confinement `fs` enforces) and `context` `compile` (upserts the durable memory store).
pub const CONNECTOR_READ_METHODS: &[&str] = &[
    "records.list",
    "records.by_task",
    "runs.list",
    "runs.latest",
    "runs.by_state",
    "roles.list",
    "state",
    "route",
    "digest",
    "search",
    "definition",
    "references",
    "health",
    "tree",
    "read_file",
];

/// Whether `/v1/hide/connector` may serve this method. Fail closed: unknown means no.
pub fn connector_method_is_read(method: &str) -> bool {
    CONNECTOR_READ_METHODS.contains(&method)
}

/// The runtime connector, optionally bound to the supervised engine. `BackendHost` re-registers it
/// with the supervisor once `maybe_boot_runtime` has run, so `state` is a real read.
pub fn runtime_connector(
    services: &BackendServices,
    runtime: Option<Arc<crate::supervisor::RuntimeSupervisor>>,
) -> RuntimeConnector {
    RuntimeConnector::new(services.role_registry.clone()).with_runtime(runtime)
}

pub fn register_backend_connectors(registry: &ConnectorRegistry, services: &BackendServices) {
    registry.register(PersonalizationConnector::new(
        services.personalization_store.clone(),
    ));
    registry.register(ResearchConnector::new(services.research_ledger.clone()));
    registry.register(runtime_connector(services, None));
    registry.register(CodeIndexConnector::new(services.code_index.clone()));
    registry.register(FsConnector::new(services.config.workspace_root.clone()));
    registry.register(ContextConnector::new(
        services.code_index.clone(),
        services.role_registry.clone(),
        services.memory_store.clone(),
    ));
    registry.register(HomeConnector::new(
        services.event_log.clone(),
        services.projection_store.clone(),
        services.config.workspace_root.clone(),
    ));
}

#[cfg(test)]
mod fs_tests {
    use super::*;

    #[test]
    fn resolve_confines_to_root() {
        let root = std::path::Path::new("/work/proj");
        assert!(workspace_resolve(root, "src/main.rs").is_ok());
        // leading slash stripped, stays in root
        assert!(workspace_resolve(root, "/src/main.rs").is_ok());
        assert!(workspace_resolve(root, "../etc/passwd").is_err());
        assert!(workspace_resolve(root, "a/../../b").is_err());
    }

    /// The `fs` connector is a READ surface. The write arm it used to carry was the app's second
    /// write channel (see the type doc); a caller reaching for it now gets the same not-found every
    /// other unknown method gets, and the save goes through `BackendHost::save_file_effect`, whose
    /// policy refusal is covered by `save_through_the_wire_path_records_a_diff_and_publishes_it`.
    #[tokio::test]
    async fn the_fs_connector_has_no_write_arm() {
        let dir = std::env::temp_dir().join(format!("hide_fs_conn_{}", hide_core::ids::now_ms()));
        std::fs::create_dir_all(&dir).unwrap();
        let fs = FsConnector::new(dir.clone());
        let err = fs
            .call("write_file", json!({ "path": "a.txt", "content": "x" }))
            .await
            .expect_err("the connector must not carry a second write channel");
        assert!(matches!(err, HideError::NotFound(_)), "{err:?}");
        assert!(!dir.join("a.txt").exists(), "nothing was written");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn lang_detection() {
        assert_eq!(lang_for("a/b.rs"), "rust");
        assert_eq!(lang_for("x.tsx"), "typescript");
        assert_eq!(lang_for("noext"), "plaintext");
    }
}

fn limit_param(params: &Value) -> Option<usize> {
    params
        .get("limit")
        .and_then(|value| value.as_u64())
        .and_then(|value| usize::try_from(value).ok())
}

fn payload_or_self(params: Value, field: &str) -> Value {
    params.get(field).cloned().unwrap_or(params)
}

pub(crate) fn choose_context_role(
    roles: &RoleRegistry,
    role_name: Option<&str>,
) -> Result<hide_core::runtime::ModelRole> {
    let all = roles.all();
    if let Some(role_name) = role_name {
        if let Some(role) = all.iter().find(|role| role.name == role_name) {
            return Ok(role.clone());
        }
        return Err(HideError::NotFound(format!("model role {role_name}")));
    }
    all.iter()
        .find(|role| role.purpose == RolePurpose::HeroCoder)
        .or_else(|| {
            all.iter()
                .find(|role| role.purpose == RolePurpose::FastDraft)
        })
        .or_else(|| all.first())
        .cloned()
        .ok_or_else(|| HideError::Config("no model roles registered".to_string()))
}

#[derive(Debug, Clone)]
pub struct StaticConnector {
    pub id: String,
    pub contributions: Vec<ExtensionContribution>,
}

impl Connector for StaticConnector {
    fn id(&self) -> &str {
        &self.id
    }

    fn status<'a>(&'a self) -> BoxFuture<'a, Result<ConnectorStatus>> {
        Box::pin(async move {
            Ok(ConnectorStatus::with_contributions(
                self.id.clone(),
                true,
                "static connector ready",
                self.contributions.clone(),
            ))
        })
    }

    fn call<'a>(&'a self, method: &'a str, params: Value) -> BoxFuture<'a, Result<Value>> {
        Box::pin(async move {
            Ok(serde_json::json!({
                "connector": self.id,
                "method": method,
                "params": params
            }))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hawking_research::{InMemoryResearchLedger, ResearchRun};
    use hide_core::runtime::InferenceRequest;
    use hide_personalize::{InMemoryPersonalizationStore, Outcome};
    use std::collections::BTreeMap;

    #[tokio::test]
    async fn registry_reports_connector_status() {
        let registry = ConnectorRegistry::default();
        registry.register(StaticConnector {
            id: "research".to_string(),
            contributions: Vec::new(),
        });
        let statuses = registry.statuses().await;
        assert_eq!(statuses.len(), 1);
        assert!(statuses[0].healthy);
    }

    #[tokio::test]
    async fn personalization_connector_appends_and_lists_records() {
        let registry = ConnectorRegistry::default();
        registry.register(PersonalizationConnector::new(Arc::new(
            InMemoryPersonalizationStore::default(),
        )));
        let record = PersonalizationRecord::accepted(TaskClass::EditCode, "prompt", "diff");

        registry
            .call(
                "personalization",
                "records.append",
                json!({ "record": record }),
            )
            .await
            .unwrap();
        let listed = registry
            .call("personalization", "records.list", json!({ "limit": 5 }))
            .await
            .unwrap();

        assert_eq!(listed["records"].as_array().unwrap().len(), 1);
        assert_eq!(listed["records"][0]["outcome"], json!(Outcome::Accepted));
    }

    #[tokio::test]
    async fn research_connector_appends_and_lists_runs() {
        let registry = ConnectorRegistry::default();
        registry.register(ResearchConnector::new(Arc::new(
            InMemoryResearchLedger::default(),
        )));
        let mut run = ResearchRun::new("connectors");
        run.state = ResearchState::Complete;

        registry
            .call("research", "runs.append", json!({ "run": run }))
            .await
            .unwrap();
        let listed = registry
            .call("research", "runs.by_state", json!({ "state": "complete" }))
            .await
            .unwrap();

        assert_eq!(listed["runs"].as_array().unwrap().len(), 1);
        assert_eq!(listed["runs"][0]["topic"], "connectors");
    }

    #[tokio::test]
    async fn runtime_connector_lists_roles_and_routes_requests() {
        let registry = ConnectorRegistry::default();
        registry.register(RuntimeConnector::new(Arc::new(
            RoleRegistry::with_default_local_roles(),
        )));

        let roles = registry
            .call("runtime", "roles.list", json!({}))
            .await
            .unwrap();
        assert!(!roles["roles"].as_array().unwrap().is_empty());

        let routed = registry
            .call(
                "runtime",
                "route",
                json!({
                    "request": InferenceRequest {
                        task_kind: "tool_call".to_string(),
                        prompt: "{}".to_string(),
                        messages: Vec::new(),
                        max_output_tokens: 128,
                        sampler: None,
                        grammar: Some("tool-call-json".to_string()),
                        want_logprobs: false,
                        metadata: BTreeMap::new(),
                    }
                }),
            )
            .await
            .unwrap();
        assert_eq!(routed["decision"]["grammar"], "tool-call-json");
    }

    #[tokio::test]
    async fn code_index_connector_indexes_and_searches_text() {
        let registry = ConnectorRegistry::default();
        registry.register(CodeIndexConnector::new(Arc::new(
            InMemoryCodeIndex::default(),
        )));

        registry
            .call(
                "code_index",
                "file.add_text",
                json!({
                    "path": "src/lib.rs",
                    "content": "pub struct ConnectorIndex {}\n// runtime connector search\n"
                }),
            )
            .await
            .unwrap();
        let results = registry
            .call(
                "code_index",
                "search",
                json!({
                    "query": SearchQuery {
                        text: "ConnectorIndex".to_string(),
                        limit: 10,
                        include_symbols: true,
                        include_lexical: true,
                        include_semantic: false,
                    }
                }),
            )
            .await
            .unwrap();

        assert!(!results["results"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn context_connector_compiles_from_code_index() {
        let registry = ConnectorRegistry::default();
        let index = Arc::new(InMemoryCodeIndex::default());
        index.add_text_file(
            "src/context.rs",
            "pub fn compile_context_bridge() {}\n// context bridge source\n",
            None,
        );
        registry.register(ContextConnector::new(
            index,
            Arc::new(RoleRegistry::with_default_local_roles()),
            Arc::new(InMemoryMemoryStore::default()),
        ));

        let compiled = registry
            .call(
                "context",
                "compile",
                json!({
                    "task": "context bridge",
                    "max_input_tokens": 1024,
                    "search_limit": 4
                }),
            )
            .await
            .unwrap();

        assert!(compiled["prompt"]
            .as_str()
            .unwrap()
            .contains("context bridge"));
        assert_eq!(
            compiled["manifest"]["retained"].as_array().unwrap().len(),
            1
        );
    }
}
