use crate::services::{BackendServices, DynMemoryStore};
use futures::future::BoxFuture;
use hawking_context::compiler::CompileInput;
use hawking_context::profiles::ContextProfile;
use hawking_context::sources::CodeIndexContextSource;
use hawking_context::{ContextCompiler, InMemoryMemoryStore, MemoryKind};
use hide_core::types::Provenance;
use hawking_index::{CodeIndex, InMemoryCodeIndex, SearchQuery};
use hawking_orch::{RoleRegistry, Router, SimpleRouter};
use hawking_research::{DynResearchLedger, ResearchRun, ResearchState};
use hide_core::error::{HideError, Result};
use hide_core::plugin::ExtensionContribution;
use hide_core::runtime::{InferenceRequest, RolePurpose};
use hide_personalize::{DynPersonalizationStore, PersonalizationRecord, TaskClass};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectorStatus {
    pub id: String,
    pub healthy: bool,
    pub detail: String,
    pub contributions: Vec<ExtensionContribution>,
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
                .unwrap_or_else(|err| ConnectorStatus {
                    id: connector.id().to_string(),
                    healthy: false,
                    detail: err.to_string(),
                    contributions: Vec::new(),
                });
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
            Ok(ConnectorStatus {
                id: self.id().to_string(),
                healthy: true,
                detail: format!("{count} personalization records available"),
                contributions: Vec::new(),
            })
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
            Ok(ConnectorStatus {
                id: self.id().to_string(),
                healthy: true,
                detail: format!("{count} research runs available"),
                contributions: Vec::new(),
            })
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
}

impl RuntimeConnector {
    pub fn new(roles: Arc<RoleRegistry>) -> Self {
        Self { roles }
    }
}

impl Connector for RuntimeConnector {
    fn id(&self) -> &str {
        "runtime"
    }

    fn status<'a>(&'a self) -> BoxFuture<'a, Result<ConnectorStatus>> {
        Box::pin(async move {
            let count = self.roles.all().len();
            Ok(ConnectorStatus {
                id: self.id().to_string(),
                healthy: count > 0,
                detail: format!("{count} model roles registered"),
                contributions: Vec::new(),
            })
        })
    }

    fn call<'a>(&'a self, method: &'a str, params: Value) -> BoxFuture<'a, Result<Value>> {
        Box::pin(async move {
            match method {
                "roles.list" => Ok(json!({ "roles": self.roles.all() })),
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
            Ok(ConnectorStatus {
                id: self.id().to_string(),
                healthy: health.degraded.is_empty(),
                detail: format!(
                    "generation={}, indexed_files={}, stale_files={}",
                    health.generation, health.indexed_files, health.stale_files
                ),
                contributions: Vec::new(),
            })
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
    pub fn new(index: Arc<InMemoryCodeIndex>, roles: Arc<RoleRegistry>, memory: DynMemoryStore) -> Self {
        Self { index, roles, memory }
    }
}

impl Connector for ContextConnector {
    fn id(&self) -> &str {
        "context"
    }

    fn status<'a>(&'a self) -> BoxFuture<'a, Result<ConnectorStatus>> {
        Box::pin(async move {
            let index_health = self.index.health().await?;
            Ok(ConnectorStatus {
                id: self.id().to_string(),
                healthy: !self.roles.all().is_empty(),
                detail: format!(
                    "roles={}, indexed_files={}",
                    self.roles.all().len(),
                    index_health.indexed_files
                ),
                contributions: Vec::new(),
            })
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

const FS_IGNORE: &[&str] = &[".git", "node_modules", "target", "dist", ".hide", ".next", "build", ".turbo"];
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

/// Real file I/O confined to the project root. Workspace-relative paths only; `..`, absolute, and
/// prefix components are rejected so a path can never escape the root.
#[derive(Clone)]
pub struct FsConnector {
    root: std::path::PathBuf,
}

impl FsConnector {
    pub fn new(root: impl Into<std::path::PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn resolve(&self, rel: &str) -> Result<std::path::PathBuf> {
        use std::path::Component;
        let rel = rel.trim_start_matches('/');
        if std::path::Path::new(rel)
            .components()
            .any(|c| matches!(c, Component::ParentDir | Component::RootDir | Component::Prefix(_)))
        {
            return Err(HideError::PolicyDenied(format!("path escapes workspace: {rel}")));
        }
        Ok(self.root.join(rel))
    }
}

fn walk_tree(dir: &std::path::Path, root: &std::path::Path, depth: usize, budget: &mut usize) -> Vec<Value> {
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
        let rel = abs.strip_prefix(root).unwrap_or(&abs).to_string_lossy().replace('\\', "/");
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
            Ok(ConnectorStatus {
                id: self.id().to_string(),
                healthy,
                detail: format!("root={}", self.root.display()),
                contributions: Vec::new(),
            })
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
                    let meta = std::fs::metadata(&abs).map_err(|e| HideError::Storage(e.to_string()))?;
                    if meta.len() > FS_MAX_READ_BYTES {
                        return Err(HideError::PolicyDenied(format!("file too large to open ({} bytes)", meta.len())));
                    }
                    let text = std::fs::read_to_string(&abs).map_err(|e| HideError::Storage(e.to_string()))?;
                    Ok(json!({ "text": text, "lang": lang_for(path), "path": path }))
                }
                "write_file" => {
                    let path = params
                        .get("path")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| HideError::Config("missing path".to_string()))?;
                    let content = params
                        .get("content")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| HideError::Config("missing content".to_string()))?;
                    let abs = self.resolve(path)?;
                    if let Some(parent) = abs.parent() {
                        std::fs::create_dir_all(parent).map_err(|e| HideError::Storage(e.to_string()))?;
                    }
                    std::fs::write(&abs, content).map_err(|e| HideError::Storage(e.to_string()))?;
                    Ok(json!({ "ok": true, "bytes": content.len() }))
                }
                other => Err(HideError::NotFound(format!("fs connector method {other}"))),
            }
        })
    }
}

pub fn register_backend_connectors(registry: &ConnectorRegistry, services: &BackendServices) {
    registry.register(PersonalizationConnector::new(
        services.personalization_store.clone(),
    ));
    registry.register(ResearchConnector::new(services.research_ledger.clone()));
    registry.register(RuntimeConnector::new(services.role_registry.clone()));
    registry.register(CodeIndexConnector::new(services.code_index.clone()));
    registry.register(FsConnector::new(services.config.workspace_root.clone()));
    registry.register(ContextConnector::new(
        services.code_index.clone(),
        services.role_registry.clone(),
        services.memory_store.clone(),
    ));
}

#[cfg(test)]
mod fs_tests {
    use super::*;

    #[test]
    fn resolve_confines_to_root() {
        let fs = FsConnector::new("/work/proj");
        assert!(fs.resolve("src/main.rs").is_ok());
        assert!(fs.resolve("/src/main.rs").is_ok()); // leading slash stripped, stays in root
        assert!(fs.resolve("../etc/passwd").is_err());
        assert!(fs.resolve("a/../../b").is_err());
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

fn choose_context_role(
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
            Ok(ConnectorStatus {
                id: self.id.clone(),
                healthy: true,
                detail: "static connector ready".to_string(),
                contributions: self.contributions.clone(),
            })
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
