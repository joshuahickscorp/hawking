use crate::personalize::{
    DynPersonalizationStore, InMemoryPersonalizationStore, JsonlPersonalizationStore,
};
use hawking_context::{
    ClassedMemorySystem, ContextCompiler, DynClassedMemory, InMemoryMemoryStore, MemoryStore,
    SqliteMemoryStore, TokenCounter,
};
use hawking_index::{CodeIndex, InMemoryCodeIndex, SqliteCodeIndex};
use hawking_orch::RoleRegistry;
use hawking_research::{DynResearchLedger, InMemoryResearchLedger, JsonlResearchLedger};
use hide_core::config::HideConfig;
use hide_core::event::JsonlEventLog;
use hide_core::ids::{now_ms, EventId, SessionId};
use hide_core::persistence::{
    DynBlobStore, DynEventLog, DynEventLogIntegrity, DynKeyValueStore, DynProjectionStore,
    FileBlobStore, FileKeyValueStore, FileProjectionStore, InMemoryBlobStore,
    InMemoryKeyValueStore, InMemoryProjectionStore,
};
use hide_core::project::WorkspaceLayout;
use hide_core::Result;
use hide_kernel::security::audit::EventChainAuditor;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Shared code-index handle consumed by grounding / context compile / connectors.
pub type DynCodeIndex = Arc<dyn CodeIndex>;

/// How many source files to index synchronously at workspace open. Beyond this
/// the open returns immediately (a failed/absent index already degrades to the
/// empty in-memory default); further files can be filled in by a later daemon
/// pass. Keeps open non-blocking on large repositories.
const OPEN_INGEST_FILE_CAP: usize = 64;

/// Directory name components / extensions skipped during bounded open ingest.
pub(crate) fn should_skip_index_path(path: &Path) -> bool {
    let mut comps = path.components();
    // Skip anything under these directory names anywhere in the relative path.
    const SKIP_DIRS: &[&str] = &[
        ".git",
        ".hide",
        "target",
        "node_modules",
        ".grok",
        "dist",
        "build",
        ".venv",
        "vendor",
    ];
    for c in comps.by_ref() {
        if let std::path::Component::Normal(s) = c {
            if SKIP_DIRS.iter().any(|d| s == *d) {
                return true;
            }
        }
    }
    // Only index text-ish source extensions (bounded, deterministic).
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => {
            const OK: &[&str] = &[
                "rs", "ts", "tsx", "js", "jsx", "py", "go", "java", "kt", "c", "h", "cc", "cpp",
                "hpp", "md", "toml", "json", "yaml", "yml", "sh", "css", "html", "txt",
            ];
            !OK.iter().any(|e| ext.eq_ignore_ascii_case(e))
        }
        None => true,
    }
}

/// Bounded, best-effort workspace ingest into a Sqlite index. Caps file count and
/// per-file size so open never blocks on a large repo. Errors on individual files
/// are skipped; a total failure leaves the index empty (still usable).
pub(crate) fn bounded_sqlite_ingest(
    index: &SqliteCodeIndex,
    root: &Path,
    max_files: usize,
    max_file_bytes: u64,
) -> usize {
    let mut indexed = 0usize;
    let walker = walkdir_shallow(root);
    for path in walker {
        if indexed >= max_files {
            break;
        }
        let rel = match path.strip_prefix(root) {
            Ok(r) => r,
            Err(_) => continue,
        };
        if should_skip_index_path(rel) {
            continue;
        }
        let meta = match std::fs::metadata(&path) {
            Ok(m) if m.is_file() => m,
            _ => continue,
        };
        if meta.len() > max_file_bytes {
            continue;
        }
        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let hash = blake3::hash(content.as_bytes()).to_hex().to_string();
        let rel_s = rel.to_string_lossy().replace('\\', "/");
        if index.index_text(&rel_s, &content, &hash).is_ok() {
            indexed += 1;
        }
    }
    indexed
}

/// Non-recursive-feeling walk that still descends, but is simple and dependency-free.
pub(crate) fn walkdir_shallow(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            // Skip hidden / heavy dirs early.
            if name == ".git"
                || name == ".hide"
                || name == "target"
                || name == "node_modules"
                || name == ".grok"
                || name == "vendor"
            {
                continue;
            }
            if path.is_dir() {
                stack.push(path);
            } else if path.is_file() {
                out.push(path);
            }
        }
    }
    // Stable order so tests are deterministic.
    out.sort();
    out
}

#[path = "services_session.rs"]
mod services_session;
pub use services_session::*;

#[path = "services_goal.rs"]
mod services_goal;
pub use services_goal::*;

#[path = "services_job.rs"]
mod services_job;
pub use services_job::*;

#[path = "services_workspace.rs"]
mod services_workspace;
pub use services_workspace::*;

pub type DynMemoryStore = Arc<dyn MemoryStore>;

#[derive(Clone)]
pub struct BackendServices {
    pub config: HideConfig,
    pub event_log: DynEventLog,
    pub(crate) verified_token_events: Arc<crate::classed_writers::VerifiedTokenEventLog>,
    /// Spine B: structured long-term memory (file-facts, decisions, test results,
    /// constraints, failed approaches) — the persistent Project Brain. Sqlite on
    /// disk via `open()`, RAM via `new()`/`with_stores()`.
    pub memory_store: DynMemoryStore,
    /// The six real HIDE memory classes (working / episodic / semantic_project /
    /// procedural / user / verification). Separate tables + write-capability
    /// boundaries — not labels on the Project Brain table above.
    pub classed_memory: DynClassedMemory,
    pub event_integrity: DynEventLogIntegrity,
    pub blob_store: DynBlobStore,
    pub projection_store: DynProjectionStore,
    pub key_value_store: DynKeyValueStore,
    pub personalization_store: DynPersonalizationStore,
    pub research_ledger: DynResearchLedger,
    pub role_registry: Arc<RoleRegistry>,
    /// Live code index used by grounding / ContextCompiler / connectors.
    /// Tests default to an empty [`InMemoryCodeIndex`]; `open` / `open_workspace`
    /// bind a durable [`SqliteCodeIndex`] with bounded ingest (degrading to the
    /// empty in-memory index on any open/ingest failure).
    pub code_index: DynCodeIndex,
    /// When the live index is the in-memory implementation (test constructors),
    /// this is `Some` so callers that need `add_text_file` can seed without a
    /// downcast. `None` when the live index is Sqlite (or the empty fallback is
    /// already empty and only reachable via the trait).
    pub memory_index: Option<Arc<InMemoryCodeIndex>>,
    /// When the live index is Sqlite (workspace open path), this is `Some` so
    /// ingest / tests can write through `index_text` without a downcast.
    pub sqlite_index: Option<Arc<SqliteCodeIndex>>,
    pub capabilities: BackendCapabilities,
    /// Stable session registry (open-or-create, not fresh-per-call).
    pub sessions: Arc<SessionRegistry>,
    /// The repo's resolved Claude Code migration instructions (CLAUDE.md tree +
    /// un-scoped rules), loaded ONCE at workspace open and cached here so the turn
    /// core folds them into the compiled context without re-parsing every turn
    /// (bible sec 20 / sec 78.1 #11). Empty for the in-memory constructors
    /// (`new`/`with_stores`); populated by `open` from the workspace root.
    /// Cache-invalidation on a live config edit is DEFERRED (reopen to refresh).
    pub repo_instructions: Arc<crate::compat_instructions::ResolvedInstructions>,
    /// Tokenizer-true token counter for context packing. Loaded once from
    /// `HIDE_TOKENIZER` / beside `HIDE_MODEL_WEIGHTS` when available; otherwise
    /// the `chars/4` heuristic (and compile reports `tokens_estimated`).
    pub token_counter: TokenCounter,
}

/// Resolve the live packing counter once at service construction.
pub(crate) fn discover_token_counter() -> TokenCounter {
    match TokenCounter::discover_from_env() {
        Some(c) => c,
        None => TokenCounter::heuristic(),
    }
}

impl BackendServices {
    /// Mint a host sink bound to the product's opaque verified-token ingress.
    pub fn verified_token_sinks(
        &self,
        session_id: hide_core::ids::SessionId,
    ) -> crate::speculation_safety::HostDurableSinks {
        crate::speculation_safety::HostDurableSinks::with_token_authority(
            session_id,
            self.verified_token_events.clone(),
        )
    }

    pub fn new(config: HideConfig, event_log: DynEventLog) -> Self {
        let memory = Arc::new(InMemoryCodeIndex::default());
        let workspace_id = config.workspace_root.display().to_string();
        let classed_memory: DynClassedMemory = Arc::new(
            ClassedMemorySystem::open_in_memory(workspace_id).expect("in-memory classed memory"),
        );
        // Mirror episodic memory off the durable event stream so any event a
        // client can read also lands in the classed store.
        let (event_log, verified_token_events) =
            crate::classed_writers::EpisodicEventMirror::wrap_with_token_authority(
                event_log,
                classed_memory.clone(),
            );
        Self {
            config,
            event_log,
            verified_token_events,
            memory_store: Arc::new(InMemoryMemoryStore::default()),
            classed_memory,
            event_integrity: Arc::new(EventChainAuditor),
            blob_store: Arc::new(InMemoryBlobStore::default()),
            projection_store: Arc::new(InMemoryProjectionStore::default()),
            key_value_store: Arc::new(InMemoryKeyValueStore::default()),
            personalization_store: Arc::new(InMemoryPersonalizationStore::default()),
            research_ledger: Arc::new(InMemoryResearchLedger::default()),
            role_registry: Arc::new(RoleRegistry::with_default_local_roles()),
            code_index: memory.clone(),
            memory_index: Some(memory),
            sqlite_index: None,
            capabilities: BackendCapabilities::wired(),
            sessions: Arc::new(SessionRegistry::default()),
            repo_instructions: Arc::new(crate::compat_instructions::ResolvedInstructions::empty()),
            token_counter: discover_token_counter(),
        }
    }

    pub fn with_stores(
        config: HideConfig,
        event_log: DynEventLog,
        blob_store: DynBlobStore,
        projection_store: DynProjectionStore,
        key_value_store: DynKeyValueStore,
        personalization_store: DynPersonalizationStore,
        research_ledger: DynResearchLedger,
    ) -> Self {
        // Tests / in-memory constructors: empty InMemoryCodeIndex stays the default.
        let memory = Arc::new(InMemoryCodeIndex::default());
        let workspace_id = config.workspace_root.display().to_string();
        let classed_memory: DynClassedMemory = Arc::new(
            ClassedMemorySystem::open_in_memory(workspace_id).expect("in-memory classed memory"),
        );
        let (event_log, verified_token_events) =
            crate::classed_writers::EpisodicEventMirror::wrap_with_token_authority(
                event_log,
                classed_memory.clone(),
            );
        Self {
            config,
            event_log,
            verified_token_events,
            memory_store: Arc::new(InMemoryMemoryStore::default()),
            classed_memory,
            event_integrity: Arc::new(EventChainAuditor),
            blob_store,
            projection_store,
            key_value_store,
            personalization_store,
            research_ledger,
            role_registry: Arc::new(RoleRegistry::with_default_local_roles()),
            code_index: memory.clone(),
            memory_index: Some(memory),
            sqlite_index: None,
            capabilities: BackendCapabilities::wired(),
            sessions: Arc::new(SessionRegistry::default()),
            repo_instructions: Arc::new(crate::compat_instructions::ResolvedInstructions::empty()),
            token_counter: discover_token_counter(),
        }
    }

    /// A [`ContextCompiler`] pre-loaded with the workspace token counter so
    /// packing is tokenizer-true whenever a real tokenizer was discovered.
    pub fn context_compiler(&self) -> ContextCompiler {
        ContextCompiler::new().with_counter(self.token_counter.clone())
    }

    pub fn open_workspace(workspace_root: impl Into<PathBuf>) -> Result<Self> {
        Self::open(HideConfig::for_workspace(workspace_root))
    }

    pub fn open(config: HideConfig) -> Result<Self> {
        // Resolve the repo's Claude Code migration instructions once, before the
        // config is moved into `with_stores`. Repo-scoped + best-effort: a repo
        // with no CLAUDE.md tree resolves empty and the turn core adds nothing.
        let repo_instructions =
            crate::compat_instructions::resolve_repo_instructions_for_root(&config.workspace_root);
        let layout = WorkspaceLayout::new(&config.workspace_root);
        std::fs::create_dir_all(&layout.hide_dir)?;
        std::fs::create_dir_all(&layout.snapshots)?;
        std::fs::create_dir_all(&layout.projections)?;
        std::fs::create_dir_all(&layout.cache)?;
        std::fs::create_dir_all(&layout.sandbox)?;
        std::fs::create_dir_all(&layout.tmp)?;

        // Raw durable log; mirrored onto classed_memory once below (not via
        // with_stores' temporary in-memory classed store).
        let raw_event_log: DynEventLog =
            Arc::new(JsonlEventLog::open(layout.event_log.join("events.jsonl"))?);
        let blob_store: DynBlobStore = Arc::new(FileBlobStore::open(&layout.blobs)?);
        let projection_store: DynProjectionStore =
            Arc::new(FileProjectionStore::open(&layout.projections)?);
        let key_value_store: DynKeyValueStore = Arc::new(FileKeyValueStore::open(&layout.kv)?);
        let personalization_store: DynPersonalizationStore =
            Arc::new(JsonlPersonalizationStore::open(
                layout
                    .hide_dir
                    .join("personalization")
                    .join("records.jsonl"),
            )?);
        let research_ledger: DynResearchLedger = Arc::new(JsonlResearchLedger::open(
            layout.hide_dir.join("research").join("runs.jsonl"),
        )?);

        // Spine B: the persistent Project Brain lives in a SQLite DB on disk.
        let memory_store: DynMemoryStore = Arc::new(SqliteMemoryStore::open(
            layout.hide_dir.join("memory").join("memory.db"),
        )?);

        // Six real memory classes: workspace tables under .hide/memory/, user
        // preferences under the user_root (cross-workspace). Degrades to
        // in-memory on open failure (same posture as the code index).
        let workspace_id = config.workspace_root.display().to_string();
        let classed_memory: DynClassedMemory = match ClassedMemorySystem::open(
            workspace_id.clone(),
            layout.hide_dir.join("memory").join("classes.db"),
            config.user_root.join("memory").join("user.db"),
        ) {
            Ok(sys) => Arc::new(sys),
            Err(e) => {
                eprintln!("warning: classed memory open failed ({e}); using in-memory six classes");
                Arc::new(
                    ClassedMemorySystem::open_in_memory(workspace_id)
                        .expect("in-memory classed memory"),
                )
            }
        };

        // with_stores will wrap raw_event_log onto a throwaway classed store; we
        // immediately rebind both fields to the durable pair so writers and
        // retrieval share one ClassedMemorySystem (single mirror layer).
        let mut services = Self::with_stores(
            config,
            raw_event_log.clone(),
            blob_store,
            projection_store,
            key_value_store,
            personalization_store,
            research_ledger,
        );
        services.memory_store = memory_store;
        services.classed_memory = classed_memory.clone();
        let (event_log, verified_token_events) =
            crate::classed_writers::EpisodicEventMirror::wrap_with_token_authority(
                raw_event_log,
                classed_memory,
            );
        services.event_log = event_log;
        services.verified_token_events = verified_token_events;
        services.repo_instructions = Arc::new(repo_instructions);

        // W4: bind the real SqliteCodeIndex at workspace open. A failed open or
        // ingest degrades to the empty in-memory index already installed by
        // with_stores — never fails the workspace open.
        services.bind_workspace_code_index(&layout);
        Ok(services)
    }

    /// Install SqliteCodeIndex + bounded workspace ingest. Best-effort: any
    /// failure leaves the empty InMemory default in place.
    pub(crate) fn bind_workspace_code_index(&mut self, layout: &WorkspaceLayout) {
        let index_dir = layout.hide_dir.join("index");
        if let Err(e) = std::fs::create_dir_all(&index_dir) {
            eprintln!("warning: code index dir create failed (using empty index): {e}");
            return;
        }
        let db_path = index_dir.join("code.sqlite");
        let sqlite = match SqliteCodeIndex::open(&db_path) {
            Ok(idx) => Arc::new(idx),
            Err(e) => {
                eprintln!("warning: SqliteCodeIndex open failed (using empty index): {e}");
                return;
            }
        };
        let max_file_bytes = self.config.index.max_file_bytes;
        let _n = bounded_sqlite_ingest(
            &sqlite,
            &self.config.workspace_root,
            OPEN_INGEST_FILE_CAP,
            max_file_bytes,
        );
        self.code_index = sqlite.clone();
        self.sqlite_index = Some(sqlite);
        self.memory_index = None;
    }

    /// Seed a text file into whichever concrete index is live. Used by tests and
    /// by anything that needs to write without knowing the backend. No-op only
    /// if both handles are somehow absent (should not happen).
    pub fn seed_code_file(&self, path: impl AsRef<str>, content: impl AsRef<str>) {
        let path = path.as_ref();
        let content = content.as_ref();
        if let Some(mem) = &self.memory_index {
            mem.add_text_file(path, content, None);
            return;
        }
        if let Some(sql) = &self.sqlite_index {
            let hash = blake3::hash(content.as_bytes()).to_hex().to_string();
            let _ = sql.index_text(path, content, &hash);
        }
    }

    pub fn layout(&self) -> WorkspaceLayout {
        WorkspaceLayout::new(&self.config.workspace_root)
    }

    /// The stable default ("primary") session. Returns the *same* id across
    /// calls (open-or-create), durably recorded so a workspace reopen recovers
    /// it — not a fresh `SessionId` per call.
    pub fn session(&self) -> SessionId {
        self.sessions
            .open_or_create(SessionRegistry::DEFAULT, Some(&self.key_value_store))
    }

    /// Open-or-create a *named* session (e.g. a second tab/run). Stable per name.
    pub fn session_named(&self, name: &str) -> SessionId {
        self.sessions
            .open_or_create(name, Some(&self.key_value_store))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendCapabilities {
    pub agent_kernel: bool,
    pub context_compiler: bool,
    pub code_index: bool,
    pub model_orchestration: bool,
    pub research_lab: bool,
    pub fleet: bool,
    pub personalization: bool,
    pub remote_protocol: bool,
}

impl BackendCapabilities {
    /// Capabilities reflecting what hide-backend *actually wires* (the audit
    /// flagged the old `Default` as overstating reality). Each flag is `true`
    /// only because a real subsystem backs it:
    ///
    /// * `agent_kernel` — `hide_kernel::AgentKernel` is constructed + driven.
    /// * `context_compiler`/`code_index` — the Context/CodeIndex connectors wrap
    ///   real `hawking-context`/`hawking-index` stores.
    /// * `model_orchestration` — `RoleRegistry` + `SimpleRouter` + (now) the HTTP
    ///   `ModelProvider`/`RuntimeSupervisor`.
    /// * `research_lab`/`personalization` — durable ledgers + connectors.
    /// * `fleet` — `hide_fleet::FleetManager` is now imported + exposed
    ///   (`BackendHost::fleet_run`); the dead dep is load-bearing.
    /// * `remote_protocol` — **false**: no remote JSON-RPC server is wired in the
    ///   shell (deferred). Honest caps over aspirational ones.
    pub fn wired() -> Self {
        Self {
            agent_kernel: true,
            context_compiler: true,
            code_index: true,
            model_orchestration: true,
            research_lab: true,
            fleet: true,
            personalization: true,
            remote_protocol: false,
        }
    }
}

impl Default for BackendCapabilities {
    fn default() -> Self {
        Self::wired()
    }
}

impl std::fmt::Debug for BackendServices {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackendServices")
            .field("workspace_root", &self.config.workspace_root)
            .field("capabilities", &self.capabilities)
            .finish()
    }
}

pub type SharedBackend = Arc<BackendServices>;

#[cfg(test)]

mod tests {
    use super::*;
    use crate::personalize::{PersonalizationRecord, TaskClass};
    use hawking_research::{ResearchRun, ResearchState};
    use hide_core::event::NewEvent;
    use hide_core::ids::now_ms;
    #[tokio::test]
    pub(crate) async fn open_workspace_wires_durable_stores() {
        let dir = std::env::temp_dir().join(format!("hide_backend_{}", now_ms()));
        let services = BackendServices::open_workspace(&dir).unwrap();
        let layout = services.layout();
        assert!(layout.hide_dir.exists());
        assert!(layout.event_log.exists());
        assert!(!services.role_registry.all().is_empty());
        let session = services.session();
        services
            .event_log
            .append(NewEvent::system(
                session.clone(),
                "backend.started",
                serde_json::json!({ "ok": true }),
            ))
            .await
            .unwrap();
        let events = services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await
            .unwrap();
        assert_eq!(events.len(), 1);
        let integrity = services.event_integrity.verify_chain(&events).unwrap();
        assert_eq!(integrity.checked_events, 1);
        let blob = services
            .blob_store
            .put(b"backend blob".to_vec(), Some("text/plain".to_string()))
            .unwrap();
        assert_eq!(
            services.blob_store.get(&blob).unwrap().unwrap(),
            b"backend blob"
        );
        services
            .projection_store
            .put_projection(&session, 1, serde_json::json!({ "view": "timeline" }))
            .unwrap();
        assert_eq!(
            services
                .projection_store
                .latest_projection(&session)
                .unwrap()
                .unwrap()
                .1["view"],
            "timeline"
        );
        services
            .key_value_store
            .put(
                "sessions",
                session.as_str(),
                serde_json::json!({ "open": true }),
            )
            .unwrap();
        assert_eq!(
            services
                .key_value_store
                .get("sessions", session.as_str())
                .unwrap()
                .unwrap()["open"],
            true
        );
        services
            .personalization_store
            .append(&PersonalizationRecord::accepted(
                TaskClass::EditCode,
                "prompt",
                "diff",
            ))
            .unwrap();
        assert_eq!(services.personalization_store.load_all().unwrap().len(), 1);
        let mut run = ResearchRun::new("backend research");
        run.state = ResearchState::Complete;
        services.research_ledger.append_run(&run).unwrap();
        assert_eq!(services.research_ledger.load_runs().unwrap().len(), 1);
        let reopened = BackendServices::open_workspace(&dir).unwrap();
        assert_eq!(reopened.personalization_store.load_all().unwrap().len(), 1);
        assert_eq!(reopened.research_ledger.load_runs().unwrap().len(), 1);
        let _ = std::fs::remove_dir_all(dir);
    }
    #[tokio::test]
    pub(crate) async fn session_is_stable_across_calls_and_reopen() {
        let dir = std::env::temp_dir().join(format!("hide_session_reg_{}", now_ms()));
        let services = BackendServices::open_workspace(&dir).unwrap();
        let a = services.session();
        let b = services.session();
        assert_eq!(a, b);
        let named = services.session_named("review-tab");
        assert_ne!(named, a);
        assert_eq!(named, services.session_named("review-tab"));
        let reopened = BackendServices::open_workspace(&dir).unwrap();
        assert_eq!(reopened.session(), a);
        let _ = std::fs::remove_dir_all(dir);
    }
}
