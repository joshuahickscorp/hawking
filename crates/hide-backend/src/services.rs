use hawking_index::InMemoryCodeIndex;
use hawking_orch::RoleRegistry;
use hawking_research::{DynResearchLedger, InMemoryResearchLedger, JsonlResearchLedger};
use hide_core::config::HideConfig;
use hide_core::event::JsonlEventLog;
use hide_core::ids::SessionId;
use hide_core::persistence::{
    DynBlobStore, DynEventLog, DynEventLogIntegrity, DynKeyValueStore, DynProjectionStore,
    FileBlobStore, FileKeyValueStore, FileProjectionStore, InMemoryBlobStore,
    InMemoryKeyValueStore, InMemoryProjectionStore,
};
use hide_core::project::WorkspaceLayout;
use hide_core::Result;
use hide_personalize::{
    DynPersonalizationStore, InMemoryPersonalizationStore, JsonlPersonalizationStore,
};
use hide_security::audit::EventChainAuditor;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone)]
pub struct BackendServices {
    pub config: HideConfig,
    pub event_log: DynEventLog,
    pub event_integrity: DynEventLogIntegrity,
    pub blob_store: DynBlobStore,
    pub projection_store: DynProjectionStore,
    pub key_value_store: DynKeyValueStore,
    pub personalization_store: DynPersonalizationStore,
    pub research_ledger: DynResearchLedger,
    pub role_registry: Arc<RoleRegistry>,
    pub code_index: Arc<InMemoryCodeIndex>,
    pub capabilities: BackendCapabilities,
}

impl BackendServices {
    pub fn new(config: HideConfig, event_log: DynEventLog) -> Self {
        Self {
            config,
            event_log,
            event_integrity: Arc::new(EventChainAuditor),
            blob_store: Arc::new(InMemoryBlobStore::default()),
            projection_store: Arc::new(InMemoryProjectionStore::default()),
            key_value_store: Arc::new(InMemoryKeyValueStore::default()),
            personalization_store: Arc::new(InMemoryPersonalizationStore::default()),
            research_ledger: Arc::new(InMemoryResearchLedger::default()),
            role_registry: Arc::new(RoleRegistry::with_default_local_roles()),
            code_index: Arc::new(InMemoryCodeIndex::default()),
            capabilities: BackendCapabilities::default(),
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
        Self {
            config,
            event_log,
            event_integrity: Arc::new(EventChainAuditor),
            blob_store,
            projection_store,
            key_value_store,
            personalization_store,
            research_ledger,
            role_registry: Arc::new(RoleRegistry::with_default_local_roles()),
            code_index: Arc::new(InMemoryCodeIndex::default()),
            capabilities: BackendCapabilities::default(),
        }
    }

    pub fn open_workspace(workspace_root: impl Into<PathBuf>) -> Result<Self> {
        Self::open(HideConfig::for_workspace(workspace_root))
    }

    pub fn open(config: HideConfig) -> Result<Self> {
        let layout = WorkspaceLayout::new(&config.workspace_root);
        std::fs::create_dir_all(&layout.hide_dir)?;
        std::fs::create_dir_all(&layout.snapshots)?;
        std::fs::create_dir_all(&layout.projections)?;
        std::fs::create_dir_all(&layout.cache)?;
        std::fs::create_dir_all(&layout.sandbox)?;
        std::fs::create_dir_all(&layout.tmp)?;

        let event_log: DynEventLog =
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

        Ok(Self::with_stores(
            config,
            event_log,
            blob_store,
            projection_store,
            key_value_store,
            personalization_store,
            research_ledger,
        ))
    }

    pub fn layout(&self) -> WorkspaceLayout {
        WorkspaceLayout::new(&self.config.workspace_root)
    }

    pub fn session(&self) -> SessionId {
        SessionId::new()
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

impl Default for BackendCapabilities {
    fn default() -> Self {
        Self {
            agent_kernel: true,
            context_compiler: true,
            code_index: true,
            model_orchestration: true,
            research_lab: true,
            fleet: true,
            personalization: true,
            remote_protocol: true,
        }
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
    use hawking_research::{ResearchRun, ResearchState};
    use hide_core::event::{EventPayload, NewEvent};
    use hide_core::ids::now_ms;
    use hide_personalize::{PersonalizationRecord, TaskClass};

    #[tokio::test]
    async fn open_workspace_wires_durable_stores() {
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
                EventPayload::Custom(serde_json::json!({ "ok": true })),
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
        assert!(integrity.ok);

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
}
