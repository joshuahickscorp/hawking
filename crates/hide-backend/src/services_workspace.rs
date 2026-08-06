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
use super::*;

// --- Multi-repo workspace graph (bible sec 35, sec 78.1 #14) -----------------

/// Whether a repo in the workspace graph has been TRUSTED (bible sec 35: the
/// trust-before-config principle). A repo is `Untrusted` until a human explicitly
/// trusts it; while untrusted its instructions/policy refs are INERT (never
/// treated active, never granted capability). Snake_case so it round-trips in the
/// KV store; `Untrusted` is the default so a record written before this field
/// existed (or a repo added with no explicit trust decision) is inert by default,
/// which is the safe direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrustState {
    #[default]
    Untrusted,
    Trusted,
}

impl TrustState {
    pub fn is_trusted(&self) -> bool {
        matches!(self, Self::Trusted)
    }
}

/// One REPOSITORY node in the multi-repo workspace graph (bible sec 35). Beyond
/// identity + location it carries the refs (instructions / index / policy) the
/// turn core would fold in, but only once the repo is TRUSTED (trust-before-
/// config; see [`RepoNode::active_instructions_ref`]). Stored in the KV
/// `workspace_repos` namespace keyed by `repo_id` so the graph survives a
/// workspace reopen.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepoNode {
    pub repo_id: String,
    pub root_path: PathBuf,
    /// Whether this repo has been trusted. Until it is, the instructions/policy
    /// refs below are inert (trust-before-config). Defaulted `Untrusted`.
    #[serde(default)]
    pub trust: TrustState,
    /// The checked-out branch, if known.
    pub branch: Option<String>,
    /// A ref (blob hash / path) to the repo's resolved instructions (its
    /// CLAUDE.md tree). Inert while untrusted.
    pub instructions_ref: Option<String>,
    /// A ref to the repo's code-index snapshot, if built.
    pub index_ref: Option<String>,
    /// A ref to the repo's policy document. Inert while untrusted.
    pub policy_ref: Option<String>,
}

impl RepoNode {
    /// A fresh, UNTRUSTED repo node (trust-before-config: a repo is inert until a
    /// human trusts it). Builder methods layer on the branch/refs.
    pub fn new(repo_id: impl Into<String>, root_path: impl Into<PathBuf>) -> Self {
        Self {
            repo_id: repo_id.into(),
            root_path: root_path.into(),
            trust: TrustState::Untrusted,
            branch: None,
            instructions_ref: None,
            index_ref: None,
            policy_ref: None,
        }
    }

    pub fn with_trust(mut self, trust: TrustState) -> Self {
        self.trust = trust;
        self
    }

    pub fn with_branch(mut self, branch: impl Into<String>) -> Self {
        self.branch = Some(branch.into());
        self
    }

    pub fn with_instructions_ref(mut self, instructions_ref: impl Into<String>) -> Self {
        self.instructions_ref = Some(instructions_ref.into());
        self
    }

    pub fn with_index_ref(mut self, index_ref: impl Into<String>) -> Self {
        self.index_ref = Some(index_ref.into());
        self
    }

    pub fn with_policy_ref(mut self, policy_ref: impl Into<String>) -> Self {
        self.policy_ref = Some(policy_ref.into());
        self
    }

    pub fn is_trusted(&self) -> bool {
        self.trust.is_trusted()
    }

    /// The instructions ref ONLY when the repo is trusted (trust-before-config):
    /// an untrusted repo's instructions are inert, never folded into a compiled
    /// context or granted capability. `None` while untrusted even if a ref is
    /// present.
    pub fn active_instructions_ref(&self) -> Option<&str> {
        if self.is_trusted() {
            self.instructions_ref.as_deref()
        } else {
            None
        }
    }

    /// The policy ref ONLY when the repo is trusted (trust-before-config). `None`
    /// while untrusted even if a ref is present.
    pub fn active_policy_ref(&self) -> Option<&str> {
        if self.is_trusted() {
            self.policy_ref.as_deref()
        } else {
            None
        }
    }
}

/// Optional RESOURCE LIMITS for an environment (bible sec 35): bounds a turn's
/// runtime may consume. All optional; an unset field means "unbounded here".
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_procs: Option<u32>,
    pub max_memory_mb: Option<u64>,
    pub max_wall_secs: Option<u64>,
}

/// One ENVIRONMENT node in the workspace graph (bible sec 35): a named execution
/// context spanning one or more filesystem roots, with its runtime, resolved
/// environment vars (by ref), network policy, granted tool scopes, and resource
/// limits. Stored in the KV `workspace_environments` namespace keyed by `env_id`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvironmentNode {
    pub env_id: String,
    /// The filesystem roots this environment exposes (may span repos).
    pub fs_roots: Vec<PathBuf>,
    /// The runtime label (e.g. `"native"`, `"container:node20"`); a display
    /// string, model-free.
    pub runtime: String,
    /// A ref (blob hash / path) to the resolved environment vars, kept out of the
    /// node so secrets are not inlined into the graph projection.
    pub vars_ref: Option<String>,
    /// The network policy label (e.g. `"deny"`, `"allow_list"`).
    pub net_policy: String,
    /// The tool scopes granted inside this environment.
    pub tool_scopes: Vec<String>,
    /// Resource bounds for turns run in this environment.
    #[serde(default)]
    pub resource_limits: ResourceLimits,
}

impl EnvironmentNode {
    /// A fresh environment with safe defaults: a `native` runtime, a `deny`
    /// network policy, no fs roots / tool scopes, no limits. Builders layer on.
    pub fn new(env_id: impl Into<String>) -> Self {
        Self {
            env_id: env_id.into(),
            fs_roots: Vec::new(),
            runtime: "native".to_string(),
            vars_ref: None,
            net_policy: "deny".to_string(),
            tool_scopes: Vec::new(),
            resource_limits: ResourceLimits::default(),
        }
    }

    pub fn with_fs_roots(mut self, fs_roots: Vec<PathBuf>) -> Self {
        self.fs_roots = fs_roots;
        self
    }

    pub fn with_runtime(mut self, runtime: impl Into<String>) -> Self {
        self.runtime = runtime.into();
        self
    }

    pub fn with_vars_ref(mut self, vars_ref: impl Into<String>) -> Self {
        self.vars_ref = Some(vars_ref.into());
        self
    }

    pub fn with_net_policy(mut self, net_policy: impl Into<String>) -> Self {
        self.net_policy = net_policy.into();
        self
    }

    pub fn with_tool_scopes(mut self, tool_scopes: Vec<String>) -> Self {
        self.tool_scopes = tool_scopes;
        self
    }

    pub fn with_resource_limits(mut self, resource_limits: ResourceLimits) -> Self {
        self.resource_limits = resource_limits;
        self
    }
}

/// A typed relationship between two repos in the workspace graph (bible sec 35).
/// Snake_case so it round-trips in the KV store and reads stably in a projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkspaceEdgeKind {
    DependsOn,
    Imports,
    Deploys,
    Documents,
    Tests,
    OwnsSchemaFor,
    ConsumesApiFrom,
    GeneratedFrom,
}

impl WorkspaceEdgeKind {
    /// The stable snake_case display string (also used inside the edge's KV key).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DependsOn => "depends_on",
            Self::Imports => "imports",
            Self::Deploys => "deploys",
            Self::Documents => "documents",
            Self::Tests => "tests",
            Self::OwnsSchemaFor => "owns_schema_for",
            Self::ConsumesApiFrom => "consumes_api_from",
            Self::GeneratedFrom => "generated_from",
        }
    }
}

/// A typed, directed edge between two repos (`from` -> `to`) in the workspace
/// graph. Stored in the KV `workspace_edges` namespace keyed by a deterministic
/// `from|kind|to` triple, so re-adding the same edge is idempotent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceEdge {
    pub from: String,
    pub to: String,
    pub kind: WorkspaceEdgeKind,
}

impl WorkspaceEdge {
    pub fn new(from: impl Into<String>, to: impl Into<String>, kind: WorkspaceEdgeKind) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            kind,
        }
    }

    /// The deterministic KV key for this edge: `from|kind|to`. Re-adding an edge
    /// with the same endpoints + kind overwrites the same key (idempotent).
    pub fn key(&self) -> String {
        format!("{}|{}|{}", self.from, self.kind.as_str(), self.to)
    }
}

/// The durable record of an ENVIRONMENT SWITCH for a session (bible sec 35.3):
/// the session moved from `previous_env` to `new_env` for a stated `reason`,
/// adopting the target environment's `fs_roots` + `tool_scopes`. Emitted as an
/// `environment.switch` event on the session's OWN log (so the session/thread is
/// not lost, the switch is a point in the same durable history) and returned to
/// the caller.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvironmentSwitch {
    pub session_id: SessionId,
    /// The environment the session was in before (`None` on the first switch).
    pub previous_env: Option<String>,
    pub new_env: String,
    pub reason: String,
    pub fs_roots: Vec<PathBuf>,
    pub tool_scopes: Vec<String>,
    pub switched_ms: u64,
}

/// A deterministic projection of the multi-repo workspace graph (bible sec 35):
/// every repo node, every environment node, and every typed edge, each ordered
/// deterministically (repos by `repo_id`, environments by `env_id`, edges by
/// `from` then `kind` then `to`) so the graph is stable across runs and reopens.
/// No model; a flat read of the durable `workspace_*` KV namespaces.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkspaceGraph {
    pub repos: Vec<RepoNode>,
    pub environments: Vec<EnvironmentNode>,
    pub edges: Vec<WorkspaceEdge>,
}

/// Durable persistence + projection for the multi-repo workspace graph (bible
/// sec 35) over the KV store. A stateless facade over three namespaces
/// (`workspace_repos`, `workspace_environments`, `workspace_edges`) plus a
/// per-session "current environment" pointer (`workspace_env_current`), mirroring
/// how [`GoalStore`]/[`CheckpointStore`] wrap their namespaces.
pub struct WorkspaceStore;

impl WorkspaceStore {
    pub const REPOS_NAMESPACE: &'static str = "workspace_repos";
    pub const ENVIRONMENTS_NAMESPACE: &'static str = "workspace_environments";
    pub const EDGES_NAMESPACE: &'static str = "workspace_edges";
    pub const CURRENT_ENV_NAMESPACE: &'static str = "workspace_env_current";

    pub fn put_repo(kv: &DynKeyValueStore, repo: &RepoNode) -> Result<()> {
        let value = serde_json::to_value(repo)?;
        kv.put(Self::REPOS_NAMESPACE, &repo.repo_id, value)
    }

    pub fn get_repo(kv: &DynKeyValueStore, repo_id: &str) -> Option<RepoNode> {
        kv.get(Self::REPOS_NAMESPACE, repo_id)
            .ok()
            .flatten()
            .and_then(|value| serde_json::from_value(value).ok())
    }

    pub fn put_environment(kv: &DynKeyValueStore, env: &EnvironmentNode) -> Result<()> {
        let value = serde_json::to_value(env)?;
        kv.put(Self::ENVIRONMENTS_NAMESPACE, &env.env_id, value)
    }

    pub fn get_environment(kv: &DynKeyValueStore, env_id: &str) -> Option<EnvironmentNode> {
        kv.get(Self::ENVIRONMENTS_NAMESPACE, env_id)
            .ok()
            .flatten()
            .and_then(|value| serde_json::from_value(value).ok())
    }

    pub fn put_edge(kv: &DynKeyValueStore, edge: &WorkspaceEdge) -> Result<()> {
        let value = serde_json::to_value(edge)?;
        kv.put(Self::EDGES_NAMESPACE, &edge.key(), value)
    }

    /// The session's current environment id, if it has switched into one.
    pub fn current_env(kv: &DynKeyValueStore, session: &SessionId) -> Option<String> {
        kv.get(Self::CURRENT_ENV_NAMESPACE, session.as_str())
            .ok()
            .flatten()
            .and_then(|value| {
                value
                    .get("env_id")
                    .and_then(|s| s.as_str())
                    .map(String::from)
            })
    }

    /// Durably record the session's current environment id (after a switch).
    pub fn set_current_env(kv: &DynKeyValueStore, session: &SessionId, env_id: &str) -> Result<()> {
        kv.put(
            Self::CURRENT_ENV_NAMESPACE,
            session.as_str(),
            serde_json::json!({ "env_id": env_id }),
        )
    }

    /// Build the deterministic [`WorkspaceGraph`] projection by walking the three
    /// durable namespaces once and sorting each collection into a stable order
    /// (repos by id, environments by id, edges by `from`/`kind`/`to`). The KV
    /// `list` order is unspecified, so the sort is what makes the graph stable
    /// across runs and reopens.
    pub fn graph(kv: &DynKeyValueStore) -> WorkspaceGraph {
        let mut repos: Vec<RepoNode> = kv
            .list(Self::REPOS_NAMESPACE)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|(_, value)| serde_json::from_value::<RepoNode>(value).ok())
            .collect();
        repos.sort_by(|a, b| a.repo_id.cmp(&b.repo_id));

        let mut environments: Vec<EnvironmentNode> = kv
            .list(Self::ENVIRONMENTS_NAMESPACE)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|(_, value)| serde_json::from_value::<EnvironmentNode>(value).ok())
            .collect();
        environments.sort_by(|a, b| a.env_id.cmp(&b.env_id));

        let mut edges: Vec<WorkspaceEdge> = kv
            .list(Self::EDGES_NAMESPACE)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|(_, value)| serde_json::from_value::<WorkspaceEdge>(value).ok())
            .collect();
        edges.sort_by(|a, b| {
            a.from
                .cmp(&b.from)
                .then_with(|| a.kind.as_str().cmp(b.kind.as_str()))
                .then_with(|| a.to.cmp(&b.to))
        });

        WorkspaceGraph {
            repos,
            environments,
            edges,
        }
    }
}
