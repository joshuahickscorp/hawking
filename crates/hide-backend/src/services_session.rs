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

/// The session registry — open-or-create stable sessions (bible ch.07).
///
/// The scaffold's `session()` minted a *fresh* `SessionId` on every call, so two
/// calls in one process never agreed on "the current session". [`SessionRegistry`]
/// keeps a named default (the "primary" session) stable for the host's lifetime
/// and records every opened session in the durable KV store under the `sessions`
/// namespace, so a reopen of the workspace recovers the same default session id.
#[derive(Default)]
pub struct SessionRegistry {
    /// Named sessions → their stable id (the "default"/"primary" lives here).
    by_name: Mutex<std::collections::HashMap<String, SessionId>>,
}

/// The typed relationship a session bears to its parent (bible sec 32-33): the
/// conversation-graph taxonomy. `origin` (a display string) is kept for
/// backward compatibility and derived from this; the graph projection keys off
/// the typed variant.
///
/// * `Root`: a new/primary session with no parent.
/// * `Fork`: a durable branch, an independent copy-forward of a parent prefix
///   ("explore an alternative from here"), read/write.
/// * `EphemeralFork`: a cheap, discardable exploration fork (same mechanics as
///   `Fork`, flagged so a client can prune it without ceremony).
/// * `SideChat`: a fork that defaults READ-ONLY and can merge a typed summary
///   back to its parent (the parent's transcript gains a cited summary; the side
///   chat is not destroyed).
/// * `VerifierBranch`: a fork dedicated to independent verification of the
///   parent's work.
/// * `MergedSummary`: a record whose typed summary has been folded back into a
///   parent.
/// * `Superseded`: a session replaced by a later one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionRelationship {
    #[default]
    Root,
    Fork,
    EphemeralFork,
    SideChat,
    VerifierBranch,
    MergedSummary,
    Superseded,
}

impl SessionRelationship {
    /// The stable snake_case display string mirrored into [`SessionRecord::origin`]
    /// (kept for backward-compat with clients that read the string `origin`).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Root => "root",
            Self::Fork => "fork",
            Self::EphemeralFork => "ephemeral_fork",
            Self::SideChat => "side_chat",
            Self::VerifierBranch => "verifier_branch",
            Self::MergedSummary => "merged_summary",
            Self::Superseded => "superseded",
        }
    }
}

/// A durable session/thread record (bible ch.07). Beyond identity it carries,
/// for a FORK, its ancestry: the parent session and the boundary (`seq` + the
/// event id it resolved from) the fork's history was folded up to. Stored in the
/// KV store under the `session_records` namespace so a client can enumerate
/// threads and render a fork's lineage (the conversation graph) after a
/// workspace reopen.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionRecord {
    pub session_id: SessionId,
    /// The session this one was forked from (`None` for a root/new session).
    pub parent_session_id: Option<SessionId>,
    /// The parent event `seq` this fork's history was folded up to (inclusive).
    pub forked_at: Option<u64>,
    /// The parent event id the boundary resolved from (when forked by event).
    pub forked_at_event: Option<EventId>,
    /// Wall-clock creation time (ms since epoch).
    pub created_ms: u64,
    /// How this session came to be: `root`, `fork`, `ephemeral_fork`,
    /// `side_chat`, ... (the display mirror of [`Self::relationship`]).
    pub origin: String,
    /// The typed relationship to the parent (the conversation-graph taxonomy).
    /// Defaulted (`Root`) so records written before this field existed still
    /// deserialize cleanly.
    #[serde(default)]
    pub relationship: SessionRelationship,
    /// A read-only session: a client should not route new turns into it. A side
    /// chat defaults to this (it merges a typed summary back instead of being
    /// written to directly). Defaulted `false` for pre-existing records.
    #[serde(default)]
    pub read_only: bool,
}

impl SessionRecord {
    /// A ROOT/new session record: no parent, no boundary, read/write.
    pub fn root(session_id: SessionId) -> Self {
        Self {
            session_id,
            parent_session_id: None,
            forked_at: None,
            forked_at_event: None,
            created_ms: now_ms(),
            origin: SessionRelationship::Root.as_str().to_string(),
            relationship: SessionRelationship::Root,
            read_only: false,
        }
    }

    /// A branch record carrying ancestry (parent + boundary) with an explicit
    /// [`SessionRelationship`] + read-only flag: the shared core for [`Self::fork`],
    /// [`Self::ephemeral_fork`], [`Self::side_chat`], and [`Self::verifier_branch`].
    pub fn branch(
        session_id: SessionId,
        parent: SessionId,
        forked_at: u64,
        forked_at_event: Option<EventId>,
        relationship: SessionRelationship,
        read_only: bool,
    ) -> Self {
        Self {
            session_id,
            parent_session_id: Some(parent),
            forked_at: Some(forked_at),
            forked_at_event,
            created_ms: now_ms(),
            origin: relationship.as_str().to_string(),
            relationship,
            read_only,
        }
    }

    /// A record for a genuine FORK: parent + boundary recorded so ancestry is
    /// durable and independent of the fork's own (fresh-lineage) event log.
    pub fn fork(
        session_id: SessionId,
        parent: SessionId,
        forked_at: u64,
        forked_at_event: Option<EventId>,
    ) -> Self {
        Self::branch(
            session_id,
            parent,
            forked_at,
            forked_at_event,
            SessionRelationship::Fork,
            false,
        )
    }

    /// A cheap, discardable EXPLORATION fork (same mechanics as a fork; flagged
    /// `EphemeralFork` so a client can prune it without ceremony).
    pub fn ephemeral_fork(
        session_id: SessionId,
        parent: SessionId,
        forked_at: u64,
        forked_at_event: Option<EventId>,
    ) -> Self {
        Self::branch(
            session_id,
            parent,
            forked_at,
            forked_at_event,
            SessionRelationship::EphemeralFork,
            false,
        )
    }

    /// A SIDE CHAT: a fork that defaults READ-ONLY and can merge a typed summary
    /// back to its parent. Ancestry is preserved exactly as for a fork.
    pub fn side_chat(
        session_id: SessionId,
        parent: SessionId,
        forked_at: u64,
        forked_at_event: Option<EventId>,
    ) -> Self {
        Self::branch(
            session_id,
            parent,
            forked_at,
            forked_at_event,
            SessionRelationship::SideChat,
            true,
        )
    }

    /// A VERIFIER branch: a fork dedicated to independently verifying the
    /// parent's work.
    pub fn verifier_branch(
        session_id: SessionId,
        parent: SessionId,
        forked_at: u64,
        forked_at_event: Option<EventId>,
    ) -> Self {
        Self::branch(
            session_id,
            parent,
            forked_at,
            forked_at_event,
            SessionRelationship::VerifierBranch,
            false,
        )
    }
}

/// One node in a conversation-graph projection (bible sec 32-33): a session +
/// its typed relationship to its parent. A flat, model-free projection of a
/// [`SessionRecord`] (or a synthesized root for an unrecorded session).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConversationNode {
    pub session_id: SessionId,
    pub parent_session_id: Option<SessionId>,
    pub relationship: SessionRelationship,
    pub origin: String,
    pub read_only: bool,
    pub created_ms: u64,
}

impl ConversationNode {
    pub(crate) fn from_record(record: &SessionRecord) -> Self {
        Self {
            session_id: record.session_id.clone(),
            parent_session_id: record.parent_session_id.clone(),
            relationship: record.relationship,
            origin: record.origin.clone(),
            read_only: record.read_only,
            created_ms: record.created_ms,
        }
    }

    /// A synthesized ROOT node for a session with no durable record (e.g. the
    /// primary session, which is tracked in the `sessions` namespace, not
    /// `session_records`). `created_ms = 0` marks it as unknown/unrecorded.
    pub(crate) fn synthetic_root(session_id: SessionId) -> Self {
        Self {
            session_id,
            parent_session_id: None,
            relationship: SessionRelationship::Root,
            origin: SessionRelationship::Root.as_str().to_string(),
            read_only: false,
            created_ms: 0,
        }
    }
}

/// A parent -> child edge in the conversation graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConversationEdge {
    pub parent: SessionId,
    pub child: SessionId,
}

/// A bounded, deterministic conversation-graph projection rooted at one session
/// (bible sec 32-33): the queried node, its ancestry chain (nearest parent
/// first, up to a root), and its DIRECT children (forks / side chats / ephemeral
/// forks / ...), plus the parent->child edges from the node to each child.
/// No model; ordering is deterministic (children/edges sorted by `created_ms`
/// then `session_id`) so the projection is stable across runs and reopens.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConversationGraph {
    pub node: ConversationNode,
    pub ancestry: Vec<ConversationNode>,
    pub children: Vec<ConversationNode>,
    pub edges: Vec<ConversationEdge>,
}

impl SessionRegistry {
    pub(crate) const DEFAULT: &'static str = "primary";
    pub(crate) const KV_NAMESPACE: &'static str = "sessions";
    pub(crate) const RECORDS_NAMESPACE: &'static str = "session_records";

    /// Open-or-create the named session. The first call mints + records it (in
    /// the KV store if present); subsequent calls return the same id.
    pub fn open_or_create(&self, name: &str, kv: Option<&DynKeyValueStore>) -> SessionId {
        let mut map = self.by_name.lock();
        if let Some(id) = map.get(name) {
            return id.clone();
        }
        // Recover a previously-recorded id from the durable KV store, else mint.
        let id = kv
            .and_then(|kv| kv.get(Self::KV_NAMESPACE, name).ok().flatten())
            .and_then(|v| {
                v.get("session_id")
                    .and_then(|s| s.as_str())
                    .map(SessionId::from)
            })
            .unwrap_or_default();
        if let Some(kv) = kv {
            let _ = kv.put(
                Self::KV_NAMESPACE,
                name,
                serde_json::json!({ "session_id": id.as_str() }),
            );
        }
        map.insert(name.to_string(), id.clone());
        id
    }

    /// Durably record a session/thread record (a fork's ancestry) in the KV
    /// store so a reopen, or a thread list, recovers it. A best-effort write: a
    /// failing KV never fails the fork that produced the record.
    pub fn record_session(&self, kv: &DynKeyValueStore, record: &SessionRecord) {
        if let Ok(value) = serde_json::to_value(record) {
            let _ = kv.put(Self::RECORDS_NAMESPACE, record.session_id.as_str(), value);
        }
    }

    /// Look up a previously-recorded session/thread record (ancestry), if any.
    pub fn session_record(
        &self,
        kv: &DynKeyValueStore,
        session_id: &SessionId,
    ) -> Option<SessionRecord> {
        kv.get(Self::RECORDS_NAMESPACE, session_id.as_str())
            .ok()
            .flatten()
            .and_then(|value| serde_json::from_value(value).ok())
    }

    /// Build a bounded, deterministic conversation-graph projection rooted at
    /// `session_id` (bible sec 32-33) by walking the durable `session_records`
    /// KV namespace: the node, its ancestry chain (nearest parent first, up to a
    /// root), and its DIRECT children (forks / side chats / ephemeral forks), plus
    /// parent->child edges. Every record is loaded once into a lookup map, so the
    /// walks are O(1); ordering is deterministic (children/edges sort by
    /// `created_ms` then `session_id`). No model; safe headless.
    ///
    /// A session with no durable record (e.g. the primary session, which lives in
    /// the `sessions` namespace) projects as a synthesized ROOT node; its children
    /// are still discovered by their `parent_session_id` back-links.
    pub fn conversation_graph(
        &self,
        kv: &DynKeyValueStore,
        session_id: &SessionId,
    ) -> ConversationGraph {
        // Load every recorded thread once (bounded by the thread count) into a
        // map keyed by session id, so ancestry/child walks are O(1) lookups.
        let records: std::collections::HashMap<SessionId, SessionRecord> = kv
            .list(Self::RECORDS_NAMESPACE)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|(_, value)| serde_json::from_value::<SessionRecord>(value).ok())
            .map(|record| (record.session_id.clone(), record))
            .collect();

        let node = records
            .get(session_id)
            .map(ConversationNode::from_record)
            .unwrap_or_else(|| ConversationNode::synthetic_root(session_id.clone()));

        // Ancestry: follow parent back-links up to a root, guarding against a
        // cycle (a corrupt record) with a visited set.
        let mut ancestry = Vec::new();
        let mut seen = std::collections::HashSet::new();
        seen.insert(session_id.clone());
        let mut cursor = node.parent_session_id.clone();
        while let Some(parent_id) = cursor {
            if !seen.insert(parent_id.clone()) {
                break; // cycle guard: never loop forever on corrupt ancestry
            }
            match records.get(&parent_id) {
                Some(record) => {
                    let parent_node = ConversationNode::from_record(record);
                    cursor = parent_node.parent_session_id.clone();
                    ancestry.push(parent_node);
                }
                None => {
                    // An unrecorded parent (a root/primary session): include it as
                    // a synthesized root and stop the walk.
                    ancestry.push(ConversationNode::synthetic_root(parent_id));
                    break;
                }
            }
        }

        // Direct children: every record whose parent back-link is this node.
        let mut children: Vec<ConversationNode> = records
            .values()
            .filter(|record| record.parent_session_id.as_ref() == Some(session_id))
            .map(ConversationNode::from_record)
            .collect();
        children.sort_by(|a, b| {
            a.created_ms
                .cmp(&b.created_ms)
                .then_with(|| a.session_id.cmp(&b.session_id))
        });

        let edges: Vec<ConversationEdge> = children
            .iter()
            .map(|child| ConversationEdge {
                parent: session_id.clone(),
                child: child.session_id.clone(),
            })
            .collect();

        ConversationGraph {
            node,
            ancestry,
            children,
            edges,
        }
    }
}
