//! Projects: unify conversations, documents, objects, connectors, plans,
//! tasks, memory, automations, agents, and artifacts under lifecycle states.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Stable project id (`prj_…`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ProjectId(pub String);

impl ProjectId {
    pub fn new() -> Self {
        Self(format!(
            "prj_{}",
            ulid::Ulid::new().to_string().to_ascii_lowercase()
        ))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for ProjectId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ProjectId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Project lifecycle states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectState {
    Explore,
    Plan,
    Execute,
    Review,
    Archive,
}

impl ProjectState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Explore => "explore",
            Self::Plan => "plan",
            Self::Execute => "execute",
            Self::Review => "review",
            Self::Archive => "archive",
        }
    }

    pub fn all() -> &'static [ProjectState] {
        &[
            Self::Explore,
            Self::Plan,
            Self::Execute,
            Self::Review,
            Self::Archive,
        ]
    }

    /// Legal transitions (forward + archive from any non-archive).
    pub fn can_transition_to(self, next: ProjectState) -> bool {
        use ProjectState::*;
        if self == next {
            return true;
        }
        if next == Archive {
            return self != Archive;
        }
        matches!(
            (self, next),
            (Explore, Plan)
                | (Plan, Explore)
                | (Plan, Execute)
                | (Execute, Plan)
                | (Execute, Review)
                | (Review, Execute)
                | (Review, Archive)
                | (Explore, Execute) // skip plan when work is obvious
        )
    }
}

/// Kinds of members a project unifies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectMemberKind {
    Conversation,
    Document,
    Object,
    Connector,
    Plan,
    Task,
    Memory,
    Automation,
    Agent,
    Artifact,
}

impl ProjectMemberKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Conversation => "conversation",
            Self::Document => "document",
            Self::Object => "object",
            Self::Connector => "connector",
            Self::Plan => "plan",
            Self::Task => "task",
            Self::Memory => "memory",
            Self::Automation => "automation",
            Self::Agent => "agent",
            Self::Artifact => "artifact",
        }
    }

    pub fn all() -> &'static [ProjectMemberKind] {
        &[
            Self::Conversation,
            Self::Document,
            Self::Object,
            Self::Connector,
            Self::Plan,
            Self::Task,
            Self::Memory,
            Self::Automation,
            Self::Agent,
            Self::Artifact,
        ]
    }
}

/// A reference to something the project unifies (by kind + external id).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectMember {
    pub kind: ProjectMemberKind,
    pub ref_id: String,
    #[serde(default)]
    pub label: Option<String>,
}

/// Unified project container.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Project {
    pub id: ProjectId,
    pub name: String,
    pub state: ProjectState,
    pub members: Vec<ProjectMember>,
    /// Free-form notes / goal for the project.
    pub summary: String,
    pub created_ms: u64,
    pub updated_ms: u64,
    /// Optional link to an active swarm.
    pub active_swarm_id: Option<String>,
    /// Metadata bag (session ids, surface tags, etc.).
    #[serde(default)]
    pub meta: BTreeMap<String, Value>,
}

impl Project {
    pub fn create(name: impl Into<String>, summary: impl Into<String>, now_ms: u64) -> Self {
        Self {
            id: ProjectId::new(),
            name: name.into(),
            state: ProjectState::Explore,
            members: Vec::new(),
            summary: summary.into(),
            created_ms: now_ms,
            updated_ms: now_ms,
            active_swarm_id: None,
            meta: BTreeMap::new(),
        }
    }

    pub fn transition(&mut self, next: ProjectState, now_ms: u64) -> crate::lenses::Result<()> {
        if !self.state.can_transition_to(next) {
            return Err(crate::lenses::YouError::InvalidState(format!(
                "cannot transition project {} from {:?} to {:?}",
                self.id, self.state, next
            )));
        }
        self.state = next;
        self.updated_ms = now_ms;
        Ok(())
    }

    pub fn attach(
        &mut self,
        kind: ProjectMemberKind,
        ref_id: impl Into<String>,
        label: Option<String>,
        now_ms: u64,
    ) {
        let ref_id = ref_id.into();
        if self
            .members
            .iter()
            .any(|m| m.kind == kind && m.ref_id == ref_id)
        {
            return;
        }
        self.members.push(ProjectMember {
            kind,
            ref_id,
            label,
        });
        self.updated_ms = now_ms;
    }

    pub fn members_of(&self, kind: ProjectMemberKind) -> impl Iterator<Item = &ProjectMember> {
        self.members.iter().filter(move |m| m.kind == kind)
    }

    pub fn link_swarm(&mut self, swarm_id: impl Into<String>, now_ms: u64) {
        self.active_swarm_id = Some(swarm_id.into());
        self.updated_ms = now_ms;
    }

    pub fn declaration(&self) -> Value {
        serde_json::json!({
            "id": self.id.as_str(),
            "name": self.name,
            "state": self.state.as_str(),
            "summary": self.summary,
            "member_counts": ProjectMemberKind::all().iter().map(|k| {
                (k.as_str(), self.members_of(*k).count())
            }).collect::<BTreeMap<_, _>>(),
            "members": self.members,
            "active_swarm_id": self.active_swarm_id,
            "created_ms": self.created_ms,
            "updated_ms": self.updated_ms,
        })
    }
}
