//! The HAIDER parallel task DAG.
//!
//! READY nodes may execute concurrently when: dependencies are satisfied, write
//! scopes are disjoint, resource classes are compatible, and the MemGate admits
//! them. Each node records the full provenance required by the spec.

use std::collections::{BTreeMap, BTreeSet};

use super::lanes::{ContextBudget, LaneRole};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub String);

impl NodeId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A read/write scope: path prefixes + resource tags. Two nodes can run
/// concurrently only if their write scopes are disjoint.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Scope {
    pub paths: BTreeSet<String>,
    pub resources: BTreeSet<String>,
}

impl Scope {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn path(mut self, p: impl Into<String>) -> Self {
        self.paths.insert(p.into());
        self
    }
    pub fn resource(mut self, r: impl Into<String>) -> Self {
        self.resources.insert(r.into());
        self
    }
    /// True if no path or resource is shared (safe to write concurrently).
    pub fn disjoint(&self, other: &Scope) -> bool {
        !Self::paths_overlap(&self.paths, &other.paths)
            && self.resources.is_disjoint(&other.resources)
    }
    pub fn is_empty(&self) -> bool {
        self.paths.is_empty() && self.resources.is_empty()
    }

    fn paths_overlap(a: &BTreeSet<String>, b: &BTreeSet<String>) -> bool {
        for x in a {
            for y in b {
                if Self::path_prefix_overlap(x, y) {
                    return true;
                }
            }
        }
        false
    }

    fn path_prefix_overlap(a: &str, b: &str) -> bool {
        if a == b {
            return true;
        }
        a.starts_with(&format!("{b}/")) || b.starts_with(&format!("{a}/"))
    }
}

/// Resource class. `GpuTiming` (protected benchmarks) is exclusive: it must not
/// overlap another `GpuTiming` or a `GpuInference` lane (a throughput benchmark
/// under another inference lane is invalid unless it measures contention).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceClass {
    Cpu,
    GpuInference,
    GpuTiming,
    Io,
    Worktree,
    Build,
    Test,
    NetworkExternal,
    ExclusiveRuntime,
}

impl ResourceClass {
    pub fn compatible(&self, other: &Self) -> bool {
        match (self, other) {
            (ResourceClass::GpuTiming, ResourceClass::GpuTiming) => false,
            (ResourceClass::GpuTiming, ResourceClass::GpuInference)
            | (ResourceClass::GpuInference, ResourceClass::GpuTiming) => false,
            (ResourceClass::ExclusiveRuntime, _) | (_, ResourceClass::ExclusiveRuntime) => false,
            _ => true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeStatus {
    Ready,
    Queued,
    Running,
    Done,
    Failed,
    Blocked,
}

/// A node's result (filled when Done).
#[derive(Clone, Debug, Default)]
pub struct NodeResult {
    pub summary: String,
    pub changed_files: BTreeSet<String>,
    pub test_results: Vec<String>,
    pub proposed_actions: Vec<String>,
}

/// One node in the HAIDER DAG.
#[derive(Clone, Debug)]
pub struct HaiderNode {
    pub id: NodeId,
    pub parent: Option<NodeId>,
    /// Node ids that must be Done before this node is Ready.
    pub deps: Vec<NodeId>,
    pub role: LaneRole,
    pub objective: String,
    pub read_scope: Scope,
    pub write_scope: Scope,
    pub resource_class: ResourceClass,
    pub context_budget: ContextBudget,
    pub worker_session: Option<String>,
    pub status: NodeStatus,
    pub result: Option<NodeResult>,
    pub receipt: Option<String>,
    pub acceptance: Option<String>,
}

/// The DAG.
#[derive(Clone, Debug, Default)]
pub struct HaiderDag {
    nodes: BTreeMap<NodeId, HaiderNode>,
}

impl HaiderDag {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, node: HaiderNode) {
        self.nodes.insert(node.id.clone(), node);
    }

    pub fn get(&self, id: &NodeId) -> Option<&HaiderNode> {
        self.nodes.get(id)
    }

    pub fn get_mut(&mut self, id: &NodeId) -> Option<&mut HaiderNode> {
        self.nodes.get_mut(id)
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Nodes whose deps are all Done and which are not yet Running/Done.
    pub fn ready_nodes(&self) -> Vec<NodeId> {
        self.nodes
            .values()
            .filter(|n| {
                matches!(n.status, NodeStatus::Ready | NodeStatus::Queued)
                    && n.deps.iter().all(|d| {
                        self.nodes
                            .get(d)
                            .map(|p| p.status == NodeStatus::Done)
                            .unwrap_or(false)
                    })
            })
            .map(|n| n.id.clone())
            .collect()
    }

    /// Whether two nodes can run concurrently (write scopes disjoint + resource
    /// classes compatible).
    pub fn can_run_concurrently(&self, a: &NodeId, b: &NodeId) -> bool {
        match (self.nodes.get(a), self.nodes.get(b)) {
            (Some(x), Some(y)) => {
                x.write_scope.disjoint(&y.write_scope)
                    && x.resource_class.compatible(&y.resource_class)
            }
            _ => false,
        }
    }

    /// Mark a node Done with its result.
    pub fn complete(&mut self, id: &NodeId, result: NodeResult) {
        if let Some(n) = self.nodes.get_mut(id) {
            n.status = NodeStatus::Done;
            n.result = Some(result);
        }
    }

    /// Mark a node Failed.
    pub fn fail(&mut self, id: &NodeId) {
        if let Some(n) = self.nodes.get_mut(id) {
            n.status = NodeStatus::Failed;
        }
    }
}
