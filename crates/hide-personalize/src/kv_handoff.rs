//! KV-cache handoff between agents (bible §11.5 + §11.4.2 Approach 2).
//!
//! When a fan-out swarm launches, every worker receives the *same* prefix (the
//! system prompt + repo-map + plan context). Re-prefilling that prefix N times
//! is pure waste. This module defines the [`KvShareGroup`] protocol: the Planner
//! checkpoints its prefix KV state under a [`KvKey`], registers a share group,
//! and every worker's [`GenerateRequest`] carries a [`KvHandle`] seed so the
//! runtime restores the shared prefix instead of re-prefilling it.
//!
//! ## Seam (important)
//!
//! This crate **does not invent KV state**. The actual block copy is performed
//! by the in-tree `copy_kv_prefix_to_slot` primitive in the live runtime
//! (`hawking-core`'s engine), exposed over `hawking-serve`'s HTTP surface. This
//! module is the *protocol + lifecycle*: the keys, the fork position, the
//! member set, the TTL/refcount bookkeeping, and the typed [`KvHandle`] that
//! rides in the handoff. The [`KvPrefixCopier`] trait is the one-method seam the
//! runtime fills in; [`copy_for_group`] drives it for every member. No fake KV
//! map is constructed here.

use serde::{Deserialize, Serialize};

/// Identifier of an agent in a swarm (e.g. `"planner:0"`, `"worker:3"`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(pub String);

impl AgentId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

/// Key under which a prefix KV state is stored in the runtime's KV store. Opaque
/// to this crate; the runtime maps it to its block table. Kept a newtype so it
/// can't be confused with an arbitrary string handle.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KvKey(pub String);

impl KvKey {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A handle the receiver uses to restore a shared prefix (§11.4.2 / §11.5.2):
/// the store key plus the token position at which divergence begins. Carried in
/// the `kv_seed` field of a worker's generate request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvHandle {
    pub store_key: KvKey,
    /// All members share tokens `0..fork_seq`; each diverges from `fork_seq` on.
    pub fork_seq: u64,
}

/// The §11.5.2 protocol object: registered with the Governor when a fan-out
/// swarm launches. The prefix KV state lives once in the store; the N members
/// share it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvShareGroup {
    /// Key under which the prefix KV state is stored.
    pub prefix_key: KvKey,
    /// Agents that share this prefix.
    pub members: Vec<AgentId>,
    /// Token position at which each member diverges (shared region is `0..`).
    pub fork_seq: u64,
    /// The shared KV state is evicted after this many ms of non-use.
    pub ttl_ms: u64,
}

impl KvShareGroup {
    pub fn new(prefix_key: KvKey, fork_seq: u64, members: Vec<AgentId>) -> Self {
        Self {
            prefix_key,
            members,
            fork_seq,
            ttl_ms: 60_000,
        }
    }

    /// The seed every member's generate request carries (§11.5.2 step 3): the
    /// same store key + fork position for all of them.
    pub fn handle(&self) -> KvHandle {
        KvHandle {
            store_key: self.prefix_key.clone(),
            fork_seq: self.fork_seq,
        }
    }

    /// Reference count = number of members still sharing the prefix. The runtime
    /// evicts the shared blocks only when this reaches zero (§11.5.2 step 4).
    pub fn refcount(&self) -> usize {
        self.members.len()
    }
}

/// The KV-seed extension to a generate request (§11.5.2). If `Some`, the runtime
/// skips prefill of `0..fork_seq` and restores from the store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub agent: AgentId,
    pub prompt_suffix: String,
    /// When set, restore the shared prefix instead of re-prefilling it.
    pub kv_seed: Option<KvHandle>,
}

/// The one-method seam to the in-tree `copy_kv_prefix_to_slot` primitive. The
/// runtime (hawking-serve / hawking-core) implements this; this crate only
/// orchestrates *which* copies happen and *when*. Returning a `Result<usize>` of
/// tokens-restored lets the caller measure the prefill it saved.
pub trait KvPrefixCopier {
    /// Copy the stored prefix at `handle.store_key` (length `handle.fork_seq`)
    /// into the decode slot for `agent`. Maps directly onto the in-tree
    /// `copy_kv_prefix_to_slot(store_key, fork_seq, slot)`.
    fn copy_kv_prefix_to_slot(
        &self,
        agent: &AgentId,
        handle: &KvHandle,
    ) -> std::result::Result<usize, String>;
}

/// Outcome of broadcasting a shared prefix to a group's members.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BroadcastReport {
    /// Members whose prefix was restored, with tokens-restored each.
    pub restored: Vec<(AgentId, usize)>,
    /// Members that failed to restore (and must fall back to full prefill).
    pub failed: Vec<(AgentId, String)>,
}

impl BroadcastReport {
    /// Total tokens of prefill skipped across the swarm — the win this protocol
    /// exists to produce.
    pub fn tokens_saved(&self) -> usize {
        self.restored.iter().map(|(_, n)| *n).sum()
    }
}

/// Drive the seam for every member of a share group (§11.5.2 step 3). A member
/// that fails to restore is reported, not panicked on, so the swarm can fall
/// back to full prefill for that worker.
pub fn copy_for_group(copier: &dyn KvPrefixCopier, group: &KvShareGroup) -> BroadcastReport {
    let handle = group.handle();
    let mut restored = Vec::new();
    let mut failed = Vec::new();
    for member in &group.members {
        match copier.copy_kv_prefix_to_slot(member, &handle) {
            Ok(n) => restored.push((member.clone(), n)),
            Err(e) => failed.push((member.clone(), e)),
        }
    }
    BroadcastReport { restored, failed }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A test double standing in for the runtime's `copy_kv_prefix_to_slot`.
    /// It records the calls and returns `fork_seq` tokens restored — it does NOT
    /// fabricate KV state, it only models the seam contract.
    struct FakeCopier {
        fail_for: Option<String>,
    }
    impl KvPrefixCopier for FakeCopier {
        fn copy_kv_prefix_to_slot(
            &self,
            agent: &AgentId,
            handle: &KvHandle,
        ) -> std::result::Result<usize, String> {
            if self.fail_for.as_deref() == Some(agent.0.as_str()) {
                Err("slot busy".into())
            } else {
                Ok(handle.fork_seq as usize)
            }
        }
    }

    #[test]
    fn share_group_handle_is_shared_across_members() {
        let group = KvShareGroup::new(
            KvKey::new("planner-0-fork"),
            2300,
            vec![AgentId::new("worker:0"), AgentId::new("worker:1")],
        );
        assert_eq!(group.refcount(), 2);
        let h = group.handle();
        assert_eq!(h.fork_seq, 2300);
        assert_eq!(h.store_key.as_str(), "planner-0-fork");
    }

    #[test]
    fn broadcast_sums_saved_prefill() {
        let group = KvShareGroup::new(
            KvKey::new("k"),
            2300,
            (0..16)
                .map(|i| AgentId::new(format!("worker:{i}")))
                .collect(),
        );
        let copier = FakeCopier { fail_for: None };
        let report = copy_for_group(&copier, &group);
        assert_eq!(report.restored.len(), 16);
        assert!(report.failed.is_empty());
        // 16 workers × 2300-token prefix all skipped.
        assert_eq!(report.tokens_saved(), 16 * 2300);
    }

    #[test]
    fn broadcast_reports_failures_without_aborting() {
        let group = KvShareGroup::new(
            KvKey::new("k"),
            100,
            vec![AgentId::new("worker:0"), AgentId::new("worker:1")],
        );
        let copier = FakeCopier {
            fail_for: Some("worker:1".into()),
        };
        let report = copy_for_group(&copier, &group);
        assert_eq!(report.restored.len(), 1);
        assert_eq!(report.failed.len(), 1);
        assert_eq!(report.tokens_saved(), 100);
    }

    #[test]
    fn generate_request_carries_seed() {
        let group = KvShareGroup::new(KvKey::new("k"), 42, vec![AgentId::new("worker:0")]);
        let req = GenerateRequest {
            agent: AgentId::new("worker:0"),
            prompt_suffix: "continue".into(),
            kv_seed: Some(group.handle()),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: GenerateRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.kv_seed.unwrap().fork_seq, 42);
    }
}
