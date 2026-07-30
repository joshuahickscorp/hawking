//! Write effects, permission gate, and audit receipts.
//!
//! Every write is an effect. A connector never silently mutates the world: it
//! prepares a [`ConnectorWriteProposal`], the [`PermissionGate`] authorizes it
//! into a [`WriteReceipt`], and only then may [`execute_with_receipt`] run.
//! Reads do not require receipts.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::connector_abi::abi::{EffectClass, FamilyId};
use crate::connector_abi::account::{AccountHandle, AccountId};
use crate::connector_abi::error::{ConnectorError, Result};

/// Kind of write a connector proposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WriteKind {
    Create,
    Update,
    Delete,
}

impl WriteKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Create => "create",
            Self::Update => "update",
            Self::Delete => "delete",
        }
    }
}

/// A prepared, un-executed connector write. Carries no authority by itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectorWriteProposal {
    pub family_id: FamilyId,
    pub account_id: AccountId,
    pub kind: WriteKind,
    pub effect: EffectClass,
    /// Human summary for the permission UI / audit log.
    pub summary: String,
    /// Opaque target locator (path, message id, ...).
    pub target: String,
    /// Opaque payload bytes as UTF-8 JSON or plain text for fixtures.
    pub payload: String,
}

/// Permission decision recorded before any write executes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionDecision {
    Allow,
    Deny,
}

/// A write receipt: proof that a proposal was authorized. Required to execute.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WriteReceipt {
    pub id: String,
    pub proposal: ConnectorWriteProposal,
    pub decision: PermissionDecision,
    pub issued_at_ms: u64,
    /// Blake3 digest of the proposal fields for tamper evidence.
    pub digest: String,
    /// Whether this receipt has already been consumed by an execute call.
    pub consumed: bool,
}

impl WriteReceipt {
    pub fn is_allow(&self) -> bool {
        self.decision == PermissionDecision::Allow && !self.consumed
    }
}

/// Result of a successfully executed write.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WriteResult {
    pub receipt_id: String,
    pub target: String,
    pub notes: String,
}

fn proposal_digest(p: &ConnectorWriteProposal) -> String {
    let mut h = blake3::Hasher::new();
    h.update(p.family_id.as_str().as_bytes());
    h.update(b"|");
    h.update(p.account_id.as_str().as_bytes());
    h.update(b"|");
    h.update(p.kind.as_str().as_bytes());
    h.update(b"|");
    h.update(p.effect.as_str().as_bytes());
    h.update(b"|");
    h.update(p.summary.as_bytes());
    h.update(b"|");
    h.update(p.target.as_bytes());
    h.update(b"|");
    h.update(p.payload.as_bytes());
    h.finalize().to_hex().to_string()
}

/// Policy for auto-allowing certain write kinds in tests. Production callers
/// set a deny-by-default gate and only allow after explicit user approval.
#[derive(Debug, Clone, Default)]
pub struct PermissionPolicy {
    /// When true, every proposal is denied unless listed in `allow_targets`.
    pub deny_by_default: bool,
    /// Explicitly allowed target locators (exact match).
    pub allow_targets: Vec<String>,
}

impl PermissionPolicy {
    pub fn deny_by_default() -> Self {
        Self {
            deny_by_default: true,
            allow_targets: Vec::new(),
        }
    }
    pub fn allow_all_for_tests() -> Self {
        Self {
            deny_by_default: false,
            allow_targets: Vec::new(),
        }
    }
    pub fn allow_target(mut self, target: impl Into<String>) -> Self {
        self.allow_targets.push(target.into());
        self
    }
}

/// Permission + receipt authority for connector writes.
pub struct PermissionGate {
    policy: PermissionPolicy,
    receipts: BTreeMap<String, WriteReceipt>,
    next: AtomicU64,
    clock_ms: AtomicU64,
}

impl PermissionGate {
    pub fn new(policy: PermissionPolicy) -> Self {
        Self {
            policy,
            receipts: BTreeMap::new(),
            next: AtomicU64::new(0),
            clock_ms: AtomicU64::new(1),
        }
    }

    /// Authorize a proposal. Returns a receipt; only `Allow` receipts may
    /// execute. Denied proposals still leave a receipt for audit.
    pub fn authorize(&mut self, proposal: ConnectorWriteProposal) -> Result<WriteReceipt> {
        if proposal.effect == EffectClass::Read {
            return Err(ConnectorError::InvalidRequest(
                "read is not a write effect".into(),
            ));
        }
        let allowed = if self.policy.deny_by_default {
            self.policy
                .allow_targets
                .iter()
                .any(|t| t == &proposal.target)
        } else {
            true
        };
        let decision = if allowed {
            PermissionDecision::Allow
        } else {
            PermissionDecision::Deny
        };
        let n = self.next.fetch_add(1, Ordering::Relaxed);
        let id = format!("wr-{}", n);
        let issued_at_ms = self.clock_ms.fetch_add(1, Ordering::Relaxed);
        let digest = proposal_digest(&proposal);
        let receipt = WriteReceipt {
            id: id.clone(),
            proposal,
            decision,
            issued_at_ms,
            digest,
            consumed: false,
        };
        self.receipts.insert(id, receipt.clone());
        Ok(receipt)
    }

    /// Consume an allow receipt. Second consume fails. Deny receipts cannot
    /// be consumed for execution.
    pub fn consume(&mut self, receipt_id: &str) -> Result<WriteReceipt> {
        let r = self
            .receipts
            .get_mut(receipt_id)
            .ok_or_else(|| ConnectorError::InvalidWriteReceipt(receipt_id.into()))?;
        if r.consumed {
            return Err(ConnectorError::InvalidWriteReceipt(receipt_id.into()));
        }
        if r.decision != PermissionDecision::Allow {
            return Err(ConnectorError::WritePermissionDenied(
                r.proposal.summary.clone(),
            ));
        }
        // Re-check digest integrity.
        let expected = proposal_digest(&r.proposal);
        if expected != r.digest {
            return Err(ConnectorError::InvalidWriteReceipt(receipt_id.into()));
        }
        r.consumed = true;
        Ok(r.clone())
    }

    pub fn get(&self, receipt_id: &str) -> Option<&WriteReceipt> {
        self.receipts.get(receipt_id)
    }

    pub fn all_receipts(&self) -> impl Iterator<Item = &WriteReceipt> {
        self.receipts.values()
    }
}

/// Execute a write only when a valid allow receipt is presented. This is the
/// sole execute entry point — there is no path that mutates without a receipt.
pub fn execute_with_receipt<F>(
    gate: &mut PermissionGate,
    receipt_id: &str,
    handle: &AccountHandle,
    mut body: F,
) -> Result<WriteResult>
where
    F: FnMut(&ConnectorWriteProposal, &AccountHandle) -> Result<WriteResult>,
{
    let receipt = gate.consume(receipt_id)?;
    if receipt.proposal.account_id != handle.account_id {
        return Err(ConnectorError::InvalidWriteReceipt(receipt_id.into()));
    }
    let mut result = body(&receipt.proposal, handle)?;
    result.receipt_id = receipt.id;
    Ok(result)
}

/// Attempt to execute without a receipt — always fails. Exists so tests can
/// prove silent execution is impossible.
pub fn execute_without_receipt(_proposal: &ConnectorWriteProposal) -> Result<WriteResult> {
    Err(ConnectorError::WriteReceiptRequired)
}
