//! Connector-scoped memory. Ingested connector content never silently enters
//! global (`user` / `semantic`) memory.
//!
//! Connector reads land in [`MemoryScope::Connector`]. Promotion to
//! [`MemoryScope::User`] or [`MemoryScope::Semantic`] requires an explicit
//! capability mint ([`UserMemoryPromotionCap`] / [`SemanticPromotionCap`]).
//! The connector read path never holds those caps.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::abi::FamilyId;
use crate::account::AccountId;
use crate::error::{ConnectorError, Result};

/// Memory scope for connector-ingested content.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryScope {
    /// Content scoped to one connector account. Default landing zone.
    Connector {
        family_id: FamilyId,
        account_id: AccountId,
    },
    /// Project / semantic memory. Requires explicit promotion.
    Semantic,
    /// User preference / standing memory. Requires explicit promotion.
    User,
}

impl MemoryScope {
    pub fn connector(family_id: FamilyId, account_id: AccountId) -> Self {
        Self::Connector {
            family_id,
            account_id,
        }
    }
    pub fn as_label(&self) -> String {
        match self {
            Self::Connector {
                family_id,
                account_id,
            } => format!("connector:{}:{}", family_id, account_id),
            Self::Semantic => "semantic".into(),
            Self::User => "user".into(),
        }
    }
}

impl std::fmt::Display for MemoryScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.as_label())
    }
}

/// One memory record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: String,
    pub scope: MemoryScope,
    pub content: String,
    pub source_object_id: String,
    pub written_at_ms: u64,
}

/// Capability: promote connector content into user memory.
/// Mint only at the explicit user-intent entry point. Connector read paths
/// must not hold this type.
#[derive(Debug, Clone, Copy)]
pub struct UserMemoryPromotionCap {
    _private: (),
}

impl UserMemoryPromotionCap {
    /// Mint only at the user-intent entry point.
    pub fn mint() -> Self {
        Self { _private: () }
    }
}

/// Capability: promote connector content into semantic memory.
#[derive(Debug, Clone, Copy)]
pub struct SemanticPromotionCap {
    _private: (),
}

impl SemanticPromotionCap {
    pub fn mint() -> Self {
        Self { _private: () }
    }
}

/// Capability held by connector read/ingest paths. Can only write connector scope.
#[derive(Debug, Clone, Copy)]
pub struct ConnectorIngestCap {
    _private: (),
}

impl ConnectorIngestCap {
    /// Mint for a connector ingest path. Does not authorize user/semantic writes.
    pub fn mint() -> Self {
        Self { _private: () }
    }
}

/// In-memory store used to prove scope isolation. Not the production memory
/// system — just the boundary for connector content.
#[derive(Default)]
pub struct ConnectorMemoryStore {
    records: BTreeMap<String, MemoryRecord>,
    next: AtomicU64,
    clock_ms: AtomicU64,
}

impl ConnectorMemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingest connector content into connector scope only.
    ///
    /// Type boundary: takes [`ConnectorIngestCap`], not a user/semantic cap.
    /// There is no overload that writes user memory from this path.
    pub fn ingest_connector(
        &mut self,
        _cap: &ConnectorIngestCap,
        family_id: FamilyId,
        account_id: AccountId,
        source_object_id: impl Into<String>,
        content: impl Into<String>,
    ) -> MemoryRecord {
        let n = self.next.fetch_add(1, Ordering::Relaxed);
        let id = format!("cm-{}", n);
        let rec = MemoryRecord {
            id: id.clone(),
            scope: MemoryScope::connector(family_id, account_id),
            content: content.into(),
            source_object_id: source_object_id.into(),
            written_at_ms: self.clock_ms.fetch_add(1, Ordering::Relaxed),
        };
        self.records.insert(id, rec.clone());
        rec
    }

    /// Attempt to write user memory from a connector ingest path.
    ///
    /// Always refuses. Exists so the safety test can name the property:
    /// a connector read cannot write `user` memory.
    pub fn ingest_as_user_from_connector(
        &mut self,
        _cap: &ConnectorIngestCap,
        _content: impl Into<String>,
    ) -> Result<MemoryRecord> {
        Err(ConnectorError::SilentMemoryPromotion {
            target: MemoryScope::User,
        })
    }

    /// Explicit promotion to user memory. Requires [`UserMemoryPromotionCap`].
    pub fn promote_to_user(
        &mut self,
        _cap: &UserMemoryPromotionCap,
        record_id: &str,
    ) -> Result<MemoryRecord> {
        let src = self
            .records
            .get(record_id)
            .ok_or_else(|| ConnectorError::NotFound(record_id.into()))?
            .clone();
        if !matches!(src.scope, MemoryScope::Connector { .. }) {
            return Err(ConnectorError::InvalidRequest(
                "only connector-scoped records can be promoted".into(),
            ));
        }
        let n = self.next.fetch_add(1, Ordering::Relaxed);
        let id = format!("um-{}", n);
        let rec = MemoryRecord {
            id: id.clone(),
            scope: MemoryScope::User,
            content: src.content,
            source_object_id: src.id,
            written_at_ms: self.clock_ms.fetch_add(1, Ordering::Relaxed),
        };
        self.records.insert(id, rec.clone());
        Ok(rec)
    }

    /// Explicit promotion to semantic memory.
    pub fn promote_to_semantic(
        &mut self,
        _cap: &SemanticPromotionCap,
        record_id: &str,
    ) -> Result<MemoryRecord> {
        let src = self
            .records
            .get(record_id)
            .ok_or_else(|| ConnectorError::NotFound(record_id.into()))?
            .clone();
        if !matches!(src.scope, MemoryScope::Connector { .. }) {
            return Err(ConnectorError::InvalidRequest(
                "only connector-scoped records can be promoted".into(),
            ));
        }
        let n = self.next.fetch_add(1, Ordering::Relaxed);
        let id = format!("sm-{}", n);
        let rec = MemoryRecord {
            id: id.clone(),
            scope: MemoryScope::Semantic,
            content: src.content,
            source_object_id: src.id,
            written_at_ms: self.clock_ms.fetch_add(1, Ordering::Relaxed),
        };
        self.records.insert(id, rec.clone());
        Ok(rec)
    }

    pub fn get(&self, id: &str) -> Option<&MemoryRecord> {
        self.records.get(id)
    }

    pub fn in_scope(&self, scope: &MemoryScope) -> Vec<&MemoryRecord> {
        self.records
            .values()
            .filter(|r| &r.scope == scope)
            .collect()
    }

    pub fn user_records(&self) -> Vec<&MemoryRecord> {
        self.records
            .values()
            .filter(|r| matches!(r.scope, MemoryScope::User))
            .collect()
    }
}
