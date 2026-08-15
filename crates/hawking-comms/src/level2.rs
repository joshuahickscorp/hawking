//! LEVEL 2 — STRUCTURED STATE: plans, evidence graphs, typed beliefs, tool results.

use crate::error::{CommsError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// What kind of structured payload is inside.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuredKind {
    Plan,
    EvidenceGraph,
    Belief,
    ToolResult,
    /// Bundle of mixed structured objects.
    Bundle,
}

/// LEVEL 2 envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuredState {
    pub schema: String,
    pub kind: StructuredKind,
    pub session_id: String,
    pub sender: String,
    pub created_unix_ms: u64,
    /// Content hash over the canonical JSON of `payload`.
    pub content_hash: String,
    pub payload: StructuredPayload,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StructuredPayload {
    /// Opaque JSON plan (typically a serialized `hide_protocol::Plan`).
    Plan {
        plan_id: String,
        plan: Value,
    },
    EvidenceGraph(EvidenceGraph),
    Belief(Belief),
    ToolResult(ToolResultPayload),
    Bundle {
        items: Vec<StructuredPayload>,
    },
}

impl StructuredState {
    pub fn from_payload(
        session_id: impl Into<String>,
        sender: impl Into<String>,
        payload: StructuredPayload,
    ) -> Self {
        let kind = payload.kind();
        let body = serde_json::to_vec(&payload).unwrap_or_default();
        let content_hash = format!("blake3:{}", blake3::hash(&body).to_hex());
        Self {
            schema: crate::STRUCTURED_STATE_SCHEMA.to_string(),
            kind,
            session_id: session_id.into(),
            sender: sender.into(),
            created_unix_ms: 0,
            content_hash,
            payload,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema != crate::STRUCTURED_STATE_SCHEMA {
            return Err(CommsError::Invalid(format!(
                "structured schema {}",
                self.schema
            )));
        }
        if self.session_id.trim().is_empty() || self.sender.trim().is_empty() {
            return Err(CommsError::Invalid(
                "structured state requires session_id and sender".into(),
            ));
        }
        if self.payload.kind() != self.kind && self.kind != StructuredKind::Bundle {
            // Bundle kind is only for Bundle payload; other kinds must match.
            if !matches!(self.payload, StructuredPayload::Bundle { .. }) {
                // kind field should match payload tag
            }
        }
        let body = serde_json::to_vec(&self.payload)
            .map_err(|e| CommsError::Invalid(format!("structured payload serialize: {e}")))?;
        let expected = format!("blake3:{}", blake3::hash(&body).to_hex());
        if self.content_hash != expected {
            return Err(CommsError::HashMismatch {
                expected,
                got: self.content_hash.clone(),
            });
        }
        self.payload.validate()
    }
}

impl StructuredPayload {
    pub fn kind(&self) -> StructuredKind {
        match self {
            Self::Plan { .. } => StructuredKind::Plan,
            Self::EvidenceGraph(_) => StructuredKind::EvidenceGraph,
            Self::Belief(_) => StructuredKind::Belief,
            Self::ToolResult(_) => StructuredKind::ToolResult,
            Self::Bundle { .. } => StructuredKind::Bundle,
        }
    }

    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Plan { plan_id, plan } => {
                if plan_id.trim().is_empty() {
                    return Err(CommsError::Invalid("plan_id empty".into()));
                }
                if plan.is_null() {
                    return Err(CommsError::Invalid("plan payload null".into()));
                }
                Ok(())
            }
            Self::EvidenceGraph(g) => g.validate(),
            Self::Belief(b) => b.validate(),
            Self::ToolResult(t) => t.validate(),
            Self::Bundle { items } => {
                if items.is_empty() {
                    return Err(CommsError::Invalid("empty structured bundle".into()));
                }
                for i in items {
                    i.validate()?;
                }
                Ok(())
            }
        }
    }
}

// ── Evidence graph ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceGraph {
    pub nodes: Vec<EvidenceNode>,
    pub edges: Vec<EvidenceEdge>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceNode {
    pub id: String,
    pub kind: EvidenceNodeKind,
    pub label: String,
    /// Optional content hash (e.g. perception citation hash / CAS pin).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
    #[serde(default)]
    pub attrs: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceNodeKind {
    Claim,
    Evidence,
    Source,
    Citation,
    Counter,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceEdge {
    pub from: String,
    pub to: String,
    pub relation: EvidenceRelation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceRelation {
    Supports,
    Refutes,
    Cites,
    DerivedFrom,
    SameAs,
}

impl EvidenceGraph {
    pub fn validate(&self) -> Result<()> {
        if self.nodes.is_empty() {
            return Err(CommsError::Invalid("evidence graph has no nodes".into()));
        }
        let ids: std::collections::BTreeSet<_> = self.nodes.iter().map(|n| n.id.as_str()).collect();
        for e in &self.edges {
            if !ids.contains(e.from.as_str()) || !ids.contains(e.to.as_str()) {
                return Err(CommsError::Invalid(format!(
                    "evidence edge {} -> {} references missing node",
                    e.from, e.to
                )));
            }
        }
        Ok(())
    }
}

// ── Typed beliefs ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Belief {
    pub id: String,
    /// Proposition in plain text (LEVEL 1 portable projection).
    pub proposition: String,
    pub polarity: BeliefPolarity,
    /// 0.0 ..= 1.0 calibrated confidence when known; uncalibrated if None.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    /// Evidence node ids that support this belief.
    #[serde(default)]
    pub evidence_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BeliefPolarity {
    Affirm,
    Deny,
    Unknown,
}

impl Belief {
    pub fn validate(&self) -> Result<()> {
        if self.id.trim().is_empty() || self.proposition.trim().is_empty() {
            return Err(CommsError::Invalid(
                "belief requires id and proposition".into(),
            ));
        }
        if let Some(c) = self.confidence {
            if !(0.0..=1.0).contains(&c) {
                return Err(CommsError::Invalid(
                    "belief confidence must be in 0..=1".into(),
                ));
            }
        }
        Ok(())
    }
}

// ── Tool results ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResultPayload {
    pub tool_name: String,
    pub call_id: String,
    pub ok: bool,
    /// Structured result (JSON). Large blobs should be CAS-pinned and referenced.
    pub result: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
}

impl ToolResultPayload {
    pub fn validate(&self) -> Result<()> {
        if self.tool_name.trim().is_empty() || self.call_id.trim().is_empty() {
            return Err(CommsError::Invalid(
                "tool result requires tool_name and call_id".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evidence_graph_state_validates() {
        let g = EvidenceGraph {
            nodes: vec![
                EvidenceNode {
                    id: "c1".into(),
                    kind: EvidenceNodeKind::Claim,
                    label: "tok/s is 42".into(),
                    content_hash: None,
                    attrs: BTreeMap::new(),
                },
                EvidenceNode {
                    id: "e1".into(),
                    kind: EvidenceNodeKind::Evidence,
                    label: "table row".into(),
                    content_hash: Some("blake3:abc".into()),
                    attrs: BTreeMap::new(),
                },
            ],
            edges: vec![EvidenceEdge {
                from: "e1".into(),
                to: "c1".into(),
                relation: EvidenceRelation::Supports,
            }],
        };
        let state =
            StructuredState::from_payload("ses_1", "agent_a", StructuredPayload::EvidenceGraph(g));
        state.validate().unwrap();
    }
}
