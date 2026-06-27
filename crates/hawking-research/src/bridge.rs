use crate::kg::{Claim, KnowledgeNode};
use hawking_context::memory::{MemoryKind, MemoryRecord};
use hawking_index::graph::Symbol;
use hide_core::types::Provenance;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FindingIssue {
    pub title: String,
    pub body: String,
    pub claim_ids: Vec<String>,
    pub suggested_labels: Vec<String>,
}

pub fn claim_to_issue(claim: &Claim) -> FindingIssue {
    FindingIssue {
        title: claim.text.chars().take(72).collect(),
        body: format!("Research claim:\n\n{}\n\nSource: {}", claim.text, claim.id),
        claim_ids: vec![claim.id.clone()],
        suggested_labels: vec!["research".to_string()],
    }
}

pub fn node_to_memory(node: &KnowledgeNode, provenance: Provenance) -> MemoryRecord {
    MemoryRecord {
        id: format!("kg:{}", node.id),
        kind: MemoryKind::Semantic,
        text: node.label.clone(),
        importance: 0.7,
        created_at_ms: node.created_at_ms,
        last_used_at_ms: None,
        provenance,
        tags: vec![format!("{:?}", node.kind)],
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodeResearchLink {
    pub claim_id: String,
    pub symbol: Symbol,
    pub relation: String,
}
