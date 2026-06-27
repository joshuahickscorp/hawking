use crate::ingest::StructuredDoc;
use hide_core::ids::{now_ms, TimestampMs};
use hide_core::types::Provenance;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeNode {
    pub id: String,
    pub kind: NodeKind,
    pub label: String,
    pub confidence: ConfidenceTier,
    pub provenance: Vec<ProvenanceSpan>,
    pub created_at_ms: TimestampMs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    Paper,
    Author,
    Venue,
    Claim,
    Method,
    Dataset,
    Metric,
    Equation,
    CodeSymbol,
    Experiment,
    Issue,
    Concept,
    Note,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfidenceTier {
    Measured,
    Extracted,
    Inferred,
    Speculative,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeEdge {
    pub id: String,
    pub from: String,
    pub to: String,
    pub kind: EdgeKind,
    pub confidence: f32,
    pub provenance: Vec<ProvenanceSpan>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    Supports,
    Refutes,
    Mentions,
    Cites,
    Implements,
    DerivedFrom,
    UsesDataset,
    ReportsMetric,
    Contradicts,
    Related,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProvenanceSpan {
    pub doc_id: String,
    pub span_id: Option<String>,
    pub char_range: Option<(usize, usize)>,
    pub citation: Option<String>,
    pub provenance: Provenance,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Claim {
    pub id: String,
    pub text: String,
    pub provenance: ProvenanceSpan,
    pub confidence: ConfidenceTier,
}

pub trait KnowledgeGraph: Send + Sync {
    fn add_node(&self, node: KnowledgeNode);
    fn add_edge(&self, edge: KnowledgeEdge);
    fn nodes_by_kind(&self, kind: NodeKind) -> Vec<KnowledgeNode>;
    fn edges_from(&self, node_id: &str) -> Vec<KnowledgeEdge>;
}

#[derive(Default)]
pub struct InMemoryKnowledgeGraph {
    nodes: RwLock<BTreeMap<String, KnowledgeNode>>,
    edges: RwLock<BTreeMap<String, KnowledgeEdge>>,
}

impl InMemoryKnowledgeGraph {
    pub fn ingest_doc_shell(&self, doc: &StructuredDoc) -> Vec<Claim> {
        let paper_id = format!("paper:{}", doc.id);
        self.add_node(KnowledgeNode {
            id: paper_id.clone(),
            kind: NodeKind::Paper,
            label: doc.title.clone(),
            confidence: ConfidenceTier::Extracted,
            provenance: vec![ProvenanceSpan {
                doc_id: doc.id.clone(),
                span_id: None,
                char_range: None,
                citation: None,
                provenance: doc.source.provenance.clone(),
            }],
            created_at_ms: now_ms(),
        });
        let mut claims = Vec::new();
        for (idx, section) in doc.sections.iter().enumerate() {
            if section.text.trim().is_empty() {
                continue;
            }
            let claim_id = format!("claim:{}:{idx}", doc.id);
            let span = ProvenanceSpan {
                doc_id: doc.id.clone(),
                span_id: section.spans.first().map(|s| s.id.clone()),
                char_range: None,
                citation: None,
                provenance: doc.source.provenance.clone(),
            };
            self.add_node(KnowledgeNode {
                id: claim_id.clone(),
                kind: NodeKind::Claim,
                label: section.text.chars().take(160).collect(),
                confidence: ConfidenceTier::Extracted,
                provenance: vec![span.clone()],
                created_at_ms: now_ms(),
            });
            self.add_edge(KnowledgeEdge {
                id: format!("edge:{paper_id}:{claim_id}"),
                from: paper_id.clone(),
                to: claim_id.clone(),
                kind: EdgeKind::Mentions,
                confidence: 0.8,
                provenance: vec![span.clone()],
            });
            claims.push(Claim {
                id: claim_id,
                text: section.text.clone(),
                provenance: span,
                confidence: ConfidenceTier::Extracted,
            });
        }
        claims
    }
}

impl KnowledgeGraph for InMemoryKnowledgeGraph {
    fn add_node(&self, node: KnowledgeNode) {
        self.nodes.write().insert(node.id.clone(), node);
    }

    fn add_edge(&self, edge: KnowledgeEdge) {
        self.edges.write().insert(edge.id.clone(), edge);
    }

    fn nodes_by_kind(&self, kind: NodeKind) -> Vec<KnowledgeNode> {
        self.nodes
            .read()
            .values()
            .filter(|n| n.kind == kind)
            .cloned()
            .collect()
    }

    fn edges_from(&self, node_id: &str) -> Vec<KnowledgeEdge> {
        self.edges
            .read()
            .values()
            .filter(|e| e.from == node_id)
            .cloned()
            .collect()
    }
}
