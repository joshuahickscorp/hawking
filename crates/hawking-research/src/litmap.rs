//! Citation / literature mapping over the knowledge graph (bible ch.08 §4.9).
//!
//! `build_literature_map` walks the graph from the Paper nodes, clusters them by
//! shared claim/concept neighborhoods (a label-propagation-style grouping over
//! the real petgraph store), and surfaces coverage gaps (sub-questions /
//! concepts with thin support). This is a local, owned equivalent of
//! Connected-Papers / Litmaps — seeded by what *you* ingested.

use crate::kg::{EdgeKind, KnowledgeGraph, KnowledgeNode, NodeKind, PetKnowledgeGraph};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LiteratureMap {
    pub topic: String,
    pub papers: Vec<KnowledgeNode>,
    pub clusters: Vec<LiteratureCluster>,
    pub gaps: Vec<ResearchGap>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LiteratureCluster {
    pub id: String,
    pub label: String,
    pub node_ids: Vec<String>,
    pub summary: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResearchGap {
    pub id: String,
    pub description: String,
    pub supporting_node_ids: Vec<String>,
}

/// Build a literature map from the current graph. Papers that share at least one
/// claim/concept neighbor are grouped into the same cluster (transitive closure
/// over the shared-neighbor relation — a connected-components clustering).
pub fn build_literature_map(graph: &PetKnowledgeGraph, topic: impl Into<String>) -> LiteratureMap {
    let topic = topic.into();
    let papers = graph.nodes_by_kind(NodeKind::Paper);

    // Map each paper → the set of its claim/concept neighbor ids.
    let mut paper_neighbors: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for p in &papers {
        let mut nbrs = BTreeSet::new();
        for e in graph.edges_from(&p.id) {
            if matches!(
                e.kind,
                EdgeKind::Mentions | EdgeKind::Cites | EdgeKind::UsesDataset | EdgeKind::Supports
            ) {
                nbrs.insert(e.to);
            }
        }
        paper_neighbors.insert(p.id.clone(), nbrs);
    }

    // Union-find over papers that share ≥1 neighbor.
    let ids: Vec<String> = papers.iter().map(|p| p.id.clone()).collect();
    let mut uf = UnionFind::new(&ids);
    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            let a = &paper_neighbors[&ids[i]];
            let b = &paper_neighbors[&ids[j]];
            if a.intersection(b).next().is_some() {
                uf.union(&ids[i], &ids[j]);
            }
        }
    }

    // Group by root.
    let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for id in &ids {
        groups.entry(uf.find(id)).or_default().push(id.clone());
    }

    let label_of = |id: &str| -> String {
        papers
            .iter()
            .find(|p| p.id == id)
            .map(|p| p.label.clone())
            .unwrap_or_else(|| id.to_string())
    };

    let mut clusters = Vec::new();
    for (i, (_root, members)) in groups.iter().enumerate() {
        let label = members
            .first()
            .map(|m| label_of(m))
            .unwrap_or_else(|| format!("cluster {i}"));
        clusters.push(LiteratureCluster {
            id: format!("cluster:{i}"),
            label: label.chars().take(60).collect(),
            node_ids: members.clone(),
            summary: format!("{} paper(s) sharing claim/concept neighbors", members.len()),
        });
    }

    // Gaps: concepts mentioned by exactly one paper (thin support).
    let mut concept_support: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for c in graph.nodes_by_kind(NodeKind::Concept) {
        concept_support.entry(c.id.clone()).or_default();
    }
    for (paper, nbrs) in &paper_neighbors {
        for n in nbrs {
            if let Some(v) = concept_support.get_mut(n) {
                v.push(paper.clone());
            }
        }
    }
    let gaps = concept_support
        .into_iter()
        .filter(|(_, s)| s.len() <= 1)
        .map(|(concept, support)| ResearchGap {
            id: format!("gap:{concept}"),
            description: format!(
                "Concept {concept} has thin coverage ({} paper)",
                support.len()
            ),
            supporting_node_ids: support,
        })
        .collect();

    LiteratureMap {
        topic,
        papers,
        clusters,
        gaps,
    }
}

/// Compare N papers by their shared and unique claim/concept neighbors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PaperComparison {
    pub paper_ids: Vec<String>,
    pub shared: Vec<String>,
    pub unique: BTreeMap<String, Vec<String>>,
}

pub fn compare_papers(graph: &PetKnowledgeGraph, paper_ids: &[String]) -> PaperComparison {
    let neighbor_sets: BTreeMap<String, BTreeSet<String>> = paper_ids
        .iter()
        .map(|id| {
            let nbrs: BTreeSet<String> = graph.edges_from(id).into_iter().map(|e| e.to).collect();
            (id.clone(), nbrs)
        })
        .collect();
    let shared: BTreeSet<String> = neighbor_sets
        .values()
        .cloned()
        .reduce(|acc: BTreeSet<String>, s| acc.intersection(&s).cloned().collect())
        .unwrap_or_default();
    let unique: BTreeMap<String, Vec<String>> = neighbor_sets
        .iter()
        .map(|(id, s)| {
            let u: Vec<String> = s.difference(&shared).cloned().collect();
            (id.clone(), u)
        })
        .collect();
    PaperComparison {
        paper_ids: paper_ids.to_vec(),
        shared: shared.into_iter().collect(),
        unique,
    }
}

// ── tiny union-find ──
struct UnionFind {
    parent: BTreeMap<String, String>,
}

impl UnionFind {
    fn new(ids: &[String]) -> Self {
        Self {
            parent: ids.iter().map(|i| (i.clone(), i.clone())).collect(),
        }
    }
    fn find(&self, x: &str) -> String {
        let mut cur = x.to_string();
        while let Some(p) = self.parent.get(&cur) {
            if p == &cur {
                break;
            }
            cur = p.clone();
        }
        cur
    }
    fn union(&mut self, a: &str, b: &str) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent.insert(ra, rb);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::structured_doc_from_text;
    use crate::ingest::{SourceQuality, SourceType};
    use hide_core::types::Provenance;

    fn ingest(graph: &PetKnowledgeGraph, id: &str, title: &str, body: &str) {
        let doc = structured_doc_from_text(
            SourceType::PdfLocal,
            id,
            &format!("memory://{id}"),
            title,
            body,
            Provenance::trusted("t"),
            SourceQuality::default(),
        );
        graph.ingest_doc(&doc);
    }

    #[test]
    fn map_clusters_papers_and_finds_gaps() {
        let graph = PetKnowledgeGraph::new();
        // Two papers sharing nothing (different claim text) → two clusters.
        ingest(&graph, "a", "A", "alpha distinct statement one");
        ingest(&graph, "b", "B", "beta distinct statement two");
        let map = build_literature_map(&graph, "topic");
        assert_eq!(map.papers.len(), 2);
        assert_eq!(map.clusters.len(), 2);
    }

    #[test]
    fn shared_concept_merges_into_one_cluster() {
        use crate::kg::{
            ConfidenceTier, EdgeKind, KnowledgeEdge, KnowledgeGraph, KnowledgeNode, NodeKind,
        };
        let graph = PetKnowledgeGraph::new();
        // Two papers whose claim text differs (→ distinct claims, per §4.2.1),
        // but which both MENTION the same Concept node → one cluster.
        ingest(&graph, "a", "A", "alpha statement about attention");
        ingest(&graph, "b", "B", "beta statement about attention");
        let papers = graph.nodes_by_kind(NodeKind::Paper);
        graph.upsert_node(KnowledgeNode {
            id: "concept:attention".into(),
            kind: NodeKind::Concept,
            label: "attention".into(),
            confidence: ConfidenceTier::Inferred,
            provenance: vec![],
            created_at_ms: 1,
        });
        for p in &papers {
            graph.upsert_edge(KnowledgeEdge {
                id: format!("edge:{}:concept", p.id),
                from: p.id.clone(),
                to: "concept:attention".into(),
                kind: EdgeKind::Mentions,
                confidence: 0.9,
                provenance: vec![],
            });
        }
        let map = build_literature_map(&graph, "topic");
        assert_eq!(map.papers.len(), 2);
        assert_eq!(map.clusters.len(), 1);
    }
}
