//! The knowledge graph (bible ch.08 §4.2, §4.8).
//!
//! A real property graph backed by [`petgraph`] (a `StableDiGraph` so node
//! indices survive deletions during entity resolution merges). On top of the
//! graph we provide:
//!
//! * **Content-addressed ids** (§4.2.1): Paper/Claim/Concept ids are derived
//!   from normalized content via [`crate::cas`], so re-ingesting identical bytes
//!   is idempotent.
//! * **Entity resolution** (§4.8): incoming nodes merge into existing ones by
//!   content-hash id collision *and* by normalized-name match within a kind,
//!   folding provenance rather than duplicating.
//! * **Query modes** (§4.8): `Local` (neighborhood expansion from seed nodes),
//!   `Global` (kind-filtered ranked listing — the map-reduce entry point), and
//!   `Path` (shortest typed path between two nodes).
//! * **Persistence** (§4.3): the whole graph round-trips to a single JSONL file
//!   so it survives process exit without a server.

use crate::cas;
use crate::ingest::StructuredDoc;
use hide_core::error::Result;
use hide_core::ids::{now_ms, TimestampMs};
use hide_core::types::Provenance;
use parking_lot::RwLock;
use petgraph::stable_graph::{NodeIndex, StableDiGraph};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeNode {
    pub id: String,
    pub kind: NodeKind,
    pub label: String,
    pub confidence: ConfidenceTier,
    pub provenance: Vec<ProvenanceSpan>,
    pub created_at_ms: TimestampMs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    SameAs,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProvenanceSpan {
    pub doc_id: String,
    pub span_id: Option<String>,
    pub char_range: Option<(usize, usize)>,
    pub citation: Option<String>,
    /// Immutable blake3 receipt of the evidence bytes (CAS), when pinned. This is
    /// what makes a citation re-verifiable (§4.7.3).
    #[serde(default)]
    pub content_hash: Option<String>,
    /// CAS blob ref for the evidence bytes, when pinned.
    #[serde(default)]
    pub evidence_blob: Option<hide_core::types::BlobRef>,
    pub provenance: Provenance,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Claim {
    pub id: String,
    pub text: String,
    pub provenance: ProvenanceSpan,
    pub confidence: ConfidenceTier,
}

/// The query surface (§4.8). Local/Global/Path are the three primitives every
/// GraphRAG-style retrieval composes from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GraphQuery {
    /// Expand `hops` outward from each seed node; return the reachable set.
    Local { seeds: Vec<String>, hops: usize },
    /// All nodes of a kind, most-recent first (the global map-reduce entry).
    Global { kind: NodeKind, limit: usize },
    /// Shortest directed path (by edge count) between two node ids.
    Path { from: String, to: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueryResult {
    pub nodes: Vec<KnowledgeNode>,
    pub edges: Vec<KnowledgeEdge>,
}

pub trait KnowledgeGraph: Send + Sync {
    fn add_node(&self, node: KnowledgeNode);
    fn add_edge(&self, edge: KnowledgeEdge);
    fn nodes_by_kind(&self, kind: NodeKind) -> Vec<KnowledgeNode>;
    fn edges_from(&self, node_id: &str) -> Vec<KnowledgeEdge>;
    fn query(&self, q: &GraphQuery) -> QueryResult;
}

/// petgraph-backed store. `StableDiGraph` keeps `NodeIndex` valid across the
/// node removals that entity-resolution merges perform.
pub struct PetKnowledgeGraph {
    inner: RwLock<GraphInner>,
}

struct GraphInner {
    graph: StableDiGraph<KnowledgeNode, KnowledgeEdge>,
    /// node id → index.
    index: HashMap<String, NodeIndex>,
    /// normalized (kind, name) → canonical node id, for name-based resolution.
    name_index: HashMap<(NodeKind, String), String>,
    /// edge id set, to dedup edges.
    edge_ids: std::collections::HashSet<String>,
}

impl Default for PetKnowledgeGraph {
    fn default() -> Self {
        Self {
            inner: RwLock::new(GraphInner {
                graph: StableDiGraph::new(),
                index: HashMap::new(),
                name_index: HashMap::new(),
                edge_ids: std::collections::HashSet::new(),
            }),
        }
    }
}

fn norm_name(label: &str) -> String {
    cas::normalize_text(label).to_lowercase()
}

impl PetKnowledgeGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert-or-merge a node by entity resolution (§4.8). A node merges into an
    /// existing one if (a) the id collides (content-addressed duplicate), or
    /// (b) the normalized (kind, name) already maps to a canonical id. On merge,
    /// provenance is unioned and the strongest confidence tier kept.
    pub fn upsert_node(&self, mut node: KnowledgeNode) -> String {
        let mut inner = self.inner.write();
        let name_key = (node.kind, norm_name(&node.label));

        // Resolve a canonical id: prefer an exact id collision, else a name match.
        let canonical = if inner.index.contains_key(&node.id) {
            Some(node.id.clone())
        } else {
            inner.name_index.get(&name_key).cloned()
        };

        if let Some(canon_id) = canonical {
            if let Some(&idx) = inner.index.get(&canon_id) {
                if let Some(existing) = inner.graph.node_weight_mut(idx) {
                    // Union provenance.
                    for span in node.provenance.drain(..) {
                        if !existing.provenance.iter().any(|s| {
                            s.doc_id == span.doc_id
                                && s.span_id == span.span_id
                                && s.content_hash == span.content_hash
                        }) {
                            existing.provenance.push(span);
                        }
                    }
                    // Keep the strongest (lowest-ordinal) confidence tier.
                    if tier_rank(node.confidence) < tier_rank(existing.confidence) {
                        existing.confidence = node.confidence;
                    }
                }
                inner.name_index.entry(name_key).or_insert(canon_id.clone());
                return canon_id;
            }
        }

        // Fresh node.
        let id = node.id.clone();
        let idx = inner.graph.add_node(node);
        inner.index.insert(id.clone(), idx);
        inner.name_index.entry(name_key).or_insert(id.clone());
        id
    }

    pub fn upsert_edge(&self, edge: KnowledgeEdge) {
        let mut inner = self.inner.write();
        if inner.edge_ids.contains(&edge.id) {
            return;
        }
        let (Some(&from), Some(&to)) =
            (inner.index.get(&edge.from), inner.index.get(&edge.to))
        else {
            return; // endpoints must exist first
        };
        inner.edge_ids.insert(edge.id.clone());
        inner.graph.add_edge(from, to, edge);
    }

    pub fn node(&self, id: &str) -> Option<KnowledgeNode> {
        let inner = self.inner.read();
        inner
            .index
            .get(id)
            .and_then(|&idx| inner.graph.node_weight(idx).cloned())
    }

    pub fn node_count(&self) -> usize {
        self.inner.read().graph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.inner.read().graph.edge_count()
    }

    /// Ingest a parsed document: mint a content-addressed Paper node plus one
    /// content-addressed Claim node per non-empty section, with `MENTIONS`
    /// edges. Returns the claims (carrying their provenance spans). Idempotent:
    /// re-ingesting the same doc merges rather than duplicates.
    pub fn ingest_doc(&self, doc: &StructuredDoc) -> Vec<Claim> {
        let paper_id = cas::composite_id("paper", &[&doc.title, &doc.id]);
        let base_span = ProvenanceSpan {
            doc_id: doc.id.clone(),
            span_id: None,
            char_range: None,
            citation: None,
            content_hash: doc.source.content_hash.clone(),
            evidence_blob: doc.blob.clone(),
            provenance: doc.source.provenance.clone(),
        };
        self.upsert_node(KnowledgeNode {
            id: paper_id.clone(),
            kind: NodeKind::Paper,
            label: doc.title.clone(),
            confidence: ConfidenceTier::Extracted,
            provenance: vec![base_span.clone()],
            created_at_ms: now_ms(),
        });

        let mut claims = Vec::new();
        for section in &doc.sections {
            if section.text.trim().is_empty() {
                continue;
            }
            let claim_id = cas::composite_id("claim", &[&section.text, &doc.id]);
            let span = ProvenanceSpan {
                doc_id: doc.id.clone(),
                span_id: section.spans.first().map(|s| s.id.clone()),
                char_range: section
                    .spans
                    .first()
                    .map(|s| (s.start_char, s.end_char)),
                citation: None,
                content_hash: doc.source.content_hash.clone(),
                evidence_blob: doc.blob.clone(),
                provenance: doc.source.provenance.clone(),
            };
            self.upsert_node(KnowledgeNode {
                id: claim_id.clone(),
                kind: NodeKind::Claim,
                label: section.text.chars().take(160).collect(),
                confidence: ConfidenceTier::Extracted,
                provenance: vec![span.clone()],
                created_at_ms: now_ms(),
            });
            self.upsert_edge(KnowledgeEdge {
                id: cas::composite_id("edge", &[&paper_id, &claim_id, "mentions"]),
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

    /// Serialize the entire graph to a JSONL file (one node/edge per line),
    /// surviving process exit (§4.3).
    pub fn save_jsonl(&self, path: &std::path::Path) -> Result<()> {
        use std::io::Write;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let inner = self.inner.read();
        let mut file = std::fs::File::create(path)?;
        for node in inner.graph.node_weights() {
            let line = serde_json::to_string(&GraphLine::Node(node.clone()))?;
            writeln!(file, "{line}")?;
        }
        for edge in inner.graph.edge_weights() {
            let line = serde_json::to_string(&GraphLine::Edge(edge.clone()))?;
            writeln!(file, "{line}")?;
        }
        file.sync_data()?;
        Ok(())
    }

    /// Load a graph from a JSONL file produced by [`Self::save_jsonl`]. Nodes are
    /// loaded first (in file order) so edges always find their endpoints.
    pub fn load_jsonl(path: &std::path::Path) -> Result<Self> {
        use std::io::BufRead;
        let graph = Self::new();
        if !path.exists() {
            return Ok(graph);
        }
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let mut edges = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<GraphLine>(&line)? {
                GraphLine::Node(n) => {
                    // Insert verbatim (preserve ids) — bypass name-merge so a
                    // saved graph reloads byte-faithfully.
                    let mut inner = graph.inner.write();
                    let name_key = (n.kind, norm_name(&n.label));
                    let id = n.id.clone();
                    let idx = inner.graph.add_node(n);
                    inner.index.insert(id.clone(), idx);
                    inner.name_index.entry(name_key).or_insert(id);
                }
                GraphLine::Edge(e) => edges.push(e),
            }
        }
        for e in edges {
            graph.upsert_edge(e);
        }
        Ok(graph)
    }

    fn local(&self, seeds: &[String], hops: usize) -> QueryResult {
        let inner = self.inner.read();
        let mut visited = std::collections::HashSet::new();
        let mut queue: VecDeque<(NodeIndex, usize)> = VecDeque::new();
        for s in seeds {
            if let Some(&idx) = inner.index.get(s) {
                if visited.insert(idx) {
                    queue.push_back((idx, 0));
                }
            }
        }
        let mut edges = Vec::new();
        while let Some((idx, depth)) = queue.pop_front() {
            if depth >= hops {
                continue;
            }
            for e in inner
                .graph
                .edges_directed(idx, Direction::Outgoing)
                .chain(inner.graph.edges_directed(idx, Direction::Incoming))
            {
                edges.push(e.weight().clone());
                let nbr = if e.source() == idx { e.target() } else { e.source() };
                if visited.insert(nbr) {
                    queue.push_back((nbr, depth + 1));
                }
            }
        }
        let nodes = visited
            .iter()
            .filter_map(|&idx| inner.graph.node_weight(idx).cloned())
            .collect();
        dedup_result(QueryResult { nodes, edges })
    }

    fn global(&self, kind: NodeKind, limit: usize) -> QueryResult {
        let mut nodes: Vec<KnowledgeNode> = self.nodes_by_kind(kind);
        nodes.sort_by(|a, b| b.created_at_ms.cmp(&a.created_at_ms));
        nodes.truncate(limit);
        QueryResult {
            nodes,
            edges: Vec::new(),
        }
    }

    fn path(&self, from: &str, to: &str) -> QueryResult {
        let inner = self.inner.read();
        let (Some(&src), Some(&dst)) = (inner.index.get(from), inner.index.get(to)) else {
            return QueryResult {
                nodes: Vec::new(),
                edges: Vec::new(),
            };
        };
        // BFS shortest path by edge count, following edges as undirected for
        // reachability but recording the typed edge traversed.
        let mut prev: HashMap<NodeIndex, (NodeIndex, KnowledgeEdge)> = HashMap::new();
        let mut visited = std::collections::HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(src);
        queue.push_back(src);
        while let Some(idx) = queue.pop_front() {
            if idx == dst {
                break;
            }
            for e in inner
                .graph
                .edges_directed(idx, Direction::Outgoing)
                .chain(inner.graph.edges_directed(idx, Direction::Incoming))
            {
                let nbr = if e.source() == idx { e.target() } else { e.source() };
                if visited.insert(nbr) {
                    prev.insert(nbr, (idx, e.weight().clone()));
                    queue.push_back(nbr);
                }
            }
        }
        if !visited.contains(&dst) {
            return QueryResult {
                nodes: Vec::new(),
                edges: Vec::new(),
            };
        }
        // Walk back from dst to src.
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut cur = dst;
        nodes.push(inner.graph.node_weight(cur).cloned().unwrap());
        while cur != src {
            let (p, e) = prev.get(&cur).cloned().unwrap();
            edges.push(e);
            nodes.push(inner.graph.node_weight(p).cloned().unwrap());
            cur = p;
        }
        nodes.reverse();
        edges.reverse();
        QueryResult { nodes, edges }
    }
}

fn tier_rank(t: ConfidenceTier) -> u8 {
    match t {
        ConfidenceTier::Measured => 0,
        ConfidenceTier::Extracted => 1,
        ConfidenceTier::Inferred => 2,
        ConfidenceTier::Speculative => 3,
    }
}

fn dedup_result(mut r: QueryResult) -> QueryResult {
    let mut seen_e = std::collections::HashSet::new();
    r.edges.retain(|e| seen_e.insert(e.id.clone()));
    r
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "rec")]
enum GraphLine {
    Node(KnowledgeNode),
    Edge(KnowledgeEdge),
}

impl KnowledgeGraph for PetKnowledgeGraph {
    fn add_node(&self, node: KnowledgeNode) {
        self.upsert_node(node);
    }

    fn add_edge(&self, edge: KnowledgeEdge) {
        self.upsert_edge(edge);
    }

    fn nodes_by_kind(&self, kind: NodeKind) -> Vec<KnowledgeNode> {
        let inner = self.inner.read();
        inner
            .graph
            .node_weights()
            .filter(|n| n.kind == kind)
            .cloned()
            .collect()
    }

    fn edges_from(&self, node_id: &str) -> Vec<KnowledgeEdge> {
        let inner = self.inner.read();
        let Some(&idx) = inner.index.get(node_id) else {
            return Vec::new();
        };
        inner
            .graph
            .edges_directed(idx, Direction::Outgoing)
            .map(|e| e.weight().clone())
            .collect()
    }

    fn query(&self, q: &GraphQuery) -> QueryResult {
        match q {
            GraphQuery::Local { seeds, hops } => self.local(seeds, *hops),
            GraphQuery::Global { kind, limit } => self.global(*kind, *limit),
            GraphQuery::Path { from, to } => self.path(from, to),
        }
    }
}

/// Back-compat alias: the prior in-memory graph type name. Tests and downstream
/// callers that used `InMemoryKnowledgeGraph` keep working; it is now the real
/// petgraph store.
pub type InMemoryKnowledgeGraph = PetKnowledgeGraph;

impl PetKnowledgeGraph {
    /// Back-compat shim for the prior `ingest_doc_shell` name.
    pub fn ingest_doc_shell(&self, doc: &StructuredDoc) -> Vec<Claim> {
        self.ingest_doc(doc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::{DocSection, DocSpan, SourceQuality, SourceRecord, SourceType};

    fn doc(id: &str, title: &str, body: &str) -> StructuredDoc {
        StructuredDoc {
            id: id.to_string(),
            source: SourceRecord {
                id: id.to_string(),
                source_type: SourceType::PdfLocal,
                title: title.to_string(),
                uri: format!("memory://{id}"),
                content_hash: Some("hash".to_string()),
                quality: SourceQuality::default(),
                provenance: Provenance::trusted("test"),
            },
            title: title.to_string(),
            sections: vec![DocSection {
                heading: "Abstract".to_string(),
                text: body.to_string(),
                spans: vec![DocSpan {
                    id: "s0".to_string(),
                    start_char: 0,
                    end_char: body.len(),
                }],
            }],
            references: Vec::new(),
            blob: None,
        }
    }

    #[test]
    fn ingest_is_idempotent_via_content_addressing() {
        let g = PetKnowledgeGraph::new();
        let d = doc("doc1", "Paged Attention", "Paged attention improves reuse.");
        let c1 = g.ingest_doc(&d);
        let n1 = g.node_count();
        let c2 = g.ingest_doc(&d);
        // Same ids the second time; node count unchanged.
        assert_eq!(c1[0].id, c2[0].id);
        assert_eq!(g.node_count(), n1);
    }

    #[test]
    fn entity_resolution_merges_by_name() {
        let g = PetKnowledgeGraph::new();
        g.upsert_node(KnowledgeNode {
            id: "concept:a".into(),
            kind: NodeKind::Concept,
            label: "KV Cache".into(),
            confidence: ConfidenceTier::Inferred,
            provenance: vec![],
            created_at_ms: 1,
        });
        // Different id, same normalized name+kind → merges.
        g.upsert_node(KnowledgeNode {
            id: "concept:b".into(),
            kind: NodeKind::Concept,
            label: "kv   cache".into(),
            confidence: ConfidenceTier::Measured,
            provenance: vec![],
            created_at_ms: 2,
        });
        assert_eq!(g.nodes_by_kind(NodeKind::Concept).len(), 1);
        // Stronger tier kept.
        assert_eq!(
            g.node("concept:a").unwrap().confidence,
            ConfidenceTier::Measured
        );
    }

    #[test]
    fn local_global_path_queries() {
        let g = PetKnowledgeGraph::new();
        g.ingest_doc(&doc("d1", "A", "alpha claim text here"));
        let paper = g.nodes_by_kind(NodeKind::Paper)[0].id.clone();
        let claim = g.nodes_by_kind(NodeKind::Claim)[0].id.clone();

        let local = g.query(&GraphQuery::Local {
            seeds: vec![paper.clone()],
            hops: 1,
        });
        assert!(local.nodes.iter().any(|n| n.id == claim));

        let global = g.query(&GraphQuery::Global {
            kind: NodeKind::Paper,
            limit: 10,
        });
        assert_eq!(global.nodes.len(), 1);

        let path = g.query(&GraphQuery::Path {
            from: paper.clone(),
            to: claim.clone(),
        });
        assert_eq!(path.nodes.len(), 2);
        assert_eq!(path.edges.len(), 1);
    }

    #[test]
    fn graph_persists_to_jsonl_and_reloads() {
        let g = PetKnowledgeGraph::new();
        g.ingest_doc(&doc("d1", "Persist Me", "a durable claim"));
        let (nc, ec) = (g.node_count(), g.edge_count());
        let dir = std::env::temp_dir().join(format!("hawking_kg_{}", now_ms()));
        let path = dir.join("graph.jsonl");
        g.save_jsonl(&path).unwrap();
        let reloaded = PetKnowledgeGraph::load_jsonl(&path).unwrap();
        assert_eq!(reloaded.node_count(), nc);
        assert_eq!(reloaded.edge_count(), ec);
        let _ = std::fs::remove_dir_all(dir);
    }
}
