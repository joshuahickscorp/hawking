//! The unified graph model + a real petgraph-backed call/import graph and a
//! personalized-PageRank repo-map (bible §4.4, §4.6).
//!
//! The DTOs (`Symbol`/`Occurrence`/`GraphEdge`/`EdgeKind`/`RepoMap*`) are the
//! cross-crate vocabulary and are preserved. `CodeGraph` is the new engine: it
//! loads edges into petgraph and runs PageRank to rank definitions, rendering a
//! token-budgeted elided signatures-only tree for the Context Compiler (ch.04).

use hide_core::types::TextRange;
use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Symbol {
    pub qualified_name: String,
    pub name: String,
    pub kind: String,
    pub file: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Occurrence {
    pub symbol: String,
    pub file: String,
    pub range: Option<TextRange>,
    pub role: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphEdge {
    pub from: String,
    pub to: String,
    pub kind: EdgeKind,
    pub weight_millis: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    Defines,
    References,
    Calls,
    Imports,
    Implements,
    Tests,
    Dataflow,
    Performance,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepoMapRequest {
    pub mentioned_files: Vec<String>,
    pub mentioned_idents: Vec<String>,
    pub max_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepoMap {
    pub rendered: String,
    pub symbols: Vec<Symbol>,
    pub estimated_tokens: usize,
}

/// A real call/import/reference graph over symbols, loaded into petgraph.
///
/// Nodes are symbol ids; edges carry an `EdgeKind` and a weight. PageRank ranks
/// nodes; `repo_map` distributes rank to definitions and renders an elided tree.
pub struct CodeGraph {
    graph: DiGraph<String, EdgeAttr>,
    index: HashMap<String, NodeIndex>,
    /// symbol id → its defining file + signature line (for rendering).
    defs: HashMap<String, RankedDef>,
}

#[derive(Debug, Clone)]
struct EdgeAttr {
    kind: EdgeKind,
    weight: f32,
}

#[derive(Debug, Clone)]
struct RankedDef {
    name: String,
    file: String,
    signature: String,
}

impl Default for CodeGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            index: HashMap::new(),
            defs: HashMap::new(),
        }
    }

    fn node(&mut self, id: &str) -> NodeIndex {
        if let Some(ix) = self.index.get(id) {
            return *ix;
        }
        let ix = self.graph.add_node(id.to_string());
        self.index.insert(id.to_string(), ix);
        ix
    }

    /// Register a definition so it can appear in the repo-map render.
    pub fn add_definition(&mut self, symbol_id: &str, name: &str, file: &str, signature: &str) {
        self.node(symbol_id);
        self.defs.insert(
            symbol_id.to_string(),
            RankedDef {
                name: name.to_string(),
                file: file.to_string(),
                signature: signature.to_string(),
            },
        );
    }

    pub fn add_edge(&mut self, from: &str, to: &str, kind: EdgeKind, weight: f32) {
        let a = self.node(from);
        let b = self.node(to);
        self.graph.add_edge(a, b, EdgeAttr { kind, weight });
    }

    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Out-neighbors of `id` restricted to one edge kind (e.g. direct callees).
    pub fn neighbors_by_kind(&self, id: &str, kind: EdgeKind) -> Vec<String> {
        let Some(&src) = self.index.get(id) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for e in self.graph.edges(src) {
            use petgraph::visit::EdgeRef;
            if e.weight().kind == kind {
                out.push(self.graph[e.target()].clone());
            }
        }
        out
    }

    /// Personalized PageRank over the graph (power iteration, alpha=0.85).
    ///
    /// `personalization` maps a node id to its teleport mass; missing nodes get
    /// uniform mass. Converges fast on sparse code graphs.
    pub fn pagerank(&self, personalization: &HashMap<String, f32>, iters: usize) -> HashMap<String, f32> {
        let n = self.graph.node_count();
        if n == 0 {
            return HashMap::new();
        }
        let alpha = 0.85f32;

        // Build personalization vector over node indices.
        let mut p: Vec<f32> = vec![0.0; n];
        let mut total_pers = 0.0f32;
        for (id, mass) in personalization {
            if let Some(ix) = self.index.get(id) {
                p[ix.index()] += *mass;
                total_pers += *mass;
            }
        }
        if total_pers <= 0.0 {
            // uniform teleport
            for x in p.iter_mut() {
                *x = 1.0 / n as f32;
            }
        } else {
            for x in p.iter_mut() {
                *x /= total_pers;
            }
        }

        // Out-weight sums per node (for weighted distribution).
        let mut out_sum: Vec<f32> = vec![0.0; n];
        for e in self.graph.edge_indices() {
            let (a, _b) = self.graph.edge_endpoints(e).unwrap();
            out_sum[a.index()] += self.graph[e].weight.max(0.0);
        }

        let mut rank: Vec<f32> = vec![1.0 / n as f32; n];
        for _ in 0..iters.max(1) {
            let mut next: Vec<f32> = vec![0.0; n];
            // dangling mass (nodes with no out-edges) redistributed via teleport.
            let mut dangling = 0.0f32;
            for ix in 0..n {
                if out_sum[ix] <= 0.0 {
                    dangling += rank[ix];
                }
            }
            for e in self.graph.edge_indices() {
                let (a, b) = self.graph.edge_endpoints(e).unwrap();
                let w = self.graph[e].weight.max(0.0);
                if out_sum[a.index()] > 0.0 {
                    next[b.index()] += alpha * rank[a.index()] * (w / out_sum[a.index()]);
                }
            }
            for ix in 0..n {
                next[ix] += alpha * dangling * p[ix];
                next[ix] += (1.0 - alpha) * p[ix];
            }
            rank = next;
        }

        self.index
            .iter()
            .map(|(id, ix)| (id.clone(), rank[ix.index()]))
            .collect()
    }

    /// Detect import cycles via strongly-connected components.
    pub fn import_cycles(&self) -> Vec<Vec<String>> {
        let sccs = petgraph::algo::tarjan_scc(&self.graph);
        sccs.into_iter()
            .filter(|c| c.len() > 1)
            .map(|c| c.iter().map(|ix| self.graph[*ix].clone()).collect())
            .collect()
    }

    /// Build a token-budgeted, signatures-only repo-map. Ranks definitions by
    /// PageRank (personalized toward mentioned idents/files), binary-searches how
    /// many fit the budget, and renders an elided tree grouped by file.
    pub fn repo_map(&self, req: &RepoMapRequest) -> RepoMap {
        // Personalization: mass to nodes whose name/file is mentioned.
        let mut pers: HashMap<String, f32> = HashMap::new();
        for (id, def) in &self.defs {
            let mut mass = 0.0f32;
            if req.mentioned_idents.iter().any(|m| m == &def.name) {
                mass += 10.0;
            }
            if req
                .mentioned_files
                .iter()
                .any(|f| def.file.contains(f.as_str()) || f.contains(def.file.as_str()))
            {
                mass += 5.0;
            }
            if mass > 0.0 {
                pers.insert(id.clone(), mass);
            }
        }

        let ranks = self.pagerank(&pers, 15);

        // Rank definitions, apply Aider-style multipliers.
        let mut ranked: Vec<(String, f32)> = self
            .defs
            .iter()
            .map(|(id, def)| {
                let mut r = *ranks.get(id).unwrap_or(&0.0);
                // distinctive long multiword identifier boost
                if def.name.len() >= 8 && is_multiword(&def.name) {
                    r *= 10.0;
                }
                // private/dunder damp
                if def.name.starts_with('_') {
                    r *= 0.1;
                }
                (id.clone(), r)
            })
            .collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        // Binary-search the count that fits the token budget.
        let budget = req.max_tokens.max(1);
        let render = |count: usize| -> (String, usize) {
            let chosen: Vec<&(String, f32)> = ranked.iter().take(count).collect();
            let text = self.render_elided(&chosen);
            let toks = estimate_tokens(&text);
            (text, toks)
        };

        let mut lo = 0usize;
        let mut hi = ranked.len();
        let mut best = String::new();
        let mut best_toks = 0usize;
        let mut best_count = 0usize;
        while lo <= hi && hi > 0 {
            let mid = (lo + hi) / 2;
            let (text, toks) = render(mid);
            if toks <= budget {
                best = text;
                best_toks = toks;
                best_count = mid;
                lo = mid + 1;
            } else {
                if mid == 0 {
                    break;
                }
                hi = mid - 1;
            }
        }

        let symbols = ranked
            .iter()
            .take(best_count)
            .filter_map(|(id, _)| {
                self.defs.get(id).map(|d| Symbol {
                    qualified_name: id.clone(),
                    name: d.name.clone(),
                    kind: "definition".to_string(),
                    file: d.file.clone(),
                })
            })
            .collect();

        RepoMap {
            rendered: best,
            symbols,
            estimated_tokens: best_toks,
        }
    }

    /// Render selected defs as an elided signatures-only tree grouped by file.
    fn render_elided(&self, chosen: &[&(String, f32)]) -> String {
        use std::collections::BTreeMap;
        let mut by_file: BTreeMap<String, Vec<&RankedDef>> = BTreeMap::new();
        for (id, _) in chosen {
            if let Some(def) = self.defs.get(id) {
                by_file.entry(def.file.clone()).or_default().push(def);
            }
        }
        let mut out = String::new();
        for (file, defs) in by_file {
            out.push_str(&file);
            out.push_str(":\n");
            for def in defs {
                let sig = truncate_line(&def.signature, 100);
                out.push_str("  ");
                out.push_str(&sig);
                out.push_str("\n    ⋮\n");
            }
        }
        out
    }
}

fn is_multiword(name: &str) -> bool {
    name.contains('_')
        || name.contains('-')
        || name
            .chars()
            .skip(1)
            .any(|c| c.is_uppercase()) // camelCase
}

fn truncate_line(line: &str, max: usize) -> String {
    let line = line.trim_end();
    if line.chars().count() <= max {
        line.to_string()
    } else {
        line.chars().take(max).collect()
    }
}

/// Cheap token estimate (chars/4, rounded up) for budget binary-search.
pub fn estimate_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph_with_calls() -> CodeGraph {
        let mut g = CodeGraph::new();
        g.add_definition("mod::core_engine", "core_engine", "src/engine.rs", "pub fn core_engine()");
        g.add_definition("mod::helper_fn", "helper_fn", "src/util.rs", "fn helper_fn()");
        g.add_definition("mod::caller_one", "caller_one", "src/a.rs", "fn caller_one()");
        // many things call core_engine → it should rank high
        g.add_edge("mod::caller_one", "mod::core_engine", EdgeKind::Calls, 1.0);
        g.add_edge("mod::helper_fn", "mod::core_engine", EdgeKind::Calls, 1.0);
        g
    }

    #[test]
    fn pagerank_ranks_popular_callee_higher() {
        let g = graph_with_calls();
        let ranks = g.pagerank(&HashMap::new(), 20);
        let engine = ranks["mod::core_engine"];
        let caller = ranks["mod::caller_one"];
        assert!(engine > caller, "popular callee should outrank its callers");
    }

    #[test]
    fn repo_map_renders_elided_tree_within_budget() {
        let g = graph_with_calls();
        let rm = g.repo_map(&RepoMapRequest {
            mentioned_files: vec![],
            mentioned_idents: vec!["core_engine".to_string()],
            max_tokens: 200,
        });
        assert!(rm.rendered.contains("core_engine"));
        assert!(rm.rendered.contains("⋮"), "bodies collapsed to ellipsis");
        assert!(rm.estimated_tokens <= 200);
        assert!(!rm.symbols.is_empty());
    }

    #[test]
    fn neighbors_filtered_by_edge_kind() {
        let mut g = CodeGraph::new();
        g.add_edge("a", "b", EdgeKind::Calls, 1.0);
        g.add_edge("a", "c", EdgeKind::Imports, 1.0);
        let calls = g.neighbors_by_kind("a", EdgeKind::Calls);
        assert_eq!(calls, vec!["b".to_string()]);
        let imports = g.neighbors_by_kind("a", EdgeKind::Imports);
        assert_eq!(imports, vec!["c".to_string()]);
    }

    #[test]
    fn import_cycles_detected() {
        let mut g = CodeGraph::new();
        g.add_edge("a", "b", EdgeKind::Imports, 1.0);
        g.add_edge("b", "a", EdgeKind::Imports, 1.0);
        let cycles = g.import_cycles();
        assert_eq!(cycles.len(), 1);
        assert_eq!(cycles[0].len(), 2);
    }
}
