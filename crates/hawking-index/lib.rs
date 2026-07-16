//! HIDE living-index (bible ch.05 · Codebase Intelligence).
//!
//! The standing organ that makes every other subsystem smarter. This crate owns
//! the query contracts and two implementations:
//!
//! - [`InMemoryCodeIndex`] — the lightweight, RAM-resident index (consumed by
//!   hide-backend / hawking-context). Now backed by REAL tree-sitter parsing, so
//!   it extracts both definitions and references.
//! - [`SqliteCodeIndex`] — the durable, index-backed implementation: a BLAKE3
//!   merkle gate ([`merkle`]), tree-sitter parsing + cAST chunking ([`parse`]),
//!   a SQLite/FTS5 + graph store ([`store`]), a petgraph PageRank repo-map
//!   ([`graph`]), a hybrid lexical⊕symbol⊕vector retriever with RRF + rerank
//!   ([`semantic`]), and an incremental [`daemon`] with generation/MVCC and
//!   crash recovery.
//!
//! Live model calls (embeddings) target `hawking-serve`'s real HTTP endpoint
//! (`POST /v1/embeddings`) behind the swappable [`semantic::EmbeddingClient`]
//! trait; tests use [`semantic::StubEmbeddingClient`] so they run offline.

#[rustfmt::skip]
pub mod daemon {
    //! The Living-Index daemon: incremental indexer (bible §4.9).
    //!
    //! Watches the workspace (notify FSEvents) with debounce, runs a merkle diff to
    //! find the exact changed files (the watcher is a *hint*; merkle decides the
    //! work), reparses only changed/renamed files, updates the store, and advances a
    //! generation with crash recovery. A runnable loop (`run_until`) is provided.

    use crate::merkle::{Blake3MerkleScanner, ChangeSet, MerkleNode, MerkleScanner};
    use crate::query::SqliteCodeIndex;
    use hide_core::Result;
    use notify::{RecommendedWatcher, RecursiveMode, Watcher};
    use serde::{Deserialize, Serialize};
    use std::path::{Path, PathBuf};
    use std::sync::mpsc::{channel, Receiver};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct IndexDaemonConfig {
        pub debounce_ms: u64,
        pub idle_reindex_after_ms: u64,
        pub max_concurrent_lsp: usize,
    }

    impl Default for IndexDaemonConfig {
        fn default() -> Self {
            Self { debounce_ms: 200, idle_reindex_after_ms: 15_000, max_concurrent_lsp: 2 }
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct IndexDaemonState {
        pub generation: u64,
        pub queue_depth: usize,
        pub is_idle: bool,
        pub last_error: Option<String>,
    }

    /// The incremental indexer. Owns the index + the merkle snapshot, and applies a
    /// changeset (add/modify/delete/rename) against the store with MVCC.
    pub struct IndexDaemon {
        root: PathBuf,
        index: Arc<SqliteCodeIndex>,
        scanner: Blake3MerkleScanner,
        /// Last committed merkle snapshot (the recovery + diff anchor).
        snapshot: Option<MerkleNode>,
        config: IndexDaemonConfig,
    }

    impl IndexDaemon {
        pub fn new(root: impl Into<PathBuf>, index: Arc<SqliteCodeIndex>) -> Self {
            let root = root.into();
            Self {
                scanner: Blake3MerkleScanner::new(root.clone()),
                root,
                index,
                snapshot: None,
                config: IndexDaemonConfig::default(),
            }
        }

        pub fn with_config(mut self, config: IndexDaemonConfig) -> Self {
            self.config = config;
            self
        }

        pub fn index(&self) -> Arc<SqliteCodeIndex> {
            self.index.clone()
        }

        pub fn root(&self) -> &Path {
            &self.root
        }

        /// Crash recovery + cold/warm start: truncate any torn generation, then
        /// merkle-diff the current tree against the last snapshot (catching edits made
        /// while the daemon was down) and apply. Returns the committed generation.
        pub fn bootstrap(&mut self) -> Result<u64> {
            // Recover the store to its last good generation (drop torn tails).
            let _recovered = self.index.store().recover()?;

            let current = self.scanner.scan_workspace()?;
            let changeset = match &self.snapshot {
                Some(prev) => self.scanner.diff(prev, &current)?,
                None => {
                    // Cold start: everything is an add.
                    let mut cs = ChangeSet::default();
                    collect_all_files(&current, &mut cs.added);
                    cs
                }
            };
            self.apply_changeset(&changeset)?;
            self.snapshot = Some(current);
            Ok(self.index.generation())
        }

        /// Apply a changeset to the index: reparse adds/modifies/rename-dests, remove
        /// deletes. (Rename sources keep their content; the dest is reparsed — a
        /// future optimization is a pure ID remap, but reparse is always correct.)
        pub fn apply_changeset(&self, cs: &ChangeSet) -> Result<()> {
            for path in &cs.removed {
                if let Some(rel) = self.rel(path) {
                    self.index.store().remove_file(&rel)?;
                }
            }
            for (old, _new) in &cs.renamed {
                // remove the old path; the new path is reparsed below via dirty_paths
                if let Some(rel) = self.rel(old) {
                    self.index.store().remove_file(&rel)?;
                }
            }
            for path in cs.dirty_paths() {
                self.index_one(&path)?;
            }
            Ok(())
        }

        /// Index one file from disk (skips unreadable / non-UTF8 files).
        pub fn index_one(&self, path: &Path) -> Result<()> {
            let rel = match self.rel(path) {
                Some(r) => r,
                None => return Ok(()),
            };
            let bytes = match std::fs::read(path) {
                Ok(b) => b,
                Err(_) => return Ok(()), // file vanished between diff and read — ok
            };
            let hash = blake3::hash(&bytes).to_hex().to_string();
            // Skip if unchanged (merkle gate already filters, but this is the backstop).
            if let Ok(Some(existing)) = self.index.store().file_hash(&rel) {
                if existing == hash {
                    return Ok(());
                }
            }
            let content = match String::from_utf8(bytes) {
                Ok(s) => s,
                Err(_) => return Ok(()), // binary file — lexical-only path could go here
            };
            self.index.index_text(&rel, &content, &hash)?;
            Ok(())
        }

        fn rel(&self, path: &Path) -> Option<String> {
            path.strip_prefix(&self.root)
                .ok()
                .map(|p| p.to_string_lossy().to_string())
                .or_else(|| Some(path.to_string_lossy().to_string()))
        }

        /// Rescan the tree, diff against the snapshot, apply, advance the snapshot.
        /// This is the "on wake" path: the FS event is only a hint, merkle decides.
        pub fn tick(&mut self) -> Result<ChangeSet> {
            let current = self.scanner.scan_workspace()?;
            let cs = match &self.snapshot {
                Some(prev) => self.scanner.diff(prev, &current)?,
                None => {
                    let mut cs = ChangeSet::default();
                    collect_all_files(&current, &mut cs.added);
                    cs
                }
            };
            if !cs.is_empty() {
                self.apply_changeset(&cs)?;
            }
            self.snapshot = Some(current);
            Ok(cs)
        }

        pub fn state(&self) -> IndexDaemonState {
            IndexDaemonState { generation: self.index.generation(), queue_depth: 0, is_idle: true, last_error: None }
        }

        /// Run the watch loop until `stop()` returns true. Sets up a notify watcher on
        /// the root (recursively, on directories), debounces bursts, and on each
        /// settled burst runs `tick()`. Blocking; intended for a dedicated thread.
        ///
        /// `stop` is polled between debounce windows so the loop is cancellable.
        pub fn run_until<F: Fn() -> bool>(&mut self, stop: F) -> Result<()> {
            self.bootstrap()?;

            let (tx, rx) = channel::<notify::Result<notify::Event>>();
            let mut watcher = RecommendedWatcher::new(
                move |res| {
                    let _ = tx.send(res);
                },
                notify::Config::default(),
            )
            .map_err(|e| hide_core::HideError::Storage(format!("watcher: {e}")))?;
            watcher
                .watch(&self.root, RecursiveMode::Recursive)
                .map_err(|e| hide_core::HideError::Storage(format!("watch root: {e}")))?;

            let debounce = Duration::from_millis(self.config.debounce_ms);
            loop {
                if stop() {
                    break;
                }
                // Block for the first event (with a timeout so we can poll `stop`).
                match rx.recv_timeout(Duration::from_millis(250)) {
                    Ok(_evt) => {
                        drain_burst(&rx, debounce);
                        // The event(s) are only a hint; merkle decides the real work.
                        let _ = self.tick();
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
            Ok(())
        }
    }

    /// Drain a burst of FS events for `window`, coalescing them (debounce).
    fn drain_burst(rx: &Receiver<notify::Result<notify::Event>>, window: Duration) {
        let deadline = Instant::now() + window;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            match rx.recv_timeout(remaining) {
                Ok(_) => continue, // keep draining within the window
                Err(_) => break,
            }
        }
    }

    fn collect_all_files(node: &MerkleNode, out: &mut Vec<PathBuf>) {
        match node.kind {
            crate::merkle::MerkleKind::File => out.push(node.path.clone()),
            crate::merkle::MerkleKind::Directory => {
                for c in &node.children {
                    collect_all_files(c, out);
                }
            }
            _ => {}
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::fs;

        fn write(dir: &Path, rel: &str, content: &str) {
            let p = dir.join(rel);
            if let Some(parent) = p.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(p, content).unwrap();
        }

        #[test]
        fn bootstrap_cold_indexes_all_files() {
            let tmp = tempfile::tempdir().unwrap();
            write(tmp.path(), "src/a.rs", "pub fn alpha() {}");
            write(tmp.path(), "src/b.rs", "pub fn beta() { alpha(); }");
            let index = Arc::new(SqliteCodeIndex::open_in_memory().unwrap());
            let mut daemon = IndexDaemon::new(tmp.path(), index.clone());
            let gen = daemon.bootstrap().unwrap();
            assert!(gen >= 2, "two files indexed → generation advanced");
            assert_eq!(index.store().file_count().unwrap(), 2);
        }

        #[test]
        fn tick_applies_incremental_change() {
            let tmp = tempfile::tempdir().unwrap();
            write(tmp.path(), "a.rs", "pub fn alpha() {}");
            let index = Arc::new(SqliteCodeIndex::open_in_memory().unwrap());
            let mut daemon = IndexDaemon::new(tmp.path(), index.clone());
            daemon.bootstrap().unwrap();
            assert_eq!(index.store().file_count().unwrap(), 1);

            // add a new file, modify the old one, then tick.
            write(tmp.path(), "b.rs", "pub fn beta() {}");
            write(tmp.path(), "a.rs", "pub fn alpha_renamed() {}");
            let cs = daemon.tick().unwrap();
            assert!(cs.added.iter().any(|p| p.ends_with("b.rs")));
            assert!(cs.modified.iter().any(|p| p.ends_with("a.rs")));
            assert_eq!(index.store().file_count().unwrap(), 2);
        }

        #[test]
        fn tick_handles_deletion_and_rename() {
            let tmp = tempfile::tempdir().unwrap();
            write(tmp.path(), "keep.rs", "pub fn keep() {}");
            write(tmp.path(), "gone.rs", "pub fn gone() {}");
            write(tmp.path(), "old.rs", "pub fn stable_unique_body() {}");
            let index = Arc::new(SqliteCodeIndex::open_in_memory().unwrap());
            let mut daemon = IndexDaemon::new(tmp.path(), index.clone());
            daemon.bootstrap().unwrap();

            fs::remove_file(tmp.path().join("gone.rs")).unwrap();
            fs::rename(tmp.path().join("old.rs"), tmp.path().join("new.rs")).unwrap();
            let cs = daemon.tick().unwrap();
            assert!(cs.removed.iter().any(|p| p.ends_with("gone.rs")));
            assert_eq!(cs.renamed.len(), 1, "rename detected, got {cs:?}");
            // store reflects: keep + new (gone removed, old→new)
            assert_eq!(index.store().file_count().unwrap(), 2);
        }

        #[test]
        fn bootstrap_recovers_from_torn_generation() {
            let tmp = tempfile::tempdir().unwrap();
            write(tmp.path(), "a.rs", "pub fn a() {}");
            let index = Arc::new(SqliteCodeIndex::open_in_memory().unwrap());
            // simulate a torn (uncommitted) generation
            index.store().begin_generation(99, "torn").unwrap();
            let mut daemon = IndexDaemon::new(tmp.path(), index.clone());
            daemon.bootstrap().unwrap();
            // recovery dropped the torn gen; committed generation is the real one
            assert!(index.store().last_committed_generation().unwrap() >= 1);
        }
    }
}
#[rustfmt::skip]
pub mod graph {
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
            Self { graph: DiGraph::new(), index: HashMap::new(), defs: HashMap::new() }
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
                RankedDef { name: name.to_string(), file: file.to_string(), signature: signature.to_string() },
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

            self.index.iter().map(|(id, ix)| (id.clone(), rank[ix.index()])).collect()
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
                if req.mentioned_files.iter().any(|f| def.file.contains(f.as_str()) || f.contains(def.file.as_str())) {
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
            ranked
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0)));

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

            RepoMap { rendered: best, symbols, estimated_tokens: best_toks }
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
        name.contains('_') || name.contains('-') || name.chars().skip(1).any(|c| c.is_uppercase())
        // camelCase
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
}
#[rustfmt::skip]
pub mod merkle {
    //! BLAKE3 merkle-DAG over the workspace tree (bible §4.8).
    //!
    //! - Leaf hash  = `BLAKE3(file_bytes)`.
    //! - Dir node   = `BLAKE3( sorted [ (name, type, child_hash) ] )` (canonical, order-independent).
    //! - O(changed) diff: compare roots, recurse only into differing subtrees.
    //! - Rename detection: a deleted leaf + an added leaf with the *same* content hash in
    //!   one changeset is reported as a rename, so the symbol graph remaps instead of
    //!   delete+reinsert (and unchanged content is never re-parsed/re-embedded).
    //!
    //! The merkle tree is the correctness backstop the watcher (§4.9) leans on: the
    //! file-system event is only a hint; the merkle diff decides the actual work.

    use ignore::WalkBuilder;
    use serde::{Deserialize, Serialize};
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    /// A node in the merkle-DAG: a file leaf or a directory.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct MerkleNode {
        pub path: PathBuf,
        /// Hex-encoded BLAKE3 hash (32 bytes → 64 hex chars).
        pub hash: String,
        pub kind: MerkleKind,
        pub size_bytes: u64,
        /// Child nodes for a directory (sorted by `path` for canonical hashing).
        /// Empty for files.
        #[serde(default)]
        pub children: Vec<MerkleNode>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum MerkleKind {
        File,
        Directory,
        Symlink,
        Missing,
    }

    /// The output of an O(changed) merkle diff.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
    pub struct ChangeSet {
        pub added: Vec<PathBuf>,
        pub modified: Vec<PathBuf>,
        pub removed: Vec<PathBuf>,
        /// `(old_path, new_path)` pairs detected by identical leaf hash.
        pub renamed: Vec<(PathBuf, PathBuf)>,
    }

    impl ChangeSet {
        pub fn is_empty(&self) -> bool {
            self.added.is_empty() && self.modified.is_empty() && self.removed.is_empty() && self.renamed.is_empty()
        }

        /// Every path whose *content* needs (re)parsing: adds + modifies + rename
        /// destinations. (Rename sources keep their content; the index just remaps.)
        pub fn dirty_paths(&self) -> Vec<PathBuf> {
            let mut out = self.added.clone();
            out.extend(self.modified.iter().cloned());
            out.extend(self.renamed.iter().map(|(_, new)| new.clone()));
            out
        }
    }

    /// Scans a workspace into a merkle-DAG and diffs two snapshots.
    pub trait MerkleScanner: Send + Sync {
        fn scan_workspace(&self) -> hide_core::Result<MerkleNode>;
        fn diff(&self, old: &MerkleNode, new: &MerkleNode) -> hide_core::Result<ChangeSet>;
    }

    /// A real BLAKE3 merkle scanner over the file system.
    ///
    /// Honors `.gitignore` (ripgrep's `ignore` crate) plus a built-in noise set
    /// (`.git`, `target`, `node_modules`, …) so generated/vendored churn never
    /// enters the index (bible §4.9, §6).
    pub struct Blake3MerkleScanner {
        root: PathBuf,
        respect_gitignore: bool,
    }

    impl Blake3MerkleScanner {
        pub fn new(root: impl Into<PathBuf>) -> Self {
            Self { root: root.into(), respect_gitignore: true }
        }

        pub fn without_gitignore(mut self) -> Self {
            self.respect_gitignore = false;
            self
        }

        pub fn root(&self) -> &Path {
            &self.root
        }
    }

    /// Built-in ignore set: directories that pollute a code index with machine noise.
    const BUILTIN_IGNORE_DIRS: &[&str] = &[
        ".git",
        ".hg",
        ".svn",
        "target",
        "node_modules",
        "dist",
        "build",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".idea",
        ".vscode",
        ".hide",
    ];

    fn hash_file(path: &Path) -> std::io::Result<(String, u64)> {
        let bytes = std::fs::read(path)?;
        let hash = blake3::hash(&bytes);
        Ok((hash.to_hex().to_string(), bytes.len() as u64))
    }

    /// Directory node hash = BLAKE3 over the canonical, sorted child digest list.
    fn hash_dir(children: &[MerkleNode]) -> String {
        let mut hasher = blake3::Hasher::new();
        // `children` is sorted by caller; encode (name, type, child_hash) per entry.
        for child in children {
            let name = child.path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            let type_tag: u8 = match child.kind {
                MerkleKind::File => 0,
                MerkleKind::Directory => 1,
                MerkleKind::Symlink => 2,
                MerkleKind::Missing => 3,
            };
            hasher.update(&(name.len() as u32).to_le_bytes());
            hasher.update(name.as_bytes());
            hasher.update(&[type_tag]);
            hasher.update(child.hash.as_bytes());
        }
        hasher.finalize().to_hex().to_string()
    }

    impl MerkleScanner for Blake3MerkleScanner {
        fn scan_workspace(&self) -> hide_core::Result<MerkleNode> {
            // Walk all non-ignored files, hashing leaves, then fold bottom-up into
            // directory nodes. `ignore`'s parallel walker yields files (not dirs),
            // so we collect leaves keyed by parent dir then build the tree.
            let mut leaves: BTreeMap<PathBuf, MerkleNode> = BTreeMap::new();

            let mut builder = WalkBuilder::new(&self.root);
            builder
                .hidden(false)
                .git_ignore(self.respect_gitignore)
                .git_global(self.respect_gitignore)
                .git_exclude(self.respect_gitignore)
                .parents(self.respect_gitignore)
                .filter_entry(|entry| {
                    if let Some(name) = entry.file_name().to_str() {
                        if entry.file_type().is_some_and(|ft| ft.is_dir()) && BUILTIN_IGNORE_DIRS.contains(&name) {
                            return false;
                        }
                    }
                    true
                });

            for result in builder.build() {
                let entry = match result {
                    Ok(e) => e,
                    Err(_) => continue, // skip unreadable entries; merkle is best-effort over readable tree
                };
                let ft = match entry.file_type() {
                    Some(ft) => ft,
                    None => continue,
                };
                if !ft.is_file() {
                    continue;
                }
                let path = entry.path().to_path_buf();
                let (hash, size) = match hash_file(&path) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                leaves.insert(
                    path.clone(),
                    MerkleNode { path, hash, kind: MerkleKind::File, size_bytes: size, children: Vec::new() },
                );
            }

            Ok(build_tree(&self.root, &leaves))
        }

        fn diff(&self, old: &MerkleNode, new: &MerkleNode) -> hide_core::Result<ChangeSet> {
            Ok(diff_nodes(old, new))
        }
    }

    /// Fold a flat leaf map into a directory tree rooted at `root`.
    fn build_tree(root: &Path, leaves: &BTreeMap<PathBuf, MerkleNode>) -> MerkleNode {
        // Group leaves by their parent directory chain. We materialize directory
        // nodes lazily: collect direct children per dir, recurse.
        fn children_of(dir: &Path, leaves: &BTreeMap<PathBuf, MerkleNode>) -> Vec<MerkleNode> {
            // Direct child dirs and files of `dir`.
            let mut child_dirs: BTreeMap<PathBuf, ()> = BTreeMap::new();
            let mut child_files: Vec<MerkleNode> = Vec::new();
            for (path, node) in leaves.range(dir.to_path_buf()..) {
                if !path.starts_with(dir) || path == dir {
                    if path > &dir.join("\u{10ffff}") {
                        break;
                    }
                    continue;
                }
                // relative component after `dir`
                let rel = match path.strip_prefix(dir) {
                    Ok(r) => r,
                    Err(_) => continue,
                };
                let mut comps = rel.components();
                let first = match comps.next() {
                    Some(c) => c.as_os_str(),
                    None => continue,
                };
                if comps.next().is_none() {
                    // direct file child
                    child_files.push(node.clone());
                } else {
                    // belongs to a subdirectory
                    child_dirs.insert(dir.join(first), ());
                }
            }

            let mut children: Vec<MerkleNode> = child_files;
            for (sub, _) in child_dirs {
                let sub_children = children_of(&sub, leaves);
                let size: u64 = sub_children.iter().map(|c| c.size_bytes).sum();
                let hash = hash_dir(&sub_children);
                children.push(MerkleNode {
                    path: sub,
                    hash,
                    kind: MerkleKind::Directory,
                    size_bytes: size,
                    children: sub_children,
                });
            }
            children.sort_by(|a, b| a.path.cmp(&b.path));
            children
        }

        let children = children_of(root, leaves);
        let size: u64 = children.iter().map(|c| c.size_bytes).sum();
        let hash = hash_dir(&children);
        MerkleNode { path: root.to_path_buf(), hash, kind: MerkleKind::Directory, size_bytes: size, children }
    }

    /// O(changed) recursive diff between two merkle nodes.
    fn diff_nodes(old: &MerkleNode, new: &MerkleNode) -> ChangeSet {
        let mut cs = ChangeSet::default();
        diff_into(old, new, &mut cs);
        detect_renames(&mut cs, old, new);
        cs
    }

    fn diff_into(old: &MerkleNode, new: &MerkleNode, cs: &mut ChangeSet) {
        // Equal subtrees prune immediately — the whole point of merkle diffing.
        if old.hash == new.hash {
            return;
        }
        match (old.kind, new.kind) {
            (MerkleKind::Directory, MerkleKind::Directory) => {
                let old_children: BTreeMap<&Path, &MerkleNode> =
                    old.children.iter().map(|c| (c.path.as_path(), c)).collect();
                let new_children: BTreeMap<&Path, &MerkleNode> =
                    new.children.iter().map(|c| (c.path.as_path(), c)).collect();

                for (path, oc) in &old_children {
                    match new_children.get(path) {
                        Some(nc) => diff_into(oc, nc, cs),
                        None => collect_removed(oc, cs),
                    }
                }
                for (path, nc) in &new_children {
                    if !old_children.contains_key(path) {
                        collect_added(nc, cs);
                    }
                }
            }
            (MerkleKind::File, MerkleKind::File) => {
                // Same path, different hash → modified.
                cs.modified.push(new.path.clone());
            }
            // Type change at a path (file↔dir): treat as remove old + add new.
            _ => {
                collect_removed(old, cs);
                collect_added(new, cs);
            }
        }
    }

    fn collect_added(node: &MerkleNode, cs: &mut ChangeSet) {
        match node.kind {
            MerkleKind::File => cs.added.push(node.path.clone()),
            MerkleKind::Directory => {
                for c in &node.children {
                    collect_added(c, cs);
                }
            }
            _ => {}
        }
    }

    fn collect_removed(node: &MerkleNode, cs: &mut ChangeSet) {
        match node.kind {
            MerkleKind::File => cs.removed.push(node.path.clone()),
            MerkleKind::Directory => {
                for c in &node.children {
                    collect_removed(c, cs);
                }
            }
            _ => {}
        }
    }

    /// Pair `removed`+`added` files with identical leaf hash into renames.
    fn detect_renames(cs: &mut ChangeSet, old: &MerkleNode, new: &MerkleNode) {
        if cs.added.is_empty() || cs.removed.is_empty() {
            return;
        }
        let old_hashes = leaf_hash_map(old);
        let new_hashes = leaf_hash_map(new);

        // hash → removed paths with that hash
        let mut removed_by_hash: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
        for p in &cs.removed {
            if let Some(h) = old_hashes.get(p) {
                removed_by_hash.entry(h.clone()).or_default().push(p.clone());
            }
        }

        let mut consumed_removed: Vec<PathBuf> = Vec::new();
        let mut consumed_added: Vec<PathBuf> = Vec::new();
        let mut renamed: Vec<(PathBuf, PathBuf)> = Vec::new();

        for added_path in &cs.added {
            if let Some(h) = new_hashes.get(added_path) {
                if let Some(candidates) = removed_by_hash.get_mut(h) {
                    // pick a not-yet-consumed source with the same content hash
                    if let Some(pos) = candidates.iter().position(|p| !consumed_removed.contains(p)) {
                        let old_path = candidates.remove(pos);
                        consumed_removed.push(old_path.clone());
                        consumed_added.push(added_path.clone());
                        renamed.push((old_path, added_path.clone()));
                    }
                }
            }
        }

        cs.removed.retain(|p| !consumed_removed.contains(p));
        cs.added.retain(|p| !consumed_added.contains(p));
        cs.renamed.extend(renamed);
    }

    fn leaf_hash_map(node: &MerkleNode) -> BTreeMap<PathBuf, String> {
        let mut out = BTreeMap::new();
        fn walk(node: &MerkleNode, out: &mut BTreeMap<PathBuf, String>) {
            match node.kind {
                MerkleKind::File => {
                    out.insert(node.path.clone(), node.hash.clone());
                }
                MerkleKind::Directory => {
                    for c in &node.children {
                        walk(c, out);
                    }
                }
                _ => {}
            }
        }
        walk(node, &mut out);
        out
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::fs;

        fn write(dir: &Path, rel: &str, content: &str) {
            let p = dir.join(rel);
            if let Some(parent) = p.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(p, content).unwrap();
        }

        #[test]
        fn identical_trees_have_equal_root_and_empty_diff() {
            let tmp = tempfile::tempdir().unwrap();
            write(tmp.path(), "src/a.rs", "fn a() {}");
            write(tmp.path(), "src/b.rs", "fn b() {}");
            let scanner = Blake3MerkleScanner::new(tmp.path());
            let t1 = scanner.scan_workspace().unwrap();
            let t2 = scanner.scan_workspace().unwrap();
            assert_eq!(t1.hash, t2.hash);
            let cs = scanner.diff(&t1, &t2).unwrap();
            assert!(cs.is_empty(), "no change → empty changeset, got {cs:?}");
        }

        #[test]
        fn detects_add_modify_delete() {
            let tmp = tempfile::tempdir().unwrap();
            write(tmp.path(), "a.rs", "fn a() {}");
            write(tmp.path(), "b.rs", "fn b() {}");
            let scanner = Blake3MerkleScanner::new(tmp.path());
            let before = scanner.scan_workspace().unwrap();

            write(tmp.path(), "a.rs", "fn a() { changed(); }"); // modify
            fs::remove_file(tmp.path().join("b.rs")).unwrap(); // delete
            write(tmp.path(), "c.rs", "fn c() {}"); // add
            let after = scanner.scan_workspace().unwrap();

            let cs = scanner.diff(&before, &after).unwrap();
            assert!(cs.modified.iter().any(|p| p.ends_with("a.rs")));
            assert!(cs.removed.iter().any(|p| p.ends_with("b.rs")));
            assert!(cs.added.iter().any(|p| p.ends_with("c.rs")));
            assert!(cs.renamed.is_empty());
        }

        #[test]
        fn detects_rename_by_identical_hash() {
            let tmp = tempfile::tempdir().unwrap();
            write(tmp.path(), "old_name.rs", "fn unchanged_content() {}");
            let scanner = Blake3MerkleScanner::new(tmp.path());
            let before = scanner.scan_workspace().unwrap();

            fs::rename(tmp.path().join("old_name.rs"), tmp.path().join("new_name.rs")).unwrap();
            let after = scanner.scan_workspace().unwrap();

            let cs = scanner.diff(&before, &after).unwrap();
            assert_eq!(cs.renamed.len(), 1, "expected one rename, got {cs:?}");
            let (old, new) = &cs.renamed[0];
            assert!(old.ends_with("old_name.rs"));
            assert!(new.ends_with("new_name.rs"));
            assert!(cs.added.is_empty() && cs.removed.is_empty());
        }

        #[test]
        fn diff_is_pruned_for_unchanged_subtrees() {
            let tmp = tempfile::tempdir().unwrap();
            write(tmp.path(), "untouched/x.rs", "fn x() {}");
            write(tmp.path(), "touched/y.rs", "fn y() {}");
            let scanner = Blake3MerkleScanner::new(tmp.path());
            let before = scanner.scan_workspace().unwrap();
            write(tmp.path(), "touched/y.rs", "fn y() { more(); }");
            let after = scanner.scan_workspace().unwrap();

            // The 'untouched' subtree hash must be unchanged across snapshots.
            let find = |t: &MerkleNode, name: &str| -> String {
                t.children.iter().find(|c| c.path.ends_with(name)).map(|c| c.hash.clone()).unwrap()
            };
            assert_eq!(find(&before, "untouched"), find(&after, "untouched"));
            assert_ne!(find(&before, "touched"), find(&after, "touched"));
        }
    }
}
#[rustfmt::skip]
pub mod parse {
    //! The parsing layer (bible §4.2, §4.3).
    //!
    //! Real tree-sitter parsing replacing the old `simple_definition` prefix scanner.
    //! For every file we run the grammar's `tags.scm` query and emit BOTH definitions
    //! and references with SCIP-style path-scoped symbol IDs, so `references()` stops
    //! returning empty and the reverse-reference / blast-radius moat is reachable.

    pub mod chunker {
        //! cAST / by-symbol chunking (bible §4.7 "Chunking").
        //!
        //! We chunk by AST symbol, not fixed line windows: each top-level definition is a
        //! chunk. A chunk that exceeds the embedding budget is split; small adjacent
        //! siblings are greedily merged to fill the budget. Each chunk carries its byte
        //! range, enclosing symbol, and a BLAKE3 content hash so unchanged chunks are
        //! never re-embedded (the dominant incremental-embedding win).

        use super::grammars::{GrammarRegistry, LangId};
        use super::{scip_symbol_id, SymKind};
        use serde::{Deserialize, Serialize};
        use std::path::Path;
        use tree_sitter::{Node, Parser};

        /// A semantic chunk: a unit of code mapped to one embedding vector.
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
        pub struct CodeChunk {
            /// Content-addressed id: BLAKE3 of the chunk text (hex).
            pub chunk_id: String,
            pub file: String,
            /// Optional symbol id this chunk corresponds to (if it's a single def).
            pub symbol: Option<String>,
            pub start_byte: usize,
            pub end_byte: usize,
            pub start_line: u32,
            pub end_line: u32,
            pub text: String,
        }

        /// Soft target for chunk size in characters (a proxy for the embedding model's
        /// token budget; ~4 chars/token → ~512 tokens). Oversized chunks are split.
        const MAX_CHUNK_CHARS: usize = 2000;
        /// Chunks smaller than this are merge candidates with their neighbors.
        const MIN_CHUNK_CHARS: usize = 200;

        /// Chunk a file into cAST-style code chunks.
        ///
        /// For known languages we chunk by top-level definition nodes; for unknown
        /// languages (or parse failure) we fall back to a fixed-window split so the
        /// semantic leg still has *something* to embed.
        pub fn chunk_file(rel_path: &str, source: &str) -> Vec<CodeChunk> {
            let lang = LangId::from_path(Path::new(rel_path));
            if !lang.is_known() {
                return window_chunks(rel_path, source);
            }
            let Some(bundle) = GrammarRegistry::bundle(lang) else {
                return window_chunks(rel_path, source);
            };
            let mut parser = Parser::new();
            if parser.set_language(&bundle.language).is_err() {
                return window_chunks(rel_path, source);
            }
            let Some(tree) = parser.parse(source, None) else {
                return window_chunks(rel_path, source);
            };

            let src = source.as_bytes();
            let mut raw: Vec<(usize, usize)> = Vec::new();
            collect_def_spans(tree.root_node(), src, &mut raw);

            if raw.is_empty() {
                return window_chunks(rel_path, source);
            }
            raw.sort_by_key(|(s, _)| *s);

            // Build a table of every definition's byte span → its SCIP id, so each chunk
            // can be tagged with the symbol whose span encloses it (bible §4.7). We walk
            // the *whole* tree (not just chunk-able defs) so that, e.g., a method chunk
            // inside a class resolves to the method symbol rather than the class.
            let mut def_symbols: Vec<DefSpan> = Vec::new();
            collect_def_symbols(tree.root_node(), src, lang, rel_path, &mut def_symbols);

            // Split oversized, then merge small adjacent siblings (cAST).
            let mut split: Vec<(usize, usize)> = Vec::new();
            for (s, e) in raw {
                if e.saturating_sub(s) > MAX_CHUNK_CHARS {
                    split_span(source, s, e, &mut split);
                } else {
                    split.push((s, e));
                }
            }
            let merged = merge_small(&split);

            merged
                .into_iter()
                .filter_map(|(s, e)| {
                    let mut chunk = make_chunk(rel_path, source, s, e)?;
                    chunk.symbol = enclosing_symbol(&def_symbols, s, e);
                    Some(chunk)
                })
                .collect()
        }

        /// A definition's byte span paired with its SCIP id.
        struct DefSpan {
            start: usize,
            end: usize,
            symbol_id: String,
        }

        /// The SCIP id of the definition that owns the chunk (bible §4.7: "the symbol
        /// whose span contains the chunk").
        ///
        /// A chunk usually IS one definition, but cAST may split an oversized def or
        /// merge small siblings; the chunk then maps to its *leading* definition — the
        /// smallest def whose span contains the chunk's start byte. For a nested form
        /// (a method inside a class) the inner, smaller def wins, so a method chunk maps
        /// to the method rather than the enclosing class. `None` only when nothing
        /// covers the start (e.g. a window-fallback fragment).
        fn enclosing_symbol(defs: &[DefSpan], start: usize, end: usize) -> Option<String> {
            defs.iter()
                // Prefer the smallest def fully containing the chunk (the clean 1:1 case),
                // then fall back to the smallest def containing just the chunk's start
                // (merged/split case).
                .filter(|d| d.start <= start && d.end >= end)
                .min_by_key(|d| d.end - d.start)
                .or_else(|| defs.iter().filter(|d| d.start <= start && d.end > start).min_by_key(|d| d.end - d.start))
                .map(|d| d.symbol_id.clone())
        }

        /// Walk the tree collecting (byte-span, SCIP id) for every named definition we
        /// can attach a symbol to. Mirrors `parse::extract_with_bundle`'s kind mapping so
        /// the ids are byte-for-byte the same as the symbols stored in the index — that's
        /// what lets a retrieval hit map back to a symbol.
        fn collect_def_symbols(node: Node, src: &[u8], lang: LangId, rel_path: &str, out: &mut Vec<DefSpan>) {
            if let Some((name, kind)) = def_name_and_kind(node, src) {
                out.push(DefSpan {
                    start: node.start_byte(),
                    end: node.end_byte(),
                    symbol_id: scip_symbol_id(lang, rel_path, &name, kind),
                });
            }
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                collect_def_symbols(child, src, lang, rel_path, out);
            }
        }

        /// Map a definition node to its (name, kind), or `None` if it isn't a named
        /// definition. Covers the Rust / Python / TS-JS forms the index emits symbols for.
        fn def_name_and_kind(node: Node, src: &[u8]) -> Option<(String, SymKind)> {
            let kind = match node.kind() {
                // rust
                "function_item" => SymKind::Function,
                "struct_item" => SymKind::Struct,
                "enum_item" => SymKind::Enum,
                "trait_item" => SymKind::Trait,
                "mod_item" => SymKind::Module,
                "macro_definition" => SymKind::Macro,
                "type_item" => SymKind::TypeAlias,
                "const_item" | "static_item" => SymKind::Constant,
                // python
                "function_definition" => SymKind::Function,
                "class_definition" => SymKind::Class,
                // typescript / javascript
                "function_declaration" | "generator_function_declaration" => SymKind::Function,
                "method_definition" => SymKind::Method,
                "class_declaration" => SymKind::Class,
                "interface_declaration" => SymKind::Interface,
                "enum_declaration" => SymKind::Enum,
                "type_alias_declaration" => SymKind::TypeAlias,
                _ => return None,
            };
            let name = node.child_by_field_name("name").and_then(|n| n.utf8_text(src).ok()).map(|s| s.to_string())?;
            if name.is_empty() {
                return None;
            }
            Some((name, kind))
        }

        /// Collect byte spans of the definition nodes we want as chunks (functions,
        /// methods, classes, structs, enums, traits, impls). We take *top-level* defs
        /// and methods, but not nested locals.
        fn collect_def_spans(node: Node, _src: &[u8], out: &mut Vec<(usize, usize)>) {
            const DEF_KINDS: &[&str] = &[
                // rust
                "function_item",
                "struct_item",
                "enum_item",
                "trait_item",
                "impl_item",
                "mod_item",
                "macro_definition",
                // python
                "function_definition",
                "class_definition",
                "decorated_definition",
                // typescript / js
                "class_declaration",
                "function_declaration",
                "method_definition",
                "interface_declaration",
                "enum_declaration",
                "lexical_declaration",
                "export_statement",
            ];

            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if DEF_KINDS.contains(&child.kind()) {
                    out.push((child.start_byte(), child.end_byte()));
                    // For impl/class bodies, also surface inner methods as their own
                    // chunks (so a big impl block isn't one giant chunk).
                    if matches!(child.kind(), "impl_item" | "class_definition" | "class_declaration") {
                        let mut inner = child.walk();
                        for grand in child.children(&mut inner) {
                            collect_def_spans(grand, _src, out);
                        }
                    }
                } else {
                    // Recurse one level into modules/exports to catch nested top-levels.
                    if matches!(child.kind(), "mod_item" | "export_statement" | "block") {
                        collect_def_spans(child, _src, out);
                    }
                }
            }
        }

        /// Split an oversized span on line boundaries into <= MAX_CHUNK_CHARS pieces.
        fn split_span(source: &str, start: usize, end: usize, out: &mut Vec<(usize, usize)>) {
            let slice = &source[start..end.min(source.len())];
            let mut cur = start;
            let mut acc = 0usize;
            let mut last_break = start;
            for (i, ch) in slice.char_indices() {
                acc += ch.len_utf8();
                if ch == '\n' && acc >= MAX_CHUNK_CHARS {
                    let break_at = start + i + 1;
                    out.push((last_break, break_at));
                    last_break = break_at;
                    acc = 0;
                    cur = break_at;
                }
            }
            if cur < end {
                out.push((last_break, end));
            }
        }

        /// Greedily merge adjacent small spans up to the budget (cAST merge step).
        fn merge_small(spans: &[(usize, usize)]) -> Vec<(usize, usize)> {
            let mut out: Vec<(usize, usize)> = Vec::new();
            for &(s, e) in spans {
                if let Some(last) = out.last_mut() {
                    let last_len = last.1 - last.0;
                    let this_len = e - s;
                    if last_len < MIN_CHUNK_CHARS && (last_len + this_len) <= MAX_CHUNK_CHARS {
                        last.1 = e;
                        continue;
                    }
                }
                out.push((s, e));
            }
            out
        }

        fn make_chunk(rel_path: &str, source: &str, start: usize, end: usize) -> Option<CodeChunk> {
            let end = end.min(source.len());
            if start >= end {
                return None;
            }
            let text = source.get(start..end)?.to_string();
            if text.trim().is_empty() {
                return None;
            }
            let chunk_id = blake3::hash(text.as_bytes()).to_hex().to_string();
            let start_line = source[..start].bytes().filter(|b| *b == b'\n').count() as u32 + 1;
            let end_line = source[..end].bytes().filter(|b| *b == b'\n').count() as u32 + 1;
            Some(CodeChunk {
                chunk_id,
                file: rel_path.to_string(),
                symbol: None,
                start_byte: start,
                end_byte: end,
                start_line,
                end_line,
                text,
            })
        }

        /// Fixed-window fallback for unparseable files (so the semantic leg still works).
        fn window_chunks(rel_path: &str, source: &str) -> Vec<CodeChunk> {
            let mut out = Vec::new();
            let mut start = 0usize;
            let bytes = source.as_bytes();
            while start < bytes.len() {
                let mut end = (start + MAX_CHUNK_CHARS).min(bytes.len());
                // snap to a char boundary
                while end < bytes.len() && !source.is_char_boundary(end) {
                    end += 1;
                }
                if let Some(c) = make_chunk(rel_path, source, start, end) {
                    out.push(c);
                }
                start = end;
            }
            out
        }

        #[cfg(test)]
        mod tests {
            use super::*;

            #[test]
            fn chunks_rust_by_definition() {
                let src = "pub fn alpha() {\n    let x = 1;\n}\n\npub fn beta() {\n    let y = 2;\n}\n";
                let chunks = chunk_file("m.rs", src);
                assert!(!chunks.is_empty());
                // each chunk is content-addressed and non-empty
                for c in &chunks {
                    assert_eq!(c.chunk_id.len(), 64);
                    assert!(!c.text.trim().is_empty());
                }
                let joined: String = chunks.iter().map(|c| c.text.clone()).collect();
                assert!(joined.contains("alpha"));
                assert!(joined.contains("beta"));
            }

            #[test]
            fn identical_text_yields_identical_chunk_id() {
                let a = chunk_file("a.rs", "pub fn f() { body(); }");
                let b = chunk_file("b.rs", "pub fn f() { body(); }");
                assert_eq!(a[0].chunk_id, b[0].chunk_id);
            }

            #[test]
            fn chunk_symbol_is_enclosing_def_scip_id() {
                use crate::parse::parse_source;
                // Two top-level fns: each chunk must carry the SCIP id of the def it covers,
                // and that id must equal the symbol the parser emits (so hits map back).
                let src = "pub fn alpha() {\n    work();\n}\n\npub fn beta() {\n    other();\n}\n";
                let chunks = chunk_file("m.rs", src);
                assert!(!chunks.is_empty());
                // every chunk that wraps a single def must have a symbol (not None)
                let with_sym: Vec<_> = chunks.iter().filter(|c| c.symbol.is_some()).collect();
                assert!(!with_sym.is_empty(), "expected at least one chunk tagged with its enclosing symbol");

                // The symbol id must be byte-identical to a parsed symbol's qualified_name.
                let parsed = parse_source("m.rs", src);
                let parsed_ids: std::collections::HashSet<&str> =
                    parsed.symbols.iter().map(|s| s.qualified_name.as_str()).collect();
                for c in &with_sym {
                    let sym = c.symbol.as_deref().unwrap();
                    assert!(
                        parsed_ids.contains(sym),
                        "chunk symbol {sym:?} must match a parsed symbol id; have {parsed_ids:?}"
                    );
                }

                // Specifically, the chunk covering `alpha` resolves to alpha's id.
                let alpha_id = scip_symbol_id(LangId::Rust, "m.rs", "alpha", SymKind::Function);
                assert!(
                    chunks.iter().any(|c| c.symbol.as_deref() == Some(alpha_id.as_str())),
                    "a chunk should map to alpha's SCIP id {alpha_id:?}"
                );
            }

            #[test]
            fn chunk_symbol_resolves_to_inner_method_not_class() {
                // A method chunk inside a class must resolve to the *method* (smallest
                // enclosing def), not the class. Make the method body large enough that it
                // survives as its own chunk (cAST won't merge a >MIN-size sibling).
                let body: String = (0..40).map(|i| format!("        line_{i}();\n")).collect();
                let src = format!("class Greeter {{\n    render() {{\n{body}    }}\n}}\n");
                let chunks = chunk_file("ui.ts", &src);
                let method_id = scip_symbol_id(LangId::TypeScript, "ui.ts", "render", SymKind::Method);
                // there must be a chunk that maps to the inner method's id (smallest
                // enclosing def), proving nested resolution prefers the method over the class.
                assert!(
                    chunks.iter().any(|c| c.symbol.as_deref() == Some(method_id.as_str())),
                    "expected a chunk mapped to inner method {method_id:?}; got {:?}",
                    chunks.iter().map(|c| c.symbol.clone()).collect::<Vec<_>>()
                );
            }

            #[test]
            fn unknown_language_uses_window_fallback() {
                let big = "x".repeat(5000);
                let chunks = chunk_file("blob.dat", &big);
                assert!(chunks.len() >= 2, "oversized unknown file should split");
            }
        }
    }
    pub mod grammars {
        //! Grammar registry: maps a language to its tree-sitter `Language` + `tags.scm`.
        //!
        //! The core set is compiled in (statically linked parsers). Tier-0 tag extraction
        //! works immediately for every registered language (bible §4.2, §7.1).

        use serde::{Deserialize, Serialize};
        use std::path::Path;
        use tree_sitter::{Language, Query};

        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(rename_all = "snake_case")]
        pub enum LangId {
            Rust,
            Python,
            TypeScript,
            /// File with no registered grammar — lexical-only (trigram over raw bytes).
            Unknown,
        }

        impl LangId {
            pub fn as_str(self) -> &'static str {
                match self {
                    LangId::Rust => "rust",
                    LangId::Python => "python",
                    LangId::TypeScript => "typescript",
                    LangId::Unknown => "unknown",
                }
            }

            /// Detect from file extension (the cheap, reliable path).
            pub fn from_path(path: &Path) -> LangId {
                match path.extension().and_then(|e| e.to_str()) {
                    Some("rs") => LangId::Rust,
                    Some("py" | "pyi") => LangId::Python,
                    Some("ts" | "tsx" | "mts" | "cts" | "js" | "jsx" | "mjs" | "cjs") => LangId::TypeScript,
                    _ => LangId::Unknown,
                }
            }

            pub fn is_known(self) -> bool {
                !matches!(self, LangId::Unknown)
            }
        }

        /// Custom TypeScript/JavaScript `tags.scm`.
        ///
        /// The grammar's *bundled* `queries/tags.scm` only matches ambient/signature
        /// declarations (`function_signature`, `method_signature`, `abstract_*`,
        /// `interface_declaration`, `module`) — so ordinary `.ts`/`.js` source (a plain
        /// `function foo() {}`, `class Bar {}`, a method body, a call site) yields ZERO
        /// defs/refs. This query covers the concrete-syntax forms that real source code
        /// actually uses, plus reference forms (call sites, `new`, ident uses), so the
        /// reverse-reference / blast-radius moat works on TS and JS. Capture names match
        /// the `definition.<kind>` / `reference.<kind>` / `@name` contract that
        /// `parse::extract_with_bundle` consumes. The TypeScript grammar is a superset of
        /// JavaScript, so the same query drives both (`.js`/`.jsx`/`.mjs` route here too).
        pub const TS_TAGS_QUERY: &str = r#"
        ; ---- definitions ----

        (function_declaration
          name: (identifier) @name) @definition.function

        (generator_function_declaration
          name: (identifier) @name) @definition.function

        (class_declaration
          name: (type_identifier) @name) @definition.class

        (method_definition
          name: (property_identifier) @name) @definition.method

        ; arrow-fn / function-expr bound to a const/let/var: `const f = (..) => ..`
        (variable_declarator
          name: (identifier) @name
          value: [(arrow_function) (function_expression)]) @definition.function

        ; interfaces / type aliases / enums (TS)
        (interface_declaration
          name: (type_identifier) @name) @definition.interface

        (type_alias_declaration
          name: (type_identifier) @name) @definition.type

        (enum_declaration
          name: (identifier) @name) @definition.enum

        ; ---- references ----

        ; direct call: `foo(..)`
        (call_expression
          function: (identifier) @name) @reference.call

        ; method/qualified call: `obj.foo(..)` — credit the property name
        (call_expression
          function: (member_expression
            property: (property_identifier) @name)) @reference.call

        ; `new Thing(..)`
        (new_expression
          constructor: (identifier) @name) @reference.class
        "#;

        /// A compiled grammar bundle. Holds the tree-sitter `Language` and a compiled
        /// `tags.scm` `Query` (defs + refs + name captures).
        pub struct GrammarBundle {
            pub lang: LangId,
            pub language: Language,
            pub tags_query: Query,
        }

        impl GrammarBundle {
            fn build(lang: LangId, language: Language, tags_src: &str) -> Option<Self> {
                let tags_query = Query::new(&language, tags_src).ok()?;
                Some(Self { lang, language, tags_query })
            }
        }

        /// Static registry. Bundles are built lazily on first request and cached in the
        /// caller (see `parse::SymbolExtractor`). Grammars themselves are cheap to clone
        /// (an `Arc` internally in tree-sitter), but `Query` compilation is not — so we
        /// compile once per registry instance.
        pub struct GrammarRegistry;

        impl GrammarRegistry {
            pub fn bundle(lang: LangId) -> Option<GrammarBundle> {
                match lang {
                    LangId::Rust => {
                        GrammarBundle::build(lang, tree_sitter_rust::LANGUAGE.into(), tree_sitter_rust::TAGS_QUERY)
                    }
                    LangId::Python => {
                        GrammarBundle::build(lang, tree_sitter_python::LANGUAGE.into(), tree_sitter_python::TAGS_QUERY)
                    }
                    LangId::TypeScript => GrammarBundle::build(
                        lang,
                        tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
                        // Custom query (the bundled `TAGS_QUERY` only matches ambient
                        // signatures, yielding ZERO defs/refs on ordinary `.ts`/`.js`).
                        TS_TAGS_QUERY,
                    ),
                    LangId::Unknown => None,
                }
            }
        }
    }

    pub use chunker::{chunk_file, CodeChunk};
    pub use grammars::{GrammarBundle, GrammarRegistry, LangId};

    use crate::graph::{Occurrence, Symbol};
    use hide_core::types::TextRange;
    use std::path::Path;
    use tree_sitter::{Node, Parser, QueryCursor, StreamingIterator};

    /// The roles an occurrence can play (mirrors SCIP role bits; stored as a string
    /// on `Occurrence::role` for backward-compat with the existing query API).
    pub const ROLE_DEFINITION: &str = "definition";
    pub const ROLE_REFERENCE: &str = "reference";

    /// One parsed file's extracted facts.
    #[derive(Debug, Clone, Default)]
    pub struct ParseOutput {
        pub lang: Option<LangId>,
        pub symbols: Vec<Symbol>,
        pub occurrences: Vec<Occurrence>,
        /// Byte ranges of ERROR / MISSING nodes (for the health surface).
        pub error_spans: Vec<(usize, usize)>,
        /// True if the whole file failed to parse into anything structural.
        pub unparseable: bool,
    }

    /// SCIP-style structured symbol ID.
    ///
    /// Format (Hawking dialect): `hawking <lang> <repo_rel_path> <descriptor>`, e.g.
    /// `hawking rust src/model/qwen.rs forward_token().`. The descriptor suffix
    /// encodes the kind: `().` method/function, `#` type, `.` term, `/` module.
    /// IDs are stable across edits as long as the qualified name is stable, so a
    /// reference resolves to its definition by string equality.
    pub fn scip_symbol_id(lang: LangId, rel_path: &str, name: &str, kind: SymKind) -> String {
        let suffix = match kind {
            SymKind::Function | SymKind::Method => "().",
            SymKind::Class | SymKind::Struct | SymKind::Enum | SymKind::Trait | SymKind::Interface => "#",
            SymKind::Module => "/",
            SymKind::Macro => "!",
            SymKind::Constant | SymKind::Field | SymKind::TypeAlias | SymKind::Unknown => ".",
        };
        format!("hawking {} {} {}{}", lang.as_str(), rel_path, name, suffix)
    }

    /// Symbol kinds (a superset of what `tags.scm` distinguishes).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum SymKind {
        Function,
        Method,
        Class,
        Struct,
        Enum,
        Trait,
        Interface,
        Module,
        Macro,
        Constant,
        Field,
        TypeAlias,
        Unknown,
    }

    impl SymKind {
        pub fn as_str(self) -> &'static str {
            match self {
                SymKind::Function => "function",
                SymKind::Method => "method",
                SymKind::Class => "class",
                SymKind::Struct => "struct",
                SymKind::Enum => "enum",
                SymKind::Trait => "trait",
                SymKind::Interface => "interface",
                SymKind::Module => "module",
                SymKind::Macro => "macro",
                SymKind::Constant => "constant",
                SymKind::Field => "field",
                SymKind::TypeAlias => "type_alias",
                SymKind::Unknown => "symbol",
            }
        }

        /// Map a `tags.scm` capture name (e.g. `definition.function`) to a kind.
        fn from_capture(capture: &str) -> Option<SymKind> {
            let suffix = capture.strip_prefix("definition.")?;
            Some(match suffix {
                "function" => SymKind::Function,
                "method" => SymKind::Method,
                "class" => SymKind::Class,
                "struct" => SymKind::Struct,
                "enum" => SymKind::Enum,
                "trait" => SymKind::Trait,
                "interface" => SymKind::Interface,
                "module" => SymKind::Module,
                "macro" => SymKind::Macro,
                "constant" => SymKind::Constant,
                "field" => SymKind::Field,
                "type" => SymKind::TypeAlias,
                _ => SymKind::Unknown,
            })
        }
    }

    /// Parse a file's source and extract symbols + occurrences via tree-sitter.
    ///
    /// `rel_path` is the workspace-relative path (used for SCIP IDs and provenance).
    /// Returns an empty/unparseable `ParseOutput` for unknown languages — the caller
    /// falls back to lexical-only indexing for those.
    pub fn parse_source(rel_path: &str, source: &str) -> ParseOutput {
        let lang = LangId::from_path(Path::new(rel_path));
        if !lang.is_known() {
            return ParseOutput { lang: Some(lang), unparseable: true, ..Default::default() };
        }
        let bundle = match GrammarRegistry::bundle(lang) {
            Some(b) => b,
            None => return ParseOutput { lang: Some(lang), unparseable: true, ..Default::default() },
        };
        extract_with_bundle(&bundle, rel_path, source)
    }

    fn extract_with_bundle(bundle: &GrammarBundle, rel_path: &str, source: &str) -> ParseOutput {
        let mut parser = Parser::new();
        if parser.set_language(&bundle.language).is_err() {
            return ParseOutput { lang: Some(bundle.lang), unparseable: true, ..Default::default() };
        }
        let tree = match parser.parse(source, None) {
            Some(t) => t,
            None => return ParseOutput { lang: Some(bundle.lang), unparseable: true, ..Default::default() },
        };

        let mut out = ParseOutput { lang: Some(bundle.lang), ..Default::default() };

        // Collect ERROR/MISSING spans (damage localization).
        collect_error_spans(tree.root_node(), &mut out.error_spans);

        let src_bytes = source.as_bytes();
        let query = &bundle.tags_query;
        let capture_names = query.capture_names();

        // Locate the `@name` capture index once.
        let name_idx = capture_names.iter().position(|n| *n == "name");

        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(query, tree.root_node(), src_bytes);

        while let Some(m) = matches.next() {
            // Find the @name node and the role/kind capture in this match.
            let mut name_text: Option<String> = None;
            let mut name_node: Option<Node> = None;
            let mut role_capture: Option<&str> = None;

            for cap in m.captures {
                let cap_name = capture_names[cap.index as usize];
                if Some(cap.index as usize) == name_idx {
                    name_node = Some(cap.node);
                    name_text = cap.node.utf8_text(src_bytes).ok().map(|s| s.to_string());
                } else if cap_name.starts_with("definition.") || cap_name.starts_with("reference.") {
                    role_capture = Some(cap_name);
                }
            }

            let (Some(name), Some(node), Some(role)) = (name_text, name_node, role_capture) else {
                continue;
            };
            if name.is_empty() {
                continue;
            }

            let range = node_to_range(&node);
            if role.starts_with("definition.") {
                let kind = SymKind::from_capture(role).unwrap_or(SymKind::Unknown);
                let symbol_id = scip_symbol_id(bundle.lang, rel_path, &name, kind);
                out.symbols.push(Symbol {
                    qualified_name: symbol_id.clone(),
                    name: name.clone(),
                    kind: kind.as_str().to_string(),
                    file: rel_path.to_string(),
                });
                out.occurrences.push(Occurrence {
                    symbol: symbol_id,
                    file: rel_path.to_string(),
                    range: Some(range),
                    role: ROLE_DEFINITION.to_string(),
                });
            } else {
                // A reference. We don't yet know which definition it binds to (that's
                // tier-1/2 resolution); store it keyed by the *bare name* so callers
                // can resolve by name equality against defs. We record a name-scoped
                // reference occurrence so `references()` returns real data.
                out.occurrences.push(Occurrence {
                    symbol: name.clone(),
                    file: rel_path.to_string(),
                    range: Some(range),
                    role: ROLE_REFERENCE.to_string(),
                });
            }
        }

        out
    }

    fn node_to_range(node: &Node) -> TextRange {
        let s = node.start_position();
        let e = node.end_position();
        TextRange {
            start_line: s.row as u32 + 1,
            start_col: s.column as u32 + 1,
            end_line: e.row as u32 + 1,
            end_col: e.column as u32 + 1,
        }
    }

    fn collect_error_spans(node: Node, out: &mut Vec<(usize, usize)>) {
        if node.is_error() || node.is_missing() {
            out.push((node.start_byte(), node.end_byte()));
            // Don't recurse into an error subtree; the span already covers it.
            return;
        }
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            collect_error_spans(child, out);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn extracts_rust_definitions_and_references() {
            let src = r#"
    pub struct Engine {
        name: String,
    }

    pub fn run_engine() {
        helper();
    }

    fn helper() {}
    "#;
            let out = parse_source("src/engine.rs", src);
            assert_eq!(out.lang, Some(LangId::Rust));
            assert!(!out.unparseable);

            // definitions present: Engine (struct), run_engine + helper (fn)
            let def_names: Vec<_> = out.symbols.iter().map(|s| s.name.as_str()).collect();
            assert!(def_names.contains(&"Engine"), "got {def_names:?}");
            assert!(def_names.contains(&"run_engine"));
            assert!(def_names.contains(&"helper"));

            // a reference to `helper` must exist (so references() is non-empty)
            let refs: Vec<_> =
                out.occurrences.iter().filter(|o| o.role == ROLE_REFERENCE).map(|o| o.symbol.as_str()).collect();
            assert!(refs.contains(&"helper"), "expected ref to helper, got {refs:?}");
        }

        #[test]
        fn scip_ids_are_stable_and_kind_scoped() {
            let id_fn = scip_symbol_id(LangId::Rust, "a.rs", "foo", SymKind::Function);
            let id_struct = scip_symbol_id(LangId::Rust, "a.rs", "Foo", SymKind::Struct);
            assert!(id_fn.ends_with("foo()."));
            assert!(id_struct.ends_with("Foo#"));
            // stable across calls
            assert_eq!(id_fn, scip_symbol_id(LangId::Rust, "a.rs", "foo", SymKind::Function));
        }

        #[test]
        fn typescript_extracts_defs_and_refs() {
            // Ordinary TS source: a function, a class with a method, and call sites.
            // The bundled grammar tags.scm would yield ZERO here (signature-only).
            let src = r#"
    function greet(name: string): string {
        return format(name);
    }

    class Greeter {
        render(): void {
            greet("world");
        }
    }

    const formatted = format("x");
    "#;
            let out = parse_source("ui.ts", src);
            assert_eq!(out.lang, Some(LangId::TypeScript));
            assert!(!out.unparseable);

            let def_names: Vec<_> = out.symbols.iter().map(|s| s.name.as_str()).collect();
            assert!(def_names.contains(&"greet"), "got defs {def_names:?}");
            assert!(def_names.contains(&"Greeter"), "got defs {def_names:?}");
            assert!(def_names.contains(&"render"), "got defs {def_names:?}");
            assert!(!out.symbols.is_empty(), "TS source must yield non-empty definitions");

            let refs: Vec<_> =
                out.occurrences.iter().filter(|o| o.role == ROLE_REFERENCE).map(|o| o.symbol.as_str()).collect();
            assert!(!refs.is_empty(), "TS source must yield non-empty references, got {refs:?}");
            assert!(refs.contains(&"greet"), "expected call ref to greet: {refs:?}");
            assert!(refs.contains(&"format"), "expected call ref to format: {refs:?}");
        }

        #[test]
        fn javascript_extracts_defs_and_refs() {
            // Plain JS (no types) routed through the TS superset grammar: a function,
            // a class + method, an arrow-fn const, and call sites.
            let src = r#"
    function add(a, b) {
        return compute(a, b);
    }

    class Calculator {
        run() {
            add(1, 2);
        }
    }

    const square = (n) => add(n, n);
    "#;
            let out = parse_source("calc.js", src);
            assert_eq!(out.lang, Some(LangId::TypeScript));
            assert!(!out.unparseable);

            let def_names: Vec<_> = out.symbols.iter().map(|s| s.name.as_str()).collect();
            assert!(def_names.contains(&"add"), "got defs {def_names:?}");
            assert!(def_names.contains(&"Calculator"), "got defs {def_names:?}");
            assert!(def_names.contains(&"run"), "got defs {def_names:?}");
            assert!(def_names.contains(&"square"), "arrow-fn const def: {def_names:?}");

            let refs: Vec<_> =
                out.occurrences.iter().filter(|o| o.role == ROLE_REFERENCE).map(|o| o.symbol.as_str()).collect();
            assert!(!refs.is_empty(), "JS source must yield non-empty references, got {refs:?}");
            assert!(refs.contains(&"compute"), "expected call ref to compute: {refs:?}");
            assert!(refs.contains(&"add"), "expected call ref to add: {refs:?}");
        }

        #[test]
        fn python_extracts_class_and_function() {
            let src = "class Widget:\n    def render(self):\n        draw()\n\ndef draw():\n    pass\n";
            let out = parse_source("ui.py", src);
            let names: Vec<_> = out.symbols.iter().map(|s| s.name.as_str()).collect();
            assert!(names.contains(&"Widget"));
            assert!(names.contains(&"draw"));
        }

        #[test]
        fn records_error_spans_for_broken_code() {
            let src = "pub fn good() {}\npub fn broken( {\n";
            let out = parse_source("x.rs", src);
            // good() still extracted despite the broken neighbor
            assert!(out.symbols.iter().any(|s| s.name == "good"));
            assert!(!out.error_spans.is_empty(), "expected an ERROR/MISSING span");
        }

        #[test]
        fn unknown_language_is_unparseable() {
            let out = parse_source("data.bin", "\u{0}\u{1}garbage");
            assert!(out.unparseable);
            assert!(out.symbols.is_empty());
        }
    }
}
#[rustfmt::skip]
pub mod query {
    //! The `CodeIndex` query facade (the surface ch.02/03/04 bind to).
    //!
    //! `InMemoryCodeIndex` keeps its public API (consumed by hawking-context and
    //! hide-backend) but is upgraded internally to use REAL tree-sitter parsing, so
    //! it now extracts both definitions and references (`references()` is no longer
    //! always-empty) and search has a symbol + lexical leg with real scoring.
    //!
    //! `SqliteCodeIndex` is the durable, index-backed implementation (FTS5 lexical +
    //! symbol/occurrence/edge schema + vectors), and `Index` is the broader ch.05
    //! query trait (`search/definition/references/repo_map/health`) with provenance
    //! and `min_generation`.

    use crate::graph::{CodeGraph, EdgeKind, Occurrence, RepoMap, RepoMapRequest, Symbol};
    use crate::parse::{self, ROLE_DEFINITION, ROLE_REFERENCE};
    use crate::semantic::{FusedHit, HybridRetriever, LegRanking, LexicalOverlapReranker, StubEmbeddingClient};
    use crate::store::SqliteStore;
    use futures::future::BoxFuture;
    use hide_core::error::Result;
    use hide_core::types::{FileSpan, TextRange};
    use parking_lot::RwLock;
    use serde::{Deserialize, Serialize};
    use std::collections::{BTreeMap, HashMap};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct SearchQuery {
        pub text: String,
        pub limit: usize,
        pub include_symbols: bool,
        pub include_lexical: bool,
        pub include_semantic: bool,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct SearchResult {
        pub span: FileSpan,
        pub title: String,
        pub snippet: String,
        pub score: f32,
        pub source: SearchResultSource,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum SearchResultSource {
        Symbol,
        Lexical,
        Semantic,
        Graph,
    }

    /// Coarse shape of a search query, used to route BEFORE retrieving (W-F2-6):
    /// route exact-symbol lookups to the precise symbol tier and only spend the
    /// fuzzy semantic leg on natural-language intent.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum QueryShape {
        /// A single symbol-like token (`Foo::bar`, `snake_case`, `camelCase`).
        ExactSymbol,
        /// One or two plain tokens — lexical territory.
        Identifier,
        /// A phrase / question — needs the hybrid (semantic) leg.
        NaturalLanguage,
    }

    fn has_internal_caps(t: &str) -> bool {
        t.chars().skip(1).any(|c| c.is_uppercase())
    }

    /// Classify a raw query string by shape (pure, deterministic).
    pub fn classify_query_shape(text: &str) -> QueryShape {
        let t = text.trim();
        let words = t.split_whitespace().count();
        if t.ends_with('?') || words >= 3 {
            return QueryShape::NaturalLanguage;
        }
        if words <= 1 {
            let symbolish = t.contains("::") || t.contains('.') || t.contains('_') || has_internal_caps(t);
            if symbolish {
                return QueryShape::ExactSymbol;
            }
        }
        QueryShape::Identifier
    }

    /// Precision rank of a result source (lower = more precise). Used to break score
    /// ties so a definition/symbol hit outranks a same-score "similar-code" semantic
    /// hit (W-F2-6).
    fn source_rank(s: SearchResultSource) -> u8 {
        match s {
            SearchResultSource::Symbol => 0,
            SearchResultSource::Lexical => 1,
            SearchResultSource::Graph => 2,
            SearchResultSource::Semantic => 3,
        }
    }

    /// Re-rank results by score (desc), breaking ties toward the more precise source
    /// so "similar function" semantic hits never displace an equally-scored
    /// definition/symbol hit (W-F2-6).
    pub fn rerank_prefer_precise(results: &mut [SearchResult]) {
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| source_rank(a.source).cmp(&source_rank(b.source)))
        });
    }

    impl SearchQuery {
        /// Build a query whose retrieval-tier flags are routed by the query's shape
        /// (W-F2-6): exact symbols skip the fuzzy legs; natural language gets the
        /// full hybrid.
        pub fn routed(text: impl Into<String>, limit: usize) -> Self {
            let text = text.into();
            let (include_symbols, include_lexical, include_semantic) = match classify_query_shape(&text) {
                QueryShape::ExactSymbol => (true, false, false),
                QueryShape::Identifier => (true, true, false),
                QueryShape::NaturalLanguage => (true, true, true),
            };
            SearchQuery { text, limit, include_symbols, include_lexical, include_semantic }
        }
    }

    #[cfg(test)]
    mod routing_tests {
        use super::*;

        #[test]
        fn classifies_query_shapes() {
            assert_eq!(classify_query_shape("CodeIndex::search"), QueryShape::ExactSymbol);
            assert_eq!(classify_query_shape("foo_bar"), QueryShape::ExactSymbol);
            assert_eq!(classify_query_shape("parseTree"), QueryShape::ExactSymbol);
            assert_eq!(classify_query_shape("parse tree"), QueryShape::Identifier);
            assert_eq!(classify_query_shape("where do we handle retries?"), QueryShape::NaturalLanguage);
            assert_eq!(classify_query_shape("how does compaction work"), QueryShape::NaturalLanguage);
        }

        #[test]
        fn routed_sets_tier_flags() {
            let exact = SearchQuery::routed("Foo::bar", 10);
            assert!(exact.include_symbols && !exact.include_lexical && !exact.include_semantic);
            let ident = SearchQuery::routed("parse tree", 10);
            assert!(ident.include_symbols && ident.include_lexical && !ident.include_semantic);
            let nl = SearchQuery::routed("how does the gate decide rollback", 10);
            assert!(nl.include_symbols && nl.include_lexical && nl.include_semantic);
        }

        #[test]
        fn precise_sources_outrank_similar_code() {
            assert!(source_rank(SearchResultSource::Symbol) < source_rank(SearchResultSource::Semantic));
            assert!(source_rank(SearchResultSource::Lexical) < source_rank(SearchResultSource::Semantic));
        }
    }

    pub trait CodeIndex: Send + Sync {
        fn search<'a>(&'a self, query: SearchQuery) -> BoxFuture<'a, Result<Vec<SearchResult>>>;
        fn definition<'a>(&'a self, symbol: &'a str) -> BoxFuture<'a, Result<Vec<Occurrence>>>;
        fn references<'a>(&'a self, symbol: &'a str) -> BoxFuture<'a, Result<Vec<Occurrence>>>;
        fn health<'a>(&'a self) -> BoxFuture<'a, Result<IndexHealth>>;
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct IndexHealth {
        pub generation: u64,
        pub indexed_files: usize,
        pub stale_files: usize,
        pub degraded: Vec<String>,
    }

    // ============================================================================
    // The broader ch.05 Index trait (§4.11): search/definition/references/repo_map/
    // health with provenance + min_generation. Additive over CodeIndex.
    // ============================================================================

    /// Per-query knobs: freshness gate + precise-only filter.
    #[derive(Debug, Clone, Default)]
    pub struct Q {
        /// Freshness gate. When set, a query errors with [`HideError::InvalidState`]
        /// unless the index is committed at >= this generation (a stale read is
        /// refused rather than silently served from an old generation).
        pub min_generation: Option<u64>,
        /// Exact-symbol-only resolution. When `true`, `definition_q`/`references_q`
        /// resolve ONLY the requested symbol id or its exact bare name — the fuzzy
        /// bare-name expansion (which can pull in same-named symbols across files) is
        /// suppressed, so a hit is a precise match for the requested symbol.
        pub precise: bool,
    }

    impl Q {
        /// Apply the freshness gate against the index's current committed generation.
        /// Returns `Err(InvalidState)` if `min_generation` is set and `current` is
        /// behind it; `Ok(())` otherwise (including when no gate is requested).
        pub fn check_fresh(&self, current: u64) -> Result<()> {
            if let Some(min) = self.min_generation {
                if current < min {
                    return Err(hide_core::HideError::InvalidState(format!(
                        "index generation {current} is behind requested min_generation {min}"
                    )));
                }
            }
            Ok(())
        }
    }

    pub trait Index: CodeIndex {
        /// Token-budgeted repo-map (the structural leg of ch.04).
        fn repo_map<'a>(&'a self, req: RepoMapRequest) -> BoxFuture<'a, Result<RepoMap>>;

        /// The committed generation this index can answer at (freshness anchor).
        fn current_generation(&self) -> u64;

        /// Definition lookup honoring the per-query [`Q`] knobs: freshness gate +
        /// precise (exact-symbol-only) resolution.
        fn definition_q<'a>(&'a self, symbol: &'a str, q: Q) -> BoxFuture<'a, Result<Vec<Occurrence>>>;

        /// Reference lookup honoring the per-query [`Q`] knobs.
        fn references_q<'a>(&'a self, symbol: &'a str, q: Q) -> BoxFuture<'a, Result<Vec<Occurrence>>>;
    }

    // ============================================================================
    // InMemoryCodeIndex — preserved public API, real tree-sitter internals.
    // ============================================================================

    #[derive(Default)]
    pub struct InMemoryCodeIndex {
        generation: RwLock<u64>,
        symbols: RwLock<BTreeMap<String, Symbol>>,
        occurrences: RwLock<BTreeMap<String, Vec<Occurrence>>>,
        files: RwLock<BTreeMap<PathBuf, IndexedFile>>,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct IndexedFile {
        pub path: PathBuf,
        pub content: String,
        pub content_hash: Option<String>,
    }

    impl InMemoryCodeIndex {
        pub fn add_symbol(&self, symbol: Symbol) {
            *self.generation.write() += 1;
            self.symbols.write().insert(symbol.qualified_name.clone(), symbol);
        }

        pub fn add_occurrence(&self, occurrence: Occurrence) {
            *self.generation.write() += 1;
            self.occurrences.write().entry(occurrence.symbol.clone()).or_default().push(occurrence);
        }

        pub fn add_text_file(
            &self,
            path: impl Into<PathBuf>,
            content: impl Into<String>,
            content_hash: Option<String>,
        ) {
            let path = path.into();
            let content = content.into();
            self.index_parsed_symbols(&path, &content);
            self.files.write().insert(path.clone(), IndexedFile { path, content, content_hash });
            *self.generation.write() += 1;
        }

        pub fn index_path(&self, path: impl AsRef<Path>) -> Result<()> {
            let path = path.as_ref();
            let content = std::fs::read_to_string(path)?;
            self.add_text_file(path.to_path_buf(), content, None);
            Ok(())
        }

        /// REAL tree-sitter extraction (replaces the old `simple_definition` prefix
        /// scanner). Produces both definition and reference occurrences with SCIP ids.
        fn index_parsed_symbols(&self, path: &Path, content: &str) {
            let rel = path.to_string_lossy().to_string();
            let parsed = parse::parse_source(&rel, content);

            for symbol in parsed.symbols {
                self.add_symbol(symbol);
            }
            for occ in parsed.occurrences {
                self.add_occurrence(occ);
            }

            // Unknown languages produce no AST; fall back to a lightweight identifier
            // sweep so grep-style symbol search still has something. (Lexical search
            // already covers raw content.)
            if parsed.unparseable {
                // nothing structural to add; lexical leg covers it.
            }
        }
    }

    impl CodeIndex for InMemoryCodeIndex {
        fn search<'a>(&'a self, query: SearchQuery) -> BoxFuture<'a, Result<Vec<SearchResult>>> {
            Box::pin(async move {
                let mut results = Vec::new();
                if query.include_symbols {
                    let needle = query.text.to_lowercase();
                    for symbol in self.symbols.read().values() {
                        if symbol.qualified_name.to_lowercase().contains(&needle)
                            || symbol.name.to_lowercase().contains(&needle)
                        {
                            // exact-name match scores higher than substring
                            let score = if symbol.name.to_lowercase() == needle { 2.0 } else { 1.2 };
                            results.push(SearchResult {
                                span: FileSpan {
                                    path: PathBuf::from(&symbol.file),
                                    range: def_range(&self.occurrences.read(), &symbol.qualified_name),
                                    content_hash: None,
                                },
                                title: symbol.qualified_name.clone(),
                                snippet: symbol.kind.clone(),
                                score,
                                source: SearchResultSource::Symbol,
                            });
                        }
                    }
                }
                if query.include_lexical && !query.text.trim().is_empty() {
                    let needle = query.text.to_lowercase();
                    for file in self.files.read().values() {
                        for (idx, line) in file.content.lines().enumerate() {
                            let lower = line.to_lowercase();
                            if let Some(col) = lower.find(&needle) {
                                let score = lexical_score(&lower, &needle);
                                results.push(SearchResult {
                                    span: FileSpan {
                                        path: file.path.clone(),
                                        range: Some(TextRange {
                                            start_line: idx as u32 + 1,
                                            start_col: col as u32 + 1,
                                            end_line: idx as u32 + 1,
                                            end_col: (col + query.text.len()) as u32 + 1,
                                        }),
                                        content_hash: file.content_hash.clone(),
                                    },
                                    title: file.path.display().to_string(),
                                    snippet: line.trim().to_string(),
                                    score,
                                    source: SearchResultSource::Lexical,
                                });
                            }
                        }
                    }
                }
                results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
                results.truncate(query.limit);
                Ok(results)
            })
        }

        fn definition<'a>(&'a self, symbol: &'a str) -> BoxFuture<'a, Result<Vec<Occurrence>>> {
            Box::pin(async move { Ok(self.lookup_occurrences(symbol, ROLE_DEFINITION)) })
        }

        fn references<'a>(&'a self, symbol: &'a str) -> BoxFuture<'a, Result<Vec<Occurrence>>> {
            Box::pin(async move { Ok(self.lookup_occurrences(symbol, ROLE_REFERENCE)) })
        }

        fn health<'a>(&'a self) -> BoxFuture<'a, Result<IndexHealth>> {
            Box::pin(async move {
                Ok(IndexHealth {
                    generation: *self.generation.read(),
                    indexed_files: self.files.read().len(),
                    stale_files: 0,
                    degraded: Vec::new(),
                })
            })
        }
    }

    impl InMemoryCodeIndex {
        /// Resolve occurrences for a symbol by id OR bare name and a given role.
        fn lookup_occurrences(&self, symbol: &str, role: &str) -> Vec<Occurrence> {
            let occs = self.occurrences.read();
            // Direct id key.
            let mut out: Vec<Occurrence> =
                occs.get(symbol).cloned().unwrap_or_default().into_iter().filter(|o| o.role == role).collect();

            // References are keyed by bare name; defs by SCIP id. Resolve the
            // alternate key: if `symbol` looks like a SCIP id, also try its bare name;
            // if it's a bare name, also try matching def symbol ids that end in it.
            let bare = bare_name(symbol);
            if bare != symbol {
                if let Some(extra) = occs.get(&bare) {
                    out.extend(extra.iter().filter(|o| o.role == role).cloned());
                }
            } else if role == ROLE_DEFINITION {
                // bare-name def lookup: scan symbol ids whose display name == symbol
                for sym in self.symbols.read().values() {
                    if sym.name == symbol {
                        if let Some(extra) = occs.get(&sym.qualified_name) {
                            out.extend(extra.iter().filter(|o| o.role == role).cloned());
                        }
                    }
                }
            }
            out
        }

        /// Precise lookup for the `Q.precise` path: resolve ONLY the exact key
        /// (`symbol` as stored), with no bare-name / cross-file expansion.
        fn lookup_occurrences_exact(&self, symbol: &str, role: &str) -> Vec<Occurrence> {
            self.occurrences
                .read()
                .get(symbol)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .filter(|o| o.role == role)
                .collect()
        }
    }

    fn def_range(occs: &BTreeMap<String, Vec<Occurrence>>, symbol_id: &str) -> Option<TextRange> {
        occs.get(symbol_id)?.iter().find(|o| o.role == ROLE_DEFINITION).and_then(|o| o.range.clone())
    }

    /// Best-effort bare name out of a SCIP id (or pass through if already bare).
    fn bare_name(symbol: &str) -> String {
        if let Some(last) = symbol.rsplit(' ').next() {
            let trimmed = last.trim_end_matches("().").trim_end_matches(['#', '.', '!', '/']);
            if trimmed != symbol && !trimmed.is_empty() {
                return trimmed.to_string();
            }
        }
        symbol.to_string()
    }

    fn lexical_score(line: &str, needle: &str) -> f32 {
        let occurrences = line.matches(needle).count().max(1) as f32;
        let density = needle.len() as f32 / line.len().max(needle.len()) as f32;
        0.5 + occurrences.min(5.0) * 0.1 + density.min(0.4)
    }

    // ============================================================================
    // SqliteCodeIndex — durable, index-backed implementation.
    // ============================================================================

    /// A durable code index backed by SQLite/FTS5 + the unified graph + vectors.
    ///
    /// Implements the full `Index` trait. Indexing goes through `index_text` (the
    /// daemon and a one-shot bootstrap both call it). Search runs the real hybrid
    /// pipeline: symbol leg + FTS5 lexical leg + vector leg → RRF → rerank.
    pub struct SqliteCodeIndex {
        store: Arc<SqliteStore>,
        graph: RwLock<CodeGraph>,
        generation: RwLock<u64>,
    }

    impl SqliteCodeIndex {
        pub fn open(path: impl AsRef<Path>) -> Result<Self> {
            let store = Arc::new(SqliteStore::open(path)?);
            Ok(Self::with_store(store))
        }

        pub fn open_in_memory() -> Result<Self> {
            let store = Arc::new(SqliteStore::open_in_memory()?);
            Ok(Self::with_store(store))
        }

        pub fn with_store(store: Arc<SqliteStore>) -> Self {
            let gen = store.last_committed_generation().unwrap_or(0);
            Self { store, graph: RwLock::new(CodeGraph::new()), generation: RwLock::new(gen) }
        }

        pub fn store(&self) -> Arc<SqliteStore> {
            self.store.clone()
        }

        pub fn generation(&self) -> u64 {
            *self.generation.read()
        }

        /// Index a single file's text at the next generation: parse, chunk, persist,
        /// and fold call edges (reference → enclosing-file definitions) into the graph.
        pub fn index_text(&self, rel_path: &str, content: &str, content_hash: &str) -> Result<u64> {
            let gen = {
                let mut g = self.generation.write();
                *g += 1;
                *g
            };
            self.store.begin_generation(gen, content_hash)?;

            let parsed = parse::parse_source(rel_path, content);
            let chunks = parse::chunk_file(rel_path, content);
            let lang = parsed.lang.map(|l| l.as_str()).unwrap_or("unknown");
            let parse_state = if parsed.unparseable {
                "unparseable"
            } else if parsed.error_spans.is_empty() {
                "ok"
            } else {
                "errors"
            };

            self.store.upsert_file(
                rel_path,
                lang,
                content_hash,
                parse_state,
                content,
                &parsed.symbols,
                &parsed.occurrences,
                &chunks,
                gen,
            )?;

            // Graph: register defs, and add Calls edges from each reference's
            // enclosing file to the matching definition (name equality) — a real
            // (approximate, tier-0) call graph.
            {
                let mut g = self.graph.write();
                let def_by_name: HashMap<String, String> =
                    parsed.symbols.iter().map(|s| (s.name.clone(), s.qualified_name.clone())).collect();
                for s in &parsed.symbols {
                    let sig = first_line_for(content, &parsed.occurrences, &s.qualified_name);
                    g.add_definition(&s.qualified_name, &s.name, &s.file, &sig);
                }
                for occ in &parsed.occurrences {
                    if occ.role == ROLE_REFERENCE {
                        if let Some(def_id) = def_by_name.get(&occ.symbol) {
                            // intra-file call edge: def → def (caller unknown at tier-0,
                            // so we credit references into the callee for ranking)
                            g.add_edge(rel_path, def_id, EdgeKind::Calls, 1.0);
                        }
                    }
                }
            }

            self.store.commit_generation(gen)?;
            Ok(gen)
        }

        /// Cross-crate edge ingestion (call/import/etc.), persisted + materialized.
        pub fn add_edge(&self, src: &str, dst: &str, kind: EdgeKind, weight: f32) -> Result<()> {
            let gen = *self.generation.read();
            self.store.add_edge(src, dst, kind, weight, gen)?;
            self.graph.write().add_edge(src, dst, kind, weight);
            Ok(())
        }

        /// Reverse call graph: who calls X (an index seek over materialized reverse
        /// edges). Used for blast-radius.
        pub fn callers_of(&self, symbol_id: &str) -> Result<Vec<String>> {
            Ok(self.store.in_edges(symbol_id, EdgeKind::Calls)?.into_iter().map(|(src, _)| src).collect())
        }

        /// Full hybrid search (symbol ⊕ lexical ⊕ vector → RRF → rerank). Uses a stub
        /// embedder by default so it works offline; swap via `search_with_embedder`.
        /// The vector (semantic) leg always runs here; use [`hybrid_search_opts`] to
        /// disable it.
        ///
        /// [`hybrid_search_opts`]: SqliteCodeIndex::hybrid_search_opts
        pub async fn hybrid_search(&self, query: &str, k_final: usize) -> Result<Vec<FusedHit>> {
            self.hybrid_search_opts(query, k_final, true).await
        }

        /// Hybrid search with an explicit `include_semantic` toggle. When `false` the
        /// vector leg (and the embedder) is skipped entirely.
        pub async fn hybrid_search_opts(
            &self,
            query: &str,
            k_final: usize,
            include_semantic: bool,
        ) -> Result<Vec<FusedHit>> {
            let embedder = StubEmbeddingClient::default();
            self.search_with_embedder(query, k_final, &embedder, include_semantic).await
        }

        pub async fn search_with_embedder<E: crate::semantic::EmbeddingClient>(
            &self,
            query: &str,
            k_final: usize,
            embedder: &E,
            include_semantic: bool,
        ) -> Result<Vec<FusedHit>> {
            // Leg B: FTS5 lexical.
            let lex_hits = self.store.lexical_search(query, 50)?;
            let mut snippets: HashMap<String, FusedHit> = HashMap::new();
            let lex_keys: Vec<String> = lex_hits
                .iter()
                .map(|h| {
                    let key = format!("{}:1", h.path);
                    snippets.entry(key.clone()).or_insert_with(|| FusedHit {
                        file: h.path.clone(),
                        start_line: 1,
                        end_line: 1,
                        snippet: first_n_lines(&h.body, 3),
                        score: 0.0,
                        legs: vec!["lexical".into()],
                    });
                    key
                })
                .collect();

            // Leg A: symbol.
            let sym_hits = self.store.symbol_search(query, 50)?;
            let sym_keys: Vec<String> = sym_hits
                .iter()
                .map(|s| {
                    let key = format!("{}:1", s.file);
                    snippets.entry(key.clone()).or_insert_with(|| FusedHit {
                        file: s.file.clone(),
                        start_line: 1,
                        end_line: 1,
                        snippet: format!("{} {}", s.kind, s.name),
                        score: 0.0,
                        legs: vec!["symbol".into()],
                    });
                    key
                })
                .collect();

            let weights = crate::semantic::HybridRetrievalWeights::default();
            let lexical = LegRanking { name: "lexical".into(), weight: weights.lexical, ranked_keys: lex_keys };
            let symbol = LegRanking { name: "symbol".into(), weight: weights.symbol, ranked_keys: sym_keys };

            let retriever = HybridRetriever::new(&self.store, embedder);
            retriever
                .search_with_legs(query, lexical, symbol, &snippets, &LexicalOverlapReranker, k_final, include_semantic)
                .await
        }
    }

    fn first_line_for(content: &str, occs: &[Occurrence], symbol_id: &str) -> String {
        if let Some(occ) = occs.iter().find(|o| o.symbol == symbol_id && o.role == ROLE_DEFINITION) {
            if let Some(range) = &occ.range {
                if let Some(line) = content.lines().nth((range.start_line.saturating_sub(1)) as usize) {
                    return line.trim().to_string();
                }
            }
        }
        String::new()
    }

    fn first_n_lines(s: &str, n: usize) -> String {
        s.lines().take(n).collect::<Vec<_>>().join("\n")
    }

    impl CodeIndex for SqliteCodeIndex {
        fn search<'a>(&'a self, query: SearchQuery) -> BoxFuture<'a, Result<Vec<SearchResult>>> {
            Box::pin(async move {
                // Honor `include_semantic`: when false, the vector leg is skipped.
                let hits = self.hybrid_search_opts(&query.text, query.limit, query.include_semantic).await?;
                Ok(hits
                    .into_iter()
                    .map(|h| SearchResult {
                        span: FileSpan {
                            path: PathBuf::from(&h.file),
                            range: Some(TextRange {
                                start_line: h.start_line,
                                start_col: 1,
                                end_line: h.end_line,
                                end_col: 1,
                            }),
                            content_hash: None,
                        },
                        title: h.file.clone(),
                        snippet: h.snippet,
                        score: h.score,
                        source: if h.legs.iter().any(|l| l == "symbol") {
                            SearchResultSource::Symbol
                        } else {
                            SearchResultSource::Lexical
                        },
                    })
                    .collect())
            })
        }

        fn definition<'a>(&'a self, symbol: &'a str) -> BoxFuture<'a, Result<Vec<Occurrence>>> {
            Box::pin(async move { self.store.definitions(symbol) })
        }

        fn references<'a>(&'a self, symbol: &'a str) -> BoxFuture<'a, Result<Vec<Occurrence>>> {
            Box::pin(async move { self.store.references(symbol) })
        }

        fn health<'a>(&'a self) -> BoxFuture<'a, Result<IndexHealth>> {
            Box::pin(async move {
                let degraded = if self.store.unparseable_count()? > 0 {
                    vec!["unparseable_files".to_string()]
                } else {
                    Vec::new()
                };
                Ok(IndexHealth {
                    generation: *self.generation.read(),
                    indexed_files: self.store.file_count()?,
                    stale_files: 0,
                    degraded,
                })
            })
        }
    }

    impl Index for SqliteCodeIndex {
        fn repo_map<'a>(&'a self, req: RepoMapRequest) -> BoxFuture<'a, Result<RepoMap>> {
            Box::pin(async move { Ok(self.graph.read().repo_map(&req)) })
        }

        fn current_generation(&self) -> u64 {
            *self.generation.read()
        }

        fn definition_q<'a>(&'a self, symbol: &'a str, q: Q) -> BoxFuture<'a, Result<Vec<Occurrence>>> {
            Box::pin(async move {
                q.check_fresh(self.current_generation())?;
                if q.precise {
                    self.store.definitions_exact(symbol)
                } else {
                    self.store.definitions(symbol)
                }
            })
        }

        fn references_q<'a>(&'a self, symbol: &'a str, q: Q) -> BoxFuture<'a, Result<Vec<Occurrence>>> {
            Box::pin(async move {
                q.check_fresh(self.current_generation())?;
                if q.precise {
                    self.store.references_exact(symbol)
                } else {
                    self.store.references(symbol)
                }
            })
        }
    }

    impl Index for InMemoryCodeIndex {
        fn repo_map<'a>(&'a self, req: RepoMapRequest) -> BoxFuture<'a, Result<RepoMap>> {
            Box::pin(async move {
                let mut g = CodeGraph::new();
                for sym in self.symbols.read().values() {
                    g.add_definition(&sym.qualified_name, &sym.name, &sym.file, &sym.name);
                }
                // reference → callee edges for ranking
                let symbols = self.symbols.read();
                let by_name: HashMap<String, String> =
                    symbols.values().map(|s| (s.name.clone(), s.qualified_name.clone())).collect();
                for occ_list in self.occurrences.read().values() {
                    for occ in occ_list {
                        if occ.role == ROLE_REFERENCE {
                            if let Some(def_id) = by_name.get(&occ.symbol) {
                                g.add_edge(&occ.file, def_id, EdgeKind::Calls, 1.0);
                            }
                        }
                    }
                }
                Ok(g.repo_map(&req))
            })
        }

        fn current_generation(&self) -> u64 {
            *self.generation.read()
        }

        fn definition_q<'a>(&'a self, symbol: &'a str, q: Q) -> BoxFuture<'a, Result<Vec<Occurrence>>> {
            Box::pin(async move {
                q.check_fresh(self.current_generation())?;
                Ok(if q.precise {
                    self.lookup_occurrences_exact(symbol, ROLE_DEFINITION)
                } else {
                    self.lookup_occurrences(symbol, ROLE_DEFINITION)
                })
            })
        }

        fn references_q<'a>(&'a self, symbol: &'a str, q: Q) -> BoxFuture<'a, Result<Vec<Occurrence>>> {
            Box::pin(async move {
                q.check_fresh(self.current_generation())?;
                Ok(if q.precise {
                    self.lookup_occurrences_exact(symbol, ROLE_REFERENCE)
                } else {
                    self.lookup_occurrences(symbol, ROLE_REFERENCE)
                })
            })
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::graph::Symbol;

        #[tokio::test]
        async fn search_finds_registered_symbol() {
            let index = InMemoryCodeIndex::default();
            index.add_symbol(Symbol {
                qualified_name: "crate::engine::Engine".to_string(),
                name: "Engine".to_string(),
                kind: "trait".to_string(),
                file: "src/engine.rs".to_string(),
            });
            let results = index
                .search(SearchQuery {
                    text: "Engine".to_string(),
                    limit: 5,
                    include_symbols: true,
                    include_lexical: false,
                    include_semantic: false,
                })
                .await
                .unwrap();
            assert_eq!(results.len(), 1);
        }

        #[tokio::test]
        async fn in_memory_extracts_real_references() {
            let index = InMemoryCodeIndex::default();
            index.add_text_file(
                "src/lib.rs",
                "pub fn helper() {}\npub fn caller() { helper(); }\n",
                Some("hash".to_string()),
            );
            // references(helper) must be non-empty now (was always empty before).
            let refs = index.references("helper").await.unwrap();
            assert!(!refs.is_empty(), "expected real references to helper");
            let defs = index.definition("helper").await.unwrap();
            assert!(!defs.is_empty(), "expected a definition of helper");
        }

        #[tokio::test]
        async fn index_text_file_supports_lexical_search_and_symbols() {
            let index = InMemoryCodeIndex::default();
            index.add_text_file(
                "src/lib.rs",
                "pub struct SearchEngine {}\nimpl SearchEngine { fn query(&self) {} }\n",
                Some("hash".to_string()),
            );
            let lexical = index
                .search(SearchQuery {
                    text: "SearchEngine".to_string(),
                    limit: 5,
                    include_symbols: false,
                    include_lexical: true,
                    include_semantic: false,
                })
                .await
                .unwrap();
            assert_eq!(lexical.len(), 2);
            assert_eq!(lexical[0].source, SearchResultSource::Lexical);

            let symbols = index
                .search(SearchQuery {
                    text: "SearchEngine".to_string(),
                    limit: 5,
                    include_symbols: true,
                    include_lexical: false,
                    include_semantic: false,
                })
                .await
                .unwrap();
            assert_eq!(symbols.len(), 1);
            assert_eq!(index.health().await.unwrap().indexed_files, 1);
        }

        #[tokio::test]
        async fn sqlite_index_search_and_nav() {
            let index = SqliteCodeIndex::open_in_memory().unwrap();
            index.index_text("src/m.rs", "pub fn target_widget() { helper(); }\nfn helper() {}\n", "hash1").unwrap();

            let defs = index.definition("target_widget").await.unwrap();
            assert!(!defs.is_empty());
            let refs = index.references("helper").await.unwrap();
            assert!(!refs.is_empty());

            let hits = index
                .search(SearchQuery {
                    text: "target_widget".to_string(),
                    limit: 5,
                    include_symbols: true,
                    include_lexical: true,
                    include_semantic: false,
                })
                .await
                .unwrap();
            assert!(hits.iter().any(|h| h.span.path.ends_with("src/m.rs")));
            assert_eq!(index.health().await.unwrap().indexed_files, 1);
        }

        #[tokio::test]
        async fn sqlite_repo_map_renders() {
            let index = SqliteCodeIndex::open_in_memory().unwrap();
            index
                .index_text(
                    "src/m.rs",
                    "pub fn popular_api() {}\npub fn user_a() { popular_api(); }\npub fn user_b() { popular_api(); }\n",
                    "h",
                )
                .unwrap();
            let rm = index
                .repo_map(RepoMapRequest {
                    mentioned_files: vec![],
                    mentioned_idents: vec!["popular_api".to_string()],
                    max_tokens: 300,
                })
                .await
                .unwrap();
            assert!(rm.rendered.contains("popular_api"));
        }
    }
}
#[rustfmt::skip]
pub mod semantic {
    //! The semantic index: hybrid retrieval (lexical ⊕ symbol ⊕ vector) → RRF →
    //! rerank (bible §4.7).
    //!
    //! - `EmbeddingClient` is a swappable trait; `HttpEmbeddingClient` talks to
    //!   `hawking-serve` `POST /v1/embeddings`; tests use `StubEmbeddingClient`.
    //! - Vectors are stored as f32 in SQLite (see `store`); recall is **cosine over
    //!   stored vectors** (no heavy ANN dep), which is exact and fits an IDE shard.
    //! - `reciprocal_rank_fusion` + a rerank actually run in `HybridRetriever::search`
    //!   (RRF is no longer dead).

    use crate::store::SqliteStore;
    use futures::future::BoxFuture;
    use hide_core::{HideError, Result};
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct EmbeddingRecord {
        pub chunk_id: String,
        pub model_id: String,
        pub dim: usize,
        pub vector: Vec<f32>,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct HybridRetrievalWeights {
        pub lexical: f32,
        pub symbol: f32,
        pub semantic: f32,
        pub graph: f32,
    }

    impl Default for HybridRetrievalWeights {
        fn default() -> Self {
            // Per ground truth + SOTA: lexical/symbol carry recall; vector re-ranks
            // (low weight today because embed() is a logits proxy).
            Self { lexical: 1.0, symbol: 1.0, semantic: 0.3, graph: 0.75 }
        }
    }

    /// `RRF(d) = Σ 1/(k + rank)`, k=60 (Cormack 2009; the Elasticsearch default).
    pub fn reciprocal_rank_fusion(ranks: &[usize], k: f32) -> f32 {
        ranks.iter().map(|rank| 1.0 / (k + *rank as f32)).sum()
    }

    pub const RRF_K: f32 = 60.0;

    /// A swappable embedding client (the live runtime is NOT up during tests).
    pub trait EmbeddingClient: Send + Sync {
        /// Embed a batch of texts. Returns one vector per input.
        fn embed<'a>(&'a self, texts: Vec<String>) -> BoxFuture<'a, Result<Vec<Vec<f32>>>>;
        /// The model id stamped on stored vectors (for versioning / lazy re-embed).
        fn model_id(&self) -> String;
    }

    /// HTTP client to `hawking-serve` `POST /v1/embeddings` (OpenAI-shaped).
    pub struct HttpEmbeddingClient {
        base_url: String,
        model_id: String,
        client: reqwest::Client,
    }

    impl HttpEmbeddingClient {
        pub fn new(base_url: impl Into<String>) -> Self {
            Self {
                base_url: base_url.into(),
                model_id: "logits-proxy:default".to_string(),
                client: reqwest::Client::new(),
            }
        }

        pub fn with_model_id(mut self, id: impl Into<String>) -> Self {
            self.model_id = id.into();
            self
        }
    }

    #[derive(Serialize)]
    struct EmbeddingsRequest {
        input: Vec<String>,
        encoding_format: &'static str,
    }

    #[derive(Deserialize)]
    struct EmbeddingsResponse {
        data: Vec<EmbeddingDatum>,
    }

    #[derive(Deserialize)]
    struct EmbeddingDatum {
        embedding: Vec<f32>,
    }

    impl EmbeddingClient for HttpEmbeddingClient {
        fn embed<'a>(&'a self, texts: Vec<String>) -> BoxFuture<'a, Result<Vec<Vec<f32>>>> {
            Box::pin(async move {
                if texts.is_empty() {
                    return Ok(Vec::new());
                }
                let url = format!("{}/v1/embeddings", self.base_url.trim_end_matches('/'));
                let resp = self
                    .client
                    .post(&url)
                    .json(&EmbeddingsRequest { input: texts, encoding_format: "float" })
                    .send()
                    .await
                    .map_err(|e| HideError::RuntimeUnavailable(format!("embeddings request: {e}")))?;
                if !resp.status().is_success() {
                    return Err(HideError::RuntimeUnavailable(format!("embeddings status {}", resp.status())));
                }
                let body: EmbeddingsResponse =
                    resp.json().await.map_err(|e| HideError::RuntimeUnavailable(format!("embeddings decode: {e}")))?;
                Ok(body.data.into_iter().map(|d| d.embedding).collect())
            })
        }

        fn model_id(&self) -> String {
            self.model_id.clone()
        }
    }

    /// A deterministic stub embedding client for tests (no live runtime).
    ///
    /// Produces a small bag-of-chars vector so cosine similarity is meaningful for
    /// tests without any network.
    pub struct StubEmbeddingClient {
        pub model_id: String,
        pub dim: usize,
    }

    impl Default for StubEmbeddingClient {
        fn default() -> Self {
            Self { model_id: "stub-embed:test".to_string(), dim: 32 }
        }
    }

    impl EmbeddingClient for StubEmbeddingClient {
        fn embed<'a>(&'a self, texts: Vec<String>) -> BoxFuture<'a, Result<Vec<Vec<f32>>>> {
            let dim = self.dim;
            Box::pin(async move { Ok(texts.iter().map(|t| bag_of_chars(t, dim)).collect()) })
        }
        fn model_id(&self) -> String {
            self.model_id.clone()
        }
    }

    fn bag_of_chars(text: &str, dim: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        for b in text.bytes() {
            v[(b as usize) % dim] += 1.0;
        }
        l2_normalize(&mut v);
        v
    }

    pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let mut dot = 0.0f32;
        let mut na = 0.0f32;
        let mut nb = 0.0f32;
        for i in 0..a.len() {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        let denom = (na.sqrt() * nb.sqrt()).max(1e-8);
        dot / denom
    }

    fn l2_normalize(v: &mut [f32]) {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        for x in v.iter_mut() {
            *x /= norm;
        }
    }

    /// One fused, reranked result.
    #[derive(Debug, Clone, PartialEq)]
    pub struct FusedHit {
        pub file: String,
        pub start_line: u32,
        pub end_line: u32,
        pub snippet: String,
        /// Combined RRF score (post-rerank ordering preserved in returned order).
        pub score: f32,
        pub legs: Vec<String>,
    }

    /// A leg's ranked output: ordered keys (a key identifies a candidate location).
    #[derive(Debug, Clone, Default)]
    pub struct LegRanking {
        pub name: String,
        pub weight: f32,
        /// Ordered candidate keys (rank 0 = best). Key = "file:start_line".
        pub ranked_keys: Vec<String>,
    }

    /// Fuse multiple leg rankings via weighted RRF.
    ///
    /// Returns candidate keys sorted by fused score descending.
    pub fn fuse_legs(legs: &[LegRanking], k: f32) -> Vec<(String, f32)> {
        use std::collections::HashMap;
        let mut scores: HashMap<String, f32> = HashMap::new();
        for leg in legs {
            for (rank, key) in leg.ranked_keys.iter().enumerate() {
                *scores.entry(key.clone()).or_insert(0.0) += leg.weight * (1.0 / (k + rank as f32));
            }
        }
        let mut out: Vec<(String, f32)> = scores.into_iter().collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
        out
    }

    /// A reranker over fused candidates. Today: a lexical-overlap precision pass
    /// (a free, deterministic stand-in for the bible's local-LLM listwise rerank;
    /// the LLM rerank slots in behind this same boundary).
    pub trait Reranker: Send + Sync {
        fn rerank(&self, query: &str, candidates: Vec<FusedHit>) -> Vec<FusedHit>;
    }

    pub struct LexicalOverlapReranker;

    impl Reranker for LexicalOverlapReranker {
        fn rerank(&self, query: &str, mut candidates: Vec<FusedHit>) -> Vec<FusedHit> {
            let q_terms: Vec<String> = query
                .split(|c: char| !c.is_alphanumeric())
                .filter(|t| !t.is_empty())
                .map(|t| t.to_lowercase())
                .collect();
            for c in candidates.iter_mut() {
                let snip = c.snippet.to_lowercase();
                let overlap = q_terms.iter().filter(|t| snip.contains(t.as_str())).count() as f32;
                // blend RRF score with overlap (rerank boosts precise term matches)
                c.score = c.score * 0.5 + overlap * 0.1;
            }
            candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            candidates
        }
    }

    /// The hybrid retriever: runs the vector leg (cosine over stored vectors),
    /// fuses with externally-supplied lexical/symbol legs via RRF, then reranks.
    pub struct HybridRetriever<'a, E: EmbeddingClient> {
        store: &'a SqliteStore,
        embedder: &'a E,
        weights: HybridRetrievalWeights,
    }

    impl<'a, E: EmbeddingClient> HybridRetriever<'a, E> {
        pub fn new(store: &'a SqliteStore, embedder: &'a E) -> Self {
            Self { store, embedder, weights: HybridRetrievalWeights::default() }
        }

        pub fn with_weights(mut self, weights: HybridRetrievalWeights) -> Self {
            self.weights = weights;
            self
        }

        /// The vector leg: embed the query, cosine over stored vectors, return the
        /// top candidate keys ordered by similarity.
        pub async fn vector_leg(&self, query: &str, k: usize) -> Result<LegRanking> {
            let qvec = self.embedder.embed(vec![query.to_string()]).await?;
            let qvec = match qvec.into_iter().next() {
                Some(v) => v,
                None => return Ok(LegRanking::default()),
            };
            let mut scored: Vec<(String, f32, String)> = self
                .store
                .all_vectors()?
                .into_iter()
                .map(|sv| {
                    let key = format!("{}:{}", sv.file, sv.start_line);
                    let sim = cosine(&qvec, &sv.vector);
                    (key, sim, sv.file)
                })
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(k);
            Ok(LegRanking {
                name: "vector".to_string(),
                weight: self.weights.semantic,
                ranked_keys: scored.into_iter().map(|(key, _, _)| key).collect(),
            })
        }

        /// Full search: vector leg + caller-supplied lexical & symbol legs → RRF →
        /// rerank. The caller passes lexical/symbol rankings (from `store`) so this
        /// stays decoupled from how those legs are produced.
        ///
        /// The vector (semantic) leg always runs; use [`search_with_legs`] to skip it.
        ///
        /// [`search_with_legs`]: HybridRetriever::search_with_legs
        pub async fn search(
            &self,
            query: &str,
            lexical: LegRanking,
            symbol: LegRanking,
            snippets: &std::collections::HashMap<String, FusedHit>,
            reranker: &dyn Reranker,
            k_final: usize,
        ) -> Result<Vec<FusedHit>> {
            self.search_with_legs(query, lexical, symbol, snippets, reranker, k_final, true).await
        }

        /// As [`search`](HybridRetriever::search), but `include_semantic` toggles the
        /// vector leg. When `false` the embedder is never invoked (no `embed()` call,
        /// no cosine pass) — the result is a pure lexical⊕symbol fusion. This is what
        /// lets `include_semantic` on a query actually turn the vector leg off.
        #[allow(clippy::too_many_arguments)]
        pub async fn search_with_legs(
            &self,
            query: &str,
            lexical: LegRanking,
            symbol: LegRanking,
            snippets: &std::collections::HashMap<String, FusedHit>,
            reranker: &dyn Reranker,
            k_final: usize,
            include_semantic: bool,
        ) -> Result<Vec<FusedHit>> {
            let mut legs = vec![lexical, symbol];
            if include_semantic {
                legs.push(self.vector_leg(query, 50).await?);
            }
            let fused = fuse_legs(&legs, RRF_K);

            let mut hits: Vec<FusedHit> = Vec::new();
            for (key, score) in fused.into_iter().take(50) {
                if let Some(base) = snippets.get(&key) {
                    let mut h = base.clone();
                    h.score = score;
                    hits.push(h);
                }
            }
            let reranked = reranker.rerank(query, hits);
            Ok(reranked.into_iter().take(k_final).collect())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::collections::HashMap;

        #[test]
        fn rrf_kernel_sums_reciprocals() {
            let v = reciprocal_rank_fusion(&[0, 1], 60.0);
            assert!((v - (1.0 / 60.0 + 1.0 / 61.0)).abs() < 1e-6);
        }

        #[test]
        fn fuse_legs_weights_and_orders() {
            let a = LegRanking { name: "lex".into(), weight: 1.0, ranked_keys: vec!["x:1".into(), "y:1".into()] };
            let b = LegRanking { name: "vec".into(), weight: 0.3, ranked_keys: vec!["y:1".into(), "x:1".into()] };
            let fused = fuse_legs(&[a, b], 60.0);
            // x:1 is rank0 in the heavier leg → should lead
            assert_eq!(fused[0].0, "x:1");
        }

        #[test]
        fn cosine_basic() {
            assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
            assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        }

        #[tokio::test]
        async fn vector_leg_uses_cosine_over_stored_vectors() {
            let store = SqliteStore::open_in_memory().unwrap();
            let out = crate::parse::parse_source("q.rs", "pub fn alpha() { compute(); }");
            let chunks = crate::parse::chunk_file("q.rs", "pub fn alpha() { compute(); }");
            store
                .upsert_file(
                    "q.rs",
                    "rust",
                    "h",
                    "ok",
                    "pub fn alpha() { compute(); }",
                    &out.symbols,
                    &out.occurrences,
                    &chunks,
                    1,
                )
                .unwrap();
            let embedder = StubEmbeddingClient::default();
            // embed and store the chunk vector
            let pending = store.pending_chunks(10).unwrap();
            let txt = "pub fn alpha() { compute(); }";
            let vecs = embedder.embed(vec![txt.to_string()]).await.unwrap();
            store.store_vector(&pending[0].chunk_id, &embedder.model_id(), &vecs[0]).unwrap();

            let retriever = HybridRetriever::new(&store, &embedder);
            let leg = retriever.vector_leg("compute alpha", 5).await.unwrap();
            assert!(!leg.ranked_keys.is_empty(), "vector leg must return hits");
        }

        /// An embedder that records how many times `embed()` was invoked, so a test
        /// can prove the vector leg was (or wasn't) run.
        struct CountingEmbedder {
            calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        }
        impl EmbeddingClient for CountingEmbedder {
            fn embed<'a>(&'a self, texts: Vec<String>) -> BoxFuture<'a, Result<Vec<Vec<f32>>>> {
                self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Box::pin(async move { Ok(texts.iter().map(|t| bag_of_chars(t, 16)).collect()) })
            }
            fn model_id(&self) -> String {
                "counting:test".into()
            }
        }

        #[tokio::test]
        async fn include_semantic_toggles_vector_leg() {
            use std::sync::atomic::Ordering;
            use std::sync::Arc;
            let store = SqliteStore::open_in_memory().unwrap();
            let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let embedder = CountingEmbedder { calls: calls.clone() };
            let retriever = HybridRetriever::new(&store, &embedder);

            let snippets: HashMap<String, FusedHit> = HashMap::new();
            let lexical = LegRanking { name: "lexical".into(), weight: 1.0, ranked_keys: vec![] };
            let symbol = LegRanking { name: "symbol".into(), weight: 1.0, ranked_keys: vec![] };

            // include_semantic = false → embedder must NOT be called.
            retriever
                .search_with_legs("q", lexical.clone(), symbol.clone(), &snippets, &LexicalOverlapReranker, 5, false)
                .await
                .unwrap();
            assert_eq!(calls.load(Ordering::SeqCst), 0, "vector leg ran despite include_semantic=false");

            // include_semantic = true → embedder IS called exactly once (the query embed).
            retriever
                .search_with_legs("q", lexical, symbol, &snippets, &LexicalOverlapReranker, 5, true)
                .await
                .unwrap();
            assert_eq!(calls.load(Ordering::SeqCst), 1, "vector leg should have embedded the query");
        }

        #[tokio::test]
        async fn full_search_fuses_and_reranks() {
            let store = SqliteStore::open_in_memory().unwrap();
            let embedder = StubEmbeddingClient::default();
            let retriever = HybridRetriever::new(&store, &embedder);

            let mut snippets = HashMap::new();
            snippets.insert(
                "a.rs:1".to_string(),
                FusedHit {
                    file: "a.rs".into(),
                    start_line: 1,
                    end_line: 2,
                    snippet: "fn target_function() {}".into(),
                    score: 0.0,
                    legs: vec![],
                },
            );
            let lexical = LegRanking { name: "lexical".into(), weight: 1.0, ranked_keys: vec!["a.rs:1".into()] };
            let symbol = LegRanking { name: "symbol".into(), weight: 1.0, ranked_keys: vec!["a.rs:1".into()] };
            let hits = retriever
                .search("target_function", lexical, symbol, &snippets, &LexicalOverlapReranker, 10)
                .await
                .unwrap();
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].file, "a.rs");
        }
    }
}
#[rustfmt::skip]
pub mod store {
    //! Durable index store: rusqlite (bundled) + FTS5 (bible §4.10).
    //!
    //! The relational source of truth. Holds the SCIP-shaped symbol/occurrence model,
    //! the unified graph edges WITH materialized reverse edges (so "who calls X?" is
    //! an index seek, not a scan), an FTS5 lexical index (BM25 over body+ident), and
    //! a generation/MVCC anchor. One writer (the daemon), many readers.

    use crate::graph::{EdgeKind, Occurrence, Symbol};
    use crate::parse::CodeChunk;
    use hide_core::types::TextRange;
    use hide_core::{HideError, Result};
    use rusqlite::{params, Connection, OptionalExtension};
    use std::path::Path;
    use std::sync::Arc;

    /// A durable, SQLite-backed index store.
    pub struct SqliteStore {
        conn: Arc<parking_lot::Mutex<Connection>>,
    }

    fn map_err(e: rusqlite::Error) -> HideError {
        HideError::Storage(e.to_string())
    }

    impl SqliteStore {
        /// Open (or create) a store at `path`. Use `:memory:` for tests.
        pub fn open(path: impl AsRef<Path>) -> Result<Self> {
            let conn = Connection::open(path).map_err(map_err)?;
            Self::init(conn)
        }

        pub fn open_in_memory() -> Result<Self> {
            let conn = Connection::open_in_memory().map_err(map_err)?;
            Self::init(conn)
        }

        fn init(conn: Connection) -> Result<Self> {
            // PRAGMAs per §4.10. `synchronous=NORMAL` + WAL is the IDE sweet spot.
            // (WAL on :memory: is a no-op but harmless.)
            conn.execute_batch(
                "PRAGMA journal_mode=WAL;
                 PRAGMA synchronous=NORMAL;
                 PRAGMA busy_timeout=5000;
                 PRAGMA foreign_keys=ON;
                 PRAGMA temp_store=MEMORY;",
            )
            .map_err(map_err)?;
            Self::create_schema(&conn)?;
            Ok(Self { conn: Arc::new(parking_lot::Mutex::new(conn)) })
        }

        fn create_schema(conn: &Connection) -> Result<()> {
            conn.execute_batch(
                r#"
                CREATE TABLE IF NOT EXISTS file (
                    file_id      INTEGER PRIMARY KEY,
                    rel_path     TEXT NOT NULL UNIQUE,
                    lang         TEXT,
                    content_hash TEXT NOT NULL,
                    parse_state  TEXT NOT NULL DEFAULT 'ok',
                    generation   INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS symbol (
                    symbol_id    TEXT PRIMARY KEY,
                    kind         TEXT NOT NULL,
                    display_name TEXT,
                    file         TEXT,
                    generation   INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS symbol_name ON symbol(display_name);

                CREATE TABLE IF NOT EXISTS occurrence (
                    occ_id     INTEGER PRIMARY KEY,
                    symbol_id  TEXT NOT NULL,
                    file       TEXT NOT NULL,
                    start_line INTEGER, start_col INTEGER,
                    end_line   INTEGER, end_col INTEGER,
                    role       TEXT NOT NULL,
                    generation INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS occ_symbol ON occurrence(symbol_id);
                CREATE INDEX IF NOT EXISTS occ_role ON occurrence(symbol_id, role);
                CREATE INDEX IF NOT EXISTS occ_file ON occurrence(file);

                -- Unified graph with MATERIALIZED reverse edges.
                CREATE TABLE IF NOT EXISTS edge (
                    src        TEXT NOT NULL,
                    dst        TEXT NOT NULL,
                    kind       TEXT NOT NULL,
                    weight     REAL NOT NULL DEFAULT 1.0,
                    generation INTEGER NOT NULL,
                    PRIMARY KEY (src, kind, dst)
                );
                CREATE INDEX IF NOT EXISTS edge_fwd ON edge(src, kind);
                CREATE INDEX IF NOT EXISTS edge_rev ON edge(dst, kind);

                CREATE TABLE IF NOT EXISTS chunk (
                    chunk_id   TEXT PRIMARY KEY,
                    file       TEXT NOT NULL,
                    symbol_id  TEXT,
                    start_byte INTEGER, end_byte INTEGER,
                    start_line INTEGER, end_line INTEGER,
                    embed_model_id TEXT,           -- NULL = pending embedding
                    generation INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS chunk_file ON chunk(file);

                -- Vectors stored alongside chunks (cosine over stored f32, no ANN dep).
                CREATE TABLE IF NOT EXISTS vector (
                    chunk_id   TEXT PRIMARY KEY REFERENCES chunk(chunk_id),
                    model_id   TEXT NOT NULL,
                    dim        INTEGER NOT NULL,
                    data       BLOB NOT NULL       -- little-endian f32 array
                );

                CREATE TABLE IF NOT EXISTS generation (
                    generation INTEGER PRIMARY KEY,
                    root_hash  TEXT NOT NULL,
                    created_ms INTEGER NOT NULL,
                    status     TEXT NOT NULL        -- 'committed' | 'in_progress'
                );

                -- FTS5: BM25 over body + identifier columns; ident weighted >> body.
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_body USING fts5(
                    body, ident, path UNINDEXED
                );
                "#,
            )
            .map_err(map_err)
        }

        // ---- writes (one writer; batch in a transaction) ----

        /// Upsert a file's parsed facts at a generation, replacing prior rows for that
        /// file. Lexical, symbol, occurrence and chunk layers committed in one txn.
        #[allow(clippy::too_many_arguments)]
        pub fn upsert_file(
            &self,
            rel_path: &str,
            lang: &str,
            content_hash: &str,
            parse_state: &str,
            body: &str,
            symbols: &[Symbol],
            occurrences: &[Occurrence],
            chunks: &[CodeChunk],
            generation: u64,
        ) -> Result<()> {
            let mut conn = self.conn.lock();
            let tx = conn.transaction().map_err(map_err)?;

            // Clear prior rows for this file (re-index of a changed file).
            tx.execute("DELETE FROM occurrence WHERE file = ?1", params![rel_path]).map_err(map_err)?;
            tx.execute("DELETE FROM symbol WHERE file = ?1", params![rel_path]).map_err(map_err)?;
            tx.execute("DELETE FROM chunk WHERE file = ?1", params![rel_path]).map_err(map_err)?;
            tx.execute("DELETE FROM fts_body WHERE path = ?1", params![rel_path]).map_err(map_err)?;

            tx.execute(
                "INSERT INTO file(rel_path, lang, content_hash, parse_state, generation)
                 VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(rel_path) DO UPDATE SET
                   lang=excluded.lang, content_hash=excluded.content_hash,
                   parse_state=excluded.parse_state, generation=excluded.generation",
                params![rel_path, lang, content_hash, parse_state, generation as i64],
            )
            .map_err(map_err)?;

            for s in symbols {
                tx.execute(
                    "INSERT OR REPLACE INTO symbol(symbol_id, kind, display_name, file, generation)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![s.qualified_name, s.kind, s.name, s.file, generation as i64],
                )
                .map_err(map_err)?;
            }

            for o in occurrences {
                let (sl, sc, el, ec) = match &o.range {
                    Some(r) => (r.start_line as i64, r.start_col as i64, r.end_line as i64, r.end_col as i64),
                    None => (0, 0, 0, 0),
                };
                tx.execute(
                    "INSERT INTO occurrence(symbol_id, file, start_line, start_col, end_line, end_col, role, generation)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                    params![o.symbol, o.file, sl, sc, el, ec, o.role, generation as i64],
                )
                .map_err(map_err)?;
            }

            for c in chunks {
                tx.execute(
                    "INSERT OR REPLACE INTO chunk(chunk_id, file, symbol_id, start_byte, end_byte, start_line, end_line, embed_model_id, generation)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, NULL, ?8)",
                    params![
                        c.chunk_id,
                        c.file,
                        c.symbol,
                        c.start_byte as i64,
                        c.end_byte as i64,
                        c.start_line as i64,
                        c.end_line as i64,
                        generation as i64
                    ],
                )
                .map_err(map_err)?;
            }

            // Lexical: index the body and an identifier column (symbol names).
            let ident_blob = symbols.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join(" ");
            tx.execute(
                "INSERT INTO fts_body(body, ident, path) VALUES (?1, ?2, ?3)",
                params![body, ident_blob, rel_path],
            )
            .map_err(map_err)?;

            tx.commit().map_err(map_err)?;
            Ok(())
        }

        /// Remove all rows for a file (deletion path).
        pub fn remove_file(&self, rel_path: &str) -> Result<()> {
            let mut conn = self.conn.lock();
            let tx = conn.transaction().map_err(map_err)?;
            for stmt in [
                "DELETE FROM occurrence WHERE file = ?1",
                "DELETE FROM symbol WHERE file = ?1",
                "DELETE FROM chunk WHERE file = ?1",
                "DELETE FROM fts_body WHERE path = ?1",
                "DELETE FROM file WHERE rel_path = ?1",
            ] {
                tx.execute(stmt, params![rel_path]).map_err(map_err)?;
            }
            tx.commit().map_err(map_err)?;
            Ok(())
        }

        /// Insert/replace a graph edge AND nothing else — the reverse direction is
        /// served by the `edge_rev` index, so a single row is materialized-reverse.
        pub fn add_edge(&self, src: &str, dst: &str, kind: EdgeKind, weight: f32, generation: u64) -> Result<()> {
            let conn = self.conn.lock();
            conn.execute(
                "INSERT OR REPLACE INTO edge(src, dst, kind, weight, generation) VALUES (?1,?2,?3,?4,?5)",
                params![src, dst, edge_kind_str(kind), weight as f64, generation as i64],
            )
            .map_err(map_err)?;
            Ok(())
        }

        pub fn store_vector(&self, chunk_id: &str, model_id: &str, vector: &[f32]) -> Result<()> {
            let conn = self.conn.lock();
            let mut bytes = Vec::with_capacity(vector.len() * 4);
            for v in vector {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            conn.execute(
                "INSERT OR REPLACE INTO vector(chunk_id, model_id, dim, data) VALUES (?1,?2,?3,?4)",
                params![chunk_id, model_id, vector.len() as i64, bytes],
            )
            .map_err(map_err)?;
            conn.execute("UPDATE chunk SET embed_model_id=?1 WHERE chunk_id=?2", params![model_id, chunk_id])
                .map_err(map_err)?;
            Ok(())
        }

        // ---- reads ----

        /// Definition occurrences for a symbol — accepts either a SCIP id or a bare
        /// name (resolves name → its definition occurrences by display_name).
        pub fn definitions(&self, symbol: &str) -> Result<Vec<Occurrence>> {
            let conn = self.conn.lock();
            // Direct id match first.
            let mut occs = query_occurrences(&conn, "symbol_id = ?1 AND role = 'definition'", symbol)?;
            if occs.is_empty() {
                // Resolve by bare name: find symbol_ids whose display_name = symbol.
                let ids = symbol_ids_by_name(&conn, symbol)?;
                for id in ids {
                    occs.extend(query_occurrences(&conn, "symbol_id = ?1 AND role = 'definition'", &id)?);
                }
            }
            Ok(occs)
        }

        /// Precise definition lookup: ONLY the exact `symbol_id` (no bare-name
        /// fallback). Used by the `Q.precise` query path so a same-named symbol in
        /// another file can't leak in.
        pub fn definitions_exact(&self, symbol: &str) -> Result<Vec<Occurrence>> {
            let conn = self.conn.lock();
            query_occurrences(&conn, "symbol_id = ?1 AND role = 'definition'", symbol)
        }

        /// Precise reference lookup: ONLY occurrences keyed by exactly `symbol`
        /// (references are stored keyed by bare name; no SCIP-suffix stripping).
        pub fn references_exact(&self, symbol: &str) -> Result<Vec<Occurrence>> {
            let conn = self.conn.lock();
            query_occurrences(&conn, "symbol_id = ?1 AND role = 'reference'", symbol)
        }

        /// Reference occurrences for a symbol — by id or bare name.
        pub fn references(&self, symbol: &str) -> Result<Vec<Occurrence>> {
            let conn = self.conn.lock();
            // References are stored keyed by bare name (tier-0); also try id.
            let name = bare_name_of(symbol);
            let mut occs = query_occurrences(&conn, "symbol_id = ?1 AND role = 'reference'", &name)?;
            occs.extend(query_occurrences(&conn, "symbol_id = ?1 AND role = 'reference'", symbol)?);
            Ok(occs)
        }

        /// Lexical search via FTS5 BM25 (ident column weighted >> body).
        pub fn lexical_search(&self, query: &str, limit: usize) -> Result<Vec<LexicalHit>> {
            let conn = self.conn.lock();
            let match_expr = fts_match_expr(query);
            if match_expr.is_empty() {
                return Ok(Vec::new());
            }
            let mut stmt = conn
                .prepare(
                    "SELECT path, body, bm25(fts_body, 1.0, 5.0) AS rank
                     FROM fts_body WHERE fts_body MATCH ?1
                     ORDER BY rank LIMIT ?2",
                )
                .map_err(map_err)?;
            let rows = stmt
                .query_map(params![match_expr, limit as i64], |row| {
                    let path: String = row.get(0)?;
                    let body: String = row.get(1)?;
                    let rank: f64 = row.get(2)?;
                    Ok(LexicalHit { path, body, rank })
                })
                .map_err(map_err)?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r.map_err(map_err)?);
            }
            Ok(out)
        }

        /// Symbols whose display_name matches the query substring (symbol leg).
        pub fn symbol_search(&self, query: &str, limit: usize) -> Result<Vec<Symbol>> {
            let conn = self.conn.lock();
            // Strip LIKE wildcards from the user input (collapsed into one pass).
            let like = format!("%{}%", query.replace(['%', '_'], ""));
            let mut stmt = conn
                .prepare(
                    "SELECT symbol_id, kind, display_name, file FROM symbol
                     WHERE display_name LIKE ?1 OR symbol_id LIKE ?1 LIMIT ?2",
                )
                .map_err(map_err)?;
            let rows = stmt
                .query_map(params![like, limit as i64], |row| {
                    Ok(Symbol { qualified_name: row.get(0)?, kind: row.get(1)?, name: row.get(2)?, file: row.get(3)? })
                })
                .map_err(map_err)?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r.map_err(map_err)?);
            }
            Ok(out)
        }

        /// Forward edges from `src` of a given kind.
        pub fn out_edges(&self, src: &str, kind: EdgeKind) -> Result<Vec<(String, f32)>> {
            self.edges_dir("src", "dst", src, kind)
        }

        /// Reverse edges into `dst` of a kind ("who calls/depends-on X?") — seek.
        pub fn in_edges(&self, dst: &str, kind: EdgeKind) -> Result<Vec<(String, f32)>> {
            self.edges_dir("dst", "src", dst, kind)
        }

        fn edges_dir(&self, key_col: &str, val_col: &str, key: &str, kind: EdgeKind) -> Result<Vec<(String, f32)>> {
            let conn = self.conn.lock();
            let sql = format!("SELECT {val_col}, weight FROM edge WHERE {key_col} = ?1 AND kind = ?2");
            let mut stmt = conn.prepare(&sql).map_err(map_err)?;
            let rows = stmt
                .query_map(params![key, edge_kind_str(kind)], |row| {
                    let v: String = row.get(0)?;
                    let w: f64 = row.get(1)?;
                    Ok((v, w as f32))
                })
                .map_err(map_err)?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r.map_err(map_err)?);
            }
            Ok(out)
        }

        /// All edges (for loading into petgraph).
        pub fn all_edges(&self) -> Result<Vec<(String, String, EdgeKind, f32)>> {
            let conn = self.conn.lock();
            let mut stmt = conn.prepare("SELECT src, dst, kind, weight FROM edge").map_err(map_err)?;
            let rows = stmt
                .query_map([], |row| {
                    let src: String = row.get(0)?;
                    let dst: String = row.get(1)?;
                    let kind: String = row.get(2)?;
                    let w: f64 = row.get(3)?;
                    Ok((src, dst, edge_kind_from_str(&kind), w as f32))
                })
                .map_err(map_err)?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r.map_err(map_err)?);
            }
            Ok(out)
        }

        /// Chunks whose vectors are present, for cosine search.
        pub fn all_vectors(&self) -> Result<Vec<StoredVector>> {
            let conn = self.conn.lock();
            let mut stmt = conn
                .prepare(
                    "SELECT v.chunk_id, v.model_id, v.dim, v.data, c.file, c.start_line, c.end_line
                     FROM vector v JOIN chunk c ON c.chunk_id = v.chunk_id",
                )
                .map_err(map_err)?;
            let rows = stmt
                .query_map([], |row| {
                    let chunk_id: String = row.get(0)?;
                    let model_id: String = row.get(1)?;
                    let dim: i64 = row.get(2)?;
                    let data: Vec<u8> = row.get(3)?;
                    let file: String = row.get(4)?;
                    let start_line: i64 = row.get(5)?;
                    let end_line: i64 = row.get(6)?;
                    let vector = bytes_to_f32(&data, dim as usize);
                    Ok(StoredVector {
                        chunk_id,
                        model_id,
                        file,
                        start_line: start_line as u32,
                        end_line: end_line as u32,
                        vector,
                    })
                })
                .map_err(map_err)?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r.map_err(map_err)?);
            }
            Ok(out)
        }

        /// Chunks lacking a vector (for the embedding daemon to fill).
        pub fn pending_chunks(&self, limit: usize) -> Result<Vec<PendingChunk>> {
            let conn = self.conn.lock();
            let mut stmt = conn
                .prepare(
                    "SELECT chunk_id, file, start_byte, end_byte FROM chunk
                     WHERE embed_model_id IS NULL LIMIT ?1",
                )
                .map_err(map_err)?;
            let rows = stmt
                .query_map(params![limit as i64], |row| {
                    Ok(PendingChunk {
                        chunk_id: row.get(0)?,
                        file: row.get(1)?,
                        start_byte: row.get::<_, i64>(2)? as usize,
                        end_byte: row.get::<_, i64>(3)? as usize,
                    })
                })
                .map_err(map_err)?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r.map_err(map_err)?);
            }
            Ok(out)
        }

        pub fn file_count(&self) -> Result<usize> {
            let conn = self.conn.lock();
            let n: i64 = conn.query_row("SELECT COUNT(*) FROM file", [], |r| r.get(0)).map_err(map_err)?;
            Ok(n as usize)
        }

        pub fn file_hash(&self, rel_path: &str) -> Result<Option<String>> {
            let conn = self.conn.lock();
            conn.query_row("SELECT content_hash FROM file WHERE rel_path = ?1", params![rel_path], |r| r.get(0))
                .optional()
                .map_err(map_err)
        }

        pub fn unparseable_count(&self) -> Result<usize> {
            let conn = self.conn.lock();
            let n: i64 = conn
                .query_row("SELECT COUNT(*) FROM file WHERE parse_state = 'unparseable'", [], |r| r.get(0))
                .map_err(map_err)?;
            Ok(n as usize)
        }

        // ---- generation / MVCC ----

        pub fn begin_generation(&self, generation: u64, root_hash: &str) -> Result<()> {
            let conn = self.conn.lock();
            conn.execute(
                "INSERT OR REPLACE INTO generation(generation, root_hash, created_ms, status)
                 VALUES (?1, ?2, ?3, 'in_progress')",
                params![generation as i64, root_hash, now_ms()],
            )
            .map_err(map_err)?;
            Ok(())
        }

        pub fn commit_generation(&self, generation: u64) -> Result<()> {
            let conn = self.conn.lock();
            conn.execute("UPDATE generation SET status='committed' WHERE generation=?1", params![generation as i64])
                .map_err(map_err)?;
            Ok(())
        }

        /// The last committed generation (crash-recovery anchor).
        pub fn last_committed_generation(&self) -> Result<u64> {
            let conn = self.conn.lock();
            let g: Option<i64> = conn
                .query_row("SELECT MAX(generation) FROM generation WHERE status='committed'", [], |r| r.get(0))
                .optional()
                .map_err(map_err)?
                .flatten();
            Ok(g.unwrap_or(0) as u64)
        }

        /// Truncate any torn (in_progress) generation rows on recovery.
        pub fn recover(&self) -> Result<u64> {
            let conn = self.conn.lock();
            conn.execute("DELETE FROM generation WHERE status='in_progress'", []).map_err(map_err)?;
            drop(conn);
            self.last_committed_generation()
        }
    }

    // ---- helpers ----

    #[derive(Debug, Clone)]
    pub struct LexicalHit {
        pub path: String,
        pub body: String,
        pub rank: f64,
    }

    #[derive(Debug, Clone)]
    pub struct StoredVector {
        pub chunk_id: String,
        pub model_id: String,
        pub file: String,
        pub start_line: u32,
        pub end_line: u32,
        pub vector: Vec<f32>,
    }

    #[derive(Debug, Clone)]
    pub struct PendingChunk {
        pub chunk_id: String,
        pub file: String,
        pub start_byte: usize,
        pub end_byte: usize,
    }

    fn query_occurrences(conn: &Connection, where_clause: &str, key: &str) -> Result<Vec<Occurrence>> {
        let sql = format!(
            "SELECT symbol_id, file, start_line, start_col, end_line, end_col, role
             FROM occurrence WHERE {where_clause}"
        );
        let mut stmt = conn.prepare(&sql).map_err(map_err)?;
        let rows = stmt
            .query_map(params![key], |row| {
                let symbol: String = row.get(0)?;
                let file: String = row.get(1)?;
                let sl: i64 = row.get(2)?;
                let sc: i64 = row.get(3)?;
                let el: i64 = row.get(4)?;
                let ec: i64 = row.get(5)?;
                let role: String = row.get(6)?;
                let range = if sl == 0 && el == 0 {
                    None
                } else {
                    Some(TextRange {
                        start_line: sl as u32,
                        start_col: sc as u32,
                        end_line: el as u32,
                        end_col: ec as u32,
                    })
                };
                Ok(Occurrence { symbol, file, range, role })
            })
            .map_err(map_err)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(map_err)?);
        }
        Ok(out)
    }

    fn symbol_ids_by_name(conn: &Connection, name: &str) -> Result<Vec<String>> {
        let mut stmt = conn.prepare("SELECT symbol_id FROM symbol WHERE display_name = ?1").map_err(map_err)?;
        let rows = stmt.query_map(params![name], |row| row.get::<_, String>(0)).map_err(map_err)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(map_err)?);
        }
        Ok(out)
    }

    /// Best-effort bare name out of a SCIP id (or pass through if it's already bare).
    fn bare_name_of(symbol: &str) -> String {
        // SCIP id: "hawking <lang> <path> <name><suffix>" — take the last token,
        // strip the descriptor suffix.
        if let Some(last) = symbol.rsplit(' ').next() {
            let trimmed = last.trim_end_matches("().").trim_end_matches(['#', '.', '!', '/']);
            if trimmed != symbol {
                return trimmed.to_string();
            }
        }
        symbol.to_string()
    }

    /// Build an FTS5 MATCH expression: tokenize on non-alphanumerics, OR the terms,
    /// escape with double-quotes to neutralize FTS operators.
    fn fts_match_expr(query: &str) -> String {
        let terms: Vec<String> = query
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|t| !t.is_empty())
            .map(|t| format!("\"{}\"", t.replace('"', "")))
            .collect();
        terms.join(" OR ")
    }

    fn bytes_to_f32(bytes: &[u8], dim: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(dim);
        for chunk in bytes.chunks_exact(4) {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        out
    }

    pub(crate) fn edge_kind_str(kind: EdgeKind) -> &'static str {
        match kind {
            EdgeKind::Defines => "defines",
            EdgeKind::References => "references",
            EdgeKind::Calls => "calls",
            EdgeKind::Imports => "imports",
            EdgeKind::Implements => "implements",
            EdgeKind::Tests => "tests",
            EdgeKind::Dataflow => "dataflow",
            EdgeKind::Performance => "performance",
        }
    }

    pub(crate) fn edge_kind_from_str(s: &str) -> EdgeKind {
        match s {
            "defines" => EdgeKind::Defines,
            "references" => EdgeKind::References,
            "calls" => EdgeKind::Calls,
            "imports" => EdgeKind::Imports,
            "implements" => EdgeKind::Implements,
            "tests" => EdgeKind::Tests,
            "dataflow" => EdgeKind::Dataflow,
            _ => EdgeKind::Performance,
        }
    }

    fn now_ms() -> i64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
    }

    /// Legacy config DTOs (kept for serde back-compat with siblings).
    #[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    pub struct IndexStoreConfig {
        pub sqlite_path: String,
        pub vector_path: String,
        pub cas_path: String,
        pub generation: u64,
    }

    #[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    pub struct StoreGeneration {
        pub generation: u64,
        pub manifest_hash: String,
        pub sealed: bool,
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::parse::parse_source;

        fn seed(store: &SqliteStore, path: &str, src: &str, gen: u64) {
            let out = parse_source(path, src);
            let chunks = crate::parse::chunk_file(path, src);
            store
                .upsert_file(
                    path,
                    out.lang.map(|l| l.as_str()).unwrap_or("unknown"),
                    "hash",
                    if out.unparseable { "unparseable" } else { "ok" },
                    src,
                    &out.symbols,
                    &out.occurrences,
                    &chunks,
                    gen,
                )
                .unwrap();
        }

        #[test]
        fn definitions_and_references_roundtrip() {
            let store = SqliteStore::open_in_memory().unwrap();
            let src = "pub fn helper() {}\npub fn caller() { helper(); }\n";
            seed(&store, "src/m.rs", src, 1);

            let defs = store.definitions("helper").unwrap();
            assert_eq!(defs.len(), 1, "one definition of helper");
            assert_eq!(defs[0].role, "definition");

            let refs = store.references("helper").unwrap();
            assert!(!refs.is_empty(), "references(helper) must be non-empty");
        }

        #[test]
        fn fts_lexical_search_finds_body() {
            let store = SqliteStore::open_in_memory().unwrap();
            seed(&store, "a.rs", "pub fn alpha_widget() { compute(); }", 1);
            seed(&store, "b.rs", "pub fn beta() { unrelated(); }", 1);
            let hits = store.lexical_search("alpha_widget", 10).unwrap();
            assert!(hits.iter().any(|h| h.path == "a.rs"));
            assert!(!hits.iter().any(|h| h.path == "b.rs"));
        }

        #[test]
        fn reverse_edges_are_a_seek() {
            let store = SqliteStore::open_in_memory().unwrap();
            store.add_edge("caller", "callee", EdgeKind::Calls, 1.0, 1).unwrap();
            let callers = store.in_edges("callee", EdgeKind::Calls).unwrap();
            assert_eq!(callers.len(), 1);
            assert_eq!(callers[0].0, "caller");
            let callees = store.out_edges("caller", EdgeKind::Calls).unwrap();
            assert_eq!(callees[0].0, "callee");
        }

        #[test]
        fn vectors_store_and_load() {
            let store = SqliteStore::open_in_memory().unwrap();
            seed(&store, "v.rs", "pub fn embed_me() { work(); }", 1);
            let pending = store.pending_chunks(10).unwrap();
            assert!(!pending.is_empty());
            store.store_vector(&pending[0].chunk_id, "logits-proxy:test", &[0.1, 0.2, 0.3]).unwrap();
            let vecs = store.all_vectors().unwrap();
            assert_eq!(vecs.len(), 1);
            assert_eq!(vecs[0].vector, vec![0.1, 0.2, 0.3]);
        }

        #[test]
        fn generation_commit_and_recovery() {
            let store = SqliteStore::open_in_memory().unwrap();
            store.begin_generation(1, "root1").unwrap();
            store.commit_generation(1).unwrap();
            store.begin_generation(2, "root2").unwrap(); // torn (never committed)
            assert_eq!(store.last_committed_generation().unwrap(), 1);
            assert_eq!(store.recover().unwrap(), 1);
        }

        #[test]
        fn remove_file_clears_rows() {
            let store = SqliteStore::open_in_memory().unwrap();
            seed(&store, "gone.rs", "pub fn x() {}", 1);
            assert_eq!(store.file_count().unwrap(), 1);
            store.remove_file("gone.rs").unwrap();
            assert_eq!(store.file_count().unwrap(), 0);
            assert!(store.definitions("x").unwrap().is_empty());
        }
    }
}

pub use query::{
    CodeIndex, InMemoryCodeIndex, Index, IndexHealth, SearchQuery, SearchResult,
    SearchResultSource, SqliteCodeIndex, Q,
};

pub use graph::{CodeGraph, EdgeKind, Occurrence, RepoMap, RepoMapRequest, Symbol};
pub use merkle::{Blake3MerkleScanner, ChangeSet, MerkleKind, MerkleNode, MerkleScanner};
pub use parse::{parse_source, scip_symbol_id, LangId, ParseOutput, SymKind};
pub use semantic::{
    cosine, fuse_legs, reciprocal_rank_fusion, EmbeddingClient, HttpEmbeddingClient,
    HybridRetrievalWeights, HybridRetriever, StubEmbeddingClient,
};
pub use store::SqliteStore;
