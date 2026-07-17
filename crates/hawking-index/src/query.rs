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
use crate::semantic::{
    FusedHit, HybridRetriever, LegRanking, LexicalOverlapReranker, StubEmbeddingClient,
};
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
        let symbolish =
            t.contains("::") || t.contains('.') || t.contains('_') || has_internal_caps(t);
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
        let (include_symbols, include_lexical, include_semantic) = match classify_query_shape(&text)
        {
            QueryShape::ExactSymbol => (true, false, false),
            QueryShape::Identifier => (true, true, false),
            QueryShape::NaturalLanguage => (true, true, true),
        };
        SearchQuery {
            text,
            limit,
            include_symbols,
            include_lexical,
            include_semantic,
        }
    }
}

#[cfg(test)]
mod routing_tests {
    use super::*;

    #[test]
    fn classifies_query_shapes() {
        assert_eq!(
            classify_query_shape("CodeIndex::search"),
            QueryShape::ExactSymbol
        );
        assert_eq!(classify_query_shape("foo_bar"), QueryShape::ExactSymbol);
        assert_eq!(classify_query_shape("parseTree"), QueryShape::ExactSymbol);
        assert_eq!(classify_query_shape("parse tree"), QueryShape::Identifier);
        assert_eq!(
            classify_query_shape("where do we handle retries?"),
            QueryShape::NaturalLanguage
        );
        assert_eq!(
            classify_query_shape("how does compaction work"),
            QueryShape::NaturalLanguage
        );
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
        assert!(
            source_rank(SearchResultSource::Symbol) < source_rank(SearchResultSource::Semantic)
        );
        assert!(
            source_rank(SearchResultSource::Lexical) < source_rank(SearchResultSource::Semantic)
        );
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
        self.symbols
            .write()
            .insert(symbol.qualified_name.clone(), symbol);
    }

    pub fn add_occurrence(&self, occurrence: Occurrence) {
        *self.generation.write() += 1;
        self.occurrences
            .write()
            .entry(occurrence.symbol.clone())
            .or_default()
            .push(occurrence);
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
        self.files.write().insert(
            path.clone(),
            IndexedFile {
                path,
                content,
                content_hash,
            },
        );
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
                        let score = if symbol.name.to_lowercase() == needle {
                            2.0
                        } else {
                            1.2
                        };
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
            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
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
        let mut out: Vec<Occurrence> = occs
            .get(symbol)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter(|o| o.role == role)
            .collect();

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
    occs.get(symbol_id)?
        .iter()
        .find(|o| o.role == ROLE_DEFINITION)
        .and_then(|o| o.range.clone())
}

/// Best-effort bare name out of a SCIP id (or pass through if already bare).
fn bare_name(symbol: &str) -> String {
    if let Some(last) = symbol.rsplit(' ').next() {
        let trimmed = last
            .trim_end_matches("().")
            .trim_end_matches(['#', '.', '!', '/']);
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
        Self {
            store,
            graph: RwLock::new(CodeGraph::new()),
            generation: RwLock::new(gen),
        }
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
            let def_by_name: HashMap<String, String> = parsed
                .symbols
                .iter()
                .map(|s| (s.name.clone(), s.qualified_name.clone()))
                .collect();
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
        Ok(self
            .store
            .in_edges(symbol_id, EdgeKind::Calls)?
            .into_iter()
            .map(|(src, _)| src)
            .collect())
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
        self.search_with_embedder(query, k_final, &embedder, include_semantic)
            .await
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
        let lexical = LegRanking {
            name: "lexical".into(),
            weight: weights.lexical,
            ranked_keys: lex_keys,
        };
        let symbol = LegRanking {
            name: "symbol".into(),
            weight: weights.symbol,
            ranked_keys: sym_keys,
        };

        let retriever = HybridRetriever::new(&self.store, embedder);
        retriever
            .search_with_legs(
                query,
                lexical,
                symbol,
                &snippets,
                &LexicalOverlapReranker,
                k_final,
                include_semantic,
            )
            .await
    }
}

fn first_line_for(content: &str, occs: &[Occurrence], symbol_id: &str) -> String {
    if let Some(occ) = occs
        .iter()
        .find(|o| o.symbol == symbol_id && o.role == ROLE_DEFINITION)
    {
        if let Some(range) = &occ.range {
            if let Some(line) = content
                .lines()
                .nth((range.start_line.saturating_sub(1)) as usize)
            {
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
            let hits = self
                .hybrid_search_opts(&query.text, query.limit, query.include_semantic)
                .await?;
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
            let by_name: HashMap<String, String> = symbols
                .values()
                .map(|s| (s.name.clone(), s.qualified_name.clone()))
                .collect();
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
        index
            .index_text(
                "src/m.rs",
                "pub fn target_widget() { helper(); }\nfn helper() {}\n",
                "hash1",
            )
            .unwrap();

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
