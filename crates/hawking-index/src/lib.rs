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

pub mod daemon;
pub mod graph;
pub mod merkle;
pub mod parse;
pub mod query;
pub mod semantic;
pub mod store;

pub use query::{
    CodeIndex, Index, IndexHealth, InMemoryCodeIndex, Q, SearchQuery, SearchResult,
    SearchResultSource, SqliteCodeIndex,
};

pub use graph::{CodeGraph, EdgeKind, Occurrence, RepoMap, RepoMapRequest, Symbol};
pub use merkle::{Blake3MerkleScanner, ChangeSet, MerkleKind, MerkleNode, MerkleScanner};
pub use parse::{parse_source, scip_symbol_id, LangId, ParseOutput, SymKind};
pub use semantic::{
    cosine, fuse_legs, reciprocal_rank_fusion, EmbeddingClient, HttpEmbeddingClient,
    HybridRetrievalWeights, HybridRetriever, StubEmbeddingClient,
};
pub use store::SqliteStore;
