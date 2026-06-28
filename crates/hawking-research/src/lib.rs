//! Headless Research & Knowledge Lab (HIDE bible chapter 08).
//!
//! This crate is the backend of the Research Lab: a model-agnostic
//! [`RuntimeClient`](runtime_client::RuntimeClient) seam (reusing
//! `hawking-orch`'s `InferenceClient`), content-addressing over
//! `hide_core::BlobStore` ([`cas`]), a real petgraph-backed knowledge graph
//! ([`kg`]) with Local/Global/Path queries + entity resolution + JSONL
//! persistence, a real arXiv ingestion adapter ([`ingest`]), an adversarial
//! verifier with CAS citation re-verification ([`verify`]), a checkpointed
//! research pipeline FSM ([`pipeline`]), literature mapping ([`litmap`]), and
//! research⇄code/issues/memory bridges ([`bridge`]). UI panels are out of scope.

pub mod bridge;
pub mod cas;
pub mod checkpoint;
pub mod experiments;
pub mod ingest;
pub mod kg;
pub mod litmap;
pub mod pipeline;
pub mod run_ledger;
pub mod runtime_client;
pub mod verify;

pub use cas::{blake3_hex, content_id, pin_evidence, verify_evidence, EvidenceCheck};
pub use checkpoint::{
    CheckpointEvent, CheckpointKind, CheckpointLedger, DynCheckpointLedger,
    InMemoryCheckpointLedger, JsonlCheckpointLedger, RunJournal,
};
pub use ingest::{ArxivAdapter, InMemorySourceAdapter, SourceAdapter, StructuredDoc};
pub use kg::{
    GraphQuery, InMemoryKnowledgeGraph, KnowledgeGraph, KnowledgeNode, NodeKind,
    PetKnowledgeGraph, QueryResult,
};
pub use litmap::{build_literature_map, compare_papers, LiteratureMap};
pub use pipeline::{ResearchBudget, ResearchPipeline, ResearchRun, ResearchState};
pub use run_ledger::{
    DynResearchLedger, InMemoryResearchLedger, JsonlResearchLedger, ResearchLedger,
};
pub use runtime_client::{stub_runtime, ChatRequest, InferenceRuntime, RuntimeClient};
pub use verify::{AdversarialVerifier, ClaimStatus, ClaimVerification};
