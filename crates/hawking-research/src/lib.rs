//! Headless Research & Knowledge Lab.
//!
//! This crate covers the backend portions of HIDE bible chapter 08: ingestion,
//! source adapters, a provenance-rich knowledge graph, research pipelines,
//! citation maps, experiments, and code/memory bridges. UI panels are out of
//! scope by design.

pub mod bridge;
pub mod experiments;
pub mod ingest;
pub mod kg;
pub mod litmap;
pub mod pipeline;
pub mod run_ledger;
pub mod verify;

pub use pipeline::{ResearchPipeline, ResearchRun, ResearchState};
pub use run_ledger::{
    DynResearchLedger, InMemoryResearchLedger, JsonlResearchLedger, ResearchLedger,
};
