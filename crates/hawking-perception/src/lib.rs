//! HCLI Perception Service (Ascension Bible §19).
//!
//! Cheap-first document pipeline — interfaces and deterministic stubs only.
//! Real OCR / vision / PDF rasterization backends are **out of scope** here
//! and must not be claimed as implemented.
//!
//! Default path:
//! ```text
//! cheap metadata/text pass
//! -> retrieve relevant pages
//! -> detect relevant regions
//! -> selective OCR or vision
//! -> parse tables/charts
//! -> structured evidence
//! -> coordinate-bound citation
//! ```
//!
//! Tool surface (stable names):
//! `document.inspect`, `document.retrieve_pages`, `document.detect_regions`,
//! `document.parse_region`, `document.parse_table`, `document.parse_chart`,
//! `document.verify_structure`, `document.cite_coordinates`.
//!
//! Reuses concepts from:
//! - `hawking-research::ingest::{StructuredDoc, SectionEvidence}` — text evidence
//!   pinning (PDF body parse is already a documented seam there).
//! - `hide_core::types::{BlobRef, Provenance}` — content-addressed blobs (mirrored
//!   lightly here so this crate stays model-free and dependency-light).
//! - `hide_backend::lenses_evidence::EvidenceTier` — claim strength ordering
//!   (mirrored as [`EvidenceStrength`] without pulling the full backend).

pub mod coords;
pub mod error;
pub mod pipeline;
pub mod service;
pub mod stubs;
pub mod tools;
pub mod types;

pub use coords::{CoordBox, PageCoord, RegionKind};
pub use error::{PerceptionError, Result};
pub use pipeline::{PerceptionPipeline, PipelineBudget, PipelineStage};
pub use service::DocumentService;
pub use stubs::StubDocumentService;
pub use tools::{DocumentTool, DocumentToolName, ToolRequest, ToolResponse};
pub use types::{
    ChartParse, Citation, DocumentHandle, DocumentMeta, EvidenceStrength, PageContent,
    PageRef, ParsedRegion, Region, RegionId, StructuredEvidence, StructureReport, TableParse,
};

/// Schema / identity markers for receipts and capability surfaces.
pub const PERCEPTION_SCHEMA: &str = "hcli.perception.v0";
pub const PERCEPTION_TOOL_NAMESPACE: &str = "document";
