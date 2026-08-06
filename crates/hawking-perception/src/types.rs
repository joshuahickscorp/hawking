//! Document / evidence types for the perception service.

use crate::coords::{CoordBox, RegionKind};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Opaque document identity (content-address or host path pin).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DocumentHandle {
    pub id: String,
    /// blake3 of source bytes when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
    /// Original URI / path (may be redacted in receipts).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
}

impl DocumentHandle {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            content_hash: None,
            uri: None,
            media_type: None,
        }
    }
}

/// Stable region id within a document (hash of page + box + kind).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RegionId(pub String);

impl RegionId {
    pub fn from_parts(doc_id: &str, page: u32, kind: RegionKind, box_: &CoordBox) -> Self {
        let payload = format!(
            "{doc_id}|{page}|{}|{:.4},{:.4},{:.4},{:.4}",
            kind.as_str(),
            box_.x0,
            box_.y0,
            box_.x1,
            box_.y1
        );
        let h = blake3::hash(payload.as_bytes());
        Self(format!("reg:{}", h.to_hex()))
    }
}

/// Cheap inspect output — never requires OCR.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocumentMeta {
    pub handle: DocumentHandle,
    pub page_count: u32,
    /// Embedded text / outline / title when free (PDF text layer, HTML, etc.).
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub has_text_layer: bool,
    #[serde(default)]
    pub has_images: bool,
    /// Per-page cheap signals (token estimate of free text, not OCR).
    #[serde(default)]
    pub page_summaries: Vec<PageSummary>,
    /// Free-text extract from metadata pass only.
    #[serde(default)]
    pub cheap_text: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PageSummary {
    pub page: u32,
    /// Characters of free text available without OCR.
    pub free_text_chars: u32,
    #[serde(default)]
    pub likely_has_figures: bool,
}

/// Reference to one retrieved page (content may still be deferred).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PageRef {
    pub document_id: String,
    pub page: u32,
}

/// Materialized page content after `retrieve_pages`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PageContent {
    pub page: u32,
    /// Free text / text-layer extract. Empty when only a raster is available.
    #[serde(default)]
    pub text: String,
    /// True if text came from an embedded layer (cheap); false if OCR.
    pub text_from_layer: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raster_ref: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Region {
    pub id: RegionId,
    pub document_id: String,
    pub kind: RegionKind,
    pub box_: CoordBox,
    /// Detector confidence 0.0 ..= 1.0.
    pub confidence: f32,
    #[serde(default)]
    pub labels: Vec<String>,
}

/// Output of selective parse (text/OCR/vision) for one region.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParsedRegion {
    pub region: Region,
    pub text: String,
    /// How expensive the parse was.
    pub cost_tier: CostTier,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
}

/// Cost ladder — higher tiers must not run before lower ones are considered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CostTier {
    /// Metadata / embedded text only.
    Metadata = 0,
    /// Page text layer / free extract.
    PageText = 1,
    /// Region-bounded OCR.
    RegionOcr = 2,
    /// Full-region multimodal vision.
    Vision = 3,
}

impl CostTier {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Metadata => "metadata",
            Self::PageText => "page_text",
            Self::RegionOcr => "region_ocr",
            Self::Vision => "vision",
        }
    }

    /// Rough relative cost units for budget accounting (scaffold defaults).
    pub fn cost_units(self) -> u64 {
        match self {
            Self::Metadata => 1,
            Self::PageText => 4,
            Self::RegionOcr => 40,
            Self::Vision => 200,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableParse {
    pub region_id: RegionId,
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
    pub box_: CoordBox,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChartParse {
    pub region_id: RegionId,
    pub title: Option<String>,
    pub chart_type: Option<String>,
    /// Series name → (x, y) or categorical points as strings.
    pub series: BTreeMap<String, Vec<ChartPoint>>,
    pub box_: CoordBox,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChartPoint {
    pub x: String,
    pub y: String,
}

/// Coordinate-bound citation ready for evidence graphs / reports.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Citation {
    pub id: String,
    pub document_id: String,
    pub box_: CoordBox,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region_id: Option<RegionId>,
    /// Quoted or reconstructed text bound to the coordinates.
    pub quote: String,
    /// blake3 of quote bytes (or region payload).
    pub content_hash: String,
    pub strength: EvidenceStrength,
}

/// Lightweight mirror of backend evidence tiers (no hide-backend dep).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceStrength {
    Asserted,
    Extracted,
    StructureVerified,
    CoordinateBound,
}

impl EvidenceStrength {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Asserted => "asserted",
            Self::Extracted => "extracted",
            Self::StructureVerified => "structure_verified",
            Self::CoordinateBound => "coordinate_bound",
        }
    }
}

/// Final structured evidence bundle produced by the pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuredEvidence {
    pub schema: String,
    pub document: DocumentHandle,
    pub pages_used: Vec<u32>,
    pub regions: Vec<ParsedRegion>,
    #[serde(default)]
    pub tables: Vec<TableParse>,
    #[serde(default)]
    pub charts: Vec<ChartParse>,
    pub citations: Vec<Citation>,
    /// Max cost tier actually spent.
    pub max_cost_tier: CostTier,
    pub notes: Vec<String>,
}

impl StructuredEvidence {
    pub fn empty(document: DocumentHandle) -> Self {
        Self {
            schema: crate::PERCEPTION_SCHEMA.to_string(),
            document,
            pages_used: Vec::new(),
            regions: Vec::new(),
            tables: Vec::new(),
            charts: Vec::new(),
            citations: Vec::new(),
            max_cost_tier: CostTier::Metadata,
            notes: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructureReport {
    pub document_id: String,
    pub ok: bool,
    pub issues: Vec<String>,
    /// Sections / headings recovered from cheap + selective passes.
    #[serde(default)]
    pub outline: Vec<String>,
}
