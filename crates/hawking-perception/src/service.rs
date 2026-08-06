//! [`DocumentService`] trait — the eight bible tools as methods.

use crate::coords::{CoordBox, RegionKind};
use crate::error::Result;
use crate::types::{
    ChartParse, Citation, DocumentHandle, DocumentMeta, PageContent, PageRef, ParsedRegion, Region,
    StructureReport, TableParse,
};

/// Perception backend. Implementors may be pure stubs, PDF text-layer extractors,
/// or (later) OCR/vision engines. Callers must not assume vision is available.
///
/// Method names mirror the tool surface:
/// `document.inspect` … `document.cite_coordinates`.
pub trait DocumentService: Send + Sync {
    /// Cheap metadata / free-text pass. Must never invoke OCR or vision.
    fn inspect(&self, handle: &DocumentHandle) -> Result<DocumentMeta>;

    /// Retrieve a subset of pages (text layer preferred; raster optional).
    fn retrieve_pages(
        &self,
        handle: &DocumentHandle,
        pages: &[u32],
    ) -> Result<Vec<PageContent>>;

    /// Detect regions on already-retrieved pages. Cheap heuristics first.
    fn detect_regions(
        &self,
        handle: &DocumentHandle,
        pages: &[PageContent],
        kinds: Option<&[RegionKind]>,
    ) -> Result<Vec<Region>>;

    /// Selective parse of one region (text extract; OCR/vision only if required
    /// and capability is present).
    fn parse_region(&self, handle: &DocumentHandle, region: &Region) -> Result<ParsedRegion>;

    fn parse_table(&self, handle: &DocumentHandle, region: &Region) -> Result<TableParse>;

    fn parse_chart(&self, handle: &DocumentHandle, region: &Region) -> Result<ChartParse>;

    /// Cross-check recovered structure (outline, table shapes) for consistency.
    fn verify_structure(
        &self,
        handle: &DocumentHandle,
        regions: &[ParsedRegion],
        tables: &[TableParse],
    ) -> Result<StructureReport>;

    /// Build a coordinate-bound citation for a quote or region.
    fn cite_coordinates(
        &self,
        handle: &DocumentHandle,
        box_: CoordBox,
        quote: &str,
        region_id: Option<&crate::types::RegionId>,
    ) -> Result<Citation>;

    /// Convenience: page refs for a document.
    fn page_refs(&self, handle: &DocumentHandle) -> Result<Vec<PageRef>> {
        let meta = self.inspect(handle)?;
        Ok((0..meta.page_count)
            .map(|page| PageRef {
                document_id: handle.id.clone(),
                page,
            })
            .collect())
    }
}
