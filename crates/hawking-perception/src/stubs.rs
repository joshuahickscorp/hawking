//! Deterministic in-memory document backend for tests and planning demos.
//!
//! No OCR, no vision, no PDF decoder — only free text fixtures. Demonstrates
//! the cheap-first path without claiming production perception capability.

use crate::coords::{CoordBox, RegionKind};
use crate::error::{PerceptionError, Result};
use crate::service::DocumentService;
use crate::types::{
    ChartParse, ChartPoint, Citation, CostTier, DocumentHandle, DocumentMeta, EvidenceStrength,
    PageContent, PageSummary, ParsedRegion, Region, RegionId, StructureReport, TableParse,
};
use std::collections::BTreeMap;
use std::sync::RwLock;

#[derive(Debug, Clone)]
struct FixtureDoc {
    handle: DocumentHandle,
    title: String,
    pages: Vec<String>,
    /// Optional synthetic tables keyed by page.
    tables: BTreeMap<u32, (Vec<String>, Vec<Vec<String>>)>,
    /// Optional synthetic chart titles keyed by page.
    charts: BTreeMap<u32, String>,
}

/// In-memory [`DocumentService`] with a small paper-like fixture.
#[derive(Debug, Default)]
pub struct StubDocumentService {
    docs: RwLock<BTreeMap<String, FixtureDoc>>,
}

impl StubDocumentService {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a multi-page text document.
    pub fn insert_text_doc(
        &self,
        id: &str,
        title: &str,
        pages: Vec<String>,
    ) -> DocumentHandle {
        let mut hasher = blake3::Hasher::new();
        for p in &pages {
            hasher.update(p.as_bytes());
            hasher.update(b"\n");
        }
        let hash = format!("blake3:{}", hasher.finalize().to_hex());
        let handle = DocumentHandle {
            id: id.to_string(),
            content_hash: Some(hash),
            uri: Some(format!("stub://{id}")),
            media_type: Some("text/plain".into()),
        };
        let doc = FixtureDoc {
            handle: handle.clone(),
            title: title.to_string(),
            pages,
            tables: BTreeMap::new(),
            charts: BTreeMap::new(),
        };
        self.docs.write().unwrap().insert(id.to_string(), doc);
        handle
    }

    /// Built-in multi-page fixture resembling a short systems paper.
    pub fn fixture_paper() -> Self {
        let s = Self::new();
        let pages = vec![
            "Abstract. We measure decode throughput on Apple silicon.\nKeywords: throughput, Metal, Gravity.".into(),
            "1. Introduction\nLocal inference requires careful memory residency.\nFigure 1 is a chart of tokens/s vs batch.".into(),
            "2. Results\nTable 1 reports median tokens/s for Qwen 30B.\nBatch 1: 42 tok/s. Batch 4: 38 tok/s.".into(),
            "3. Conclusion\nSelective vision on charts beats full-page OCR for cost.".into(),
        ];
        let handle = s.insert_text_doc("doc:fixture-paper", "Fixture Throughput Paper", pages);
        {
            let mut guard = s.docs.write().unwrap();
            let doc = guard.get_mut(&handle.id).unwrap();
            doc.tables.insert(
                2,
                (
                    vec!["batch".into(), "tok_s".into()],
                    vec![
                        vec!["1".into(), "42".into()],
                        vec!["4".into(), "38".into()],
                    ],
                ),
            );
            doc.charts
                .insert(1, "tokens/s vs batch (synthetic)".into());
        }
        s
    }

    fn get(&self, handle: &DocumentHandle) -> Result<FixtureDoc> {
        self.docs
            .read()
            .unwrap()
            .get(&handle.id)
            .cloned()
            .ok_or_else(|| PerceptionError::DocumentNotFound(handle.id.clone()))
    }
}

impl DocumentService for StubDocumentService {
    fn inspect(&self, handle: &DocumentHandle) -> Result<DocumentMeta> {
        let doc = self.get(handle)?;
        let page_summaries = doc
            .pages
            .iter()
            .enumerate()
            .map(|(i, t)| PageSummary {
                page: i as u32,
                free_text_chars: t.chars().count() as u32,
                likely_has_figures: doc.charts.contains_key(&(i as u32))
                    || t.to_lowercase().contains("figure")
                    || t.to_lowercase().contains("chart"),
            })
            .collect();
        let cheap_text = doc.pages.join("\n\n");
        Ok(DocumentMeta {
            handle: doc.handle,
            page_count: doc.pages.len() as u32,
            title: Some(doc.title),
            has_text_layer: true,
            has_images: !doc.charts.is_empty(),
            page_summaries,
            cheap_text: Some(cheap_text),
        })
    }

    fn retrieve_pages(
        &self,
        handle: &DocumentHandle,
        pages: &[u32],
    ) -> Result<Vec<PageContent>> {
        let doc = self.get(handle)?;
        let mut out = Vec::with_capacity(pages.len());
        for &page in pages {
            let text = doc.pages.get(page as usize).ok_or(PerceptionError::PageOutOfRange {
                page,
                pages: doc.pages.len() as u32,
            })?;
            out.push(PageContent {
                page,
                text: text.clone(),
                text_from_layer: true,
                raster_ref: None,
            });
        }
        Ok(out)
    }

    fn detect_regions(
        &self,
        handle: &DocumentHandle,
        pages: &[PageContent],
        kinds: Option<&[RegionKind]>,
    ) -> Result<Vec<Region>> {
        let doc = self.get(handle)?;
        let mut regions = Vec::new();
        for page in pages {
            // Full-page text region (always).
            let text_box = CoordBox::full_page(page.page);
            let text_region = Region {
                id: RegionId::from_parts(&handle.id, page.page, RegionKind::Text, &text_box),
                document_id: handle.id.clone(),
                kind: RegionKind::Text,
                box_: text_box,
                confidence: 1.0,
                labels: vec!["text_layer".into()],
            };
            if kinds.map(|k| k.contains(&RegionKind::Text)).unwrap_or(true) {
                regions.push(text_region);
            }

            if doc.tables.contains_key(&page.page)
                && kinds.map(|k| k.contains(&RegionKind::Table)).unwrap_or(true)
            {
                let box_ = CoordBox::new(page.page, 0.1, 0.4, 0.9, 0.75);
                regions.push(Region {
                    id: RegionId::from_parts(&handle.id, page.page, RegionKind::Table, &box_),
                    document_id: handle.id.clone(),
                    kind: RegionKind::Table,
                    box_,
                    confidence: 0.85,
                    labels: vec!["synthetic_table".into()],
                });
            }

            if doc.charts.contains_key(&page.page)
                && kinds.map(|k| k.contains(&RegionKind::Chart)).unwrap_or(true)
            {
                let box_ = CoordBox::new(page.page, 0.15, 0.2, 0.85, 0.55);
                regions.push(Region {
                    id: RegionId::from_parts(&handle.id, page.page, RegionKind::Chart, &box_),
                    document_id: handle.id.clone(),
                    kind: RegionKind::Chart,
                    box_,
                    confidence: 0.8,
                    labels: vec!["synthetic_chart".into()],
                });
            }
        }
        Ok(regions)
    }

    fn parse_region(&self, handle: &DocumentHandle, region: &Region) -> Result<ParsedRegion> {
        let doc = self.get(handle)?;
        let page = region.box_.page;
        let text = doc
            .pages
            .get(page as usize)
            .cloned()
            .unwrap_or_default();
        let content_hash = Some(format!("blake3:{}", blake3::hash(text.as_bytes()).to_hex()));
        Ok(ParsedRegion {
            region: region.clone(),
            text,
            cost_tier: CostTier::PageText,
            content_hash,
        })
    }

    fn parse_table(&self, handle: &DocumentHandle, region: &Region) -> Result<TableParse> {
        let doc = self.get(handle)?;
        let page = region.box_.page;
        let (headers, rows) = doc.tables.get(&page).cloned().ok_or_else(|| {
            PerceptionError::InvalidRequest(format!("no table on page {page}"))
        })?;
        let body = format!("{headers:?}\n{rows:?}");
        Ok(TableParse {
            region_id: region.id.clone(),
            headers,
            rows,
            box_: region.box_,
            content_hash: Some(format!("blake3:{}", blake3::hash(body.as_bytes()).to_hex())),
        })
    }

    fn parse_chart(&self, handle: &DocumentHandle, region: &Region) -> Result<ChartParse> {
        let doc = self.get(handle)?;
        let page = region.box_.page;
        let title = doc.charts.get(&page).cloned().ok_or_else(|| {
            PerceptionError::InvalidRequest(format!("no chart on page {page}"))
        })?;
        let mut series = BTreeMap::new();
        series.insert(
            "tok_s".into(),
            vec![
                ChartPoint {
                    x: "1".into(),
                    y: "42".into(),
                },
                ChartPoint {
                    x: "4".into(),
                    y: "38".into(),
                },
            ],
        );
        let body = format!("{title}:{series:?}");
        Ok(ChartParse {
            region_id: region.id.clone(),
            title: Some(title),
            chart_type: Some("line".into()),
            series,
            box_: region.box_,
            content_hash: Some(format!("blake3:{}", blake3::hash(body.as_bytes()).to_hex())),
        })
    }

    fn verify_structure(
        &self,
        handle: &DocumentHandle,
        regions: &[ParsedRegion],
        tables: &[TableParse],
    ) -> Result<StructureReport> {
        let doc = self.get(handle)?;
        let mut outline = Vec::new();
        for page in &doc.pages {
            for line in page.lines() {
                let t = line.trim();
                if t.starts_with(|c: char| c.is_ascii_digit()) && t.contains('.') {
                    outline.push(t.to_string());
                }
            }
        }
        let mut issues = Vec::new();
        for t in tables {
            if t.headers.is_empty() {
                issues.push(format!("table {} has empty headers", t.region_id.0));
            }
            let width = t.headers.len();
            for (i, row) in t.rows.iter().enumerate() {
                if row.len() != width {
                    issues.push(format!(
                        "table {} row {i} width {} != headers {width}",
                        t.region_id.0,
                        row.len()
                    ));
                }
            }
        }
        if regions.is_empty() && tables.is_empty() {
            issues.push("no regions or tables to verify".into());
        }
        Ok(StructureReport {
            document_id: handle.id.clone(),
            ok: issues.is_empty(),
            issues,
            outline,
        })
    }

    fn cite_coordinates(
        &self,
        handle: &DocumentHandle,
        box_: crate::coords::CoordBox,
        quote: &str,
        region_id: Option<&RegionId>,
    ) -> Result<Citation> {
        if !box_.is_normalized() {
            return Err(PerceptionError::InvalidRequest(
                "citation box must be normalized 0..=1".into(),
            ));
        }
        let content_hash = format!("blake3:{}", blake3::hash(quote.as_bytes()).to_hex());
        let id_src = format!(
            "{}|{}|{:.4},{:.4},{:.4},{:.4}|{}",
            handle.id, box_.page, box_.x0, box_.y0, box_.x1, box_.y1, content_hash
        );
        let id = format!("cite:{}", blake3::hash(id_src.as_bytes()).to_hex());
        Ok(Citation {
            id,
            document_id: handle.id.clone(),
            box_,
            region_id: region_id.cloned(),
            quote: quote.to_string(),
            content_hash,
            strength: EvidenceStrength::CoordinateBound,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{DocumentTool, ToolRequest};

    #[test]
    fn inspect_is_cheap_and_has_text_layer() {
        let s = StubDocumentService::fixture_paper();
        let h = DocumentHandle::new("doc:fixture-paper");
        let meta = s.inspect(&h).unwrap();
        assert_eq!(meta.page_count, 4);
        assert!(meta.has_text_layer);
        assert!(meta.cheap_text.unwrap().contains("throughput"));
    }

    #[test]
    fn tool_dispatch_inspect_round_trips_json() {
        let s = StubDocumentService::fixture_paper();
        let req = ToolRequest::Inspect {
            handle: DocumentHandle::new("doc:fixture-paper"),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: ToolRequest = serde_json::from_str(&json).unwrap();
        let resp = DocumentTool::dispatch(&s, back).unwrap();
        let resp_json = serde_json::to_string(&resp).unwrap();
        assert!(resp_json.contains("Fixture Throughput Paper"));
    }
}
